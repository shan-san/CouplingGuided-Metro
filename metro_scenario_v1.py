from typing import Dict, List, Optional, Union
import torch
from metro_world import MetroWorldV1
from metro_agent_v1 import MetroAgentV1 as MetroAgent
from linedata import routes
from timetable import TimetableStop
from train_run_config import TrainRunConfig
from buildtopology import MAX_BFS_NODES
from topology.node import is_whitelist_node

class MetroScenarioV1:
    """Metro system scenario class, responsible for managing environment states and interaction rules"""
    
    def __init__(self, **kwargs):
        self.world = None
        self.n_agents = kwargs["n_agents"]
        self.headway = kwargs.get("headway", 240)
        self._max_steps = kwargs.get("max_steps")
        raw_config = kwargs.pop("train_run_config", None)
        if raw_config is None or isinstance(raw_config, TrainRunConfig):
            self.train_run_config: Optional[TrainRunConfig] = raw_config
        elif isinstance(raw_config, dict):
            self.train_run_config = TrainRunConfig(**raw_config)
        else:
            self.train_run_config = None
        self._init_scenario()
        self._assigned_runs: List = []
        self._run_ready: bool = False

    def _init_scenario(self):
        # Initialize state trackers
        self.passenger_count = {}  # Waiting passengers at each station
        self.congestion = {}      # Congestion levels
        self.delays = {}          # Delay situations
        # 缓存：用于观测的BFS相关张量缓存，按节点ID缓存，避免每步重建
        self._obs_cache = {
            'bfs_tree': {},
            'bfs_weight': {},
            'bfs_weight_ub': {},
        }
        self._position_scale: float = 1.0
        self._max_congestion: float = 1.0

    def env_make_world(self, num_envs: int, device: torch.device, **kwargs):
        """Create vectorized world states"""
        self.num_envs = num_envs
        self.device = device
        if "train_run_config" in kwargs and self.train_run_config is None:
            alt_config = kwargs.pop("train_run_config")
            if isinstance(alt_config, TrainRunConfig):
                self.train_run_config = alt_config
            elif isinstance(alt_config, dict):
                self.train_run_config = TrainRunConfig(**alt_config)
        
        # Initialize world state
        self.world = MetroWorldV1(
            num_envs=self.num_envs,
            device=self.device,
            **kwargs
        )

        for i in range(self.n_agents):
            agent = MetroAgent(
                f"agent_{i}",
                num_envs,
                device
            )
            agent.agent_index = i
            agent.route = routes[0]  # 分配默认 route
            # agent.init_random_position(len(self.world.platform_nodes))
            self.world.register_agent(agent)
        self._setup_runs()
        self.env_reset_world_at()

        # 预构建 BFS 特征张量银行，按 node_id 直接索引（缺失ID填零）
        max_id = max(self.world.nodes.keys()) if len(self.world.nodes) else 0
        self._bfs_bank = torch.zeros(max_id + 1, 100, device=device)
        self._bfs_w_bank = torch.zeros(max_id + 1, 100, device=device)
        self._bfs_ub_bank = torch.zeros(max_id + 1, 100, device=device)
        for nid, node in self.world.nodes.items():
            self._bfs_bank[nid] = torch.as_tensor(node.bfs_tree, device=device)
            self._bfs_w_bank[nid] = torch.as_tensor(node.bfs_tree_weight, device=device)
            self._bfs_ub_bank[nid] = torch.as_tensor(node.bfs_tree_weight_upper_bound, device=device)


        return self.world

    def env_reset_world_at(self, env_index=None):
        # init random weights
        self.world.init_random_weights()
        # Regenerate timetable and run assignments on full reset to ensure randomness
        if env_index is None:
            self._setup_runs()
        # Reset train positions
        for agent in self.world.agents:
            agent.reset_state(env_index)

        import random, torch
        world = self.world
        platform_dict = getattr(world, "platform_nodes", None)
        if not platform_dict:
            raise RuntimeError("未找到 world.platform_nodes")

        # 仅白名单平台（按 type 判断）
        allowed_station_ids = [
            nid for nid, node in platform_dict.items()
            if is_whitelist_node(node)
        ]
        if len(allowed_station_ids) < self.n_agents:
            raise RuntimeError(f"白名单站台数量不足用于初始化: 可用={len(allowed_station_ids)}, 需要={self.n_agents}")


        if self._run_ready and self._assigned_runs:
            self._reset_agents_to_runs(env_index)
        else:
            self._reset_agents_randomly(env_index, allowed_station_ids)

    def _setup_runs(self):
        self._assigned_runs = []
        self._run_ready = False

        config = self.train_run_config
        if config is None:
            return

        route_ref = config.route_id if config.route_id is not None else (self.world.routes[0] if self.world.routes else None)
        if route_ref is None:
            return

        timetable = self.world.build_and_store_timetable(
            route_ref=route_ref,
            first_start_time=config.first_start_time,
            headway=config.headway,
            num_runs=self.n_agents,
            loops_per_run=config.loops_per_run,
        )

        if timetable is None or not getattr(timetable, "runs", None):
            return

        self._truncate_initial_segments(timetable)
        valid_runs = [run for run in timetable.runs if getattr(run, "stops", None)]
        if not valid_runs:
            return

        run_lookup = {run.run_id: run for run in valid_runs}
        assignments = []
        for idx, agent in enumerate(self.world.agents):
            run = None
            if config.assignments:
                desired = config.assignments.get(idx)
                if desired is not None:
                    run = run_lookup.get(desired)
            if run is None:
                run = valid_runs[idx % len(valid_runs)]
            agent.assign_run(run, self.world)
            assignments.append(run)

        self._assigned_runs = assignments
        self._run_ready = True

    def _reset_agents_to_runs(self, env_index=None):
        if env_index is None:
            for agent in self.world.agents:
                start_node = agent.get_run_start_node()
                if start_node is None:
                    continue
                positions = torch.full((self.num_envs,), float(start_node), device=self.device)
                agent.init_position(positions)
            self.world.refresh_reservations()
        else:
            for agent in self.world.agents:
                start_node = agent.get_run_start_node()
                if start_node is None:
                    continue
                positions = agent.state.position.clone()
                positions[env_index] = float(start_node)
                agent.set_position(positions, env_mask=env_index)
            self.world.refresh_reservations(env_index)

    def _reset_agents_randomly(self, env_index=None, allowed_station_ids=None):
        import random

        if env_index is None:
            random_positions = []
            for env in range(self.num_envs):
                random_positions.append(random.sample(allowed_station_ids, self.n_agents))
            # 逐agent赋值
            for i, agent in enumerate(self.world.agents):
                positions = torch.zeros(self.num_envs, device=self.device)
                for env in range(self.num_envs):
                    positions[env] = random_positions[env][i]
                agent.init_position(positions)
            self.world.refresh_reservations()

            agent_count = len(self.world.agents)
            if agent_count and getattr(self.world, "routes", None):
                timetable = self.world.build_and_store_timetable(
                    route_ref=self.world.routes[0],
                    first_start_time=0,
                    headway=self.headway,
                    num_runs=agent_count,
                    loops_per_run=5,
                )
                self._truncate_initial_segments(timetable)
        else:
            # 单环境指定索引的重置
            chosen = random.sample(allowed_station_ids, self.n_agents)
            for i, agent in enumerate(self.world.agents):
                t = agent.state.position.clone()
                t[env_index] = chosen[i]
                agent.init_position(t)


    def pre_step(self):
        for agent in self.world.agents:
            potential_next_pos = torch.zeros(self.num_envs, device=self.device)
            for env in range(self.num_envs):
                if agent.action[env].item() == 1:
                    current_pos = agent.state.position[env].item()
                    next_nodes = self.world.get_next_nodes(current_pos)
                    
                    # 修复：检查是否有可用的下一节点
                    if len(next_nodes) > 0:
                        potential_next_pos[env] = next_nodes[0].id
                    else:
                        # 如果没有可用的下一节点，保持在当前位置
                        potential_next_pos[env] = current_pos
                        # print(f"警告: 环境 {env} 中智能体 {agent.name} 在位置 {current_pos} 没有可用的下一节点")
                else:
                    # 动作为停留(0)，保持当前位置
                    potential_next_pos[env] = agent.state.position[env].item()
            agent.potential_step(potential_next_pos.clone())

    def post_step(self):
        pass

    def env_process_action(self, agent: MetroAgent):
        pass

    def _ensure_env_vector(self, value, dtype=torch.float32) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(self.device)
        else:
            tensor = torch.as_tensor(value, device=self.device)

        tensor = tensor.to(dtype)
        tensor = tensor.reshape(-1)

        numel = tensor.numel()
        if numel == self.num_envs:
            return tensor
        if numel == 1:
            return tensor.expand(self.num_envs)

        if numel < self.num_envs:
            padded = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
            padded[:numel] = tensor
            return padded

        return tensor[:self.num_envs]

    def _compute_agent_features(self, agent: MetroAgent) -> Dict[str, torch.Tensor]:
        position = self._ensure_env_vector(agent.state.position)
        dwell_time = self._ensure_env_vector(agent.state.dwell_time)
        expected_dwell = self._ensure_env_vector(agent.state.current_expected_dwell_time)
        arrival_time_vec = self._ensure_env_vector(agent.state.arrival_time)
        schedule_time_vec = self._ensure_env_vector(agent.state.schedule_time)
        # Use this tensor to store the signed gap between arrival time and schedule time.
        # It remains 0 when not on a platform or when no valid run segment exists.
        dwell_ratio = torch.zeros(self.num_envs, device=self.device)

        on_route = self._ensure_env_vector(agent.state.on_assigned_route)
        run_segment_index = self._ensure_env_vector(agent.state.run_segment_index)

        segment_norm = torch.full((self.num_envs,), -1.0, device=self.device)
        segment_progress = torch.zeros(self.num_envs, device=self.device)
        schedule_alignment = torch.zeros(self.num_envs, device=self.device)
        congestion = torch.zeros(self.num_envs, device=self.device)
        is_platform = torch.zeros(self.num_envs, device=self.device)

        plan = getattr(agent, "_active_run_plan", None)
        segments = getattr(plan, "segments", None) if plan is not None else None
        total_segments = len(segments) if segments else 0

        for env_index in range(self.num_envs):
            node_id = int(position[env_index].item())
            if self.world and node_id in self.world.platform_nodes:
                is_platform[env_index] = 1.0

            if self.world:
                congestion[env_index] = float(
                    self.world.get_reservation_overlap_count(agent, env_index)
                )

            if not total_segments:
                continue

            seg_idx = int(run_segment_index[env_index].item())
            if seg_idx < 0 or seg_idx >= total_segments:
                segment_norm[env_index] = -1.0
                continue

            if total_segments > 1:
                segment_norm[env_index] = seg_idx / float(total_segments - 1)
            else:
                segment_norm[env_index] = 1.0

            # Compute arrival-schedule gap only when on a platform and with a valid run segment
            if is_platform[env_index] > 0.0:
                dwell_ratio[env_index] = float(
                    arrival_time_vec[env_index] - schedule_time_vec[env_index]
                )

            relative_time = agent._relative_time(env_index, plan)
            if relative_time is None:
                continue

            segment = segments[seg_idx]
            duration = max(segment.schedule_end - segment.schedule_start, 1.0)
            progress = (relative_time - segment.schedule_start) / duration
            segment_progress[env_index] = float(max(0.0, min(progress, 1.0)))

            time_gap = agent._time_distance(segment, relative_time, plan.cycle_time)
            schedule_alignment[env_index] = float(max(0.0, 1.0 - time_gap / 600.0))

        return {
            "position": position,
            "dwell_time": dwell_time,
            "expected_dwell": expected_dwell,
            "dwell_ratio": dwell_ratio,
            "on_route": on_route,
            "run_segment_index": run_segment_index,
            "segment_norm": segment_norm,
            "segment_progress": segment_progress,
            "schedule_alignment": schedule_alignment,
            "action": self._ensure_env_vector(agent.action),
            "action_result": self._ensure_env_vector(agent.action_result),
            "is_platform": is_platform,
            "congestion": congestion,
        }

    def _compute_conflict_risks(
        self, feature_map: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Identify potential node contention risk between trains.

        A risk is flagged when two trains have overlapping BFS frontiers but
        neither one's current node lies in the other's BFS tree, meaning they
        may converge onto the same downstream node from different edges.
        """

        conflict_map: Dict[str, torch.Tensor] = {
            name: torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            for name in feature_map
        }
        if self.world is None or len(feature_map) < 2:
            return conflict_map

        bfs_cache: Dict[int, frozenset[int]] = {}

        def get_bfs_set(node_id: int) -> frozenset[int]:
            cached = bfs_cache.get(node_id)
            if cached is not None:
                return cached

            if node_id not in self.world.nodes:
                tree = frozenset()
            else:
                node = self.world.get_node(node_id)
                bfs_tree_nodes = getattr(node, "bfs_tree_nodes", None) or []
                tree = frozenset(
                    int(val)
                    for val in bfs_tree_nodes
                    if isinstance(val, (int, float))
                )
            bfs_cache[node_id] = tree
            return tree

        agent_names = list(feature_map.keys())
        for env_index in range(self.num_envs):
            env_nodes: List[tuple[str, int]] = []
            for name in agent_names:
                position_tensor = feature_map[name].get("position")
                if position_tensor is None or env_index >= position_tensor.shape[0]:
                    continue
                value = position_tensor[env_index]
                if not torch.isfinite(value):
                    continue
                node_id = int(value.item())
                if node_id < 0 or node_id not in self.world.nodes:
                    continue
                env_nodes.append((name, node_id))

            if len(env_nodes) < 2:
                continue

            for idx_a in range(len(env_nodes) - 1):
                name_a, node_a = env_nodes[idx_a]
                bfs_a = get_bfs_set(node_a)
                if not bfs_a:
                    continue

                for idx_b in range(idx_a + 1, len(env_nodes)):
                    name_b, node_b = env_nodes[idx_b]
                    bfs_b = get_bfs_set(node_b)
                    if not bfs_b:
                        continue
                    if node_a in bfs_b or node_b in bfs_a:
                        continue
                    if bfs_a.intersection(bfs_b):
                        conflict_map[name_a][env_index] = 1.0
                        conflict_map[name_b][env_index] = 1.0

        return conflict_map

    def _build_central_state(self, feature_list: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        if not feature_list:
            return torch.zeros(self.num_envs, 1, device=self.device)

        position_stack = torch.stack(
            [feat["position"] / max(self._position_scale, 1.0) for feat in feature_list],
            dim=-1,
        )
        dwell_ratio_stack = torch.stack([feat["dwell_ratio"] for feat in feature_list], dim=-1)
        progress_stack = torch.stack([feat["segment_progress"] for feat in feature_list], dim=-1)
        alignment_stack = torch.stack([feat["schedule_alignment"] for feat in feature_list], dim=-1)
        on_route_stack = torch.stack([feat["on_route"] for feat in feature_list], dim=-1)
        action_stack = torch.stack([feat["action"] for feat in feature_list], dim=-1)
        action_result_stack = torch.stack([feat["action_result"] for feat in feature_list], dim=-1)
        congestion_stack = torch.stack([feat["congestion"] for feat in feature_list], dim=-1)

        mean_position = position_stack.mean(dim=-1)
        std_position = torch.nan_to_num(position_stack.std(dim=-1, unbiased=False))
        mean_dwell_ratio = dwell_ratio_stack.mean(dim=-1)
        mean_progress = progress_stack.mean(dim=-1)
        mean_alignment = alignment_stack.mean(dim=-1)
        fraction_on_route = on_route_stack.mean(dim=-1)
        fraction_move_cmd = action_stack.mean(dim=-1)
        fraction_move_success = (action_result_stack == 2).to(torch.float32).mean(dim=-1)
        mean_congestion = (congestion_stack / self._max_congestion).mean(dim=-1)
        max_congestion = (congestion_stack / self._max_congestion).amax(dim=-1)

        central_state = torch.stack(
            [
                mean_position,
                std_position,
                mean_dwell_ratio,
                mean_progress,
                mean_alignment,
                fraction_on_route,
                fraction_move_cmd,
                fraction_move_success,
                mean_congestion,
                max_congestion,
            ],
            dim=-1,
        )
        return central_state.to(torch.float32)

    def observation(self, agent: MetroAgent) -> Dict[str, torch.Tensor]:
        """Generate agent observations"""
        per_agent_features: Dict[str, Dict[str, torch.Tensor]] = {}
        ordered_features: List[Dict[str, torch.Tensor]] = []
        if self.world is not None:
            for other in self.world.agents:
                feats = self._compute_agent_features(other)
                per_agent_features[other.name] = feats
                ordered_features.append(feats)

            conflict_map = self._compute_conflict_risks(per_agent_features)
            for name, risk in conflict_map.items():
                per_agent_features[name]["conflict_risk"] = risk
        else:
            feats = self._compute_agent_features(agent)
            risk = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            feats["conflict_risk"] = risk
            per_agent_features[agent.name] = feats
            ordered_features.append(feats)

        target_features = per_agent_features.get(agent.name)
        if target_features is None:
            target_features = self._compute_agent_features(agent)

        conflict_risk = target_features.get("conflict_risk")
        if conflict_risk is None:
            conflict_risk = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            target_features["conflict_risk"] = conflict_risk

        state = torch.stack(
            [
                target_features["position"],
                target_features["dwell_time"],
                target_features["expected_dwell"],
                target_features["dwell_ratio"],
                target_features["on_route"],
                # target_features["run_segment_index"],
                # target_features["segment_norm"],
                # target_features["segment_progress"],
                target_features["schedule_alignment"],
                target_features["action"],
                target_features["action_result"],
                target_features["is_platform"],
                target_features["congestion"],
                conflict_risk,
            ],
            dim=-1,
        )

        central_state = self._build_central_state(ordered_features)

        bfs_nodes = MAX_BFS_NODES
        bfs_flat_dim = bfs_nodes * bfs_nodes
        bfs_tree = torch.zeros(self.num_envs, bfs_flat_dim, device=self.device)
        for i in range(self.num_envs):
            node_id = int(agent.state.position[i].item())
            node = self.world.get_node(node_id)
            if node is None:
                continue

            adjacency = getattr(node, "bfs_tree", None)
            if adjacency is not None:
                adjacency_tensor = torch.as_tensor(
                    adjacency, dtype=torch.float32, device=self.device
                )
                bfs_tree[i] = adjacency_tensor.reshape(-1)
        return {
            # Train state
            "train_state": state,
            "central_state": central_state,
            "train_bfs_tree": bfs_tree,
            "available_routes_count": agent.get_available_paths_count(),

            # Global state
            # "global_state": torch.stack([
            #     self.world.adjacency_matrix
            # ], dim=-1)

            # Agent's Future
            
        }

    def reward(self, agent: MetroAgent) -> torch.Tensor:
        """Calculate agent rewards"""
        reward = agent.calc_reward()
        done_mask = self.done()
        if done_mask is not None and done_mask.any():
            reward = reward + agent.calc_terminal_reward(done_mask)
        return reward

    def done(self) -> torch.Tensor:
        """Determine if scenario is finished"""
        device = self.device if getattr(self, "device", None) is not None else torch.device("cpu")
        num_envs = getattr(self, "num_envs", 0)
        if num_envs <= 0:
            return torch.zeros(0, dtype=torch.bool, device=device)

        if self.device is None or self.world is None or self._max_steps is None:
            return torch.zeros(num_envs, dtype=torch.bool, device=device)

        sim_time = getattr(self.world, "sim_time", None)
        if not isinstance(sim_time, torch.Tensor):
            return torch.zeros(num_envs, dtype=torch.bool, device=device)

        max_steps_tensor = self._ensure_env_vector(self._max_steps, dtype=torch.float32)
        if max_steps_tensor.numel() == 1:
            max_steps_tensor = max_steps_tensor.expand_as(sim_time)

        return (sim_time >= max_steps_tensor).to(torch.bool)

    def set_max_steps(self, max_steps) -> None:
        self._max_steps = max_steps

    def info(self, agent) -> Dict[str, torch.Tensor]:
        return {
        }



    def _truncate_initial_segments(self, timetable):
        """Trim leading/trailing segments so runs share a common start and end time."""
        if timetable is None or not getattr(timetable, "runs", None):
            return

        runs = [run for run in timetable.runs if getattr(run, "stops", None)]
        if not runs:
            return

        original_starts = [run.start_time for run in runs]
        baseline_start = min(original_starts)
        trim_start = max(original_starts)
        common_end = min(run.stops[-1].departure_time for run in runs)

        if common_end <= trim_start:
            return

        trimmed_runs: list[tuple] = []
        for idx, run in enumerate(runs):
            trimmed: list[TimetableStop] = []
            for stop in run.stops:
                if stop.departure_time <= trim_start:
                    continue
                if stop.arrival_time >= common_end:
                    break

                arrival = max(stop.arrival_time, trim_start) - trim_start
                departure = min(stop.departure_time, common_end) - trim_start
                if departure <= arrival:
                    continue

                trimmed.append(
                    TimetableStop(
                        platform_id=stop.platform_id,
                        arrival_time=arrival,
                        departure_time=departure,
                        platform_name=stop.platform_name,
                    )
                )

            headway_offset = original_starts[idx] - baseline_start
            trimmed_runs.append((run, trimmed, headway_offset))

        seen_platforms: set[int] = set()
        for run, trimmed, headway_offset in trimmed_runs:
            # Drop leading stops until this run starts from an unused platform.
            while trimmed and trimmed[0].platform_id in seen_platforms:
                trimmed.pop(0)

            if trimmed:
                seen_platforms.add(trimmed[0].platform_id)

            run.start_time = headway_offset
            run.stops = trimmed
            if trimmed:
                run.cycle_time = max(0, trimmed[-1].departure_time - headway_offset)
            else:
                run.cycle_time = 0

        timetable.runs = [run for run in runs if run.stops]
