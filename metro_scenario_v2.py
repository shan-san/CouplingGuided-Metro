"""Scenario wiring for the headway-focused MetroAgentV2."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Union

import torch

from metro_agent_v2 import MetroAgentV2, PlatformHeadwayTracker
from metro_world import MetroWorldV1
from linedata import routes


class MetroScenarioV2:
    """Scenario that spawns MetroAgentV2 instances without timetables."""

    def __init__(self, **kwargs):
        self.world: Optional[MetroWorldV1] = None
        self.n_agents = kwargs["n_agents"]
        self.headway = kwargs.get("headway", 240)
        self.device: Optional[torch.device] = None
        self.num_envs: int = 0

        self._headway_tracker: Optional[PlatformHeadwayTracker] = None
        self._position_scale = 1.0
        self._max_congestion = 1.0
        self._max_steps = kwargs.get("max_steps")
        self.use_float_boarding = kwargs.get("use_float_boarding", False)

    # ------------------------------------------------------------------
    # Environment lifecycle
    # ------------------------------------------------------------------
    def env_make_world(self, num_envs: int, device: torch.device, **kwargs) -> MetroWorldV1:
        self.num_envs = int(num_envs)
        self.device = device

        self.world = MetroWorldV1(num_envs=self.num_envs, device=device, **kwargs)
        self._headway_tracker = PlatformHeadwayTracker(self.headway, self.num_envs, device)
        self._headway_tracker.bootstrap(self.world.platform_nodes.keys())

        for index in range(self.n_agents):
            agent = MetroAgentV2(
                name=f"agent_{index}",
                num_envs=self.num_envs,
                device=device,
                headway=self.headway,
                use_float_boarding=self.use_float_boarding,
            )
            agent.agent_index = index
            agent.route = routes[0] if routes else None
            self.world.register_agent(agent)
            agent.set_headway_tracker(self._headway_tracker)

        self.env_reset_world_at()
        if self.world.pending_events is not None:
            self.world.load_events(self.world.pending_events)
        if self.world.dispatch_config is not None and self.world.dispatch is None:
            self.world.configure_dispatch(self.world.dispatch_config)

        if self.world.nodes:
            self._position_scale = float(max(self.world.nodes.keys())) or 1.0
        self._max_congestion = float(max(1, len(self.world.agents) - 1))

        return self.world

    def env_reset_world_at(self, env_index: Optional[int] = None) -> None:
        if self.world is None:
            return

        dispatch_config = self.world.dispatch_config
        if dispatch_config is not None and self.world.num_envs == 1:
            env_index = None
        self.world.dispatch = None
        if dispatch_config is not None:
            for node in self.world.platform_nodes.values():
                node.init_passenger_tensor(self.num_envs, self.device)
        self.world.init_random_weights()
        if env_index is None:
            self.world.current_time = 0
            self.world.disruption_manager.update_disruptions(0)
        if hasattr(self.world, "sim_time") and isinstance(self.world.sim_time, torch.Tensor):
            if env_index is None:
                self.world.sim_time.zero_()
            else:
                idx = int(env_index)
                if 0 <= idx < self.world.sim_time.shape[0]:
                    self.world.sim_time[idx] = 0.0
        if self._headway_tracker is not None:
            self._headway_tracker.reset(env_index)

        for agent in self.world.agents:
            agent.reset_state(env_index)

        self._reset_agents_randomly(env_index)
        if dispatch_config is not None:
            self.world.configure_dispatch(dispatch_config)
            from topology.metro_lines import LINE_SPECS
            if self.world.topology in LINE_SPECS and dispatch_config.get("metadata", {}).get("line_key"):
                from scenarios.prepare_dispatch import initialize_stock
                initialize_stock(self.world)

    def _reset_agents_randomly(self, env_index: Optional[int] = None) -> None:
        if self.world is None:
            return

        # node_ids = list(self.world.nodes.keys())
        node_ids = list(self.world.platform_nodes.keys())
        depots = {d["node_id"] for d in (self.world.dispatch_config or {}).get("depots", {}).values()}
        node_ids = [n for n in node_ids if n not in depots]
        if env_index is None:
            random_positions = []
            for env_idx in range(self.num_envs):
                platform_nodes = node_ids
                if len(platform_nodes) >= self.n_agents:
                    candidates = platform_nodes
                else:
                    candidates = [n for n in self.world.nodes if n not in depots]
                random_positions.append(random.sample(candidates, self.n_agents))
            for agent_idx, agent in enumerate(self.world.agents):
                positions = torch.zeros(
                        self.num_envs,
                        device=self.device,
                    )
                for env_idx in range(self.num_envs):
                    positions[env_idx] = random_positions[env_idx][agent_idx]
                agent.init_position(positions)
            self.world.refresh_reservations()
        else:
            env_index = int(env_index)
            chosen_positions = random.sample(node_ids, len(self.world.agents))
            for agent, node_id in zip(self.world.agents, chosen_positions):
                positions = agent.state.position.clone()
                positions[env_index] = float(node_id)
                agent.set_position(positions, env_mask=env_index)
            self.world.refresh_reservations(env_index)

    # ------------------------------------------------------------------
    # Step coordination
    # ------------------------------------------------------------------
    def pre_step(self) -> None:
        if self.world is None:
            return
        for agent in self.world.agents:
            agent.potential_step()

    def post_step(self) -> None:
        pass

    def env_process_action(self, agent: MetroAgentV2) -> None:
        pass

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def _ensure_env_vector(self, value, dtype=torch.float32) -> torch.Tensor:
        tensor = value.to(self.device) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=self.device)
        tensor = tensor.to(dtype).reshape(-1)
        if tensor.numel() == self.num_envs:
            return tensor
        if tensor.numel() == 1:
            return tensor.expand(self.num_envs)
        if tensor.numel() < self.num_envs:
            padded = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
            padded[: tensor.numel()] = tensor
            return padded
        return tensor[: self.num_envs]

    def _compute_agent_features(self, agent: MetroAgentV2) -> Dict[str, torch.Tensor]:
        position = self._ensure_env_vector(agent.state.position)
        dwell_time = self._ensure_env_vector(agent.state.dwell_time)
        time_since_arrival = self._ensure_env_vector(agent.time_since_arrival)
        dwell_target = self._ensure_env_vector(agent.dwell_target)
        target_departure = self._ensure_env_vector(agent.target_departure_time)
        dwell_min_limit = self._ensure_env_vector(agent.dwell_min_limit)
        dwell_max_limit = self._ensure_env_vector(agent.dwell_max_limit)
        last_dwell_duration = self._ensure_env_vector(agent.last_dwell_duration)
        arrival_headway = self._ensure_env_vector(agent.state.arrival_headway)
        headway_gap = self._ensure_env_vector(agent.headway_gap)
        action = self._ensure_env_vector(agent.action)
        action_result = self._ensure_env_vector(agent.action_result)

        current_time = self._ensure_env_vector(self.world.sim_time, dtype=torch.float32)
        time_to_target = torch.zeros(self.num_envs, device=self.device)
        is_platform = torch.zeros(self.num_envs, device=self.device)
        congestion = torch.zeros(self.num_envs, device=self.device)
        has_target = torch.zeros(self.num_envs, device=self.device)

        platform_nodes = getattr(self.world, "platform_nodes", {}) if self.world else {}

        for env_index in range(self.num_envs):
            node_id = int(position[env_index].item())
            if node_id in platform_nodes:
                is_platform[env_index] = 1.0
            congestion[env_index] = float(self.world.get_reservation_overlap_count(agent, env_index))

            target_value = float(target_departure[env_index].item())
            current_value = float(current_time[env_index].item())
            if target_value > 0.0:
                has_target[env_index] = 1.0
                time_to_target[env_index] = target_value - current_value
            else:
                time_to_target[env_index] = 0.0

        return {
            "position": position,
            "dwell_time": dwell_time,
            "time_since_arrival": time_since_arrival,
            "dwell_target": dwell_target,
            "target_departure": target_departure,
            "time_to_target": time_to_target,
            "headway_gap": headway_gap,
            "dwell_min_limit": dwell_min_limit,
            "dwell_max_limit": dwell_max_limit,
            "last_dwell_duration": last_dwell_duration,
            "arrival_headway": arrival_headway,
            "action": action,
            "action_result": action_result,
            "is_platform": is_platform,
            "congestion": congestion,
            "has_target": has_target,
        }

    def _compute_conflict_risks(
        self, feature_map: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
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
                bfs_cache[node_id] = frozenset()
                return bfs_cache[node_id]
            node = self.world.get_node(node_id)
            bfs_tree_nodes = getattr(node, "bfs_tree_nodes", None) or []
            tree = frozenset(int(val) for val in bfs_tree_nodes if isinstance(val, (int, float)))
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
            return torch.zeros(self.num_envs, 7, device=self.device)

        time_to_target_stack = torch.stack([feat["time_to_target"] for feat in feature_list], dim=-1)
        time_since_arrival_stack = torch.stack([feat["time_since_arrival"] for feat in feature_list], dim=-1)
        dwell_target_stack = torch.stack([feat["dwell_target"] for feat in feature_list], dim=-1)
        dwell_error_stack = time_since_arrival_stack - dwell_target_stack
        headway_gap_stack = torch.stack([feat["headway_gap"] for feat in feature_list], dim=-1)
        action_stack = torch.stack([feat["action"] for feat in feature_list], dim=-1)
        action_result_stack = torch.stack([feat["action_result"] for feat in feature_list], dim=-1)
        congestion_stack = torch.stack([feat["congestion"] for feat in feature_list], dim=-1)

        mean_time_to_target = time_to_target_stack.mean(dim=-1)
        mean_abs_dwell_error = dwell_error_stack.abs().mean(dim=-1)
        mean_headway_gap = headway_gap_stack.mean(dim=-1)
        fraction_move_cmd = action_stack.mean(dim=-1)
        fraction_move_success = (action_result_stack == 2).to(torch.float32).mean(dim=-1)
        mean_congestion = (congestion_stack / max(self._max_congestion, 1.0)).mean(dim=-1)
        max_congestion = (congestion_stack / max(self._max_congestion, 1.0)).amax(dim=-1)

        return torch.stack(
            [
                mean_time_to_target,
                mean_abs_dwell_error,
                mean_headway_gap,
                fraction_move_cmd,
                fraction_move_success,
                mean_congestion,
                max_congestion,
            ],
            dim=-1,
        )

    def observation(self, agent: MetroAgentV2) -> Dict[str, torch.Tensor]:
        if self.world is None:
            raise RuntimeError("World not initialised")

        per_agent_features: Dict[str, Dict[str, torch.Tensor]] = {}
        ordered_features: List[Dict[str, torch.Tensor]] = []

        for other in self.world.agents:
            feats = self._compute_agent_features(other)
            per_agent_features[other.name] = feats
            ordered_features.append(feats)

        # conflict_map = self._compute_conflict_risks(per_agent_features)
        # for name, risk in conflict_map.items():
        #     per_agent_features[name]["conflict_risk"] = risk

        target_features = per_agent_features.get(agent.name)
        if target_features is None:
            target_features = self._compute_agent_features(agent)
        #     target_features["conflict_risk"] = torch.zeros(self.num_envs, device=self.device)

        # conflict_risk = target_features.get("conflict_risk")
        # if conflict_risk is None:
        #     conflict_risk = torch.zeros(self.num_envs, device=self.device)
        #     target_features["conflict_risk"] = conflict_risk

        current_time = self._ensure_env_vector(self.world.sim_time, dtype=torch.float32)

        headway_scale = float(self.headway) if getattr(self, "headway", None) else 0.0
        headway_scale = max(headway_scale, 1.0)

        train_state = torch.stack(
            [
                # target_features["position"],
                target_features["dwell_time"],
                # target_features["time_since_arrival"],
                # target_features["dwell_target"],
                # target_features["time_since_arrival"] - target_features["dwell_target"],
                target_features["dwell_min_limit"],
                target_features["dwell_max_limit"],
                # target_features["last_dwell_duration"],
                # target_features["target_departure"],
                # target_features["time_to_target"],
                # target_features["has_target"],
                # target_features["headway_gap"],
                # target_features["arrival_headway"],
                target_features["is_platform"],
                # target_features["congestion"],
                # target_features["action"],
                # target_features["action_result"],
                # conflict_risk,
                # current_time,
            ],
            dim=-1,
        )

        central_state = self._build_central_state(ordered_features)

        return {
            "train_state": train_state,
            "central_state": central_state,
            "available_routes_count": agent.get_available_paths_count(),
        }

    # ------------------------------------------------------------------
    # API stubs mirroring scenario v1
    # ------------------------------------------------------------------
    def reward(self, agent: MetroAgentV2) -> torch.Tensor:
        reward = agent.calc_reward()
        done_mask = self.done()
        if done_mask is not None and done_mask.any():
            reward = reward + agent.calc_terminal_reward(done_mask)
        return reward

    def done(self) -> torch.Tensor:
        device = self.device if self.device is not None else torch.device("cpu")
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

        if self.world.dispatch:
            from dispatch_timing import decision_seconds
            time_index = sim_time / decision_seconds(self.world)
        else:
            time_index = sim_time
        return (time_index >= max_steps_tensor).to(torch.bool)

    def set_max_steps(self, max_steps) -> None:
        self._max_steps = max_steps

    def info(self, agent: MetroAgentV2) -> Dict[str, torch.Tensor]:
        return {}
