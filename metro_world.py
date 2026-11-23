import random
import os
import copy
import torch
from typing import Dict, List, Optional
import math
from collections import deque
from linedata import platforms, LineDisruption, routes
from buildtopology import build_topology, calc_coordinates_with_networkx, build_adjacency_matrix
from timetable import Timetable, build_timetable as build_route_timetable
from logger import logger
from experiment_scenarios import apply_passenger_variant


FRONT_RESERVE = 5
REAR_RESERVE = 2
SAFE_CONFLICT_SECONDS = 30  # 最小追踪间隔（秒）


class MetroWorldV1:
    """
    The world class for the metro simulation.
    Manages all simulation elements, including trains, stations, tracks, and disruptions.
    """
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        **kwargs
    ):
        self.num_envs = num_envs
        self.device = device
        requested_interval = kwargs.get("default_interval", 1) or 1
        self.second_precision = bool(kwargs.get("second_precision", kwargs.get("dispatch_config") is not None))
        tick_seconds = 1 if self.second_precision else requested_interval
        self.tick_seconds = tick_seconds
        self.decision_seconds = float(kwargs.get("decision_seconds", requested_interval) or 1)
        self.topology = kwargs.get("topology", 1)  # 保存topology编号

        self.nodes, self.platform_nodes, self.edges, self.segments, self.node2segments, self.segment2nodes, self.switch_nodes = build_topology(
            platforms,
            default_interval=tick_seconds,
        )
        self.adjacency_matrix = build_adjacency_matrix(self.nodes, self.edges, sparse=self.second_precision)
        self.nodes = calc_coordinates_with_networkx(self.nodes, self.edges, bounded=self.second_precision)
        # 基础线段缓存（LineSegment 对象集合；动态生成后存放在 self.segments 的 key 中）
        self.base_segments = list(self.segments.keys())
        # 便捷映射：segment_id -> segment 对象
        self._segment_id_map = {seg.id: seg for seg in self.base_segments}
        self._step_segment = {}
        self._edge_stage = {}
        for segment, sequence in self.segment2nodes.items():
            for index, (origin, dest) in enumerate(zip(sequence, sequence[1:])):
                self._step_segment[(origin.id, dest.id)] = segment
                self._edge_stage[(origin.id, dest.id)] = (index, len(sequence) - 1)
        if not self.base_segments:
            logger.warning("警告: base_segments 为空，请检查 routes 定义是否正确。")
        # 预计算：邻接加速结构（按节点ID索引的下一节点ID列表/集合）
        self._max_node_id = max(self.nodes.keys()) if len(self.nodes) else 0
        self._next_nodes_id_map = {nid: [] for nid in self.nodes.keys()}
        for edge in self.edges:
            self._next_nodes_id_map[edge.start_node.id].append(edge.end_node.id)
        # 用于快速判断是否存在边（如需要）
        self._has_edge = {
            start: set(ends) for start, ends in self._next_nodes_id_map.items()
        }
        self.routes = list(routes)
        self.timetables = {}
        self.active_timetable = None
        self.segment_nodes: Dict[tuple[int, int], List[int]] = {}
        self.segment_travel_time: Dict[tuple[int, int], int] = {}
        self._build_segment_lookup()
        self.sim_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # 智能体列表
        self.agents = []
        self._reservations = {}
        self._route_prev_map = {}
        self._build_route_predecessors()
        
        # 时间步计数器
        self.current_time = 0
        
        # 区段中断管理
        self.disruption_manager = LineDisruption()
        self.event_resources = kwargs.get("event_resources", {})
        self.disruption_manager.resources = self.event_resources
        self.clock_start_seconds = float(kwargs.get("clock_start_seconds", 21600))
        self.boarding_rate_persons_per_second = float(kwargs.get("boarding_rate_persons_per_second", 42))
        self.safe_platform_queue_capacity = float(kwargs.get("safe_platform_queue_capacity", 1200))
        if (not math.isfinite(self.boarding_rate_persons_per_second) or self.boarding_rate_persons_per_second < 0
                or not math.isfinite(self.safe_platform_queue_capacity) or self.safe_platform_queue_capacity < 0):
            raise ValueError("Invalid boarding rate or platform queue capacity")
        for node in self.platform_nodes.values():
            node.passenger_flow.world = self
        
        # 保存中断配置参数
        self.enable_random_blockages = kwargs.get('enable_random_blockages', True)
        self.enable_random_speed_limits = kwargs.get('enable_random_speed_limits', True)
        self.enable_simulation_blockages = kwargs.get('enable_simulation_blockages', False)
        self.enable_simulation_speed_limits = kwargs.get('enable_simulation_speed_limits', False)
        self.verbose = kwargs.get('verbose', False)
        self.custom_disruptions = kwargs.get('custom_disruptions')
        self.passenger_demand_scale = kwargs.get('passenger_demand_scale', 1.0)
        self.passenger_od_variant = kwargs.get('passenger_od_variant', 'baseline')
        self.scenario_name = kwargs.get('scenario_name')

        apply_passenger_variant(self)
        
        # 初始化物理约束
        self._init_constraints()

        # 初始化中断设置
        self._init_disruptions()
        self.pending_events = kwargs.get("events", kwargs.get("events_file", os.environ.get("METRO_EVENTS_FILE")))
        self._configured_events = None
        self.dispatch = None
        self.dispatch_config = kwargs.get("dispatch_config", os.environ.get("METRO_DISPATCH_CONFIG"))
        if isinstance(self.dispatch_config, (str, bytes)) or hasattr(self.dispatch_config, "__fspath__"):
            import json
            with open(self.dispatch_config) as stream:
                self.dispatch_config = json.load(stream)
        
    def _init_constraints(self):
        """初始化物理约束"""
        pass
    
    def _build_route_predecessors(self):
        """Pre-compute previous-node lookups per route."""
        self._route_prev_map = {}
        for node in self.nodes.values():
            next_map = node.getnext() or {}
            for route_key, next_node_id in next_map.items():
                if next_node_id is None:
                    continue
                key = (route_key, next_node_id)
                prev_list = self._route_prev_map.setdefault(key, [])
                if node.id not in prev_list:
                    prev_list.append(node.id)

    def register_agent(self, agent):
        """Register an agent with the world and initialise reservations."""
        if agent not in self.agents:
            self.agents.append(agent)
        self._ensure_agent_entry(agent)
        agent.world = self
        self.refresh_reservations()

    def refresh_reservations(self, env_index=None):
        """Recompute reservation regions for all agents."""
        env_indices = range(self.num_envs) if env_index is None else [env_index]
        for agent in self.agents:
            self._ensure_agent_entry(agent)
            for env in env_indices:
                current_pos = int(agent.state.position[env].item())
                if current_pos not in self.nodes:
                    self._reservations[agent][env] = set()
                    continue
                route_key = self._get_route_key(agent, env, current_pos)
                reservation = self._compute_reservation(agent, env, current_pos, route_key)
                self._reservations[agent][env] = reservation

    def _ensure_agent_entry(self, agent):
        """Ensure reservation storage exists for agent."""
        existing = self._reservations.get(agent)
        if existing is None or len(existing) != self.num_envs:
            self._reservations[agent] = [set() for _ in range(self.num_envs)]

    def _get_route_key(self, agent, env_index: int, node_id: int):
        if node_id not in self.nodes:
            return None

        if self.dispatch:
            nxt = int(agent.potential_next_pos[env_index])
            if nxt != node_id:
                for rid, successor in (self.nodes[node_id].getnext() or {}).items():
                    if successor == nxt:
                        return rid

        route_key = None
        if hasattr(agent, "route_adapter") and agent.route_adapter is not None:
            route_key = agent.route_adapter.ensure_route(agent, env_index, node_id)
        elif hasattr(agent, "route") and agent.route is not None:
            route_key = getattr(agent.route, "id", agent.route)
        return route_key

    def _get_next_on_route(self, current_node_id: int, route_key):
        if route_key is None:
            return None
        node = self.nodes.get(current_node_id)
        if node is None:
            return None
        return node.getnext(route_key)

    def _get_prev_on_route(self, node_id: int, route_key):
        if route_key is None:
            return None
        prev_candidates = self._route_prev_map.get((route_key, node_id), [])
        if not prev_candidates:
            return None
        return prev_candidates[0]

    def _compute_reservation(self, agent, env_index: int, center_node_id: int, route_key=None):
        if center_node_id not in self.nodes:
            return set()
        if self.dispatch and not self.dispatch.blocks_track(agent) and center_node_id == int(agent.state.position[env_index]):
            return set()

        route_key = route_key if route_key is not None else self._get_route_key(agent, env_index, center_node_id)
        region = {center_node_id}

        # Forward reservation
        forward_node = center_node_id
        for _ in range(FRONT_RESERVE):
            next_node = self._get_next_on_route(forward_node, route_key)
            if next_node is None:
                break
            if self.dispatch and not self.dispatch.receiver_allowed(agent.name, next_node):
                break
            if next_node in region:
                break
            region.add(next_node)
            forward_node = next_node

        # Backward reservation
        backward_node = center_node_id
        for _ in range(REAR_RESERVE):
            prev_node = self._get_prev_on_route(backward_node, route_key)
            if prev_node is None:
                break
            if prev_node in region:
                break
            region.add(prev_node)
            backward_node = prev_node

        return region

    def get_reservation_overlap_count(self, agent, env_index: int) -> int:
        """Return how many other agents reserve overlapping nodes with the given agent."""
        if env_index < 0 or env_index >= self.num_envs:
            return 0

        self._ensure_agent_entry(agent)
        regions = self._reservations.get(agent)
        if regions is None or env_index >= len(regions):
            return 0

        region = regions[env_index]
        if not region:
            return 0

        count = 0
        for other_agent, reservation_list in self._reservations.items():
            if other_agent is agent:
                continue
            if env_index >= len(reservation_list):
                continue
            if region.intersection(reservation_list[env_index]):
                count += 1
        return count

    def _build_segment_lookup(self):
        for segment, nodes in self.segment2nodes.items():
            if not nodes:
                continue
            sequence = [node.id for node in nodes]
            start = sequence[0]
            end = sequence[-1]
            self.segment_nodes[(start, end)] = sequence
            self.segment_nodes[(end, start)] = list(reversed(sequence))
            travel_time = getattr(segment, "weight", None)
            if travel_time is not None:
                self.segment_travel_time[(start, end)] = travel_time
                self.segment_travel_time[(end, start)] = travel_time

    def get_segment_sequence(self, start_node_id: int, end_node_id: int) -> Optional[List[int]]:
        return self.segment_nodes.get((start_node_id, end_node_id))

    def sync_sim_time(self, time_tensor, env_index: Optional[int] = None) -> None:
        if env_index is None:
            tensor = time_tensor if isinstance(time_tensor, torch.Tensor) else torch.as_tensor(time_tensor)
            self.sim_time = tensor.to(self.device, dtype=torch.float32)
        else:
            if env_index < 0 or env_index >= self.num_envs:
                return
            if self.sim_time.device != self.device or self.sim_time.shape[0] != self.num_envs:
                self.sim_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            value_tensor = time_tensor if isinstance(time_tensor, torch.Tensor) else torch.as_tensor(time_tensor)
            self.sim_time[env_index] = float(value_tensor)
    
    def _collect_base_segments(self):
        """返回基础线段列表（防御式刷新）。"""
        if not self.base_segments and self.segments:
            self.base_segments = list(self.segments.keys())
            self._segment_id_map = {seg.id: seg for seg in self.base_segments}
        return self.base_segments

    def _get_segment_by_id(self, seg_id: str):
        """按 'start-end' 形式的 segment_id 获取线段对象。"""
        return self._segment_id_map.get(seg_id)

    def _init_disruptions(self):
        """初始化中断设置，确保仿真和随机事件互斥"""
        # 清空现有中断
        self.disruption_manager.clear()
        
        # 优先处理固定的仿真场景
        if self.enable_simulation_blockages or self.enable_simulation_speed_limits:
            self._setup_simulation_disruptions()
            return  # 如果设置了仿真场景，则不应再添加随机事件
        
        # 如果没有仿真场景，再处理随机事件
        if self.enable_random_blockages:
            self._setup_random_blockages()
        if self.enable_random_speed_limits:
            self._setup_random_speed_limits()

    def _setup_simulation_disruptions(self):
        """
        设置固定的、用于测试和评估的线路中断场景。
        支持不同topology的特定配置。
        """
        logger.info(f"正在加载固定的线路中断/限速场景... (Topology: {self.topology})")
        segments = self._collect_base_segments()
        if not segments:
            logger.warning("无法设置固定中断：无可用基础线段。")
            return

        if self.custom_disruptions:
            logger.info("应用自定义中断/限速场景: %s", self.scenario_name or "custom")
            available_ids_once = False
            for config in self.custom_disruptions.get("blockages", []):
                target_segment = self._get_segment_by_id(config["segment_id"])
                if target_segment:
                    self.disruption_manager.add_disruption(
                        target_segment.id, config["start_time"], config["end_time"], "BLOCK"
                    )
                    logger.info(
                        "已添加自定义中断: %s, 时间: %s-%s",
                        target_segment.id,
                        config["start_time"],
                        config["end_time"],
                    )
                elif not available_ids_once:
                    logger.warning(
                        "未找到自定义中断区段 %s；可用: %s",
                        config["segment_id"],
                        sorted(self._segment_id_map.keys()),
                    )
                    available_ids_once = True

            for config in self.custom_disruptions.get("speed_limits", []):
                target_segment = self._get_segment_by_id(config["segment_id"])
                if target_segment:
                    self.disruption_manager.add_disruption(
                        target_segment.id,
                        config["start_time"],
                        config["end_time"],
                        "SPEED_LIMIT",
                        config.get("factor", 2.0),
                    )
                    logger.info(
                        "已添加自定义限速: %s, 因子: %s, 时间: %s-%s",
                        target_segment.id,
                        config.get("factor", 2.0),
                        config["start_time"],
                        config["end_time"],
                    )
                elif not available_ids_once:
                    logger.warning(
                        "未找到自定义限速区段 %s；可用: %s",
                        config["segment_id"],
                        sorted(self._segment_id_map.keys()),
                    )
                    available_ids_once = True
            return
        
        # Topology 3的特殊配置
        if self.topology == 3:
            if self.enable_simulation_blockages:
                logger.info("应用 Topology 3 的固定阻塞配置")
                fixed_blockage_configs = [
                    {"segment_id": "17-19", "start_time": 1000, "end_time": 2000},
                    {"segment_id": "32-33", "start_time": 3000, "end_time": 4000},
                    {"segment_id": "87-48", "start_time": 5000, "end_time": 6000},
                ]
                for config in fixed_blockage_configs:
                    target_segment = self._get_segment_by_id(config["segment_id"])
                    if target_segment:
                        self.disruption_manager.add_disruption(
                            target_segment.id, config['start_time'], config['end_time'], "BLOCK"
                        )
                        logger.info(f"已添加固定中断: {target_segment.id}, 时间: {config['start_time']}-{config['end_time']}")
                    else:
                        logger.warning(f"未找到固定中断区段 {config['segment_id']}")
            
            if self.enable_simulation_speed_limits:
                logger.info("应用 Topology 3 的固定限速配置")
                fixed_speed_limit_configs = [
                    {"segment_id": "13-14", "start_time": 4000, "end_time": 4500, "factor": 2.0},
                ]
                for config in fixed_speed_limit_configs:
                    target_segment = self._get_segment_by_id(config["segment_id"])
                    if target_segment:
                        self.disruption_manager.add_disruption(
                            target_segment.id,
                            config['start_time'],
                            config['end_time'],
                            "SPEED_LIMIT",
                            factor=config['factor']
                        )
                        logger.info(f"已添加固定限速: {target_segment.id}, 时间: {config['start_time']}-{config['end_time']}, 因子: {config['factor']}")
                    else:
                        logger.warning(f"未找到固定限速区段 {config['segment_id']}")
            return
        
        # 其他topology的默认配置
        available_ids_once = False
        if self.enable_simulation_blockages:
            # 定义需要中断的区段ID和时间
            fixed_blockage_configs = [
                {"segment_id": "1-2", "start_time": 1000, "end_time": 2000},
                {"segment_id": "8-14", "start_time": 2000, "end_time": 3000},
            ]
            for config in fixed_blockage_configs:
                target_segment = self._get_segment_by_id(config["segment_id"])
                if target_segment:
                    self.disruption_manager.add_disruption(
                        target_segment.id, config['start_time'], config['end_time'], "BLOCK"
                    )
                    logger.info(f"已添加固定中断: {target_segment.id}, 时间: {config['start_time']}-{config['end_time']}")
                else:
                    if not available_ids_once:
                        logger.warning(f"未找到固定中断区段 {config['segment_id']}；可用: {sorted(self._segment_id_map.keys())}")
                        available_ids_once = True

        if self.enable_simulation_speed_limits:
            # 定义需要限速的区段ID和参数
            fixed_speed_limit_configs = [
                {"segment_id": "3-4", "start_time": 4000, "end_time": 4500, "factor": 2.0},
            ]
            for config in fixed_speed_limit_configs:
                target_segment = self._get_segment_by_id(config["segment_id"])
                if target_segment:
                    self.disruption_manager.add_disruption(
                        target_segment.id,
                        config['start_time'],
                        config['end_time'],
                        "SPEED_LIMIT",
                        config.get("factor", 2.0)
                    )
                    logger.info(f"已添加固定限速: {target_segment.id}, 因子: {config['factor']}, 时间: {config['start_time']}-{config['end_time']}")
                else:
                    if not available_ids_once:
                        logger.warning(f"未找到固定限速区段 {config['segment_id']}；可用: {sorted(self._segment_id_map.keys())}")
                        available_ids_once = True

    def _setup_random_blockages(self):
        """设置随机中断（训练时使用）——改为使用 self.base_segments"""
        segments = self._collect_base_segments()
        if not segments:
            logger.warning("随机中断跳过：无基础线段。")
            return
        count = min(len(segments), random.randint(1, 2))
        for seg in random.sample(segments, count):
            start_time = random.randint(50, 5000)
            # 随机持续时间（500-1000步）
            duration = random.randint(500, 1000)
            end_time = start_time + duration
            self.disruption_manager.add_disruption(seg.id, start_time, end_time, "BLOCK")
            logger.info(f"已添加随机中断: {getattr(seg.start_platform,'id',None)}->{getattr(seg.end_platform,'id',None)} ({seg.id}) 时间:{start_time}-{end_time}")

    def _setup_random_speed_limits(self):
        """设置随机限速（训练时使用）——改为使用 self.base_segments"""
        segments = self._collect_base_segments()
        if not segments:
            logger.warning("随机限速跳过：无基础线段。")
            return
        count = min(len(segments), random.randint(1,1))
        for seg in random.sample(segments, count):
            start_time = random.randint(50, 5000)
            duration = random.randint(500, 1000)
            end_time = start_time + duration
            # 减速因子，例如1.5倍或2倍时间
            speed_factor = random.choice([2.0, 3.0])
            self.disruption_manager.add_disruption(seg.id, start_time, end_time, "SPEED_LIMIT", speed_factor)
            logger.info(f"已添加随机限速: {getattr(seg.start_platform,'id',None)}->{getattr(seg.end_platform,'id',None)} ({seg.id}) 因子:{speed_factor} 时间:{start_time}-{end_time}")

    def reset_world(self):
        """重置世界状态，包括重新设置中断"""
        # 重置时间
        self.current_time = 0
        
        # 重新初始化中断
        self._init_disruptions()
        if self._configured_events is not None:
            self.load_events(self._configured_events, resources=self.event_resources)
        logger.info(f"世界已重置，当前时间: {self.current_time}")
    
    def step(self, *, arrival_inputs=None, dispatch_targets=None):
        if self.dispatch and self.second_precision:
            from dispatch_timing import advance_window
            return advance_window(self, arrival_inputs, dispatch_targets)
        return self._step_physical(arrival_inputs=arrival_inputs, dispatch_targets=dispatch_targets)

    def _step_physical(self, *, arrival_inputs=None, dispatch_targets=None):
        if self.dispatch:
            self.disruption_manager.update_disruptions(float(self.sim_time[0]))
            log_start = len(self.dispatch.log)
            start_time = float(self.sim_time[0])
            self.dispatch.pre_step()
            for node, arrivals in (arrival_inputs or {}).items():
                self.platform_nodes[node].passenger_flow.add_arrivals(arrivals)
            targets = dispatch_targets if dispatch_targets is not None else getattr(self, "_dispatch_stage_targets", None)
            self._dispatch_stage_targets = None
            for name, target in (targets or {}).items():
                agent = self.dispatch.records[name].agent
                current, proposed = int(agent.state.position[0]), int(agent.potential_next_pos[0])
                op = self.dispatch.operation_for(name)
                if op and target not in (current, proposed):
                    raise ValueError("A scheduled movement cannot bypass the active operation phase")
                if target != current and target not in self._has_edge.get(current, ()):
                    raise ValueError("A scheduled movement must follow a directed physical edge")
                agent.potential_next_pos[0] = target
            counts = self.dispatch.objective.passenger_state(self)
            recorder = getattr(getattr(self, "execution_records", None), "result_writer", None)
            if recorder is not None:
                recorder.accrue(counts, float(self.tick_seconds or 1))
            self.last_platform_waiting_person_seconds = (counts["inside"] + counts["outside"]) * float(self.tick_seconds or 1)
            waiting_seconds = self.last_platform_waiting_person_seconds
            onboard_seconds = counts["onboard"] * float(self.tick_seconds or 1)
            for node in self.platform_nodes.values():
                node.passenger_flow.accrue_waiting(float(self.tick_seconds or 1))
            self._transition_arrival_time = start_time + float(self.tick_seconds or 1)
            self._update_train_positions()
            headway = sum(abs(float(a.state.headway_gap[0])) for a in self.agents
                          if int(a.action_result[0]) == 2 and bool(a.state.just_departed[0])
                          and float(a.state.last_departure_time[0]) == start_time)
            self.sim_time.fill_(start_time + float(self.tick_seconds or 1))
            self.current_time = int(round(float(self.sim_time[0]) / float(self.tick_seconds or 1)))
            self.disruption_manager.update_disruptions(float(self.sim_time[0]))
            self.dispatch.post_step()
            started = [e["action"] for e in self.dispatch.log[log_start:] if e["event"] == "command_started"]
            self.dispatch.interval_cost = self.dispatch.objective.interval(
                waiting_seconds, headway, started, onboard_person_seconds=onboard_seconds)
            expected = getattr(self, "_dispatch_expected_positions", None)
            if expected is not None:
                actual = {a.name: int(a.state.position[0]) for a in self.agents}
                self.last_dispatch_execution = dict(positions_match=actual == expected,
                    mismatches={name: dict(expected=pos, actual=actual[name]) for name, pos in expected.items() if actual[name] != pos},
                    constraint_interventions=[name for name, item in self.last_transition_audit.items() if item["conflict"]])
            self._dispatch_expected_positions = None
            self.dispatch.version += 1
            self.refresh_reservations()
            if getattr(self, "execution_records", None) is not None:
                self.execution_records.observe(self)
            return
        self.current_time += 1
        self.sim_time.fill_(self.current_time * float(self.tick_seconds or 1))
        # 更新中断状态
        self.disruption_manager.update_disruptions(float(self.sim_time[0]))
        self._update_train_positions()

    def configure_dispatch(self, config):
        """Opt in to timed operations without replacing the existing agents."""
        import json
        from dispatch_operations import DispatchOperations
        if isinstance(config, (str, bytes)) or hasattr(config, "__fspath__"):
            with open(config) as stream:
                config = json.load(stream)
        if self.current_time != 0 or self.dispatch is not None:
            raise ValueError("Dispatch configuration may only initialize a reset world")
        trial = copy.deepcopy(self)
        trial.dispatch = DispatchOperations(trial, config)
        trial.dispatch.check_invariants()
        rates = {int(p): rate for p, rate in config.get("passenger_arrival_rates", {}).items()}
        if any(p not in self.platform_nodes or isinstance(rate, bool) or not isinstance(rate, (int, float))
               or not math.isfinite(rate) or rate < 0 for p, rate in rates.items()):
            raise ValueError("Current platform arrival rates must be finite nonnegative persons/second")
        self.dispatch_config = copy.deepcopy(config)
        self.execution_records = None
        self.dispatch = DispatchOperations(self, config)
        for p, rate in rates.items():
            self.platform_nodes[p].passenger_flow.arrival_rate = float(rate)
        self.dispatch.check_invariants()
        self.refresh_reservations()

    def submit_dispatch_plan(self, plan):
        if self.dispatch is None:
            raise ValueError("Configure structured dispatch operations first")
        return self.dispatch.submit_plan(plan)

    def dispatch_snapshot(self):
        if self.dispatch is None:
            raise ValueError("Configure structured dispatch operations first")
        return self.dispatch.snapshot(published=True)

    def load_events(self, events, *, resources=None, replace=True):
        """Validate the whole batch before changing the live event registry."""
        from delay_events import DelayEvents, read_event_file
        if isinstance(events, (str, bytes)) or hasattr(events, "__fspath__"):
            data = read_event_file(events)
            if data.get("clock_start_seconds", self.clock_start_seconds) != self.clock_start_seconds:
                raise ValueError("Event file and simulator use different operating clock origins")
            resources = data.get("resources", resources)
            events = data["events"]
        manager = DelayEvents()
        manager.resources = copy.deepcopy(resources if resources is not None else self.event_resources)
        routes_by_id = {route.id: route for route in self.routes}
        for spec in manager.resources.get("switches", {}).values():
            entries = spec.setdefault("route_entry_nodes", {})
            for rid in spec.get("route_ids", []):
                route = routes_by_id.get(rid)
                # A macroscopic turnback has a different entry platform in
                # each direction; node_id identifies the station, not both gates.
                origin = (route.node_list[0].id if route and (route.name or "").startswith("turnback")
                          else spec["node_id"])
                if str(rid) in entries and entries[str(rid)] != origin:
                    raise ValueError("Switch route entry disagrees with the physical topology")
                if origin not in self.nodes or rid not in (self.nodes[origin].getnext() or {}):
                    raise ValueError("Switch route has no physical entry at its declared node")
                entries[str(rid)] = origin
        records = list(events) if replace else list(self.disruption_manager.disruptions.values()) + list(events)
        manager.add_events(records)
        targets = {"segment": set(self._segment_id_map), "platform": set(self.platform_nodes),
                   "passenger_origin": set(self.platform_nodes), "vehicle": {a.name for a in self.agents}}
        for group, kind in (("switches", "switch"), ("turnbacks", "turnback"), ("depots", "depot")):
            targets[kind] = set(manager.resources.get(group, {}))
        for e in records:
            target = e["target"]
            if target["kind"] != "vehicle" or self.agents:
                if target["id"] not in targets[target["kind"]]:
                    raise ValueError(f"Unknown {target['kind']} target: {target['id']}")
            if e["type"] == "od_shift" and not set(e["parameters"]["preferred_destination_ids"]) <= set(self.platform_nodes):
                raise ValueError("OD shift references an unknown passenger destination")
            if e["type"] == "switch_failure":
                spec = manager.resources["switches"][target["id"]]
                allowed = set(spec.get("route_ids", self.nodes[spec["node_id"]].getnext()))
                if not set(e["parameters"]["disabled_route_ids"]) <= allowed:
                    raise ValueError("Switch failure references an unknown route")
            if e["type"] == "turnback_capacity_loss":
                base = manager.resources["turnbacks"][target["id"]].get("capacity", 1)
                if e["parameters"]["capacity"] > base:
                    raise ValueError("A turnback capacity loss cannot increase capacity")
        for spec in manager.resources.get("switches", {}).values():
            if spec["node_id"] not in self.nodes:
                raise ValueError("Unknown switch node")
        for group in ("turnbacks", "depots", "shared_routes"):
            for spec in manager.resources.get(group, {}).values():
                ids = spec.get("segments", []) + spec.get("entry_segments", []) + spec.get("exit_segments", [])
                if not set(ids) <= set(self._segment_id_map):
                    raise ValueError("Resource references an unknown directed segment")
                if group in ("turnbacks", "shared_routes"):
                    spec["interior_nodes"] = sorted({n.id for sid in spec["segments"]
                                                      for n in self.segment2nodes[self._segment_id_map[sid]][1:-1]})
        manager.update_disruptions(self.disruption_manager.current_time)
        self.disruption_manager = manager
        self.event_resources = manager.resources
        self._configured_events = copy.deepcopy(list(manager.disruptions.values()))

    def event_view(self, *, published=False):
        return self.disruption_manager.published_view() if published else self.disruption_manager.physical_view()

    def get_published_events(self):
        return self.disruption_manager.published_records()

    def waiting_at(self, node_id, env_index=0):
        flow = self.platform_nodes[node_id].passenger_flow
        return float(flow.waiting_passengers[env_index]) + flow.outside_count(env_index)

    def edge_event_allowed(self, agent, origin, dest, *, view=None, check_load=True, env_index=0, allow_tow=False):
        view = self.event_view() if view is None else view
        if dest not in self._has_edge.get(origin, ()):
            return False
        segment = self._step_segment.get((origin, dest))
        route_ids = [rid for rid, nxt in (self.nodes[origin].getnext() or {}).items() if nxt == dest]
        if not view.edge_allowed(agent.name, origin, dest, segment.id if segment else None, route_ids, allow_immobilized=allow_tow):
            return False
        if view.active("vehicle_breakdown", "vehicle", agent.name) and not allow_tow:
            # A movable faulty train is empty and may only progress towards an
            # explicitly declared depot gate, not continue a passenger loop.
            gates = {self._segment_id_map[sid].end_platform.id
                     for spec in view.resources.get("depots", {}).values() for sid in spec.get("entry_segments", [])}
            if origin in gates or check_load and float(agent.passenger_count[env_index]) > 1e-6:
                return False
            distances = {n: 0 for n in gates}
            queue = deque(gates)
            reverse = {n: [] for n in self.nodes}
            for node, successors in self._next_nodes_id_map.items():
                for successor in successors:
                    reverse[successor].append(node)
            while queue:
                node = queue.popleft()
                for previous in reverse[node]:
                    if previous not in distances:
                        distances[previous] = distances[node] + 1
                        queue.append(previous)
            if distances.get(dest, math.inf) >= distances.get(origin, math.inf):
                return False
        return True

    def minimum_dwell_steps(self, agent, platform_id, *, view=None):
        view = self.event_view() if view is None else view
        base, _, _ = agent._compute_platform_dwell_limits(self.platform_nodes[platform_id])
        return math.ceil(base) + math.ceil(view.dwell_addition(agent.name, platform_id) / float(self.tick_seconds or 1))

    def edge_delay_steps(self, agent, origin, dest, *, view=None):
        view = self.event_view() if view is None else view
        segment = self._step_segment.get((origin, dest))
        if segment is None:
            return 0
        info = view.get_segment_disruption_info(segment.id, agent.name)
        factor = info["factor"] if info else 1.
        if self.dispatch:
            factor = max(factor, self.dispatch.multiplier(agent, segment))
        steps = max(1, len(self.segment2nodes[segment]) - 1)
        if self.second_precision:
            index, steps = self._edge_stage[origin, dest]
            duration = steps * factor + (info["running_time_add_seconds"] if info else 0.)
            # Cumulative rounding preserves ceil(total running seconds), unlike
            # independently rounding a 1.5x multiplier at every one-second edge.
            end = math.ceil((index + 1) * duration / steps - 1e-10)
            start = math.ceil(index * duration / steps - 1e-10)
            if steps > 1:
                first_extra = max(0, math.ceil(duration / steps - 1e-10) - 1)
                if index == 0:
                    return 0
                if index == steps - 1:
                    end += first_extra
            return max(0, end - start - 1)
        extra = (info["running_time_add_seconds"] if info else 0.) / (steps * float(self.tick_seconds or 1))
        return max(0, int(math.ceil(factor - 1 + extra - 1e-10)))
    
    def _can_use_edge(self, edge):
        """
        检查边是否可以通过（考虑中断）。
        使用更高效和准确的查找方法。
        """
        # 一个边连接两个节点，这两个节点必须属于同一个线段。
        # 我们可以通过边的起始节点来找到它所属的线段。
        start_node_id = edge.start_node.id
        
        # 使用 node2segments 映射直接查找线段
        segment = self.node2segments.get(start_node_id)
        if isinstance(segment, list):  # 兼容列表形式
            segment = segment[0] if segment else None
        if segment is None:
            return True
        # 获取该线段的中断信息
        disruption_info = self.disruption_manager.get_segment_disruption_info(segment.id)
        
        # 如果存在中断且类型为'BLOCK'，则该边不可用
        if disruption_info and disruption_info['type'] == 'BLOCK':
            # print(f"DEBUG: Edge {edge.id} ({edge.start_node.id}->{edge.end_node.id}) on segment {segment.id} is BLOCKED.")
            return False
            
        # 否则，该边可用（限速不影响边的可用性，只影响移动后的延误）
        return True
    
    def _get_forward_distance(self, start_node_id, end_node_id, max_dist=30):
        """
        使用广度优先搜索（BFS）重写，以正确处理分支和交汇点。
        计算从 start_node_id 沿着轨道方向到 end_node_id 的前进距离。
        如果 end_node 在 start_node 的前进路径上，则返回距离；否则返回无穷大。
        """
        return self._get_forward_distances(start_node_id, max_dist).get(end_node_id, float('inf'))

    def _get_forward_distances(self, start_node_id, max_dist):
        """One directed BFS for all targets in the same separation check."""
        cache = self.dispatch._forward_distance_cache if self.dispatch else None
        key = (start_node_id, max_dist)
        if cache is not None and key in cache:
            return cache[key]
        queue = deque([start_node_id])
        distances = {start_node_id: 0}
        while queue:
            current_id = queue.popleft()
            distance = distances[current_id]
            if distance >= max_dist:
                continue
            for nid in self._next_nodes_id_map.get(current_id, []):
                if nid not in distances:
                    distances[nid] = distance + 1
                    queue.append(nid)
        if cache is not None:
            if len(cache) >= 8192:
                cache.clear()
            cache[key] = distances
        return distances

    def _update_train_positions(self):
        # for each env, check all the agent's potential next position
        # if there is overlap position, the potential move failed, and the agent's position remains the same
        # if there is no overlap position, the agent's position is updated to the potential next position
        if not self.agents:
            return

        new_positions = []
        num_agents = len(self.agents)
        duplicate_masks = []
        disruption_masks = []  # 记录因中断而无法移动的掩码
        speed_limit_masks = []  # 记录因限速而无法移动的掩码
        dwell_hold_masks = []
        platform_nodes = getattr(self, "platform_nodes", {})
        events = self.event_view()
        self.last_transition_audit = {}
        
        for env in range( self.num_envs):
            # Positions and directed adjacency do not change within this pass.
            # Discard these queries each tick so no dynamic restriction is cached.
            physical_positions = [int(a.state.position[env].item()) for a in self.agents]
            blocking = [not self.dispatch or self.dispatch.blocks_track(a) for a in self.agents]
            forward_queries = {}
            potential_next_pos = torch.zeros(num_agents, device=self.device)
            for i, agent in enumerate(self.agents):
                potential_next_pos[i] = agent.potential_next_pos[env]
            requested_positions = potential_next_pos.clone()
            speed_limit_mask = torch.zeros(len(self.agents), dtype=torch.bool, device=self.device)
            dwell_hold_mask = torch.zeros(len(self.agents), dtype=torch.bool, device=self.device)


            current_positions = []
            current_regions = []
            desired_regions = []
            route_keys = []

            # check if there is any other agent in the same potential next positions
            duplicate_mask = torch.zeros(len(self.agents), dtype=torch.bool, device=self.device)
            disruption_mask = torch.zeros(len(self.agents), dtype=torch.bool, device=self.device)
            dwell_hold_mask = torch.zeros(len(self.agents), dtype=torch.bool, device=self.device)


            
            for i, agent in enumerate(self.agents):

                # 如果智能体当前正处于限速延误中，则强制停留
                if agent.state.speed_limit_delay[env].item() > 0:
                    speed_limit_mask[i] = True
                    agent.state.speed_limit_delay[env] = max(0, agent.state.speed_limit_delay[env] - 1)

                    # 强制停留在当前节点
                    potential_next_pos[i] = agent.state.position[env].item()
                    # 仍需补齐对齐信息，防止 route_keys 等长度不足
                    self._ensure_agent_entry(agent)
                    current_pos_int = int(agent.state.position[env].item())
                    current_positions.append(current_pos_int)
                    route_key = self._get_route_key(agent, env, current_pos_int)
                    route_keys.append(route_key)
                    # 当前位置的 reservation（保持不动）
                    current_regions.append(self._compute_reservation(agent, env, current_pos_int, route_key))
                    # 这些掩码位置保持 False（未参与冲突/中断检测）
                    # 直接进入下一智能体
                    continue

                # 检查智能体是否需要因停站时间不足而继续停留
                current_pos_val = agent.state.position[env].item()
                current_pos_int = int(current_pos_val)
                min_limit = 0.0
                dwell_min_tensor = getattr(agent.state, "dwell_min_limit", None)
                if dwell_min_tensor is not None:
                    min_limit = max(min_limit, float(dwell_min_tensor[env].item()))
                if current_pos_int in platform_nodes:
                    min_limit = max(min_limit, self.minimum_dwell_steps(agent, current_pos_int, view=events))
                if self.dispatch and not self.dispatch.phase_allows_dwell(agent, current_pos_int):
                    min_limit = 0
                dwell_time = float(agent.state.dwell_time[env].item())
                if min_limit > 0.0 and dwell_time + 1e-6 < min_limit and current_pos_int in platform_nodes:
                    dwell_hold_mask[i] = True
                    potential_next_pos[i] = current_pos_val

                # 修正安全距离检查逻辑，使其与智能体的移动意图对齐
                current_pos_id = agent.state.position[env].item()
                potential_next_id = potential_next_pos[i].item()

                # 只在智能体尝试移动时，才检查其前进路径上的冲突
                if current_pos_id != potential_next_id:
                    segment = self._step_segment.get((int(current_pos_id), int(potential_next_id)))
                    required_seconds = events.safety_headway(segment.id if segment else None, SAFE_CONFLICT_SECONDS)
                    conflict_distance_limit = max(1, math.ceil(required_seconds / float(self.tick_seconds or 1)))
                    query = (potential_next_id, conflict_distance_limit)
                    if query not in forward_queries:
                        forward_queries[query] = self._get_forward_distances(*query)
                    distances = forward_queries[query]
                    for j in range(len(self.agents)):
                        if i == j:
                            continue
                        other_agent = self.agents[j]
                        if self.dispatch and (not blocking[j] or self.dispatch.ignore_pair(agent, other_agent)):
                            continue
                        other_agent_pos_id = physical_positions[j]
                        
                        # 计算从智能体i的 *下一个潜在位置* 到智能体j的距离。
                        # 这样可以确保我们只检查选定路径上的冲突。
                        # 我们将最大距离减1，因为我们是从下一个节点开始计算的。
                        forward_distance = distances.get(other_agent_pos_id, float('inf'))
                        
                        # 如果j在i的选定路径前方且距离小于30，则i需要等待
                        # 根据当前拓扑的 tick_seconds 折算需要保持的“节点距离”阈值
                        if forward_distance < conflict_distance_limit:
                             duplicate_mask[i] = True # 标记为冲突，强制等待
                             break # 找到一个就足够了
                
                if duplicate_mask[i]:
                    potential_next_pos[i] = agent.state.position[env].item() # 强制停留

                # 中断检查：直接根据当前节点所属区段判断是否BLOCK，避免扫描所有边
                current_pos = agent.state.position[env].item()
                next_pos = potential_next_pos[i].item()
                if current_pos != next_pos:  # 只有在尝试移动时才检查
                    towing = bool(self.dispatch and self.dispatch.towed(agent.name))
                    if not self.edge_event_allowed(agent, int(current_pos), int(next_pos), view=events, env_index=env, allow_tow=towing):
                        disruption_mask[i] = True
                    if self.dispatch and not self.dispatch.move_allowed(agent, int(current_pos), int(next_pos)):
                        disruption_mask[i] = True
                    if disruption_mask[i]:
                        # A denied route cannot claim its speculative target
                        # while another train is clearing the same junction.
                        potential_next_pos[i] = current_pos

                self._ensure_agent_entry(agent)
                current_pos = int(agent.state.position[env].item())
                current_positions.append(current_pos)
                route_key = self._get_route_key(agent, env, current_pos)
                route_keys.append(route_key)
                current_regions.append(self._compute_reservation(agent, env, current_pos, route_key))

            for i, agent in enumerate(self.agents):
                next_pos = int(potential_next_pos[i].item())
                desired_regions.append(self._compute_reservation(agent, env, next_pos, route_keys[i]))

            allowed = [True] * num_agents

            for i in range(num_agents):
                for j in range(num_agents):
                    if i == j:
                        continue
                    if self.dispatch and self.dispatch.ignore_pair(self.agents[i], self.agents[j]):
                        continue
                    if desired_regions[i].intersection(current_regions[j]):
                        allowed[i] = False
                        break

            # Resolve multiple agents racing for the same target node.
            node_claims: Dict[int, List[int]] = {}

            for i in range(num_agents):
                if not allowed[i]:
                    continue

                next_pos = int(potential_next_pos[i].item())
                if next_pos == current_positions[i]:
                    continue  # Agent stays put; no contention on a freed node

                node_claims.setdefault(next_pos, []).append(i)

            for _, indices in node_claims.items():
                if len(indices) <= 1:
                    continue

                # Give priority to the train that has waited the longest at its node.
                winner = indices[0]
                winner_score = (
                    float(self.agents[winner].state.dwell_time[env].item()),
                    -winner,
                )

                for candidate in indices[1:]:
                    candidate_score = (
                        float(self.agents[candidate].state.dwell_time[env].item()),
                        -candidate,
                    )
                    if candidate_score > winner_score:
                        winner = candidate
                        winner_score = candidate_score

                for candidate in indices:
                    if candidate != winner:
                        allowed[candidate] = False

            for i in range(num_agents):
                if not allowed[i]:
                    continue
                for j in range(i + 1, num_agents):
                    if not allowed[j]:
                        continue
                    if self.dispatch and self.dispatch.ignore_pair(self.agents[i], self.agents[j]):
                        continue
                    if desired_regions[i].intersection(desired_regions[j]):
                        allowed[j] = False

            # Crossovers and depot approaches share their physical conflict capacity.
            shared = list(self.event_resources.get("turnbacks", {}).items())
            shared += list(self.event_resources.get("shared_routes", {}).items())
            for rid, spec in shared:
                interiors = self.dispatch._shared_interiors[rid] if self.dispatch else {
                    n.id for sid in spec["segments"]
                    for n in self.segment2nodes[self._segment_id_map[sid]][1:-1]}
                capacity = (events.turnback_capacity(rid, spec.get("capacity", 1))
                            if rid in self.event_resources.get("turnbacks", {}) else spec.get("capacity", 1))
                occupied = sum(pos in interiors for pos in current_positions)
                entrants = [i for i in range(num_agents) if allowed[i] and current_positions[i] not in interiors
                            and int(potential_next_pos[i]) in interiors]
                if self.dispatch:
                    def group(index):
                        op = self.dispatch.operation_for(self.agents[index].name)
                        return op.id if op and op.action == "rescue_tow" else self.agents[index].name
                    owners = {group(i) for i, pos in enumerate(current_positions) if pos in interiors}
                    for i in entrants:
                        if group(i) not in owners:
                            if len(owners) >= capacity:
                                allowed[i] = False
                            else:
                                owners.add(group(i))
                    continue
                for i in entrants[max(0, capacity - occupied):]:
                    allowed[i] = False

            if self.dispatch:
                self.dispatch.atomic_tow(allowed, current_positions, potential_next_pos,
                    (duplicate_mask, disruption_mask, dwell_hold_mask, speed_limit_mask))
            new_position = torch.zeros(num_agents, device=self.device)

            for i, agent in enumerate(self.agents):
                # 检查是否因为限速而停留
                is_stopped_by_speed_limit = bool(speed_limit_mask[i].item())
                is_stopped_by_min_dwell = bool(dwell_hold_mask[i].item())

                if not duplicate_mask[i].item() and not disruption_mask[i].item() and not dwell_hold_mask[i].item():
                    previous_pos = agent.state.position[env].item()
                    new_pos = potential_next_pos[i].item()
                    new_position[i] = previous_pos
                    if previous_pos != new_pos:
                        next_pos = int(potential_next_pos[i].item())
                        if allowed[i]:
                            new_position[i] = next_pos
                            if next_pos != current_positions[i]:
                                agent.action_result[env] = 2
                        else:
                            agent.action_result[env] = 0
                    else:

                        if is_stopped_by_speed_limit:
                            agent.action_result[env] = 4  # 因限速停留
                        elif disruption_mask[i].item():
                            agent.action_result[env] = 3
                        elif duplicate_mask[i].item():
                            agent.action_result[env] = 0
                        else:
                            agent.action_result[env] = 1  # 原地停留
                    self._reservations[agent][env] = (desired_regions[i] if allowed[i] else current_regions[i]).copy()
                else:
                    previous_pos = agent.state.position[env].item()
                    new_position[i] = previous_pos
                    if is_stopped_by_min_dwell:
                            agent.action_result[env] = 5  # 因不满足最小停站时间停留
                    elif disruption_mask[i].item():
                        agent.action_result[env] = 3  # 因中断而被阻止移动
                    else:
                        agent.action_result[env] = 0  # 因冲突而被阻止移动
                if self.dispatch:
                    self.last_transition_audit[agent.name] = dict(
                        origin=current_positions[i], requested=int(requested_positions[i]), actual=int(new_position[i]),
                        conflict=bool(int(requested_positions[i]) != current_positions[i]
                            and not speed_limit_mask[i] and not dwell_hold_mask[i] and not disruption_mask[i]
                            and (duplicate_mask[i] or not allowed[i])),
                        dwell_wait=bool(dwell_hold_mask[i]), travel_wait=bool(speed_limit_mask[i]),
                        event_or_control_wait=bool(disruption_mask[i]))
            
            duplicate_masks.append(duplicate_mask)
            disruption_masks.append(disruption_mask)
            speed_limit_masks.append(speed_limit_mask)
            dwell_hold_masks.append(dwell_hold_mask)
            new_positions.append(new_position)

            if not new_positions:
                return

            stacked_positions = torch.stack(new_positions)  # (num_envs, num_agents)
            duplicate_tensor = torch.stack(duplicate_masks)  # (num_envs, num_agents)
            disruption_tensor = torch.stack(disruption_masks)  # (num_envs, num_agents)
            speed_limit_tensor = torch.stack(speed_limit_masks)
            dwell_hold_tensor = torch.stack(dwell_hold_masks)


        # transform the duplicate masks
        for agent_idx, agent in enumerate(self.agents):
            agent_positions = stacked_positions[:, agent_idx]  # shape: (num_envs,)
            agent.set_position(agent_positions.to(agent.device))

            agent_duplicate = duplicate_tensor[:, agent_idx]
            agent_disruption = disruption_tensor[:, agent_idx]
            agent_speed_limit = speed_limit_tensor[:, agent_idx]
            agent_dwell_hold = dwell_hold_tensor[:, agent_idx]


            if not hasattr(agent, "disruption_mask"):
                agent.disruption_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=agent.device)
            agent.disruption_mask.copy_(agent_disruption.to(agent.device))

            if not hasattr(agent, "speed_limit_mask"):
                agent.speed_limit_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=agent.device)
            agent.speed_limit_mask.copy_(agent_speed_limit.to(agent.device))

            if not hasattr(agent, "dwell_hold_mask"):
                agent.dwell_hold_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=agent.device)
            agent.dwell_hold_mask.copy_(agent_dwell_hold.to(agent.device))

            for env in range(self.num_envs):
                if agent.action_result[env].item() == 2:
                    previous = int(agent.state.previous_position[env].item())
                    current = int(agent.state.position[env].item())
                    agent.state.speed_limit_delay[env] = 0 if self.dispatch else self.edge_delay_steps(agent, previous, current, view=events)

    def to(self, device: torch.device):
        self.device = device
        for agent in self.agents:
            agent.to(device)

    def _collect_line_segments(self):
        return list(self.segments.keys())

    def _collect_dwell_overrides(self):
        overrides = {}
        for platform_id, node in self.platform_nodes.items():
            dwell = getattr(node, "weight", None)
            if dwell is None:
                continue
            overrides[platform_id] = int(dwell)
        return overrides

    def _resolve_route(self, route_ref):
        if route_ref is None:
            raise ValueError("Route reference cannot be None")
        if hasattr(route_ref, "node_list"):
            return route_ref
        for route in self.routes:
            if route_ref in (getattr(route, "id", None), getattr(route, "name", None)):
                return route
        raise ValueError(f"Unknown route reference: {route_ref!r}")

    def build_timetable(self, route_ref, **kwargs) -> Timetable:
        route = self._resolve_route(route_ref)
        user_overrides = kwargs.pop("dwell_overrides", None)
        dwell_overrides = self._collect_dwell_overrides()
        if user_overrides:
            dwell_overrides.update(user_overrides)

        return build_route_timetable(
            route=route,
            line_segments=self._collect_line_segments(),
            dwell_overrides=dwell_overrides,
            **kwargs,
        )

    def build_and_store_timetable(self, route_ref, **kwargs) -> Timetable:
        timetable = self.build_timetable(route_ref, **kwargs)
        if timetable is not None:
            self.timetables[timetable.route_id] = timetable
            self.active_timetable = timetable
        return timetable

    def get_node(self, node_id):
        return self.nodes[node_id]

    def init_random_weights(self):
        for node in self.platform_nodes.values():
            # generate random weights for each node
            # a int value between 1 and 10
            node.weight = random.randint(30, 60)

    def get_next_nodes(self, current_node):
        """
        Get all possible next nodes that a train can move to from its current position
        考虑中断的情况，返回可用的下一节点
        
        Args:
            current_node: 当前节点ID
            
        Returns:
            list: List of available next nodes considering disruptions
        """
        current_node_obj = self.nodes[current_node]
        next_node_map = current_node_obj.getnext() or {}
        if not next_node_map:
            return []

        possible_nodes = []

        # 首先检查节点是否存在
        if current_node not in self.nodes:
            print(f"错误: 节点 {current_node} 不存在于网络中")
            return possible_nodes

        # 使用预计算邻接映射
        next_ids = self._next_nodes_id_map.get(current_node, [])
        if next_ids:
            for nid in next_ids:
                possible_nodes.append(self.nodes[nid])

        return possible_nodes
    
    def get_disrupted_segments(self):
        """Controller-visible restrictions; simulator truth stays in the manager."""
        view = self.event_view(published=True)
        ids = {e["target"]["id"] for e in view.active(target_kind="segment")}
        return {sid: view.get_segment_disruption_info(sid) for sid in sorted(ids)}
    
    def is_segment_disrupted(self, segment):
        """检查指定区段是否中断"""
        return self.event_view(published=True).get_segment_disruption_info(str(getattr(segment, "id", segment))) is not None
