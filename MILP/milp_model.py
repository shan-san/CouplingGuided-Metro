"""Time-expanded MILP solver for metro dispatching.

This module replaces the original myopic MILP with a rolling-horizon
formulation that keeps track of passenger flows so that we can optimise a
multi-step reward objective.  The solver builds a time-expanded network,
enforces train/track capacity, models passenger waiting, boarding and
alighting, and records per-step rewards so downstream benchmarks can compare
against theoretical optima within the chosen horizon.

此外，本模块支持收集求解过程信息（如变量重要性、约束紧度、分支决策等），
用于生成微调数据，训练LLM学习如何快速引导MILP求解。
"""

from __future__ import annotations

import json
import copy
import math
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from MILP import cplex_backend as mip
import numpy as np
import torch


@dataclass
class MILPSolveTrace:
    """记录MILP求解过程中的关键信息，用于生成微调数据。"""
    
    # 问题规模信息
    num_agents: int = 0
    num_nodes: int = 0
    num_platforms: int = 0
    time_horizon: int = 0
    num_variables: int = 0
    num_constraints: int = 0
    
    # 环境状态摘要
    state_summary: str = ""
    disruptions: List[Dict[str, Any]] = field(default_factory=list)
    agent_positions: Dict[str, int] = field(default_factory=dict)
    platform_waiting: Dict[int, float] = field(default_factory=dict)
    
    # 求解过程信息
    solve_time: float = 0.0
    gap: float = 0.0
    node_count: int = 0  # 分支定界节点数
    iteration_count: int = 0
    objective_value: Optional[float] = None
    
    # 变量分析 - 关键变量的最优值和reduced cost
    key_variables: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # 格式: {"move_0_1_2_0": {"value": 1.0, "rc": 0.0, "is_fixed": False}, ...}
    
    # 约束分析 - 紧约束(binding constraints)
    binding_constraints: List[str] = field(default_factory=list)
    slack_distribution: Dict[str, float] = field(default_factory=dict)
    
    # 求解策略建议 (由后处理生成)
    warm_start_hint: Dict[str, int] = field(default_factory=dict)  # 建议的初始解
    variable_fixing_hint: Dict[str, int] = field(default_factory=dict)  # 建议固定的变量
    priority_hint: Dict[str, int] = field(default_factory=dict)  # 分支优先级建议
    
    # 最终决策
    actions: Dict[str, int] = field(default_factory=dict)
    action_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，便于JSON序列化。"""
        return asdict(self)
    
    def generate_llm_guidance_target(self) -> Dict[str, Any]:
        """
        生成LLM应该学习输出的"求解指导"目标。
        这是微调数据中的target部分。
        """
        guidance = {
            "warm_start": self.warm_start_hint,
            "variable_fixing": self.variable_fixing_hint,
            "branching_priority": self.priority_hint,
            "expected_solve_time": self.solve_time,
            "strategy_reasoning": self._generate_strategy_reasoning()
        }
        return guidance
    
    def _generate_strategy_reasoning(self) -> str:
        """基于求解结果生成策略推理说明。"""
        reasons = []
        
        # 分析中断情况
        if self.disruptions:
            reasons.append(f"存在{len(self.disruptions)}处线路中断，需要重点处理受影响列车的路径规划。")
        
        # 分析拥挤站点
        high_wait_stations = [
            (nid, wait) for nid, wait in self.platform_waiting.items() 
            if wait > 50
        ]
        if high_wait_stations:
            station_str = ", ".join([f"节点{nid}({int(wait)}人)" for nid, wait in high_wait_stations[:3]])
            reasons.append(f"高等待站点: {station_str}，应优先调度列车前往服务。")
        
        # 分析求解效率
        if self.node_count > 1000:
            reasons.append(f"分支定界探索了{self.node_count}个节点，可通过固定部分变量加速。")
        
        if self.gap > 0.01:
            reasons.append(f"求解gap为{self.gap:.2%}，存在优化空间。")
            
        return " ".join(reasons) if reasons else "标准求解，无特殊策略建议。"


def _safe_tensor_item(tensor: Optional[torch.Tensor], index: int = 0, default: float = 0.0) -> float:
    """Safely extract a float from a tensor."""
    if tensor is None:
        return default
    try:
        if tensor.dim() == 0:
            return float(tensor.item())
        if index >= tensor.shape[0]:
            return default
        return float(tensor[index].item())
    except Exception:
        return default


def _safe_reduced_cost(var) -> float:
    """MIP solutions do not provide reduced costs."""
    try:
        return float(var.reduced_cost)
    except Exception:
        return 0.0


def _normalise_distribution(dist: Dict[int, float], exclude: Optional[int] = None) -> Dict[int, float]:
    """Return a probability distribution over destinations."""
    filtered = {k: max(float(v), 0.0) for k, v in dist.items() if k != exclude}
    total = sum(filtered.values())
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in filtered.items()}


class MILPSolver:
    """Long-horizon MILP dispatch solver with passenger flow tracking."""

    # Online hierarchical calls pass focus agents. Keep that subproblem small;
    # full ground-truth MILP calls do not pass focus agents and are unaffected.
    MAX_ONLINE_FOCUS_AGENTS = 12
    MAX_ONLINE_FOCUS_HORIZON = 20

    # Reward weights (match metro_agent_v2 defaults where possible)
    BASE_MOVE_BONUS = 0.05
    NON_PLATFORM_HOLD_PENALTY = 0.05
    PASSENGER_BOARD_REWARD = 0.01
    PASSENGER_ALIGHT_REWARD = 0.01
    PASSENGER_WAIT_PENALTY = 0.005
    DWELL_STEP_SCALE = 0.05
    DWELL_THRESHOLD_PENALTY = 0.05
    DWELL_DEPART_BONUS = 0.05
    DWELL_TOLERANCE = 1e-3
    HEADWAY_MAX_REWARD = 0.1

    def __init__(self, time_horizon: int = 20, time_limit: int = 1000) -> None:
        self.time_horizon = int(time_horizon)
        self.time_limit = int(time_limit)
        self.last_solution: Dict[str, Union[float, Dict[str, Union[int, float]], List[float]]] = {}
        self.last_solve_trace: Optional[MILPSolveTrace] = None  # 新增：保存最近一次求解的trace

        try:
            with mip.Env(empty=True) as env:
                env.start()
            print("CPLEX environment is valid.")
        except mip.SolverError as exc:  # pragma: no cover - environment dependent
            print(f"Error initializing CPLEX: {exc}")
            print("Please ensure you have a valid CPLEX license installed and configured.")
            raise

    # ------------------------------------------------------------------
    # Helpers for passenger flows
    # ------------------------------------------------------------------
    def _expected_arrival(self, *, node, base_time: float, tick_seconds: float, step_index: int) -> float:
        """Declared planning intensity; never read future sampled arrivals."""

        arrival_rate = max(float(getattr(node, "passenger_arrival_rate", 0.0)), 0.0)
        current_time = base_time + step_index * tick_seconds
        seconds_in_day = 24 * 60 * 60
        seconds_since_midnight = current_time % seconds_in_day
        hour = int((seconds_since_midnight // 3600) % 24)

        if 7 <= hour < 9 or 17 <= hour < 19:
            arrival_rate *= 3.0
        elif 11 <= hour < 13:
            arrival_rate *= 1.5

        # 如果时间步长大于1秒，则对期望值按比例缩放
        if tick_seconds > 1:
            arrival_rate *= tick_seconds

        return arrival_rate

    def _build_destination_matrix(
        self,
        *,
        platform_ids: List[int],
        platform_nodes,
    ) -> List[List[float]]:
        """Construct origin-destination probability matrix for passenger boarding."""

        platform_idx = {pid: idx for idx, pid in enumerate(platform_ids)}
        matrix = [[0.0 for _ in platform_ids] for _ in platform_ids]

        for origin_id in platform_ids:
            node = platform_nodes[origin_id]
            dest_dist = dict(getattr(node, "passenger_destinations", {}))
            normalised = _normalise_distribution(dest_dist, exclude=origin_id)

            converted: Dict[int, float] = {}
            for dest_id, prob in normalised.items():
                dest_key: Optional[int] = None
                if isinstance(dest_id, (int, float)):
                    dest_key = int(dest_id)
                elif hasattr(dest_id, "id"):
                    try:
                        dest_key = int(getattr(dest_id, "id"))
                    except (TypeError, ValueError):
                        dest_key = None
                else:
                    try:
                        dest_key = int(str(dest_id))
                    except (TypeError, ValueError):
                        dest_key = None

                if dest_key is None or dest_key not in platform_idx or dest_key == origin_id:
                    continue
                converted[dest_key] = max(float(prob), 0.0)

            row = matrix[platform_idx[origin_id]]

            row_total = sum(converted.values())
            if row_total > 0.0:
                for dest_key, prob in converted.items():
                    row[platform_idx[dest_key]] = prob / row_total
            else:
                candidates = [pid for pid in platform_ids if pid != origin_id]
                if candidates:
                    fallback_prob = 1.0 / len(candidates)
                    for pid in candidates:
                        row[platform_idx[pid]] = fallback_prob
                else:
                    row[platform_idx[origin_id]] = 1.0

        return matrix

    def _rank_focus_agents_for_online_milp(
        self,
        *,
        focus_agent_indices: Set[int],
        agents: Sequence[Any],
        platform_nodes: Dict[int, Any],
        adjacency: Dict[int, List[int]],
        env_index: int,
        limit: int,
    ) -> Set[int]:
        """Shrink pathological SLM focus outputs while preserving critical trains."""
        waiting_by_platform = {
            platform_id: _safe_tensor_item(
                getattr(node, "waiting_passengers", None),
                env_index,
                default=0.0,
            )
            for platform_id, node in platform_nodes.items()
        }
        high_wait_platforms = [
            platform_id
            for platform_id, waiting in sorted(
                waiting_by_platform.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if waiting > 0.0
        ][:8]

        def _distance_to_targets(start_id: int, max_depth: int = 120) -> float:
            if not high_wait_platforms:
                return float(max_depth)
            if start_id in high_wait_platforms:
                return 0.0
            queue = deque([(start_id, 0)])
            visited = {start_id}
            while queue:
                node_id, dist = queue.popleft()
                if dist >= max_depth:
                    continue
                for next_id in adjacency.get(node_id, []):
                    if next_id in high_wait_platforms:
                        return float(dist + 1)
                    if next_id not in visited:
                        visited.add(next_id)
                        queue.append((next_id, dist + 1))
            return float(max_depth)

        scored_agents: List[Tuple[float, int]] = []
        for agent_idx in focus_agent_indices:
            if agent_idx < 0 or agent_idx >= len(agents):
                continue
            agent = agents[agent_idx]
            current_id = int(agent.state.position[env_index].item())
            passenger_count = _safe_tensor_item(getattr(agent, "passenger_count", None), env_index, default=0.0)
            dwell_time = _safe_tensor_item(getattr(agent.state, "dwell_time", None), env_index, default=0.0)
            expected_dwell = _safe_tensor_item(
                getattr(agent.state, "current_expected_dwell_time", None),
                env_index,
                default=0.0,
            )
            local_wait = waiting_by_platform.get(current_id, 0.0)
            nearest_wait_distance = _distance_to_targets(current_id)

            score = 0.0
            score += min(passenger_count, 100.0) * 0.5
            score += min(local_wait, 200.0) * 2.0
            if expected_dwell > 0.0 and dwell_time > expected_dwell:
                score += min(dwell_time - expected_dwell, 300.0)
            if high_wait_platforms:
                score += max(0.0, 120.0 - nearest_wait_distance) * 0.25
            scored_agents.append((-score, agent_idx))

        scored_agents.sort()
        return {agent_idx for _score, agent_idx in scored_agents[: max(1, int(limit))]}

    # ------------------------------------------------------------------
    # Main solve entry point
    # ------------------------------------------------------------------
    def solve(
        self,
        env_state,
        focus_agent_ids: Optional[Sequence[Union[int, str]]] = None,
        focus_node_ids: Optional[Sequence[Union[int, str]]] = None,
        time_horizon: Optional[int] = None,
        time_limit: Optional[int] = None,
        objective_type: str = "minimize_wait_time",
        collect_trace: bool = False,  # 新增：是否收集求解trace用于微调
        llm_guidance: Optional[Dict[str, Any]] = None,  # 新增：LLM提供的求解指导
        fixed_train_trajectories: Optional[Dict[str, Sequence[int]]] = None,
        dispatch_candidates: Optional[Dict[str, Any]] = None,
        passenger_inputs: Optional[List[Dict[int, Dict[int, float]]]] = None,
        candidate_start_stages: Sequence[int] = (0,),
        max_evaluations: Optional[int] = None,
        reference_actions: Optional[Dict[str, Sequence[int]]] = None,
        focus_platform_ids: Optional[Sequence[int]] = None,
        reference_dispatch_packets: Optional[List[Dict[str, Any]]] = None,
        dispatch_optimization_mode: str = "joint",
    ) -> Dict[str, int]:
        """
        求解MILP调度问题。
        
        Args:
            env_state: 环境状态
            focus_agent_ids: 焦点列车ID列表
            focus_node_ids: 焦点节点ID列表
            time_horizon: 规划时间范围
            time_limit: 求解时间限制
            objective_type: 目标类型
            collect_trace: 是否收集求解过程信息用于微调数据生成
            llm_guidance: LLM提供的求解加速指导，包含:
                - warm_start: Dict[str, int] 初始解建议
                - variable_fixing: Dict[str, int] 变量固定建议
                - branching_priority: Dict[str, int] 分支优先级建议
        
        Returns:
            Dict[str, int]: 各列车的动作
        """
        start_time = time.perf_counter()
        dispatch = getattr(env_state.scenario.world, "dispatch", None)
        if dispatch:
            return self._solve_dispatch(env_state.scenario.world, time_horizon=time_horizon,
                time_limit=time_limit, focus=focus_agent_ids, candidates=dispatch_candidates,
                passenger_inputs=passenger_inputs, fixed_paths=fixed_train_trajectories,
                start_stages=candidate_start_stages, max_evaluations=max_evaluations,
                guidance=llm_guidance, collect_trace=collect_trace, reference_actions=reference_actions,
                focus_platforms=focus_platform_ids, reference_packets=reference_dispatch_packets,
                optimization_mode=dispatch_optimization_mode)
        
        # 初始化trace记录器
        trace = MILPSolveTrace() if collect_trace else None

        model = mip.Model("metro_dispatch_long_horizon")
        model.setParam("OutputFlag", 0)

        world = env_state.scenario.world
        agents = world.agents
        nodes = world.nodes
        platform_nodes = world.platform_nodes
        events = world.disruption_manager.published_view()
        if getattr(self, "last_model", None) is not None:
            self.last_model.dispose()
        self.last_model = model
        from delay_events import EVENT_TARGETS
        event_manifest = {
            "model": "metro_dispatch_long_horizon",
            "supported_event_families": sorted(EVENT_TARGETS),
            "published_events": copy.deepcopy(events.records),
            "restriction_continuation": "active published restrictions persist until restoration is reported",
            "demand_input": "declared current arrival intensity and published fresh-OD weights; not realized future arrivals",
        }

        node_ids = sorted(nodes.keys())
        node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}

        platform_ids = sorted(platform_nodes.keys())
        platform_index = {pid: idx for idx, pid in enumerate(platform_ids)}

        def _iter_segments(entry):
            if entry is None:
                return []
            if isinstance(entry, (list, tuple, set)):
                return [seg for seg in entry if seg is not None]
            return [entry]

        adjacency = {nid: [next_node.id for next_node in world.get_next_nodes(nid)] for nid in node_ids}

        horizon = int(time_horizon) if time_horizon is not None else self.time_horizon
        horizon = max(2, horizon)  # ensure we have at least two steps
        num_agents = len(agents)
        num_nodes = len(nodes)
        num_platforms = len(platform_ids)

        tick_seconds = float(getattr(world, "tick_seconds", 1.0) or 1.0)
        env_index = 0  # 当前所有接口均以单环境为主

        if hasattr(world, "sim_time") and isinstance(world.sim_time, torch.Tensor):
            base_time = _safe_tensor_item(world.sim_time, env_index, default=float(getattr(world, "current_time", 0)))
        else:
            base_time = float(getattr(world, "current_time", 0))

        # 焦点智能体解析
        focus_agent_indices: Optional[set[int]] = None
        if focus_agent_ids:
            focus_agent_indices = set()
            for identifier in focus_agent_ids:
                try:
                    if isinstance(identifier, str):
                        match = re.search(r"(\d+)", identifier)
                        if match:
                            idx = int(match.group(1))
                        else:
                            continue
                    else:
                        idx = int(identifier)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < num_agents:
                    focus_agent_indices.add(idx)

            if horizon > self.MAX_ONLINE_FOCUS_HORIZON:
                print(
                    f"[模型优化] focus模式下规划时间域过大: {horizon} -> "
                    f"{self.MAX_ONLINE_FOCUS_HORIZON}"
                )
                horizon = self.MAX_ONLINE_FOCUS_HORIZON

            if len(focus_agent_indices) > self.MAX_ONLINE_FOCUS_AGENTS:
                focus_agent_indices = self._rank_focus_agents_for_online_milp(
                    focus_agent_indices=focus_agent_indices,
                    agents=agents,
                    platform_nodes=platform_nodes,
                    adjacency=adjacency,
                    env_index=env_index,
                    limit=self.MAX_ONLINE_FOCUS_AGENTS,
                )
                print(
                    f"[模型优化] focus agent数量过大，已收缩到 "
                    f"{len(focus_agent_indices)} 个: {sorted(focus_agent_indices)}"
                )

        focus_node_idx: Optional[set[int]] = None
        if focus_node_ids:
            focus_node_idx = set()
            for identifier in focus_node_ids:
                try:
                    nid = int(identifier)
                except (TypeError, ValueError):
                    continue
                if nid in node_index:
                    focus_node_idx.add(node_index[nid])

        guidance_dict = llm_guidance if isinstance(llm_guidance, dict) else {}
        soft_focus_node_weights: Dict[int, float] = {}
        for identifier in guidance_dict.get("soft_focus_node_ids", []) or []:
            try:
                node_id = int(identifier)
            except (TypeError, ValueError):
                continue
            if node_id in node_index:
                soft_focus_node_weights[node_id] = max(soft_focus_node_weights.get(node_id, 0.0), 1.0)
        raw_soft_weights = guidance_dict.get("soft_focus_node_weights", {})
        if isinstance(raw_soft_weights, dict):
            for raw_node_id, raw_weight in raw_soft_weights.items():
                try:
                    node_id = int(raw_node_id)
                    weight = float(raw_weight)
                except (TypeError, ValueError):
                    continue
                if node_id in node_index and weight > 0.0:
                    soft_focus_node_weights[node_id] = max(soft_focus_node_weights.get(node_id, 0.0), weight)

        soft_focus_node_idx = {node_index[node_id] for node_id in soft_focus_node_weights if node_id in node_index}
        try:
            max_search_nodes_per_agent = int(guidance_dict.get("max_search_nodes_per_agent", 80))
        except (TypeError, ValueError):
            max_search_nodes_per_agent = 80
        max_search_nodes_per_agent = max(20, max_search_nodes_per_agent)
        use_terminal_focus_bonus = bool(guidance_dict.get("terminal_focus_bonus", True)) and bool(soft_focus_node_weights)
        if soft_focus_node_weights:
            print(
                "[模型优化] soft focus nodes: "
                f"{len(soft_focus_node_weights)}, max_search_nodes_per_agent={max_search_nodes_per_agent}"
            )

        def _compute_reachable_nodes(agent_idx: int) -> Set[int]:
            """BFS reachable node indices within the current horizon."""
            reachable: Set[int] = set()
            current_pos_id = int(agents[agent_idx].state.position[env_index].item())
            if current_pos_id not in node_index:
                return reachable

            visited: Set[int] = set()
            queue = deque([(current_pos_id, 0)])  # (node_id, steps)
            while queue:
                node_id, steps = queue.popleft()
                if node_id in visited or steps > horizon:
                    continue
                visited.add(node_id)
                reachable.add(node_index[node_id])

                if steps < horizon:
                    for next_id in adjacency.get(node_id, []):
                        if next_id not in visited:
                            queue.append((next_id, steps + 1))
            return reachable

        def _compute_soft_search_nodes(agent_idx: int, reachable: Set[int]) -> Set[int]:
            """Keep the forward path pipe to soft hotspots plus a small halo."""
            if not soft_focus_node_weights or not reachable:
                return reachable

            current_pos_id = int(agents[agent_idx].state.position[env_index].item())
            if current_pos_id not in node_index:
                return reachable

            parents: Dict[int, Optional[int]] = {current_pos_id: None}
            distances: Dict[int, int] = {current_pos_id: 0}
            queue = deque([current_pos_id])
            best_target: Optional[int] = None
            best_score = 0.0
            if current_pos_id in soft_focus_node_weights:
                best_target = current_pos_id
                best_score = float(soft_focus_node_weights[current_pos_id])

            while queue:
                node_id = queue.popleft()
                dist = distances[node_id]
                if dist >= horizon:
                    continue
                for next_id in adjacency.get(node_id, []):
                    if next_id in parents:
                        continue
                    next_idx = node_index.get(next_id)
                    if next_idx is None or next_idx not in reachable:
                        continue
                    parents[next_id] = node_id
                    distances[next_id] = dist + 1
                    weight = soft_focus_node_weights.get(next_id, 0.0)
                    if weight > 0.0:
                        score = weight / max(1.0, float(dist + 1))
                        if score > best_score:
                            best_score = score
                            best_target = next_id
                    queue.append(next_id)

            if best_target is None:
                return reachable

            pipe_node_ids: Set[int] = set()
            cursor: Optional[int] = best_target
            while cursor is not None:
                pipe_node_ids.add(cursor)
                cursor = parents.get(cursor)

            halo_ids = set(pipe_node_ids)
            for node_id in list(pipe_node_ids):
                for next_id in adjacency.get(node_id, []):
                    if next_id in node_index and node_index[next_id] in reachable:
                        halo_ids.add(next_id)

            selected_idx = {node_index[node_id] for node_id in halo_ids if node_id in node_index}
            selected_idx &= reachable
            current_idx = node_index[current_pos_id]
            selected_idx.add(current_idx)

            if len(selected_idx) < min(3, len(reachable)):
                return reachable
            if len(selected_idx) <= max_search_nodes_per_agent:
                return selected_idx

            def _priority(node_idx: int) -> Tuple[int, float, int]:
                node_id = node_ids[node_idx]
                in_pipe = 0 if node_id in pipe_node_ids else 1
                weight = soft_focus_node_weights.get(node_id, 0.0)
                dist = distances.get(node_id, horizon + 1)
                return (in_pipe, -weight, dist)

            trimmed = set(sorted(selected_idx, key=_priority)[:max_search_nodes_per_agent])
            trimmed.add(current_idx)
            return trimmed

        # 确定参与优化的agent集合（如果未指定focus，则优化所有agent）
        active_agent_indices: Set[int] = set(range(num_agents)) if focus_agent_indices is None else set(focus_agent_indices)
        print(f"[模型优化] 参与优化的agent数量: {len(active_agent_indices)}/{num_agents}")

        # 计算每个active agent的可达范围（horizon步内），大幅减少变量数量
        agent_reachable_nodes: Dict[int, Set[int]] = {}
        for agent_idx in active_agent_indices:
            reachable = _compute_reachable_nodes(agent_idx)
            if focus_agent_indices is not None and soft_focus_node_idx:
                reachable = _compute_soft_search_nodes(agent_idx, reachable)
            agent_reachable_nodes[agent_idx] = reachable
            print(f"  agent_{agent_idx}: 可达节点数 {len(reachable)}")
        
        current_node_idx = []
        pos_counts = defaultdict(list)
        # 详细调试：打印每个agent的位置信息
        print(f"[调试] 检查agent位置 (env_index={env_index}):")
        print(f"[调试] 总节点数: {len(nodes)}, 总agent数: {len(agents)}")
        pos_id_list = []
        for agent in agents:
            pos_id = int(agent.state.position[env_index].item())
            pos_id_list.append(pos_id)
            idx = node_index.get(pos_id)
            current_node_idx.append(idx)
            print(f"  {agent.name}: pos_id={pos_id}, node_idx={idx}")
            if idx is not None:
                pos_counts[idx].append(agent.name)
        
        # 检查是否有重复的pos_id
        from collections import Counter
        pos_counter = Counter(pos_id_list)
        duplicates = {pos: count for pos, count in pos_counter.items() if count > 1}
        if duplicates:
            print(f"[严重错误] 发现重复的位置ID: {duplicates}")
            for pos_id, count in duplicates.items():
                agents_at_pos = [agent.name for agent in agents 
                                 if int(agent.state.position[env_index].item()) == pos_id]
                print(f"  位置 {pos_id} 有 {count} 个agent: {agents_at_pos}")
        
        for idx, agent_names in pos_counts.items():
            if len(agent_names) > 1:
                print(f"WARNING: Multiple agents at node {node_ids[idx]} (idx {idx}): {agent_names}")

        # ==== 提前检测 force_dwell 冲突：确定哪些 agent 需要被释放移动 ====
        # 这必须在 node_allowed 和 valid_move_keys 生成之前完成
        force_dwell_conflict_released: Set[int] = set()  # 因force_dwell冲突需要移动的agent
        initial_conflict_released: Set[int] = set()      # 因初始位置重叠需要移动的agent
        force_dwell_conflict_nodes: Set[int] = set()  # 发生冲突的节点idx
        
        # 1. 初始位置重叠检测 (Collision at t=0)
        # 如果多个agent在同一个节点，必须允许它们（全部）移动以解开冲突
        for node_idx, agent_names in pos_counts.items():
            if len(agent_names) > 1:
                print(f"  [初始冲突] 节点idx={node_idx} 有 {len(agent_names)} 个agent，必须允许移动")
                force_dwell_conflict_nodes.add(node_idx)
                # 找出这些agent的index
                for agent_idx in range(num_agents):
                    if current_node_idx[agent_idx] == node_idx:
                        initial_conflict_released.add(agent_idx)

        # 2. Force Dwell 冲突检测
        if horizon > 1:
            # 收集所有需要 force_dwell 的 agent 及其节点
            _force_dwell_agents_at_node: Dict[int, List[int]] = {}
            for agent_idx, agent in enumerate(agents):
                current_node_id = int(agent.state.position[env_index].item())
                if current_node_id not in platform_index:
                    continue
                dwell_time = _safe_tensor_item(getattr(agent.state, "dwell_time", None), env_index, default=0.0)
                expected_dwell = _safe_tensor_item(getattr(agent.state, "current_expected_dwell_time", None), env_index, default=0.0)
                if dwell_time + 1e-6 < expected_dwell:
                    node_idx = node_index[current_node_id]
                    if node_idx not in _force_dwell_agents_at_node:
                        _force_dwell_agents_at_node[node_idx] = []
                    _force_dwell_agents_at_node[node_idx].append(agent_idx)
            
            # 检测冲突
            for node_idx, agents_list in _force_dwell_agents_at_node.items():
                if len(agents_list) > 1:
                    force_dwell_conflict_nodes.add(node_idx)
                    # 第一个保留force_dwell，其余必须移动
                    for agent_idx in agents_list[1:]:
                        force_dwell_conflict_released.add(agent_idx)
                        print(f"  [提前检测] agent_{agent_idx} 因force_dwell冲突将被释放移动 (节点idx={node_idx})")
        
        # 合并所有需要释放的agent
        all_released_agents = force_dwell_conflict_released.union(initial_conflict_released)

        # 如果这些agent不在focus列表中，必须加入，否则无法生成移动变量
        if focus_agent_indices is not None and all_released_agents:
            print(f"  [自动调整] 将冲突涉及的agent加入focus列表: {all_released_agents}")
            before_update = set(focus_agent_indices)
            focus_agent_indices.update(all_released_agents)
            added_agents = focus_agent_indices - before_update
            active_agent_indices = set(focus_agent_indices)
            for agent_idx in added_agents:
                reachable = _compute_reachable_nodes(agent_idx)
                if focus_agent_indices is not None and soft_focus_node_idx:
                    reachable = _compute_soft_search_nodes(agent_idx, reachable)
                agent_reachable_nodes[agent_idx] = reachable
                print(f"  agent_{agent_idx}: 可达节点数 {len(agent_reachable_nodes[agent_idx])}")
            print(f"[模型优化] 调整后参与优化的agent数量: {len(active_agent_indices)}/{num_agents}")

        # 为冲突节点收集相邻节点，供被释放的agent逃离使用
        escape_node_idx: Set[int] = set()
        for conflict_node_idx in force_dwell_conflict_nodes:
            conflict_node_id = node_ids[conflict_node_idx]
            neighbors = adjacency.get(conflict_node_id, [])
            print(f"  [逃离节点] 冲突节点 {conflict_node_id} (idx={conflict_node_idx}) 的相邻节点: {neighbors}")
            for neighbor_id in neighbors:
                neighbor_idx = node_index.get(neighbor_id)
                if neighbor_idx is not None:
                    escape_node_idx.add(neighbor_idx)
        if escape_node_idx:
            print(f"  [逃离节点] 可用的逃离节点idx集合: {escape_node_idx}")
        # ==== 提前检测结束 ====

        def node_allowed(agent_idx: int, n_idx: int) -> bool:
            # 非active agent不参与模型
            if agent_idx not in active_agent_indices:
                return False
            
            # 检查可达范围
            if agent_idx in agent_reachable_nodes:
                if n_idx not in agent_reachable_nodes[agent_idx]:
                    return False
            
            # 检查focus_node限制
            if focus_node_idx is not None:
                current_idx = current_node_idx[agent_idx]
                # 如果agent因冲突被释放，允许它移动到逃离节点
                if agent_idx in all_released_agents:
                    if n_idx in escape_node_idx or n_idx == current_idx:
                        return True
                    # 也允许移动到focus_nodes
                    if n_idx in focus_node_idx:
                        return True
                    return False
                if current_idx is None:
                    return n_idx in focus_node_idx
                return n_idx in focus_node_idx or n_idx == current_idx
            
            return True

        # 处理速度限制与中断 -> 当前实现假设在规划期内保持不变
        edge_travel_time: Dict[Tuple[int, int], int] = {}
        blocked_segments: set[str] = set()
        blocked_edges: set[Tuple[int, int]] = set()

        segment_edge_map: Dict[Tuple[int, int], str] = {}
        segment_nodes = getattr(world, "segment2nodes", {}) or {}
        for segment, node_list in segment_nodes.items():
            if not node_list:
                continue
            for idx in range(len(node_list) - 1):
                start_id = getattr(node_list[idx], "id", None)
                end_id = getattr(node_list[idx + 1], "id", None)
                if start_id is None or end_id is None:
                    continue
                segment_edge_map[(start_id, end_id)] = segment.id

        for edge in world.edges:
            start_id = edge.start_node.id
            end_id = edge.end_node.id
            # The online environment advances one graph edge per simulation tick.
            # Some edge weights are physical seconds or segment metadata, but the
            # MILP state model has no in-transit variable; using weight/tick as a
            # multi-tick travel time leaves the train at no node between departure
            # and arrival and collapses feasible movement into long stays.
            travel_time = 1
            segment_id = segment_edge_map.get((start_id, end_id))

            if segment_id is None:
                segment_entry = world.node2segments.get(start_id)
                if isinstance(segment_entry, list):
                    if segment_entry:
                        segment_id = getattr(segment_entry[0], "id", None)
                elif segment_entry is not None:
                    segment_id = getattr(segment_entry, "id", None)

            if segment_id is not None:
                info = events.get_segment_disruption_info(segment_id)
                if info:
                    disruption_type = info.get("type")
                    if disruption_type == "SPEED_LIMIT":
                        # The simulator applies speed limits after a successful
                        # move as a forced hold on later ticks. Keep the topology
                        # transition itself one tick so MILP actions remain
                        # executable in the same semantics as the environment.
                        pass
                    if disruption_type == "BLOCK":
                        blocked_segments.add(segment_id)
                        blocked_edges.add((start_id, end_id))

            edge_travel_time[(start_id, end_id)] = max(1, travel_time)

        if blocked_segments:
            print(f"MILP模型识别到中断区段: {blocked_segments}")

        valid_move_keys: List[Tuple[int, int, int, int]] = []
        for agent_idx in active_agent_indices:
            for origin_idx in agent_reachable_nodes.get(agent_idx, set()):
                origin_id = node_ids[origin_idx]
                if not node_allowed(agent_idx, origin_idx):
                    continue

                for dest_id in adjacency.get(origin_id, []):
                    dest_idx = node_index[dest_id]
                    if (origin_id, dest_id) in blocked_edges:
                        continue
                    if not node_allowed(agent_idx, dest_idx):
                        continue
                    if not world.edge_event_allowed(agents[agent_idx], origin_id, dest_id, view=events, check_load=False):
                        continue
                    travel_time = edge_travel_time.get((origin_id, dest_id), 1)
                    for t in range(max(0, horizon - travel_time)):
                        valid_move_keys.append((agent_idx, origin_idx, dest_idx, t))
        
        print(f"[模型优化] 生成的move变量数: {len(valid_move_keys)}")

        # ------------------------------
        # Decision variables (只为active agent生成)
        # ------------------------------
        # 生成x变量的合法key
        x_keys = [(agent_idx, node_idx, t) 
                  for agent_idx in active_agent_indices
                  for node_idx in agent_reachable_nodes.get(agent_idx, set())
                  for t in range(horizon)]
        x = model.addVars(x_keys, vtype=mip.BINARY, name="x")
        
        # 生成stay变量的合法key
        stay_keys = [(agent_idx, node_idx, t)
                     for agent_idx in active_agent_indices
                     for node_idx in agent_reachable_nodes.get(agent_idx, set())
                     for t in range(horizon - 1)]
        stay = model.addVars(stay_keys, vtype=mip.BINARY, name="stay")
        
        move = model.addVars(valid_move_keys, vtype=mip.BINARY, name="move")
        move_keys_by_agent_origin_time: Dict[Tuple[int, int, int], List[Tuple[int, int, int, int]]] = defaultdict(list)
        incoming_move_keys_by_agent_dest_time: Dict[Tuple[int, int, int], List[Tuple[int, int, int, int]]] = defaultdict(list)
        for key in valid_move_keys:
            agent_idx, origin_idx, dest_idx, t = key
            move_keys_by_agent_origin_time[(agent_idx, origin_idx, t)].append(key)
            travel_time = edge_travel_time.get((node_ids[origin_idx], node_ids[dest_idx]), 1)
            arrival_t = t + travel_time
            if 0 <= arrival_t < horizon:
                incoming_move_keys_by_agent_dest_time[(agent_idx, dest_idx, arrival_t)].append(key)

        # Every slowed graph-edge transition is followed by a mandatory hold.
        # The same discretisation is used by MetroWorldV1.edge_delay_steps.
        moves_by_agent_time = defaultdict(list)
        for key in valid_move_keys:
            moves_by_agent_time[key[0], key[3]].append(key)
        for key in valid_move_keys:
            i, origin, dest, t = key
            delay = world.edge_delay_steps(agents[i], node_ids[origin], node_ids[dest], view=events)
            for following in range(t + 1, min(horizon - 1, t + delay + 1)):
                model.addConstr(move[key] + model.sum(move[k] for k in moves_by_agent_time[i, following]) <= 1,
                                f"running_time_{i}_{origin}_{dest}_{t}_{following}")
            if node_ids[dest] in platform_nodes:
                minimum = world.minimum_dwell_steps(agents[i], node_ids[dest], view=events)
                for following in range(t + 1, min(horizon - 1, t + max(1, minimum))):
                    model.addConstr(move[key] + model.sum(move[k] for k in moves_by_agent_time[i, following]) <= 1,
                                    f"dwell_after_arrival_{i}_{dest}_{t}_{following}")

        # ------------------------------
        # Safety-distance consistency constraints
        # ------------------------------
        # 环境在执行移动时会检查“前向最小安全间隔”（默认30秒）。
        # 若MILP不显式建模该规则，就会出现“求解器给出move，但环境大量拦截为blocked”。
        safe_conflict_seconds = max([30.] + [events.safety_headway(s.id) for s in world.base_segments])
        conflict_distance_limit = max(1, int(math.ceil(safe_conflict_seconds / max(tick_seconds, 1e-6))))

        # 仅在当前问题相关节点上构造冲突图，避免约束爆炸。
        candidate_node_idx: Set[int] = set()
        for reachable in agent_reachable_nodes.values():
            candidate_node_idx.update(reachable)
        idx_values = current_node_idx.values() if hasattr(current_node_idx, "values") else current_node_idx
        for idx_val in idx_values:
            if idx_val is not None:
                candidate_node_idx.add(idx_val)

        move_dest_indices = {dest_idx for (_, _, dest_idx, _) in valid_move_keys}
        conflict_ahead_by_dest: Dict[int, Set[int]] = {dest_idx: set() for dest_idx in move_dest_indices}

        if hasattr(world, "_get_forward_distance"):
            for dest_idx in move_dest_indices:
                dest_node_id = node_ids[dest_idx]
                for other_idx in candidate_node_idx:
                    other_node_id = node_ids[other_idx]
                    forward_distance = world._get_forward_distance(dest_node_id, other_node_id, conflict_distance_limit)
                    if forward_distance < conflict_distance_limit:
                        conflict_ahead_by_dest[dest_idx].add(other_idx)

        # move[a,o,d,t] + x[b,n,t] <= 1, 当 n 在 d 的前向冲突域内且 b!=a
        # 近似复现环境冲突规则，降低“计划可行但执行被拦截”的落差。
        for key in valid_move_keys:
            agent_idx, _origin_idx, dest_idx, t = key
            conflict_nodes = conflict_ahead_by_dest.get(dest_idx, set())
            if not conflict_nodes:
                continue

            for other_agent_idx in active_agent_indices:
                if other_agent_idx == agent_idx:
                    continue
                for node_idx in conflict_nodes:
                    x_key = (other_agent_idx, node_idx, t)
                    if x_key in x:
                        model.addConstr(
                            move[key] + x[x_key] <= 1,
                            f"safe_conflict_{agent_idx}_{other_agent_idx}_{dest_idx}_{node_idx}_{t}",
                        )

        wait = model.addVars(num_platforms, horizon + 1, lb=0.0, name="wait")
        
        # board/alight/load只为active agent生成
        board_keys = [(agent_idx, p_idx, t)
                      for agent_idx in active_agent_indices
                      for p_idx in range(num_platforms)
                      for t in range(horizon)]
        board = model.addVars(board_keys, lb=0.0, name="board")
        
        alight_keys = [(agent_idx, p_idx, t)
                       for agent_idx in active_agent_indices
                       for p_idx in range(num_platforms)
                       for t in range(horizon)]
        alight = model.addVars(alight_keys, lb=0.0, name="alight")
        
        load_keys = [(agent_idx, t)
                     for agent_idx in active_agent_indices
                     for t in range(horizon + 1)]
        load = model.addVars(load_keys, lb=0.0, name="load")
        
        load_by_dest_keys = [(agent_idx, p_idx, t)
                             for agent_idx in active_agent_indices
                             for p_idx in range(num_platforms)
                             for t in range(horizon + 1)]
        load_by_dest = model.addVars(load_by_dest_keys, lb=0.0, name="destload")
        
        print(f"[模型优化] 变量数统计:")
        print(f"  x: {len(x_keys)}, stay: {len(stay_keys)}, move: {len(valid_move_keys)}")
        print(f"  board: {len(board_keys)}, alight: {len(alight_keys)}")
        print(f"  load: {len(load_keys)}, load_by_dest: {len(load_by_dest_keys)}")

        passenger_capacity = {agent_idx: float(getattr(agents[agent_idx], "passenger_capacity", 0.0)) 
                              for agent_idx in active_agent_indices}
        passenger_capacity = {k: max(v, 0.0) for k, v in passenger_capacity.items()}

        # ------------------------------
        # Initial conditions (只为active agent添加)
        # ------------------------------
        for agent_idx in active_agent_indices:
            agent = agents[agent_idx]
            initial_pos_id = int(agent.state.position[env_index].item())
            if initial_pos_id not in node_index:
                continue
            pos_idx = node_index[initial_pos_id]
            
            # 只约束可达节点
            if (agent_idx, pos_idx, 0) in x:
                model.addConstr(x[agent_idx, pos_idx, 0] == 1, f"initial_pos_{agent_idx}")
            
            for other_idx in agent_reachable_nodes.get(agent_idx, set()):
                if other_idx == pos_idx:
                    continue
                if (agent_idx, other_idx, 0) in x:
                    model.addConstr(x[agent_idx, other_idx, 0] == 0, f"initial_pos_z_{agent_idx}_{other_idx}")

            initial_load = _safe_tensor_item(agent.passenger_count, env_index, default=0.0)
            if (agent_idx, 0) in load:
                model.addConstr(load[agent_idx, 0] == initial_load, f"load_init_{agent_idx}")

            passengers_by_dest = getattr(agent, "passengers_by_destination", {}) or {}
            initial_dest_values = []
            for platform_id in platform_ids:
                dest_tensor = passengers_by_dest.get(platform_id)
                value = _safe_tensor_item(dest_tensor, env_index, default=0.0)
                initial_dest_values.append(max(0.0, value))

            # Normalize initial_dest_values so they sum exactly to initial_load
            current_total = sum(initial_dest_values)
            
            if initial_load <= 1e-6:
                initial_dest_values = [0.0] * len(initial_dest_values)
            elif current_total < 1e-6:
                # Load > 0 but no destination info -> distribute evenly
                if num_platforms > 0:
                    val = initial_load / num_platforms
                    initial_dest_values = [val] * num_platforms
            else:
                # Scale to match initial_load
                scale = initial_load / current_total
                initial_dest_values = [v * scale for v in initial_dest_values]
            
            # Absorb floating point error in the last element
            if num_platforms > 0:
                diff = initial_load - sum(initial_dest_values)
                initial_dest_values[-1] += diff

            for dest_idx, value in enumerate(initial_dest_values):
                if (agent_idx, dest_idx, 0) in load_by_dest:
                    model.addConstr(load_by_dest[agent_idx, dest_idx, 0] == value, f"dest_init_{agent_idx}_{dest_idx}")

        for platform_id in platform_ids:
            p_idx = platform_index[platform_id]
            node = platform_nodes[platform_id]
            initial_wait = _safe_tensor_item(node.waiting_passengers, env_index, default=0.0)
            model.addConstr(wait[p_idx, 0] == initial_wait, f"wait_init_{p_idx}")

        # ------------------------------
        # Constraints: positioning and movement
        # ------------------------------
        # ==== 使用提前检测的 force_dwell 冲突结果 ====
        # all_released_agents 已在 node_allowed 定义前计算
        skip_force_dwell: Set[int] = all_released_agents  # 直接使用提前检测的结果
        
        # 添加限速延误约束：当列车处于限速延误状态时，必须强制停留
        public_cooldown = {}
        if horizon > 1:
            for agent_idx in active_agent_indices:
                agent = agents[agent_idx]
                previous = int(_safe_tensor_item(getattr(agent.state, "previous_position", None), env_index, -1))
                current = int(agent.state.position[env_index])
                elapsed = max(0., _safe_tensor_item(agent.state.dwell_time, env_index) - 1)
                speed_limit_delay = max(0., world.edge_delay_steps(agent, previous, current, view=events) - elapsed)
                public_cooldown[agent_idx] = speed_limit_delay
                if speed_limit_delay > 0:
                    current_node_id = int(agent.state.position[env_index].item())
                    if current_node_id in node_index:
                        node_idx = node_index[current_node_id]
                        # 在延误期间强制停留
                        delay_steps = min(int(math.ceil(speed_limit_delay)), horizon - 1)
                        for t in range(delay_steps):
                            if (agent_idx, node_idx, t) in stay:
                                model.addConstr(stay[agent_idx, node_idx, t] == 1, 
                                              f"speed_limit_delay_{agent_idx}_{t}")
                        print(f"  [限速延误] agent_{agent_idx} 需要强制停留 {delay_steps} 步")
        
        # 添加 force_dwell 约束（跳过冲突的）
        if horizon > 1:
            for agent_idx in active_agent_indices:
                agent = agents[agent_idx]
                current_node_id = int(agent.state.position[env_index].item())
                if current_node_id not in platform_index:
                    continue
                dwell_time = _safe_tensor_item(getattr(agent.state, "dwell_time", None), env_index, default=0.0)
                expected_dwell = max(_safe_tensor_item(getattr(agent.state, "dwell_min_limit", None), env_index),
                                     world.minimum_dwell_steps(agent, current_node_id, view=events))
                if dwell_time + 1e-6 < expected_dwell:
                    if agent_idx in skip_force_dwell:
                        print(f"  [force_dwell] 跳过agent_{agent_idx}的force_dwell约束（冲突释放，需要移动）")
                        continue  # 跳过冲突的agent
                    node_idx = node_index[current_node_id]
                    for t in range(min(horizon - 1, int(math.ceil(expected_dwell - dwell_time)))):
                        if (agent_idx, node_idx, t) in stay:
                            model.addConstr(stay[agent_idx, node_idx, t] == 1, f"force_dwell_{agent_idx}_{t}")

        for agent_idx in active_agent_indices:
            for t in range(horizon):
                # 只在可达节点上求和
                reachable = agent_reachable_nodes.get(agent_idx, set())
                if reachable:
                    x_sum = model.sum(x[agent_idx, node_idx, t]
                                       for node_idx in reachable 
                                       if (agent_idx, node_idx, t) in x)
                    model.addConstr(x_sum == 1, f"position_unique_{agent_idx}_{t}")

        # 决策约束：stay或move（只为active agent）
        for agent_idx in active_agent_indices:
            for t in range(horizon - 1):
                for node_idx in agent_reachable_nodes.get(agent_idx, set()):
                    if (agent_idx, node_idx, t) not in stay:
                        continue
                    if (agent_idx, node_idx, t) not in x:
                        continue
                    
                    decision_expr = stay[agent_idx, node_idx, t]
                    move_sum = model.sum(
                        move[key]
                        for key in move_keys_by_agent_origin_time.get((agent_idx, node_idx, t), [])
                    )
                    decision_expr += move_sum
                    model.addConstr(
                        decision_expr == x[agent_idx, node_idx, t],
                        f"decision_{agent_idx}_{node_idx}_{t}"
                    )

        # 流守恒约束（只为active agent，只在可达节点）
        for agent_idx in active_agent_indices:
            for t in range(1, horizon):
                for node_idx in agent_reachable_nodes.get(agent_idx, set()):
                    if (agent_idx, node_idx, t) not in x:
                        continue
                    
                    incoming = model.sum(
                        move[key]
                        for key in incoming_move_keys_by_agent_dest_time.get((agent_idx, node_idx, t), [])
                    )
                    
                    # stay项
                    stay_term = stay[agent_idx, node_idx, t - 1] if (agent_idx, node_idx, t - 1) in stay else 0
                    model.addConstr(
                        x[agent_idx, node_idx, t] == stay_term + incoming,
                        f"flow_{agent_idx}_{node_idx}_{t}"
                    )

        # Non-focus trains use an explicitly supplied trajectory. Without one,
        # hold their current footprints conservatively, never erase occupancy.
        non_active_occupancy: Dict[Tuple[int, int], int] = {}
        fixed_paths = fixed_train_trajectories or {}
        for i, agent in enumerate(agents):
            if i in active_agent_indices:
                continue
            current = int(agent.state.position[env_index])
            path = list(fixed_paths.get(agent.name, [current] * horizon))
            if len(path) < horizon or path[0] != current:
                raise ValueError("Fixed train trajectory must cover the horizon and start at the current node")
            for origin, dest in zip(path[:horizon], path[1:horizon]):
                if dest != origin and dest not in adjacency.get(origin, []):
                    raise ValueError("Non-directed fixed train trajectory")
                if dest != origin and not world.edge_event_allowed(agent, origin, dest, view=events):
                    raise ValueError("Fixed train trajectory violates a published restriction")
            for t, nid in enumerate(path[:horizon]):
                if nid not in node_index:
                    raise ValueError("Unknown node in fixed train trajectory")
                key = (node_index[nid], t)
                non_active_occupancy[key] = non_active_occupancy.get(key, 0) + 1

        capacity_slacks = []
        capacity_node_indices = set(candidate_node_idx) | {n for n, _ in non_active_occupancy}
        for node_idx in sorted(capacity_node_indices):
            for t in range(1, horizon):
                occupation = model.sum(x[i, node_idx, t] for i in active_agent_indices if (i, node_idx, t) in x)
                model.addConstr(occupation + non_active_occupancy.get((node_idx, t), 0) <= 1,
                                f"node_capacity_{node_idx}_{t}")
        for key in valid_move_keys:
            i, origin, dest, t = key
            if any(non_active_occupancy.get((n, t), 0) for n in conflict_ahead_by_dest.get(dest, {dest})):
                model.addConstr(move[key] == 0, f"fixed_train_separation_{i}_{origin}_{dest}_{t}")

        shared = list(events.resources.get("turnbacks", {}).items())
        shared += list(events.resources.get("shared_routes", {}).items())
        for rid, spec in shared:
            interior = {node_index[n.id] for sid in spec["segments"]
                        for n in world.segment2nodes[world._segment_id_map[sid]][1:-1]}
            initial = sum(idx in interior for idx in current_node_idx)
            capacity = (max(events.turnback_capacity(rid, spec.get("capacity", 1)), initial)
                        if rid in events.resources.get("turnbacks", {}) else spec.get("capacity", 1))
            for t in range(1, horizon):
                use = model.sum(x[i, n, t] for i in active_agent_indices for n in interior if (i, n, t) in x)
                fixed = sum(non_active_occupancy.get((n, t), 0) for n in interior)
                model.addConstr(use + fixed <= capacity, f"turnback_resource_{rid}_{t}")

        # ------------------------------
        # Passenger flow constraints
        # ------------------------------
        dest_matrix = self._build_destination_matrix(platform_ids=platform_ids, platform_nodes=platform_nodes)
        arrivals = []
        fresh_od = {}
        for p, pid in enumerate(platform_ids):
            arrival = [self._expected_arrival(node=platform_nodes[pid], base_time=base_time + world.clock_start_seconds,
                                             tick_seconds=tick_seconds, step_index=t)
                       * events.demand_multiplier(pid) for t in range(horizon)]
            arrivals.append(arrival)
            fresh_od[p] = events.od_weights(pid, {platform_ids[r]: w for r, w in enumerate(dest_matrix[p])})
        qod = model.addVars(num_platforms, num_platforms, horizon + 1, lb=0., name="queue_od")
        outside = model.addVars(num_platforms, num_platforms, horizon + 1, lb=0., name="outside_od")
        admit = model.addVars(num_platforms, num_platforms, horizon, lb=0., name="admit_od")
        bod = model.addVars([(i, p, r, t) for i in active_agent_indices for p in range(num_platforms)
                             for r in range(num_platforms) for t in range(horizon)], lb=0., name="board_od")
        for p, pid in enumerate(platform_ids):
            flow = platform_nodes[pid].passenger_flow
            flow.ensure_od_queues()
            for r, rid in enumerate(platform_ids):
                initial_q = _safe_tensor_item(flow.waiting_by_destination.get(rid), env_index)
                initial_outside = _safe_tensor_item(flow.outside_by_destination.get(rid), env_index)
                model.addConstr(qod[p, r, 0] == initial_q, f"initial_queue_od_{p}_{r}")
                model.addConstr(outside[p, r, 0] == initial_outside, f"initial_outside_od_{p}_{r}")
            queue_cap = events.queue_capacity(pid, world.safe_platform_queue_capacity)
            for t in range(horizon + 1):
                model.addConstr(wait[p, t] == model.sum(qod[p, r, t] for r in range(num_platforms)),
                                f"queue_total_{p}_{t}")
            for t in range(horizon):
                board_total = model.sum(board[i, p, t] for i in active_agent_indices)
                admit_total = model.sum(admit[p, r, t] for r in range(num_platforms))
                if math.isfinite(queue_cap):
                    total_bound = queue_cap + sum(arrivals[p]) + _safe_tensor_item(flow.waiting_passengers, env_index) + flow.outside_count(env_index)
                    room = model.addVar(lb=-total_bound, ub=total_bound, name=f"station_room_{p}_{t}")
                    positive_room = model.addVar(lb=0., ub=total_bound, name=f"station_admission_space_{p}_{t}")
                    model.addConstr(room == queue_cap - wait[p, t] + board_total)
                    model.addGenConstrMax(positive_room, [room], constant=0., name=f"station_space_{p}_{t}")
                    model.addConstr(admit_total <= positive_room, f"station_capacity_{p}_{t}")
                if not events.boarding_allowed(pid):
                    model.addConstr(admit_total == 0, f"closed_station_admission_{p}_{t}")
                for r, rid in enumerate(platform_ids):
                    boarded = model.sum(bod[i, p, r, t] for i in active_agent_indices)
                    demand = arrivals[p][t] * fresh_od[p].get(rid, 0.)
                    model.addConstr(boarded <= qod[p, r, t], f"boarding_demand_{p}_{r}_{t}")
                    model.addConstr(admit[p, r, t] <= outside[p, r, t] + demand, f"admission_demand_{p}_{r}_{t}")
                    model.addConstr(qod[p, r, t + 1] == qod[p, r, t] - boarded + admit[p, r, t],
                                    f"queue_od_balance_{p}_{r}_{t}")
                    model.addConstr(outside[p, r, t + 1] == outside[p, r, t] + demand - admit[p, r, t],
                                    f"outside_od_balance_{p}_{r}_{t}")

        for i in active_agent_indices:
            agent = agents[i]
            capacity = events.vehicle_capacity(agent.name, passenger_capacity[i])
            initial_load = _safe_tensor_item(agent.passenger_count, env_index)
            # Existing passengers survive a capacity reduction; new boarding
            # cannot fill an already overloaded vehicle beyond its new limit.
            load_bound = max(capacity, initial_load)
            for t in range(horizon + 1):
                model.addConstr(load[i, t] <= load_bound, f"cap_{i}_{t}")
                model.addConstr(model.sum(load_by_dest[i, r, t] for r in range(num_platforms)) == load[i, t],
                                f"dest_sum_{i}_{t}")
            for t in range(horizon):
                boards = model.sum(board[i, p, t] for p in range(num_platforms))
                alights = model.sum(alight[i, p, t] for p in range(num_platforms))
                if events.active("vehicle_breakdown", "vehicle", agent.name):
                    for key in moves_by_agent_time[i, t]:
                        model.addConstr(load[i, t] <= load_bound * (1 - move[key]),
                                        f"faulty_vehicle_empty_{i}_{key[1]}_{key[2]}_{t}")
                room = model.addVar(lb=-load_bound, ub=max(load_bound, 1.), name=f"vehicle_room_{i}_{t}")
                positive_room = model.addVar(lb=0., ub=max(load_bound, 1.), name=f"vehicle_boarding_space_{i}_{t}")
                model.addConstr(room == capacity - load[i, t] + alights)
                model.addGenConstrMax(positive_room, [room], constant=0., name=f"vehicle_space_{i}_{t}")
                model.addConstr(boards <= positive_room, f"capacity_fill_{i}_{t}")
                model.addConstr(load[i, t + 1] == load[i, t] - alights + boards, f"load_update_{i}_{t}")
                for p, pid in enumerate(platform_ids):
                    location = x[i, node_index[pid], t] if (i, node_index[pid], t) in x else 0
                    service = events.boarding_allowed(pid, agent.name)
                    model.addConstr(board[i, p, t] <= capacity * location * int(service), f"board_loc_{i}_{p}_{t}")
                    rate = world.boarding_rate_persons_per_second * tick_seconds * events.boarding_multiplier(pid)
                    model.addConstr(board[i, p, t] <= rate * location, f"boarding_rate_{i}_{p}_{t}")
                    model.addConstr(board[i, p, t] == model.sum(bod[i, p, r, t] for r in range(num_platforms)),
                                    f"board_od_sum_{i}_{p}_{t}")
                    model.addConstr(alight[i, p, t] <= load_bound * location, f"alight_loc_{i}_{p}_{t}")
                    model.addConstr(alight[i, p, t] <= load_by_dest[i, p, t], f"alight_dest_{i}_{p}_{t}")
                    model.addConstr(alight[i, p, t] >= load_by_dest[i, p, t] - load_bound * (1 - location),
                                    f"alight_destination_service_{i}_{p}_{t}")
                    inflow = model.sum(bod[i, o, p, t] for o in range(num_platforms))
                    model.addConstr(load_by_dest[i, p, t + 1] == load_by_dest[i, p, t] - alight[i, p, t] + inflow,
                                    f"dest_update_{i}_{p}_{t}")

        # ------------------------------
        # Objective: Sum of per-step rewards
        # ------------------------------
        objective = model.linear_expr()

        # Penalize capacity violations heavily
        for slack in capacity_slacks:
            objective -= 100000.0 * slack

        platform_set = set(platform_ids)

        def _extract_reward_context(agent_idx: int, agent) -> Optional[Dict[str, float]]:
            node_idx = current_node_idx[agent_idx]
            if node_idx is None:
                return None
            node_id = node_ids[node_idx]
            context = {
                "node_idx": node_idx,
                "node_id": node_id,
                "on_platform": node_id in platform_set,
                "current_dwell": _safe_tensor_item(getattr(agent.state, "time_since_arrival", None), env_index, 0.0),
                "dwell_min": _safe_tensor_item(getattr(agent.state, "dwell_min_limit", None), env_index, 0.0),
                "dwell_max": _safe_tensor_item(getattr(agent.state, "dwell_max_limit", None), env_index, 0.0),
                "dwell_target": _safe_tensor_item(getattr(agent.state, "dwell_target", None), env_index, 0.0),
                "headway_gap": _safe_tensor_item(getattr(agent.state, "headway_gap", None), env_index, 0.0),
                "headway_seconds": float(getattr(agent, "headway_seconds", 0.0) or 0.0),
                "waiting_now": 0.0,
            }
            if context["on_platform"]:
                node = platform_nodes.get(node_id)
                waiting_tensor = getattr(node, "waiting_passengers", None)
                context["waiting_now"] = _safe_tensor_item(waiting_tensor, env_index, 0.0)
            return context

        agent_reward_ctx: Dict[int, Optional[Dict[str, float]]] = {}
        for agent_idx in active_agent_indices:
            agent_reward_ctx[agent_idx] = _extract_reward_context(agent_idx, agents[agent_idx])

        stay_reward_coeffs: Dict[Tuple[int, int, int], float] = defaultdict(float)
        if horizon > 1:
            for agent_idx in active_agent_indices:
                ctx = agent_reward_ctx.get(agent_idx)
                if ctx is None or not ctx["on_platform"]:
                    continue
                current_dwell = ctx["current_dwell"]
                dwell_min = ctx["dwell_min"]
                dwell_max = ctx["dwell_max"]
                node_idx = ctx["node_idx"]
                for t in range(horizon - 1):
                    dwell_before = current_dwell + t * tick_seconds
                    dwell_after = dwell_before + tick_seconds
                    if dwell_min > 0.0 and dwell_before + self.DWELL_TOLERANCE < dwell_min:
                        delta = dwell_min - dwell_before
                        step_progress = min(delta, 1.0)
                        bonus = self.DWELL_STEP_SCALE * step_progress
                        stay_reward_coeffs[(agent_idx, node_idx, t)] += bonus
                    if dwell_max > 0.0 and dwell_after - self.DWELL_TOLERANCE > dwell_max:
                        delta = dwell_after - dwell_max
                        step_excess = min(delta, 1.0)
                        penalty = self.DWELL_STEP_SCALE * step_excess
                        stay_reward_coeffs[(agent_idx, node_idx, t)] -= penalty

        stay_reward_by_time: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
        for (agent_idx, node_idx, t), coeff in stay_reward_coeffs.items():
            stay_reward_by_time[t].append((agent_idx, node_idx, coeff))

        move_coeffs: Dict[Tuple[int, int, int, int], float] = {}

        for t in range(horizon):
            # boarding/alighting rewards (只计算active agent)
            objective += self.PASSENGER_BOARD_REWARD * model.sum(
                board[agent_idx, p_idx, t]
                for agent_idx in active_agent_indices
                for p_idx in range(num_platforms)
                if (agent_idx, p_idx, t) in board
            )
            objective += self.PASSENGER_ALIGHT_REWARD * model.sum(
                alight[agent_idx, p_idx, t]
                for agent_idx in active_agent_indices
                for p_idx in range(num_platforms)
                if (agent_idx, p_idx, t) in alight
            )

            # move bonus / non-platform dwell penalty (只计算active agent)
            for key in valid_move_keys:
                i_idx, origin_idx, dest_idx, step = key
                if step != t:
                    continue
                coeff = self.BASE_MOVE_BONUS
                ctx = agent_reward_ctx.get(i_idx)
                if ctx is not None and ctx["on_platform"] and origin_idx == ctx["node_idx"]:
                    dwell_duration = ctx["current_dwell"] + step * tick_seconds
                    dwell_target = ctx["dwell_target"]
                    if dwell_target > 0.0 and dwell_duration > 0.0:
                        dwell_error = dwell_duration - dwell_target
                        abs_error = abs(dwell_error)
                        if abs_error <= self.DWELL_TOLERANCE:
                            coeff += self.DWELL_DEPART_BONUS
                        else:
                            penalty = min(self.DWELL_THRESHOLD_PENALTY, self.DWELL_STEP_SCALE * abs_error)
                            coeff -= penalty
                    headway_seconds = ctx["headway_seconds"]
                    if headway_seconds > 0.0:
                        gap_after_wait = ctx["headway_gap"] + step * tick_seconds
                        
                        # Adjust weight based on objective
                        weight = self.HEADWAY_MAX_REWARD
                        if objective_type == "minimize_headway_deviation":
                            weight = 2.0  # Significantly increase weight for headway deviation

                        gap_score = min(weight, abs(gap_after_wait) / headway_seconds * weight)
                        if gap_after_wait < 0:
                            coeff += gap_score
                        elif gap_after_wait > 0:
                            coeff -= gap_score
                move_coeffs[key] = coeff
                objective += coeff * move[key]

            # 非站台停留惩罚 (只计算active agent)
            for agent_idx in active_agent_indices:
                for node_idx, node_id in enumerate(node_ids):
                    if node_id in platform_set:
                        continue
                    if t < horizon - 1:
                        if (agent_idx, node_idx, t) in stay:
                            objective -= self.NON_PLATFORM_HOLD_PENALTY * stay[agent_idx, node_idx, t]

        for (agent_idx, node_idx, t), coeff in stay_reward_coeffs.items():
            if t < horizon - 1:
                # 检查键是否存在（稀疏变量可能不包含所有组合）
                if (agent_idx, node_idx, t) in stay:
                    objective += coeff * stay[agent_idx, node_idx, t]

        # 等待乘客惩罚：对每个时刻每个站台的等待人数进行惩罚
        # 这会鼓励模型尽早调度列车去接乘客，减少总等待时间
        # 使用 wait[p_idx, t] 变量（线性），避免二次项
        # 对未来时刻应用折扣因子，使近期等待惩罚更重要
        wait_discount_factor = 0.95  # 每步折扣
        for t in range(horizon + 1):
            discount = wait_discount_factor ** t
            for p_idx in range(num_platforms):
                # 对每个站台的等待人数施加惩罚
                # 惩罚 = PASSENGER_WAIT_PENALTY * 等待人数 * 折扣
                objective -= self.PASSENGER_WAIT_PENALTY * discount * wait[p_idx, t]
                objective -= self.PASSENGER_WAIT_PENALTY * discount * model.sum(
                    outside[p_idx, r, t] for r in range(num_platforms))

        if use_terminal_focus_bonus and horizon > 1:
            terminal_t = horizon - 1
            distance_cache: Dict[int, Tuple[float, float]] = {}

            def _soft_focus_potential(node_id: int, max_depth: int = 20) -> Tuple[float, float]:
                if node_id in distance_cache:
                    return distance_cache[node_id]
                if node_id in soft_focus_node_weights:
                    result = (0.0, float(soft_focus_node_weights[node_id]))
                    distance_cache[node_id] = result
                    return result
                visited = {node_id}
                queue = deque([(node_id, 0)])
                best_distance = float("inf")
                best_weight = 0.0
                while queue:
                    current_id, dist = queue.popleft()
                    if dist >= max_depth:
                        continue
                    for next_id in adjacency.get(current_id, []):
                        if next_id in visited:
                            continue
                        next_dist = dist + 1
                        weight = soft_focus_node_weights.get(next_id, 0.0)
                        if weight > 0.0:
                            score = weight / max(1.0, float(next_dist))
                            best_score = best_weight / max(1.0, best_distance) if best_distance < float("inf") else 0.0
                            if score > best_score:
                                best_distance = float(next_dist)
                                best_weight = float(weight)
                        visited.add(next_id)
                        if next_dist < max_depth:
                            queue.append((next_id, next_dist))
                result = (best_distance, best_weight)
                distance_cache[node_id] = result
                return result

            terminal_terms = 0
            for agent_idx in active_agent_indices:
                for node_idx in agent_reachable_nodes.get(agent_idx, set()):
                    x_key = (agent_idx, node_idx, terminal_t)
                    if x_key not in x:
                        continue
                    node_id = node_ids[node_idx]
                    distance, weight = _soft_focus_potential(node_id, max_depth=horizon)
                    if distance == float("inf") or weight <= 0.0:
                        continue
                    capped_weight = min(float(weight), 5000.0)
                    coeff = 0.002 * capped_weight / (1.0 + float(distance))
                    objective += coeff * x[x_key]
                    terminal_terms += 1
            if terminal_terms:
                print(f"[模型优化] soft focus terminal bonus terms: {terminal_terms}")

        model.setObjective(objective, mip.MAXIMIZE)

        # ------------------------------
        # 收集trace基础信息 (求解前)
        # ------------------------------
        if trace:
            trace.num_agents = num_agents
            trace.num_nodes = num_nodes
            trace.num_platforms = num_platforms
            trace.time_horizon = horizon
            
            # 记录环境状态
            for agent_idx, agent in enumerate(agents):
                pos_id = int(agent.state.position[env_index].item())
                trace.agent_positions[agent.name] = pos_id
            
            for platform_id in platform_ids:
                node = platform_nodes[platform_id]
                waiting = _safe_tensor_item(node.waiting_passengers, env_index, default=0.0)
                trace.platform_waiting[platform_id] = waiting
            
            # 记录中断信息
            for seg_id in blocked_segments:
                info = events.get_segment_disruption_info(seg_id)
                if info:
                    trace.disruptions.append({"segment_id": seg_id, **info})
            
            # 生成状态摘要
            trace.state_summary = self._generate_state_summary(
                agents, platform_nodes, blocked_segments, world, env_index
            )

        # ------------------------------
        # 应用LLM求解指导 (如果提供)
        # ------------------------------
        if llm_guidance:
            # 必须先调用 update() 才能通过名称访问变量
            model.update()
            self._apply_llm_guidance(
                model, llm_guidance, 
                move, stay, x,
                num_agents, num_nodes, horizon,
                node_ids, node_index, current_node_idx, adjacency
            )

        def _build_heuristic_solution(reason: str) -> Dict[str, int]:
            """Return executable solver-side fallback actions instead of MARL fallback.

            This is used only when CPLEX cannot produce an incumbent within the
            requested time. It respects the same immediate hard constraints the
            simulator will enforce: minimum dwell, blocked edges, forward safety
            distance, and duplicate next-node claims.
            """
            waiting_by_platform = {
                platform_id: _safe_tensor_item(
                    getattr(platform_nodes[platform_id], "waiting_passengers", None),
                    env_index,
                    default=0.0,
                )
                for platform_id in platform_ids
            }
            target_platforms = [
                platform_id
                for platform_id, waiting in sorted(
                    waiting_by_platform.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                if waiting > 0.0
            ][:8]

            def _action_for_next(current_id: int, next_id: int) -> int:
                node_obj = world.get_node(current_id) if hasattr(world, "get_node") else None
                route_map = node_obj.getnext() if node_obj is not None and hasattr(node_obj, "getnext") else None
                route_map = route_map or {}
                available = {key: value for key, value in route_map.items() if value is not None}
                if available:
                    route_keys = sorted(available.keys(), key=lambda x: str(x))
                    for direction_idx, route_key in enumerate(route_keys):
                        try:
                            if int(available[route_key]) == int(next_id):
                                return direction_idx + 1
                        except (TypeError, ValueError):
                            continue

                for direction_idx, candidate_id in enumerate(adjacency.get(current_id, [])):
                    if int(candidate_id) == int(next_id):
                        return direction_idx + 1
                return 0

            def _distance_to_target(start_id: int, target_id: int, max_depth: int = 200) -> float:
                if start_id == target_id:
                    return 0.0
                queue = deque([(start_id, 0)])
                visited = {start_id}
                while queue:
                    node_id, dist = queue.popleft()
                    if dist >= max_depth:
                        continue
                    for next_id in adjacency.get(node_id, []):
                        if (node_id, next_id) in blocked_edges:
                            continue
                        if next_id == target_id:
                            return float(dist + 1)
                        if next_id not in visited:
                            visited.add(next_id)
                            queue.append((next_id, dist + 1))
                return float("inf")

            def _is_safe_next(agent_idx: int, current_id: int, next_id: int, claimed: Set[int]) -> bool:
                if next_id == current_id:
                    return True
                if next_id in claimed:
                    return False
                if (current_id, next_id) in blocked_edges:
                    return False
                if not world.edge_event_allowed(agents[agent_idx], current_id, next_id, view=events):
                    return False
                if hasattr(world, "_get_forward_distance"):
                    for other_idx, other_agent in enumerate(agents):
                        if other_idx == agent_idx:
                            continue
                        other_pos = int(other_agent.state.position[env_index].item())
                        forward_distance = world._get_forward_distance(next_id, other_pos, conflict_distance_limit)
                        if forward_distance < conflict_distance_limit:
                            return False
                return True

            fallback_actions: Dict[str, int] = {}
            fallback_targets: Dict[str, int] = {}
            fallback_details: Dict[str, Dict[str, Union[int, float, str]]] = {}
            claimed_next_nodes: Set[int] = set()

            for agent_idx, agent in enumerate(agents):
                agent_name = agent.name
                current_id = int(agent.state.position[env_index].item())

                if agent_idx not in active_agent_indices:
                    fallback_actions[agent_name] = 0
                    fallback_targets[agent_name] = current_id
                    fallback_details[agent_name] = {
                        "action": 0,
                        "next_node": current_id,
                        "reason": f"{reason}; non-active agent left to outer policy",
                    }
                    continue

                on_platform = current_id in platform_index
                dwell_time = _safe_tensor_item(getattr(agent.state, "dwell_time", None), env_index, 0.0)
                time_since_arrival = _safe_tensor_item(
                    getattr(agent.state, "time_since_arrival", None),
                    env_index,
                    dwell_time,
                )
                dwell_min = _safe_tensor_item(getattr(agent.state, "dwell_min_limit", None), env_index, 0.0)
                expected_dwell = _safe_tensor_item(
                    getattr(agent.state, "current_expected_dwell_time", None),
                    env_index,
                    0.0,
                )
                if on_platform:
                    node_min_seconds = float(getattr(platform_nodes[current_id], "weight", 0) or 0.0)
                    if node_min_seconds > 0.0:
                        dwell_min = max(dwell_min, math.ceil(node_min_seconds / max(tick_seconds, 1e-6)))
                    dwell_min = max(dwell_min, world.minimum_dwell_steps(agent, current_id, view=events))

                must_hold = on_platform and max(dwell_time, time_since_arrival) + 1e-6 < max(dwell_min, expected_dwell)
                must_hold = must_hold or public_cooldown.get(agent_idx, 0) > 0
                if must_hold:
                    fallback_actions[agent_name] = 0
                    fallback_targets[agent_name] = current_id
                    fallback_details[agent_name] = {
                        "action": 0,
                        "next_node": current_id,
                        "reason": f"{reason}; minimum dwell hold",
                    }
                    continue

                neighbors = [next_id for next_id in adjacency.get(current_id, []) if (current_id, next_id) not in blocked_edges]
                best_next: Optional[int] = None
                best_score = float("inf")
                candidate_targets = target_platforms or platform_ids

                for next_id in neighbors:
                    if not _is_safe_next(agent_idx, current_id, next_id, claimed_next_nodes):
                        continue
                    distance = min(
                        (_distance_to_target(next_id, target_id) for target_id in candidate_targets if target_id != current_id),
                        default=0.0,
                    )
                    if distance < best_score:
                        best_score = distance
                        best_next = next_id

                if best_next is None:
                    action_value = 0
                    best_next = current_id
                else:
                    action_value = _action_for_next(current_id, best_next)
                    if action_value <= 0:
                        best_next = current_id
                    else:
                        claimed_next_nodes.add(best_next)

                fallback_actions[agent_name] = int(action_value)
                fallback_targets[agent_name] = int(best_next)
                fallback_details[agent_name] = {
                    "action": int(action_value),
                    "next_node": int(best_next),
                    "reason": reason,
                }

            self.last_solution = {
                "status": "heuristic_fallback",
                "event_inputs": event_manifest,
                "milp_incumbent": False,
                "objective": None,
                "step_rewards": [],
                "actions": fallback_details,
                "all_step_actions": [fallback_actions],
                "all_step_targets": [fallback_targets],
                "solve_time": time.perf_counter() - start_time,
            }
            return fallback_actions

        # ------------------------------
        # Solve
        # ------------------------------
        resolved_time_limit = int(
            time_limit
            if time_limit is not None
            else self.time_limit
            if self.time_limit is not None
            else 1
        )
        if focus_agent_indices is not None:
            resolved_time_limit = max(1, min(resolved_time_limit, 30))
        try:
            model.update()
        except mip.SolverError:
            pass
        elapsed_before_optimize = time.perf_counter() - start_time
        search_node_count = sum(len(nodes_set) for nodes_set in agent_reachable_nodes.values())
        print(
            "[模型优化] 规模统计: "
            f"focus_agents={len(active_agent_indices)}, horizon={horizon}, "
            f"search_nodes={search_node_count}, capacity_nodes={len(capacity_node_indices)}, "
            f"x={len(x_keys)}, stay={len(stay_keys)}, move={len(valid_move_keys)}, "
            f"vars={model.NumVars}, constrs={model.NumConstrs}, "
            f"build_time={elapsed_before_optimize:.2f}s"
        )
        remaining_time_limit = max(1, int(resolved_time_limit - elapsed_before_optimize))
        if elapsed_before_optimize >= resolved_time_limit:
            print(
                f"MILP构模已耗尽总预算: build_time={elapsed_before_optimize:.2f}s, "
                f"limit={resolved_time_limit}s。使用内部启发式解。"
            )
            return _build_heuristic_solution("MILP build exceeded wall-clock budget")

        model.setParam("TimeLimit", remaining_time_limit)
        model.setParam("MIPFocus", 1)
        model.setParam("MIPGap", 0.02)
        model.setParam("Threads", 1)
        model.setParam("Heuristics", 0.5)
        model.setParam("Presolve", 2)
        # CPLEX has no direct equivalent to NoRelHeurTime.
        print(
            f"[模型优化] 构模耗时 {elapsed_before_optimize:.2f}s, "
            f"CPLEX剩余时限 {remaining_time_limit}s"
        )
        model.optimize()

        solve_time = time.perf_counter() - start_time

        actions: Dict[str, int] = {}
        solution_details: Dict[str, Dict[str, Union[int, float]]] = {}
        step_rewards: List[float] = []
        # 新增：保存所有时间步的动作，供缓存使用
        all_step_actions: List[Dict[str, int]] = []
        all_step_targets: List[Dict[str, int]] = []

        if model.status not in (mip.OPTIMAL, mip.TIME_LIMIT):
            print(f"未找到MILP可行解 (状态码: {model.status})，使用内部启发式解。")
            
            # Handle INF_OR_UNBD
            if model.status == mip.INF_OR_UNBD:
                print("Model status is INF_OR_UNBD. Re-optimizing with DualReductions=0 to distinguish...")
                model.setParam("DualReductions", 0)
                model.optimize()
                print(f"New status: {model.status}")

            # Conflict refinement is useful offline, but expensive in the online
            # dispatch loop. Only collect it when trace/debug collection is
            # explicitly requested.
            if model.status == mip.INFEASIBLE and collect_trace:
                print("Running CPLEX conflict refiner to diagnose infeasibility...")
                try:
                    model.computeIIS()
                    model.write("milp_infeasible.ilp")
                    print("CPLEX conflict refiner output written to 'milp_infeasible.ilp'. Conflicting constraints:")
                    for c in model.getConstrs():
                        if model.is_conflict(c):
                            print(f"  {c.name}")
                except mip.SolverError as e:
                    print(f"CPLEX conflict refiner failed: {e}")

            return _build_heuristic_solution(f"MILP no incumbent/status={model.status}")

        if model.status == mip.TIME_LIMIT:
            if model.SolCount == 0:
                print("MILP在时限内未找到可行解 (SolCount=0)。")
                return _build_heuristic_solution("MILP time limit without incumbent")
            print("MILP在时限内找到次优解。")
        else:
            print("MILP求解成功。")
        print(f"模型构建+求解耗时: {solve_time:.2f}s")

        # ==== 提取所有时间步的动作（供缓存使用） ====
        # 直接从 x[a,n,t] 的最优值反解动作，避免本地位置递推误差导致缓存动作漂移。
        for t in range(horizon - 1):  # 只到 horizon-1，因为最后一步没有决策
            step_actions_t: Dict[str, int] = {}
            step_targets_t: Dict[str, int] = {}

            for agent_idx, agent in enumerate(agents):
                # 非active agent保持停留
                if agent_idx not in active_agent_indices:
                    step_actions_t[agent.name] = 0
                    current_pos = int(agent.state.position[env_index].item())
                    step_targets_t[agent.name] = current_pos
                    continue

                reachable_nodes = agent_reachable_nodes.get(agent_idx, set())
                current_idx = None
                next_idx = None

                # 从x变量中读取 t 时刻位置
                for node_idx in reachable_nodes:
                    key_now = (agent_idx, node_idx, t)
                    if key_now in x and x[key_now].solution_value > 0.5:
                        current_idx = node_idx
                        break

                # 从x变量中读取 t+1 时刻位置
                for node_idx in reachable_nodes:
                    key_next = (agent_idx, node_idx, t + 1)
                    if key_next in x and x[key_next].solution_value > 0.5:
                        next_idx = node_idx
                        break

                # 防御式回退：如果读取失败，保持停留
                if current_idx is None or next_idx is None:
                    step_actions_t[agent.name] = 0
                    current_pos = int(agent.state.position[env_index].item())
                    step_targets_t[agent.name] = current_pos
                    continue

                # 位置未变 -> 停留
                if current_idx == next_idx:
                    step_actions_t[agent.name] = 0
                    step_targets_t[agent.name] = int(node_ids[current_idx])
                    continue

                current_id = node_ids[current_idx]
                next_id = node_ids[next_idx]
                step_targets_t[agent.name] = int(next_id)

                # 方向编码必须与 RouteSelectionAdapter.next_node_by_direction 一致：
                # 按 route_key 的字符串排序来映射动作 1/2/3...
                chosen_action = 0
                node_obj = world.get_node(current_id)
                route_map = node_obj.getnext() if node_obj is not None else None
                route_map = route_map or {}
                available = {key: value for key, value in route_map.items() if value is not None}
                route_keys = sorted(available.keys(), key=lambda x: str(x))
                for direction_idx, route_key in enumerate(route_keys):
                    if int(available[route_key]) == next_id:
                        chosen_action = direction_idx + 1
                        break

                # 若 next_id 非当前节点邻居，按停留处理，避免输出非法动作
                step_actions_t[agent.name] = chosen_action

            all_step_actions.append(step_actions_t)
            all_step_targets.append(step_targets_t)
        
        # 第一步动作（兼容原有逻辑）
        actions = all_step_actions[0] if all_step_actions else {}
        
        # 解析第一步动作详情
        for agent_idx, agent in enumerate(agents):
            current_id = int(agent.state.position[env_index].item())
            current_idx = node_index[current_id]

            if agent_idx not in active_agent_indices:
                solution_details[agent.name] = {
                    "action": 0,
                    "next_node": current_id,
                    "reason": "Non-active agent fixed"
                }
                continue

            action_val = actions.get(agent.name, 0)
            if action_val == 0:
                solution_details[agent.name] = {
                    "action": 0,
                    "next_node": current_id,
                    "reason": "Optimal stay"
                }
            else:
                # 环境支持多方向动作空间，动作方向以 route_key 排序为准。
                node_obj = world.get_node(current_id)
                route_map = node_obj.getnext() if node_obj is not None else None
                route_map = route_map or {}
                available = {key: value for key, value in route_map.items() if value is not None}
                route_keys = sorted(available.keys(), key=lambda x: str(x))
                direction_idx = action_val - 1  # 转换回0-based索引

                if 0 <= direction_idx < len(route_keys):
                    chosen_next = int(available[route_keys[direction_idx]])
                else:
                    # 方向不存在，回退到第一个方向
                    chosen_next = int(available[route_keys[0]]) if route_keys else current_id
                    action_val = 1 if route_keys else 0
                    
                solution_details[agent.name] = {
                    "action": action_val,
                    "next_node": chosen_next,
                    "reason": f"Optimal move to direction {action_val}"
                }

        # 计算逐步奖励以便对比 (只计算active agent)
        for t in range(horizon):
            reward_t = 0.0

            board_sum = sum(
                board[agent_idx, p_idx, t].solution_value
                for agent_idx in active_agent_indices 
                for p_idx in range(num_platforms)
                if (agent_idx, p_idx, t) in board
            )
            alight_sum = sum(
                alight[agent_idx, p_idx, t].solution_value
                for agent_idx in active_agent_indices
                for p_idx in range(num_platforms)
                if (agent_idx, p_idx, t) in alight
            )
            move_sum = sum(move_coeffs.get(key, self.BASE_MOVE_BONUS) * move[key].solution_value
                          for key in valid_move_keys if key[3] == t)
            non_platform_stay = 0.0
            if t < horizon - 1:
                for agent_idx in active_agent_indices:
                    for node_idx, node_id in enumerate(node_ids):
                        if node_id in platform_set:
                            continue
                        if (agent_idx, node_idx, t) in stay:
                            non_platform_stay += stay[agent_idx, node_idx, t].solution_value

            reward_t += self.PASSENGER_BOARD_REWARD * board_sum
            reward_t += self.PASSENGER_ALIGHT_REWARD * alight_sum
            reward_t += move_sum
            reward_t -= self.NON_PLATFORM_HOLD_PENALTY * non_platform_stay

            stay_bonus = 0.0
            if t < horizon - 1:
                for agent_idx, node_idx, coeff in stay_reward_by_time.get(t, []):
                    if (agent_idx, node_idx, t) in stay:
                        stay_bonus += coeff * stay[agent_idx, node_idx, t].solution_value
            reward_t += stay_bonus

            # 等待乘客惩罚：对每个站台的等待人数进行惩罚（与目标函数一致）
            # 使用折扣因子，近期等待惩罚更重要
            wait_discount_factor = 0.95
            discount = wait_discount_factor ** t
            waiting_penalty = 0.0
            for p_idx in range(num_platforms):
                wait_count = wait[p_idx, t].solution_value
                waiting_penalty += self.PASSENGER_WAIT_PENALTY * discount * wait_count
            reward_t -= waiting_penalty

            step_rewards.append(reward_t)

        print("\n--- MILP 决策结果 ---")
        for agent_name, action in actions.items():
            action_str = "移动" if action > 0 else "停留"
            print(f"列车 {agent_name}: 动作 -> {action_str} ({action})")
        print(f"总计规划 {len(all_step_actions)} 步动作")
        print("----------------------")

        self.last_solution = {
            "status": model.status,
            "solver_backend": "cplex",
            "cplex_status": model.cplex_status,
            "solver_parameter_mapping": model.parameter_mapping,
            "solver_version": model.native_model.get_cplex().get_version(),
            "milp_incumbent": True,
            "event_inputs": event_manifest,
            "passenger_totals": {
                "platform": [sum(wait[p, t].solution_value for p in range(num_platforms)) for t in range(horizon + 1)],
                "outside": [sum(outside[p, r, t].solution_value for p in range(num_platforms) for r in range(num_platforms)) for t in range(horizon + 1)],
                "onboard": [sum(load[i, t].solution_value for i in active_agent_indices) for t in range(horizon + 1)],
            },
            "objective": float(model.ObjVal) if model.SolCount else None,
            "step_rewards": step_rewards,
            "actions": solution_details,
            "all_step_actions": all_step_actions,  # 新增：所有时间步的动作
            "all_step_targets": all_step_targets,
            "solve_time": solve_time,
        }

        # ------------------------------
        # 完成trace信息收集 (求解后)
        # ------------------------------
        if trace:
            trace.solve_time = solve_time
            trace.objective_value = float(model.ObjVal) if model.SolCount else None
            trace.gap = model.MIPGap if hasattr(model, 'MIPGap') and model.SolCount else 0.0
            trace.node_count = int(model.NodeCount) if hasattr(model, 'NodeCount') else 0
            trace.iteration_count = int(model.IterCount) if hasattr(model, 'IterCount') else 0
            trace.num_variables = model.NumVars
            trace.num_constraints = model.NumConstrs
            
            # 记录关键变量值和reduced cost
            trace.key_variables = self._collect_key_variables(
                model, move, stay, x,
                num_agents, current_node_idx, node_ids, adjacency, horizon
            )
            
            # 记录紧约束
            trace.binding_constraints = self._collect_binding_constraints(model)
            
            # 生成求解策略建议（用作微调target）
            trace.warm_start_hint, trace.variable_fixing_hint, trace.priority_hint = \
                self._generate_solve_hints(
                    model, move, stay, x,
                    num_agents, current_node_idx, node_ids, adjacency, horizon,
                    actions, solution_details
                )
            
            # 记录最终动作
            trace.actions = actions.copy()
            trace.action_details = {k: dict(v) for k, v in solution_details.items()}
            
            self.last_solve_trace = trace

        return actions

    def solve_prepared_dispatch(self, problem, *, time_limit, guidance=None, max_evaluations=None,
                                seed_selections=None, optimization_mode="joint", stop_on_background_conflict=False):
        """Continue a request-local search without replaying its checked baseline.

        Each attempt gets a fresh master. Only exact point checks from the same
        state, arrivals, catalog and background are shared, never search bounds.
        """
        from MILP.operation_model import JointDispatchModel, guard_matches, state_guard
        if not isinstance(problem, JointDispatchModel) or not guard_matches(
                state_guard(problem.base), state_guard(problem.world)):
            raise ValueError("Prepared dispatch problem is stale or invalid")
        return self._solve_dispatch(problem.world, time_horizon=problem.horizon, time_limit=time_limit,
            focus=sorted(problem.focus), focus_platforms=sorted(problem.focus_platforms), candidates=None,
            passenger_inputs=problem.inputs, fixed_paths=problem.fixed_paths, start_stages=(0,),
            max_evaluations=max_evaluations, guidance=guidance, collect_trace=False,
            optimization_mode=optimization_mode, prepared_problem=problem, seed_selections=seed_selections,
            stop_on_background_conflict=stop_on_background_conflict)

    def _solve_dispatch(self, world, *, time_horizon, time_limit, focus, candidates,
                        passenger_inputs, fixed_paths, start_stages, max_evaluations, guidance, collect_trace,
                        reference_actions=None, focus_platforms=None, reference_packets=None, optimization_mode="joint",
                        prepared_problem=None, seed_selections=None, stop_on_background_conflict=False):
        from MILP.operation_model import JointDispatchModel, EvaluationTimeout
        started = time.perf_counter()
        budget = float(self.time_limit if time_limit is None else time_limit)
        horizon = self.time_horizon if time_horizon is None else time_horizon
        if optimization_mode not in ("joint", "staged"):
            raise ValueError("Unknown native dispatch optimization mode")
        if (isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1
                or not math.isfinite(budget) or budget < 0
                or max_evaluations is not None and (not isinstance(max_evaluations, int) or isinstance(max_evaluations, bool) or max_evaluations < 0)):
            raise ValueError("Invalid joint-dispatch horizon, budget or evaluation limit")
        if getattr(self, "last_model", None) is not None:
            self.last_model.dispose()
        self.last_model = None
        self.last_solve_trace = None
        self.last_joint_model = None
        try:
            problem = (prepared_problem.fork(deadline=started + budget, reuse_evaluations=True)
                if prepared_problem is not None else
                JointDispatchModel(world, horizon, focus=focus, focus_platforms=focus_platforms, candidates=candidates,
                    passenger_inputs=passenger_inputs, fixed_paths=fixed_paths, start_stages=start_stages,
                    reference_actions=reference_actions, reference_packets=reference_packets, deadline=started + budget))
            self.last_joint_model = problem
            expected_catalog = (guidance or {}).get("catalog_fingerprint")
            mismatch = bool(expected_catalog and expected_catalog != problem.catalog_fingerprint)
            if mismatch:
                guidance = {}
            if optimization_mode == "staged":
                from MILP.staged_dispatch import solve_staged
                self.last_model, self.last_solution = solve_staged(problem, max_evaluations=max_evaluations, guidance=guidance)
            else:
                self.last_model, self.last_solution = problem.solve(max_evaluations=max_evaluations, guidance=guidance,
                    seed_selections=seed_selections, adaptive_seed_completion=bool(seed_selections),
                    stop_on_background_conflict=stop_on_background_conflict)
            self.last_solution["optimization_mode"] = optimization_mode
            self.last_solution["guidance_catalog_mismatch"] = mismatch
        except EvaluationTimeout:
            self.last_solution = dict(status="no_incumbent_limit", termination="time_limit_during_build",
                milp_incumbent=False, objective=None, cost=None, cost_lower_bound=0., cost_upper_bound=None,
                gap=None, solve_time=time.perf_counter() - started, dispatch_packets=[],
                certificate_scope="no feasible trajectory certified before the build budget expired")
        finally:
            if prepared_problem is not None and self.last_joint_model is not None:
                prepared_problem.evaluated.update(self.last_joint_model.evaluated)
        self.last_solution["solve_time"] = time.perf_counter() - started
        if collect_trace and self.last_model is not None and self.last_solution.get("milp_incumbent"):
            self.last_solve_trace = MILPSolveTrace(num_agents=len(world.agents), num_nodes=len(world.nodes),
                num_platforms=len(world.platform_nodes), time_horizon=horizon,
                num_variables=self.last_model.NumVars, num_constraints=self.last_model.NumConstrs,
                solve_time=self.last_solution["solve_time"], objective_value=self.last_solution["objective"], gap=self.last_solution["gap"],
                actions=(self.last_solution.get("all_step_actions") or [{}])[0])
        return (self.last_solution.get("all_step_actions") or [{}])[0]

    def apply_dispatch_solution(self, world, stage=0):
        from MILP.operation_model import apply_dispatch_step
        return apply_dispatch_step(world, self.last_solution, stage)

    def audit_dispatch_solution(self, world, stage=0):
        from MILP.operation_model import audit_dispatch_step
        return audit_dispatch_step(world, self.last_solution, stage)

    # ------------------------------------------------------------------
    # 辅助方法：生成状态摘要
    # ------------------------------------------------------------------
    def _generate_state_summary(
        self, agents, platform_nodes, blocked_segments, world, env_index
    ) -> str:
        """生成用于LLM输入的环境状态摘要。"""
        lines = []
        
        # 中断信息
        if blocked_segments:
            lines.append("[线路中断]")
            for seg_id in blocked_segments:
                info = world.disruption_manager.published_view().get_segment_disruption_info(seg_id)
                if info:
                    dtype = info.get("type", "UNKNOWN")
                    lines.append(f"  - 区段 {seg_id}: {dtype}")
        
        # 列车状态
        lines.append("\n[列车状态]")
        for idx, agent in enumerate(agents):
            pos = int(agent.state.position[env_index].item())
            passengers = int(_safe_tensor_item(agent.passenger_count, env_index, 0))
            dwell = _safe_tensor_item(getattr(agent.state, "dwell_time", None), env_index, 0)
            expected_dwell = _safe_tensor_item(
                getattr(agent.state, "current_expected_dwell_time", None), env_index, 0
            )
            lines.append(
                f"  列车{idx}: 位置={pos}, 乘客={passengers}, "
                f"停留时间={int(dwell)}/{int(expected_dwell)}s"
            )
        
        # 站台等待情况（只显示等待人数>10的站点）
        lines.append("\n[站台等待]")
        for node_id, node in platform_nodes.items():
            waiting = _safe_tensor_item(node.waiting_passengers, env_index, 0)
            if waiting > 10:
                lines.append(f"  节点{node_id}: {int(waiting)}人等待")
        
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 辅助方法：应用LLM求解指导
    # ------------------------------------------------------------------
    def _apply_llm_guidance(
        self, model, llm_guidance: Dict[str, Any],
        move, stay, x,
        num_agents: int, num_nodes: int, horizon: int,
        node_ids: List[int], node_index: Dict[int, int],
        current_node_idx: List[int], adjacency: Dict[int, List[int]]
    ) -> None:
        """应用LLM提供的求解加速指导。
        
        关键优化：当固定stay变量时，自动推导并固定相关的x和move变量，
        大幅减少未固定变量数量，加速求解。
        """

        def _preview_items(d: Dict[str, Any], limit: int = 8) -> List[Tuple[str, Any]]:
            items = list(d.items())
            return items[: max(int(limit), 0)]
        
        def _parse_stay_var(var_name: str) -> Optional[Tuple[int, int, int]]:
            """解析stay[agent,node,t]变量名，返回(agent_idx, node_idx, t)"""
            import re
            match = re.match(r'stay\[(\d+),(\d+),(\d+)\]', var_name)
            if match:
                return int(match.group(1)), int(match.group(2)), int(match.group(3))
            return None
        
        # 收集需要额外固定的变量
        extra_fixing: Dict[str, int] = {}
        
        # 1. 应用warm start（初始解建议）
        warm_start = llm_guidance.get("warm_start", {})
        if warm_start:
            applied_count = 0
            print(f"尝试应用LLM warm start建议: {len(warm_start)}个变量")
            for var_name, value in _preview_items(warm_start):
                print(f"  - {var_name} = {value}")
            if len(warm_start) > 8:
                print(f"  ... 其余 {len(warm_start) - 8} 个已省略")
            for var_name, value in warm_start.items():
                try:
                    var = model.getVarByName(var_name)
                    if var is not None:
                        model.set_start(var, value)
                        applied_count += 1
                    else:
                        print(f"  [警告] 变量不存在: {var_name}")
                except Exception as e:
                    print(f"  [错误] 应用warm start失败 {var_name}: {e}")
            print(f"  => 成功应用 {applied_count}/{len(warm_start)} 个warm start")
        
        # 2. 应用变量固定建议 + 自动推导相关变量
        variable_fixing = llm_guidance.get("variable_fixing", {})
        if variable_fixing:
            applied_count = 0
            skipped_conflict = 0
            print(f"尝试应用LLM变量固定建议: {len(variable_fixing)}个变量")
            for var_name, value in _preview_items(variable_fixing):
                print(f"  - FIX {var_name} = {value}")
            if len(variable_fixing) > 8:
                print(f"  ... 其余 {len(variable_fixing) - 8} 个已省略")
            
            # ==== 新增：预先检测节点容量冲突 ====
            # 收集每个(node_idx, t)上被固定stay=1的agent列表
            node_time_agents: Dict[Tuple[int, int], List[int]] = {}
            for var_name, value in variable_fixing.items():
                if value == 1:
                    parsed = _parse_stay_var(var_name)
                    if parsed:
                        agent_idx, node_idx, t = parsed
                        key = (node_idx, t)
                        if key not in node_time_agents:
                            node_time_agents[key] = []
                        node_time_agents[key].append(agent_idx)
            
            # 找出冲突的(node, t)：有多个agent被固定在同一节点同一时刻
            conflicting_positions: Set[Tuple[int, int]] = set()
            for (node_idx, t), agents_list in node_time_agents.items():
                if len(agents_list) > 1:
                    conflicting_positions.add((node_idx, t))
                    print(f"  [冲突检测] 节点{node_idx}时刻{t}有多个agent被固定: {agents_list}，将跳过")
            
            # 对于冲突位置，只保留第一个agent的固定，跳过其余
            # 记录哪些(agent, node, t)需要跳过
            skip_fixing: Set[Tuple[int, int, int]] = set()
            for (node_idx, t), agents_list in node_time_agents.items():
                if len(agents_list) > 1:
                    # 保留列表中第一个（通常是当前位置的agent），跳过其余
                    for agent_idx in agents_list[1:]:
                        skip_fixing.add((agent_idx, node_idx, t))
            # ==== 冲突检测结束 ====
            
            for var_name, value in variable_fixing.items():
                try:
                    # 检查是否需要跳过该固定（冲突）
                    parsed = _parse_stay_var(var_name)
                    if parsed and value == 1:
                        agent_idx, node_idx, t = parsed
                        if (agent_idx, node_idx, t) in skip_fixing:
                            skipped_conflict += 1
                            continue
                    
                    var = model.getVarByName(var_name)
                    if var is not None:
                        var.lb = value
                        var.ub = value
                        applied_count += 1
                        
                        # 关键优化：当固定stay[a,n,t]=1时，自动推导并固定相关变量
                        # 注意：x[a,n,t+1] 会通过 flow 约束自动推导，不需要显式固定
                        if parsed and value == 1:
                            agent_idx, node_idx, t = parsed
                            
                            # stay[a,n,t]=1 => x[a,n,t]=1 (当前时刻在节点n)
                            # x[a,n,t+1] 会通过 flow 约束自动得到，不需要显式固定
                            # 这样避免了与停站结束后移动的冲突
                            extra_fixing[f"x[{agent_idx},{node_idx},{t}]"] = 1
                            
                            # stay[a,n,t]=1 => 所有 move[a,n,*,t]=0 (当前时刻不移动)
                            origin_id = node_ids[node_idx] if node_idx < len(node_ids) else None
                            if origin_id is not None and origin_id in adjacency:
                                for dest_id in adjacency[origin_id]:
                                    dest_idx = node_index.get(dest_id)
                                    if dest_idx is not None:
                                        extra_fixing[f"move[{agent_idx},{node_idx},{dest_idx},{t}]"] = 0
                            
                            # x[a,n,t]=1 => 其他节点 x[a,other,t]=0 (只固定当前时刻)
                            for other_idx in range(num_nodes):
                                if other_idx != node_idx:
                                    extra_fixing[f"x[{agent_idx},{other_idx},{t}]"] = 0
                    else:
                        print(f"  [警告] 变量不存在: {var_name}")
                except Exception as e:
                    print(f"  [错误] 固定变量失败 {var_name}: {e}")
            print(f"  => 成功固定 {applied_count}/{len(variable_fixing)} 个变量")
            if skipped_conflict > 0:
                print(f"  => 跳过 {skipped_conflict} 个冲突的固定建议（多agent同节点）")
            
            # 应用推导出的额外固定
            if extra_fixing:
                extra_applied = 0
                for var_name, value in extra_fixing.items():
                    try:
                        var = model.getVarByName(var_name)
                        if var is not None:
                            var.lb = value
                            var.ub = value
                            extra_applied += 1
                    except Exception:
                        pass
                print(f"  => 自动推导并固定了额外 {extra_applied} 个相关变量")
        
        # 3. 应用分支优先级建议
        branching_priority = llm_guidance.get("branching_priority", {})
        if branching_priority:
            applied_count = 0
            print(f"尝试应用LLM分支优先级建议: {len(branching_priority)}个变量")
            for var_name, priority in _preview_items(branching_priority):
                print(f"  - PRIORITY {var_name} = {priority}")
            if len(branching_priority) > 8:
                print(f"  ... 其余 {len(branching_priority) - 8} 个已省略")
            for var_name, priority in branching_priority.items():
                try:
                    var = model.getVarByName(var_name)
                    if var is not None:
                        model.set_priority(var, priority)
                        applied_count += 1
                    else:
                        print(f"  [警告] 变量不存在: {var_name}")
                except Exception as e:
                    print(f"  [错误] 设置优先级失败 {var_name}: {e}")
            print(f"  => 成功设置 {applied_count}/{len(branching_priority)} 个优先级")

    # ------------------------------------------------------------------
    # 辅助方法：收集关键变量信息
    # ------------------------------------------------------------------
    def _collect_key_variables(
        self, model, move, stay, x,
        num_agents: int, current_node_idx: List[int], 
        node_ids: List[int], adjacency: Dict[int, List[int]], horizon: int
    ) -> Dict[str, Dict[str, Any]]:
        """收集关键决策变量的最优值和reduced cost。"""
        key_vars = {}
        
        # 只收集t=0时刻的关键决策变量（第一步动作）
        for agent_idx in range(num_agents):
            cur_idx = current_node_idx[agent_idx]
            if cur_idx is None:
                continue
            
            cur_node_id = node_ids[cur_idx]
            
            # stay变量
            if horizon > 1:
                stay_key = (agent_idx, cur_idx, 0)
                if stay_key in stay:
                    var = stay[stay_key]
                    try:
                        key_vars[var.name] = {
                            "value": var.solution_value,
                            "rc": _safe_reduced_cost(var),
                            "agent": agent_idx,
                            "type": "stay"
                        }
                    except Exception:
                        pass
            
            # move变量
            for neighbor_id in adjacency.get(cur_node_id, []):
                neighbor_idx = {v: k for k, v in enumerate(node_ids)}.get(neighbor_id)
                if neighbor_idx is None:
                    continue
                move_key = (agent_idx, cur_idx, neighbor_idx, 0)
                if move_key in move:
                    var = move[move_key]
                    try:
                        key_vars[var.name] = {
                            "value": var.solution_value,
                            "rc": _safe_reduced_cost(var),
                            "agent": agent_idx,
                            "type": "move",
                            "to_node": neighbor_id
                        }
                    except Exception:
                        pass
        
        return key_vars

    # ------------------------------------------------------------------
    # 辅助方法：收集紧约束信息
    # ------------------------------------------------------------------
    def _collect_binding_constraints(self, model) -> List[str]:
        """收集紧约束（slack接近0的约束）。"""
        binding = []
        try:
            for constr in model.getConstrs():
                slack = abs(constr.slack_value) if hasattr(constr, 'slack_value') else float('inf')
                if slack < 1e-6:
                    binding.append(constr.name)
        except Exception:
            pass
        return binding[:50]  # 只保留前50个以控制大小

    # ------------------------------------------------------------------
    # 辅助方法：生成求解策略建议
    # ------------------------------------------------------------------
    def _generate_solve_hints(
        self, model, move, stay, x,
        num_agents: int, current_node_idx: List[int],
        node_ids: List[int], adjacency: Dict[int, List[int]], horizon: int,
        actions: Dict[str, int], solution_details: Dict[str, Dict]
    ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
        """
        基于求解结果生成策略建议，这些建议将作为微调target。
        
        Returns:
            - warm_start_hint: 建议的初始解
            - variable_fixing_hint: 建议固定的变量
            - priority_hint: 分支优先级建议
        """
        warm_start = {}
        variable_fixing = {}
        priority = {}
        
        # 基于最优解生成warm start建议
        for agent_idx in range(num_agents):
            cur_idx = current_node_idx[agent_idx]
            if cur_idx is None:
                continue
            
            cur_node_id = node_ids[cur_idx]
            
            # 对于t=0时刻的决策变量
            if horizon > 1:
                stay_key = (agent_idx, cur_idx, 0)
                if stay_key in stay:
                    var = stay[stay_key]
                    try:
                        val = int(round(var.solution_value))
                        warm_start[var.name] = val
                        # 如果变量值确定性高（reduced cost大），建议固定
                        rc = abs(_safe_reduced_cost(var))
                        if rc > 0.1:
                            variable_fixing[var.name] = val
                            priority[var.name] = 100  # 高优先级
                    except Exception:
                        pass
            
            # move变量
            for neighbor_id in adjacency.get(cur_node_id, []):
                neighbor_idx = {v: k for k, v in enumerate(node_ids)}.get(neighbor_id)
                if neighbor_idx is None:
                    continue
                move_key = (agent_idx, cur_idx, neighbor_idx, 0)
                if move_key in move:
                    var = move[move_key]
                    try:
                        val = int(round(var.solution_value))
                        warm_start[var.name] = val
                        rc = abs(_safe_reduced_cost(var))
                        if rc > 0.1:
                            variable_fixing[var.name] = val
                            priority[var.name] = 100
                    except Exception:
                        pass
        
        return warm_start, variable_fixing, priority
