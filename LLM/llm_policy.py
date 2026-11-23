from regex import F
import torch
import json
import re
import copy
import math
import ast
import os
import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import AIMessage, HumanMessage
except ModuleNotFoundError as exc:
    if not exc.name.startswith("langchain"):
        raise
    ChatOpenAI = ChatPromptTemplate = SystemMessagePromptTemplate = HumanMessagePromptTemplate = None
    StrOutputParser = AIMessage = HumanMessage = None
from tensordict import TensorDict

from MILP.milp_model import MILPSolver
from LLM.rag.rag_manager import RAGManager
from LLM.strategic_context import (
    DEFAULT_INPUT_TOKENS, DEFAULT_OUTPUT_TOKENS, INSTRUCTIONS, LLMInputTooLong,
    check_messages, context_audit, dumps as context_dumps, strategy_context,
)

logger = logging.getLogger(__name__)


class OperationsResearchToolkit:
    """一组可复用的运筹学工具，供LLM Agent在工具3中选择使用。"""

    def __init__(self, policy: "LLMAgentPolicy") -> None:
        self.policy = policy
        self.default_horizon = policy.local_milp_time_horizon * 2
        self.default_limit = max(policy.local_milp_time_limit * 2, 5)

    # --- 对外接口 ---

    def solve(
        self,
        method: str,
        config: Dict[str, Any],
        state_description: str,
    ) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Dict[str, Any]]:
        """根据指定方法返回全局调度方案。"""

        method_lower = (method or "").strip().lower() or "lagrangian_relaxation"
        dispatch = {
            "lagrangian": self._solve_lagrangian_relaxation,
            "lagrangian_relaxation": self._solve_lagrangian_relaxation,
            "column_generation": self._solve_column_generation,
            "benders": self._solve_benders_decomposition,
            "benders_decomposition": self._solve_benders_decomposition,
            "queueing": self._solve_queueing_analysis,
            "queueing_analysis": self._solve_queueing_analysis,
        }
        solver_fn = dispatch.get(method_lower)
        if solver_fn is None:
            return None, {"error": f"未知的运筹学方法: {method}"}

        try:
            plan = solver_fn(config)
            return plan, {"method": method_lower, "config": config}
        except Exception as exc:  # noqa: BLE001
            return None, {"error": f"{method_lower} 求解失败: {exc}"}

    # --- 内部工具函数 ---

    def _prepare_solver(self, config: Dict[str, Any]) -> MILPSolver:
        horizon = int(config.get("time_horizon") or self.default_horizon)
        time_limit = int(config.get("time_limit") or self.default_limit)
        return MILPSolver(time_horizon=horizon, time_limit=time_limit)

    def _actions_to_plan(
        self,
        actions_dict: Dict[str, Any],
        solver: MILPSolver,
        method_tag: str,
        fallback_reason: str = "",
    ) -> Dict[str, Any]:
        world = self.policy.env.base_env.scenario.world
        plan: Dict[str, Dict[str, Any]] = {}
        solution_details = getattr(solver, "last_solution", {}) or {}

        for idx, agent in enumerate(world.agents):
            agent_name = agent.name
            key = f"agent_{idx}"
            action_val = int(actions_dict.get(agent_name, 0))
            current_pos = int(agent.state.position[0].item())
            next_node = current_pos

            detail = solution_details.get(agent_name, {})
            if action_val > 0:
                candidate_nodes = [n.id for n in world.get_next_nodes(current_pos)]
                if detail.get("next_node") is not None:
                    next_node = int(detail.get("next_node"))
                elif action_val - 1 < len(candidate_nodes):
                    next_node = int(candidate_nodes[action_val - 1])

            reasoning = detail.get("reason") or detail.get("reasoning")
            if not reasoning:
                if fallback_reason:
                    reasoning = fallback_reason
                else:
                    reasoning = f"{method_tag}: action={action_val}" if action_val else f"{method_tag}: stay"

            plan[key] = {
                "action": action_val,
                "next_node": int(next_node),
                "reasoning": reasoning,
            }

        return {"steps": [plan]}

    def _solve_lagrangian_relaxation(self, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        base_env = self.policy.env.base_env
        world = base_env.scenario.world

        iterations = int(config.get("iterations") or 2)
        local_objective = config.get("objective", "minimize_wait_time")
        lambda_step = float(config.get("lambda_step") or 0.5)

        stay_penalty_cfg = float(config.get("stay_penalty", 0.5))
        passenger_weight = float(config.get("passenger_weight", 0.05))
        move_bonus_cfg = float(config.get("move_bonus", 0.1))
        demand_balance_weight = float(config.get("demand_balance_weight", 0.2))

        default_horizon = getattr(self.policy, "plan_horizon", 1)
        plan_horizon_raw = config.get("plan_horizon")
        try:
            plan_horizon = int(plan_horizon_raw) if plan_horizon_raw is not None else int(default_horizon)
        except (TypeError, ValueError):
            plan_horizon = int(default_horizon)
        plan_horizon = max(plan_horizon, 1)

        tick_seconds = int(getattr(world, "tick_seconds", 1) or 1)

        def _resolve_expected_dwell(node_id: int, fallback: int = 1) -> int:
            node = world.platform_nodes.get(node_id)
            if node is None:
                return 0
            weight = getattr(node, "weight", None)
            if weight is not None and weight > 0:
                return int(weight)
            dwell_seconds = getattr(node, "dwell", None) or getattr(node, "dwell_seconds", None)
            if dwell_seconds is None:
                return fallback
            return max(1, int(math.ceil(float(dwell_seconds) / tick_seconds)))

        sim_agent_states: List[Dict[str, Any]] = []
        for agent in world.agents:
            position = int(agent.state.position[0].item())
            dwell_time = int(agent.state.dwell_time[0].item())
            expected_dwell = int(agent.state.current_expected_dwell_time[0].item())
            passenger_count = int(agent.passenger_count[0].item())
            capacity = getattr(agent, "passenger_capacity", 2520)
            fallback_expected = _resolve_expected_dwell(position, fallback=1)
            resolved_expected = expected_dwell if expected_dwell > 0 else fallback_expected
            sim_agent_states.append({
                "position": position,
                "dwell_time": dwell_time,
                "expected_dwell": max(0, resolved_expected),
                "default_expected": max(1, fallback_expected),
                "passenger_count": passenger_count,
                "capacity": capacity,
            })

        sim_waiting: Dict[int, float] = {
            node_id: float(node.waiting_passengers[0].item())
            for node_id, node in world.platform_nodes.items()
        }

        service_capacity = max(0, int(config.get("service_capacity", 80)))
        arrival_growth = float(config.get("arrival_growth", 0.0))

        # Capture per-node arrival rates
        node_arrival_rates: Dict[int, float] = {}
        for nid, node in world.platform_nodes.items():
            node_arrival_rates[nid] = float(getattr(node, "passenger_arrival_rate", 0.0))

        cumulative_penalties: Dict[int, float] = defaultdict(float)
        plan_steps: List[Dict[str, Dict[str, Any]]] = []

        # 增加搜索深度和折扣因子，以便在微观路网中能“看到”远处的下一站
        discount = 0.99
        max_depth = 100

        num_agents = len(sim_agent_states)

        def _estimate_node_reward(node_id: int, cache: Dict[int, float]) -> float:
            if node_id in cache:
                return cache[node_id]

            max_reward = 0.0
            visited: Set[int] = {node_id}
            queue: deque[Tuple[int, int]] = deque([(node_id, 0)])

            while queue:
                current, depth = queue.popleft()
                if current in world.platform_nodes:
                    waiting_val = sim_waiting.get(current)
                    if waiting_val is None:
                        waiting_val = world.platform_nodes[current].waiting_passengers[0].item()
                    discounted = float(waiting_val) * (discount ** depth)
                    if discounted > max_reward:
                        max_reward = discounted
                if depth >= max_depth:
                    continue
                for successor in world.get_next_nodes(current):
                    succ_id = successor.id
                    if succ_id not in visited:
                        visited.add(succ_id)
                        queue.append((succ_id, depth + 1))

            cache[node_id] = max_reward
            return max_reward

        def _run_single_step(step_idx: int) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, float]]:
            demand_cache: Dict[int, float] = {}
            dual_penalties_step: Dict[int, float] = defaultdict(float)
            dual_penalties_step.update(cumulative_penalties)

            best_plan_local: Dict[str, Dict[str, Any]] = {}

            for iteration in range(iterations):
                candidate_plan: Dict[int, Dict[str, Any]] = {}
                occupancy: Dict[int, List[int]] = defaultdict(list)

                for agent_id in range(num_agents):
                    agent_state = sim_agent_states[agent_id]
                    current_pos = int(agent_state["position"])
                    dwell_time = int(agent_state["dwell_time"])
                    expected_dwell = int(agent_state["expected_dwell"])
                    passenger_count = int(agent_state["passenger_count"])

                    forced_stay = current_pos in world.platform_nodes and dwell_time < expected_dwell
                    stay_penalty = stay_penalty_cfg if not forced_stay else 0.0
                    base_reward = _estimate_node_reward(current_pos, demand_cache)
                    base_score = base_reward + passenger_count * passenger_weight - stay_penalty

                    action = 0
                    next_node = current_pos
                    best_score = base_score
                    reasoning = (
                        f"Lagrangian 第{iteration + 1}轮: 默认停留 (评分={best_score:.2f})"
                        + (" | 停站时间不足，强制停留" if forced_stay else "")
                    )

                    if not forced_stay:
                        candidate_nodes = world.get_next_nodes(current_pos)
                        disrupted_segments = world.get_disrupted_segments()
                        
                        for successor in candidate_nodes:
                            successor_id = successor.id
                            
                            # Check for blockages
                            seg_fwd = f"{current_pos}-{successor_id}"
                            seg_bwd = f"{successor_id}-{current_pos}"
                            is_blocked = False
                            if seg_fwd in disrupted_segments and disrupted_segments[seg_fwd]['type'] == 'BLOCKAGE':
                                is_blocked = True
                            elif seg_bwd in disrupted_segments and disrupted_segments[seg_bwd]['type'] == 'BLOCKAGE':
                                is_blocked = True
                            
                            if is_blocked:
                                continue

                            successor_reward = _estimate_node_reward(successor_id, demand_cache)
                            
                            # Capacity constraint: limit reward if full
                            capacity = sim_agent_states[agent_id]["capacity"]
                            available = max(0, capacity - passenger_count)
                            if available <= 0:
                                # If full, reduce attraction to demand (only move to alight)
                                successor_reward *= 0.1
                            
                            delta_demand = successor_reward - base_reward
                            score = (
                                successor_reward
                                + passenger_count * passenger_weight
                                + move_bonus_cfg
                                + demand_balance_weight * delta_demand
                                - dual_penalties_step[agent_id]
                            )
                            if score > best_score + 1e-6:
                                best_score = score
                                action = 1
                                next_node = successor_id
                                reasoning = (
                                    f"Lagrangian 第{iteration + 1}轮: 选择区段 {current_pos}->{successor_id} "
                                    f"(评分={best_score:.2f}, 罚项={dual_penalties_step[agent_id]:.2f})"
                                )

                    candidate_plan[agent_id] = {
                        "action": action,
                        "next_node": next_node,
                        "reasoning": reasoning,
                    }

                    if action == 1:
                        occupancy[next_node].append(agent_id)

                conflict_count = 0
                for node_id, agents_at_node in occupancy.items():
                    if len(agents_at_node) <= 1:
                        continue

                    conflict_count += 1
                    agents_sorted = sorted(
                        agents_at_node,
                        key=lambda idx: (
                            sim_agent_states[idx]["passenger_count"],
                            -dual_penalties_step[idx],
                        ),
                        reverse=True,
                    )
                    keeper = agents_sorted[0]
                    for loser in agents_sorted[1:]:
                        dual_penalties_step[loser] += lambda_step
                        current_pos = int(sim_agent_states[loser]["position"])
                        candidate_plan[loser] = {
                            "action": 0,
                            "next_node": current_pos,
                            "reasoning": "Lagrangian 调整: 避免冲突, 将列车保持在当前位置",
                        }
                    candidate_plan[keeper]["reasoning"] += " | 获得区段优先权"

                if iteration == iterations - 1 or conflict_count == 0:
                    best_plan_local = {
                        f"agent_{aid}": copy.deepcopy(plan_data)
                        for aid, plan_data in candidate_plan.items()
                    }
                    break

            for agent_key, decision in best_plan_local.items():
                decision_reason = decision.get("reasoning")
                if decision_reason:
                    decision["reasoning"] = f"{decision_reason} | 计划步{step_idx + 1}"

            return best_plan_local, dual_penalties_step

        def _apply_step(step_plan: Dict[str, Dict[str, Any]]) -> None:
            for agent_id, agent_state in enumerate(sim_agent_states):
                agent_key = f"agent_{agent_id}"
                decision = step_plan.get(agent_key, {})
                action = int(decision.get("action", 0))
                raw_next = decision.get("next_node", agent_state["position"])
                try:
                    next_node = int(raw_next)
                except (TypeError, ValueError):
                    next_node = int(agent_state["position"])

                if action == 1:
                    agent_state["position"] = next_node
                    agent_state["dwell_time"] = 0
                    expected = _resolve_expected_dwell(next_node, agent_state["default_expected"])
                    agent_state["expected_dwell"] = max(0, expected)
                    
                    # Simulate alighting (heuristic: 20% alight)
                    agent_state["passenger_count"] = int(agent_state["passenger_count"] * 0.8)
                    
                    # Simulate boarding
                    if next_node in sim_waiting and service_capacity > 0:
                        capacity = agent_state["capacity"]
                        available = max(0, capacity - agent_state["passenger_count"])
                        pickup = min(sim_waiting[next_node], available, service_capacity)
                        sim_waiting[next_node] = max(0.0, sim_waiting[next_node] - pickup)
                        agent_state["passenger_count"] += pickup
                else:
                    agent_state["dwell_time"] = min(
                        agent_state["expected_dwell"],
                        agent_state["dwell_time"] + 1,
                    )
                    pos = agent_state["position"]
                    
                    # Simulate boarding during dwell
                    if pos in sim_waiting and service_capacity > 0:
                        capacity = agent_state["capacity"]
                        available = max(0, capacity - agent_state["passenger_count"])
                        pickup = min(sim_waiting[pos], available, service_capacity)
                        sim_waiting[pos] = max(0.0, sim_waiting[pos] - pickup)
                        agent_state["passenger_count"] += pickup

            # Apply arrival growth (per-node rate + global bias)
            for node_id in sim_waiting:
                rate = node_arrival_rates.get(node_id, 0.0)
                sim_waiting[node_id] = max(0.0, sim_waiting[node_id] + rate * tick_seconds + arrival_growth)

        for step_idx in range(plan_horizon):
            step_plan, step_penalties = _run_single_step(step_idx)
            if not step_plan:
                # 若未生成计划，确保至少提供停留策略
                fallback_step: Dict[str, Dict[str, Any]] = {}
                for agent_id, agent_state in enumerate(sim_agent_states):
                    current_pos = int(agent_state["position"])
                    fallback_step[f"agent_{agent_id}"] = {
                        "action": 0,
                        "next_node": current_pos,
                        "reasoning": f"Lagrangian: 缺省停留策略 | 计划步{step_idx + 1}",
                    }
                step_plan = fallback_step

            plan_steps.append(step_plan)

            for idx, penalty_value in step_penalties.items():
                cumulative_penalties[idx] = penalty_value

            _apply_step(step_plan)

        return {"steps": plan_steps}

    def _solve_column_generation(self, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        base_env = self.policy.env.base_env
        world = base_env.scenario.world

        max_iterations = int(config.get("iterations") or 3)
        initial_columns = int(config.get("initial_columns") or 3)
        max_columns_cfg = config.get("max_columns")
        try:
            max_columns = int(max_columns_cfg) if max_columns_cfg is not None else len(world.platform_nodes)
        except (TypeError, ValueError):
            max_columns = len(world.platform_nodes)
        max_columns = max(1, max_columns)
        solver = self._prepare_solver(config)

        default_horizon = getattr(self.policy, "plan_horizon", 1)
        plan_horizon_raw = config.get("plan_horizon")
        try:
            plan_horizon = int(plan_horizon_raw) if plan_horizon_raw is not None else int(default_horizon)
        except (TypeError, ValueError):
            plan_horizon = int(default_horizon)
        plan_horizon = max(plan_horizon, 1)

        service_capacity = max(0, int(config.get("service_capacity", 80)))
        arrival_growth = float(config.get("arrival_growth", 0.0))

        # Capture per-node arrival rates
        node_arrival_rates: Dict[int, float] = {}
        for nid, node in world.platform_nodes.items():
            node_arrival_rates[nid] = float(getattr(node, "passenger_arrival_rate", 0.0))

        def _snapshot_agents() -> List[Dict[str, torch.Tensor]]:
            snapshots: List[Dict[str, torch.Tensor]] = []
            for agent in world.agents:
                snapshots.append(
                    {
                        "position": agent.state.position.clone(),
                        "dwell_time": agent.state.dwell_time.clone(),
                        "expected_dwell": agent.state.current_expected_dwell_time.clone(),
                        "passengers": agent.passenger_count.clone(),
                    }
                )
            return snapshots

        def _snapshot_waiting() -> Dict[int, torch.Tensor]:
            return {
                node_id: node.waiting_passengers.clone()
                for node_id, node in world.platform_nodes.items()
            }

        agent_snapshots = _snapshot_agents()
        waiting_snapshots = _snapshot_waiting()

        def _restore_world() -> None:
            for agent, snap in zip(world.agents, agent_snapshots):
                agent.state.position.copy_(snap["position"])
                agent.state.dwell_time.copy_(snap["dwell_time"])
                agent.state.current_expected_dwell_time.copy_(snap["expected_dwell"])
                agent.passenger_count.copy_(snap["passengers"])
            for node_id, tensor in waiting_snapshots.items():
                if node_id in world.platform_nodes:
                    world.platform_nodes[node_id].waiting_passengers.copy_(tensor)

        tick_seconds = int(getattr(world, "tick_seconds", 1) or 1)

        def _resolve_expected_dwell(node_id: int, fallback: int = 1) -> int:
            node = world.platform_nodes.get(node_id)
            if node is None:
                return 0
            weight = getattr(node, "weight", None)
            if weight is not None and weight > 0:
                return int(weight)
            dwell_seconds = getattr(node, "dwell", None) or getattr(node, "dwell_seconds", None)
            if dwell_seconds is None:
                return fallback
            return max(1, int(math.ceil(float(dwell_seconds) / tick_seconds)))

        sim_agent_states: List[Dict[str, Any]] = []
        for agent in world.agents:
            position = int(agent.state.position[0].item())
            dwell_time = int(agent.state.dwell_time[0].item())
            expected_dwell = int(agent.state.current_expected_dwell_time[0].item())
            passenger_count = int(agent.passenger_count[0].item())
            capacity = getattr(agent, "passenger_capacity", 2520)
            fallback_expected = _resolve_expected_dwell(position, fallback=1)
            resolved_expected = expected_dwell if expected_dwell > 0 else fallback_expected
            sim_agent_states.append(
                {
                    "position": position,
                    "dwell_time": dwell_time,
                    "expected_dwell": max(0, resolved_expected),
                    "default_expected": max(1, fallback_expected),
                    "passenger_count": passenger_count,
                    "capacity": capacity,
                }
            )

        waiting_map: Dict[int, float] = {
            node_id: float(node.waiting_passengers[0].item())
            for node_id, node in world.platform_nodes.items()
        }
        for node_id in world.platform_nodes.keys():
            waiting_map.setdefault(node_id, 0.0)

        def _apply_sim_state_to_world() -> None:
            for agent, state in zip(world.agents, sim_agent_states):
                agent.state.position[0] = float(state["position"])
                agent.state.dwell_time[0] = float(state["dwell_time"])
                agent.state.current_expected_dwell_time[0] = float(state["expected_dwell"])
                agent.passenger_count[0] = float(state["passenger_count"])
            for node_id, node in world.platform_nodes.items():
                node.waiting_passengers[0] = float(waiting_map.get(node_id, 0.0))

        plan_steps: List[Dict[str, Dict[str, Any]]] = []

        try:
            for step_idx in range(plan_horizon):
                _apply_sim_state_to_world()

                waiting_sorted = sorted(waiting_map.items(), key=lambda item: item[1], reverse=True)
                focus_nodes = [nid for nid, wait in waiting_sorted[:initial_columns] if wait > 0]
                if not focus_nodes:
                    focus_nodes = [nid for nid, _ in waiting_sorted[:initial_columns]]
                if not focus_nodes:
                    focus_nodes = list(world.platform_nodes.keys())[:initial_columns]

                if not focus_nodes:
                    focus_nodes = list(world.platform_nodes.keys())

                focus_nodes = focus_nodes[:max_columns]

                step_plan: Dict[str, Dict[str, Any]] = {}

                for iteration in range(max_iterations):
                    try:
                        actions = solver.solve(
                            base_env,
                            focus_agent_ids=None,
                            focus_node_ids=focus_nodes,
                            time_horizon=config.get("time_horizon"),
                            time_limit=config.get("time_limit"),
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"!! 列生成第{iteration + 1}轮求解失败: {exc}")
                        break

                    plan_wrapper = self._actions_to_plan(
                        actions,
                        solver,
                        method_tag=f"列生成-第{iteration + 1}轮",
                    )
                    candidate_steps = plan_wrapper.get("steps") if isinstance(plan_wrapper, dict) else None
                    candidate_plan = (candidate_steps or [{}])[0] if candidate_steps else {}
                    step_plan = candidate_plan or {}

                    served_nodes = {
                        info.get("next_node")
                        for info in step_plan.values()
                        if info.get("action", 0) == 1
                    }
                    uncovered = [nid for nid, _ in waiting_sorted if nid not in served_nodes]
                    if not uncovered:
                        break
                    newly_added = False
                    for nid in uncovered:
                        if nid not in focus_nodes:
                            focus_nodes.append(nid)
                            newly_added = True
                            if len(focus_nodes) >= max_columns:
                                break
                    if not newly_added:
                        break

                if not step_plan:
                    fallback_step: Dict[str, Dict[str, Any]] = {}
                    for agent_id, state in enumerate(sim_agent_states):
                        current_pos = int(state["position"])
                        fallback_step[f"agent_{agent_id}"] = {
                            "action": 0,
                            "next_node": current_pos,
                            "reasoning": f"列生成: 缺省停留策略 | 计划步{step_idx + 1}",
                        }
                    step_plan = fallback_step

                for decision in step_plan.values():
                    reason = decision.get("reasoning", "")
                    suffix = f"列生成 计划步{step_idx + 1}"
                    if suffix not in reason:
                        decision["reasoning"] = f"{reason} | {suffix}" if reason else suffix

                plan_steps.append(step_plan)

                for agent_id, state in enumerate(sim_agent_states):
                    decision = step_plan.get(f"agent_{agent_id}", {})
                    action = int(decision.get("action", 0))
                    current_pos = int(state["position"])
                    raw_next = decision.get("next_node", current_pos)
                    try:
                        target_node = int(raw_next)
                    except (TypeError, ValueError):
                        target_node = current_pos

                    if action == 1:
                        state["position"] = target_node
                        state["dwell_time"] = 0
                        expected = _resolve_expected_dwell(target_node, state["default_expected"])
                        state["expected_dwell"] = max(0, expected)
                        
                        # Simulate alighting
                        state["passenger_count"] = int(state["passenger_count"] * 0.8)
                        
                        if service_capacity > 0:
                            capacity = state["capacity"]
                            available = max(0, capacity - state["passenger_count"])
                            pickup = min(waiting_map.get(target_node, 0.0), available, service_capacity)
                            waiting_map[target_node] = max(0.0, waiting_map.get(target_node, 0.0) - pickup)
                            state["passenger_count"] += pickup
                    else:
                        state["dwell_time"] = min(state["expected_dwell"], state["dwell_time"] + 1)
                        if service_capacity > 0:
                            capacity = state["capacity"]
                            available = max(0, capacity - state["passenger_count"])
                            pickup = min(waiting_map.get(current_pos, 0.0), available, service_capacity)
                            waiting_map[current_pos] = max(0.0, waiting_map.get(current_pos, 0.0) - pickup)
                            state["passenger_count"] += pickup

                # Apply arrival growth (per-node rate + global bias)
                for node_id in list(waiting_map.keys()):
                    rate = node_arrival_rates.get(node_id, 0.0)
                    waiting_map[node_id] = max(0.0, waiting_map[node_id] + rate * tick_seconds + arrival_growth)

            return {"steps": plan_steps}
        finally:
            _restore_world()

    def _solve_benders_decomposition(self, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        base_env = self.policy.env.base_env
        world = base_env.scenario.world

        def _coerce_int_list(raw: Any) -> Set[int]:
            result: Set[int] = set()
            if raw is None:
                return result
            if isinstance(raw, (list, tuple, set)):
                for item in raw:
                    try:
                        result.add(int(item))
                    except (TypeError, ValueError):
                        continue
                return result
            if isinstance(raw, str):
                cleaned = raw.strip().strip("[]{}")
                for token in re.split(r"[,\s]+", cleaned):
                    if not token:
                        continue
                    try:
                        result.add(int(token))
                    except (TypeError, ValueError):
                        continue
            return result

        focus_nodes = _coerce_int_list(config.get("focus_nodes"))
        focus_bonus = float(config.get("focus_bonus", 12.0))
        conflict_penalty = float(config.get("conflict_penalty", 8.0))
        travel_bias = float(config.get("travel_bias", 0.5))
        stay_focus_weight = float(config.get("stay_focus_weight", 0.5))
        dwell_slack = int(config.get("dwell_slack", 0))
        subproblem_horizon = config.get("subproblem_horizon")
        subproblem_limit = config.get("subproblem_limit")
        validate_subproblem = subproblem_horizon is not None or subproblem_limit is not None

        default_horizon = getattr(self.policy, "plan_horizon", 1)
        plan_horizon_raw = config.get("plan_horizon")
        try:
            plan_horizon = int(plan_horizon_raw) if plan_horizon_raw is not None else int(default_horizon)
        except (TypeError, ValueError):
            plan_horizon = int(default_horizon)
        plan_horizon = max(plan_horizon, 1)

        tick_seconds = int(getattr(world, "tick_seconds", 1) or 1)

        def _resolve_expected_dwell(node_id: int, fallback: int = 1) -> int:
            node = world.platform_nodes.get(node_id)
            if node is None:
                return 0
            weight = getattr(node, "weight", None)
            if weight is not None and weight > 0:
                return int(weight)
            dwell_seconds = getattr(node, "dwell", None) or getattr(node, "dwell_seconds", None)
            if dwell_seconds is None:
                return fallback
            return max(1, int(math.ceil(float(dwell_seconds) / tick_seconds)))

        def _snapshot_agents() -> List[Dict[str, torch.Tensor]]:
            return [
                {
                    "position": agent.state.position.clone(),
                    "dwell_time": agent.state.dwell_time.clone(),
                    "expected_dwell": agent.state.current_expected_dwell_time.clone(),
                    "passengers": agent.passenger_count.clone(),
                }
                for agent in world.agents
            ]

        def _snapshot_waiting() -> Dict[int, torch.Tensor]:
            return {
                node_id: node.waiting_passengers.clone()
                for node_id, node in world.platform_nodes.items()
            }

        agent_snapshots = _snapshot_agents()
        waiting_snapshots = _snapshot_waiting()

        def _restore_world() -> None:
            for agent, snap in zip(world.agents, agent_snapshots):
                agent.state.position.copy_(snap["position"])
                agent.state.dwell_time.copy_(snap["dwell_time"])
                agent.state.current_expected_dwell_time.copy_(snap["expected_dwell"])
                agent.passenger_count.copy_(snap["passengers"])
            for node_id, tensor in waiting_snapshots.items():
                if node_id in world.platform_nodes:
                    world.platform_nodes[node_id].waiting_passengers.copy_(tensor)

        sim_agent_states: List[Dict[str, Any]] = []
        for agent in world.agents:
            position = int(agent.state.position[0].item())
            dwell_time = int(agent.state.dwell_time[0].item())
            expected_dwell = int(agent.state.current_expected_dwell_time[0].item())
            passenger_count = int(agent.passenger_count[0].item())
            capacity = getattr(agent, "passenger_capacity", 2520)
            fallback_expected = _resolve_expected_dwell(position, fallback=1)
            resolved_expected = expected_dwell if expected_dwell > 0 else fallback_expected
            sim_agent_states.append(
                {
                    "position": position,
                    "dwell_time": dwell_time,
                    "expected_dwell": max(0, resolved_expected),
                    "default_expected": max(1, fallback_expected),
                    "passenger_count": passenger_count,
                    "capacity": capacity,
                }
            )

        waiting_map: Dict[int, float] = {
            node_id: float(node.waiting_passengers[0].item())
            for node_id, node in world.platform_nodes.items()
        }
        for node_id in world.platform_nodes.keys():
            waiting_map.setdefault(node_id, 0.0)

        service_capacity = max(0, int(config.get("service_capacity", 80)))
        arrival_growth = float(config.get("arrival_growth", 0.0))

        # Capture per-node arrival rates
        node_arrival_rates: Dict[int, float] = {}
        for nid, node in world.platform_nodes.items():
            node_arrival_rates[nid] = float(getattr(node, "passenger_arrival_rate", 0.0))

        def _apply_sim_state_to_world() -> None:
            for agent, state in zip(world.agents, sim_agent_states):
                agent.state.position[0] = float(state["position"])
                agent.state.dwell_time[0] = float(state["dwell_time"])
                agent.state.current_expected_dwell_time[0] = float(state["expected_dwell"])
                agent.passenger_count[0] = float(state["passenger_count"])
            for node_id, node in world.platform_nodes.items():
                node.waiting_passengers[0] = float(waiting_map.get(node_id, 0.0))

        def _lookahead_demand(start_node_id: int, limit: int = 100) -> float:
            """BFS lookahead to find downstream demand."""
            queue = deque([(start_node_id, 0)])
            visited = {start_node_id}
            while queue:
                curr, depth = queue.popleft()
                if curr in waiting_map and waiting_map[curr] > 0:
                    # Found demand, return discounted value
                    return waiting_map[curr] * (0.99 ** depth)
                if depth >= limit:
                    continue
                
                node_obj = world.nodes.get(curr)
                if node_obj:
                    next_map = node_obj.getnext() or {}
                    for next_id in next_map.values():
                        if next_id is not None and next_id not in visited:
                            visited.add(next_id)
                            queue.append((next_id, depth + 1))
            return 0.0

        plan_steps: List[Dict[str, Dict[str, Any]]] = []

        try:
            for step_idx in range(plan_horizon):
                _apply_sim_state_to_world()
                agent_candidates: Dict[str, Dict[str, Any]] = {}
                node_claims: Dict[int, List[str]] = defaultdict(list)

                for agent_id, state in enumerate(sim_agent_states):
                    key = f"agent_{agent_id}"
                    current_pos = int(state["position"])
                    dwell_time = int(state["dwell_time"])
                    expected_dwell = int(state["expected_dwell"])
                    required_dwell = max(0, expected_dwell - dwell_slack)

                    if current_pos in world.platform_nodes and dwell_time < required_dwell:
                        reasoning = f"Benders 主问题: 完成停站需求 ({dwell_time}/{required_dwell})"
                        agent_candidates[key] = {
                            "action": 0,
                            "next_node": current_pos,
                            "score": -math.inf,
                            "reasoning": reasoning,
                        }
                        continue

                    next_nodes = world.get_next_nodes(current_pos)
                    
                    # Filter blockages
                    disrupted_segments = world.get_disrupted_segments()
                    valid_next_nodes = []
                    for node in next_nodes:
                        node_id = int(node.id)
                        seg_fwd = f"{current_pos}-{node_id}"
                        seg_bwd = f"{node_id}-{current_pos}"
                        is_blocked = False
                        if seg_fwd in disrupted_segments and disrupted_segments[seg_fwd]['type'] == 'BLOCKAGE':
                            is_blocked = True
                        elif seg_bwd in disrupted_segments and disrupted_segments[seg_bwd]['type'] == 'BLOCKAGE':
                            is_blocked = True
                        
                        if not is_blocked:
                            valid_next_nodes.append(node)
                    next_nodes = valid_next_nodes

                    if not next_nodes:
                        reasoning = "Benders 主问题: 无可行去向(或受阻)，保持停留"
                        agent_candidates[key] = {
                            "action": 0,
                            "next_node": current_pos,
                            "score": -math.inf,
                            "reasoning": reasoning,
                        }
                        continue

                    best_target = current_pos
                    best_score = -math.inf
                    best_reason = ""
                    for node in next_nodes:
                        node_id = int(node.id)
                        # Use lookahead to find downstream demand
                        demand = _lookahead_demand(node_id)
                        if focus_nodes and node_id in focus_nodes:
                            demand += focus_bonus
                        pending_penalty = conflict_penalty * len(node_claims[node_id])
                        score = demand - pending_penalty
                        if score > best_score:
                            best_score = score
                            best_target = node_id
                            best_reason = f"需求 {demand:.1f}"

                    stay_score = waiting_map.get(current_pos, 0.0)
                    if focus_nodes and current_pos in focus_nodes:
                        stay_score += focus_bonus * stay_focus_weight

                    move_score = best_score + travel_bias
                    if current_pos not in world.platform_nodes:
                        move_score += travel_bias

                    should_move = (
                        best_target != current_pos
                        and (
                            move_score > stay_score
                            or current_pos not in world.platform_nodes
                        )
                        and best_score > -math.inf
                    )

                    if not should_move:
                        delta = move_score - stay_score
                        reasoning = f"Benders 主问题: 保持 {current_pos} (Δ={delta:.1f})"
                        if focus_nodes and current_pos in focus_nodes:
                            reasoning += " | 焦点站点服务"
                        agent_candidates[key] = {
                            "action": 0,
                            "next_node": current_pos,
                            "score": stay_score,
                            "reasoning": reasoning,
                        }
                        continue

                    if focus_nodes and best_target in focus_nodes:
                        best_reason += " | 焦点站点"
                    if not best_reason:
                        best_reason = f"需求 {waiting_map.get(best_target, 0.0):.1f}"

                    delta = move_score - stay_score
                    reasoning = f"Benders 主问题: 前往 {best_target} ({best_reason}, Δ={delta:.1f})"
                    agent_candidates[key] = {
                        "action": 1,
                        "next_node": best_target,
                        "score": move_score,
                        "reasoning": reasoning,
                    }
                    node_claims[best_target].append(key)

                # 冲突处理：同一节点仅允许得分最高的列车前往
                for node_id, claimants in node_claims.items():
                    if len(claimants) <= 1:
                        continue
                    sorted_claimants = sorted(
                        claimants,
                        key=lambda agent_key: agent_candidates[agent_key]["score"],
                        reverse=True,
                    )
                    keepers = set(sorted_claimants[:1])
                    for agent_key in claimants:
                        if agent_key in keepers:
                            agent_candidates[agent_key]["reasoning"] += " | 冲突获批"
                            continue
                        agent_candidates[agent_key] = {
                            "action": 0,
                            "next_node": int(sim_agent_states[int(agent_key.split('_')[1])]["position"]),
                            "score": agent_candidates[agent_key]["score"],
                            "reasoning": "Benders 主问题: 冲突回退，保持停留",
                        }

                step_plan: Dict[str, Dict[str, Any]] = {
                    agent_key: {
                        "action": data["action"],
                        "next_node": data["next_node"],
                        "reasoning": data["reasoning"],
                    }
                    for agent_key, data in agent_candidates.items()
                }

                # 子问题校验：对焦点节点或显式配置的列车执行 MILP 验证
                if validate_subproblem:
                    sub_payload = {
                        "objective": config.get("objective", "minimize_wait_time"),
                    }
                    if focus_nodes:
                        sub_payload["focus_nodes"] = list(focus_nodes)
                    if subproblem_horizon is not None:
                        sub_payload["time_horizon"] = subproblem_horizon
                    if subproblem_limit is not None:
                        sub_payload["time_limit"] = subproblem_limit

                    for agent_key, decision in step_plan.items():
                        if decision.get("action", 0) != 1:
                            continue

                        agent_idx = int(agent_key.split("_")[-1])
                        target_node = decision.get("next_node")
                        if focus_nodes and target_node not in focus_nodes:
                            continue

                        local = self.policy._solve_local_problem_with_gurobi(
                            agent_idx,
                            json.dumps(sub_payload),
                        )
                        if local.get("action", 0) == 0:
                            current_pos = int(sim_agent_states[agent_idx]["position"])
                            step_plan[agent_key] = {
                                "action": 0,
                                "next_node": current_pos,
                                "reasoning": f"{decision.get('reasoning', '')} | 子问题否决移动",
                            }
                        else:
                            approved_target = local.get("next_node", target_node)
                            step_plan[agent_key] = {
                                "action": 1,
                                "next_node": int(approved_target),
                                "reasoning": f"{decision.get('reasoning', '')} | 子问题确认可行",
                            }

                if not step_plan:
                    fallback_step: Dict[str, Dict[str, Any]] = {}
                    for agent_id, state in enumerate(sim_agent_states):
                        current_pos = int(state["position"])
                        fallback_step[f"agent_{agent_id}"] = {
                            "action": 0,
                            "next_node": current_pos,
                            "reasoning": f"Benders: 缺省停留策略 | 计划步{step_idx + 1}",
                        }
                    step_plan = fallback_step

                for decision in step_plan.values():
                    reason = decision.get("reasoning", "")
                    suffix = f"Benders 计划步{step_idx + 1}"
                    if suffix not in reason:
                        decision["reasoning"] = f"{reason} | {suffix}" if reason else suffix

                plan_steps.append(step_plan)

                for agent_id, state in enumerate(sim_agent_states):
                    decision = step_plan.get(f"agent_{agent_id}", {})
                    action = int(decision.get("action", 0))
                    current_pos = int(state["position"])
                    raw_next = decision.get("next_node", current_pos)
                    try:
                        target_node = int(raw_next)
                    except (TypeError, ValueError):
                        target_node = current_pos

                    if action == 1:
                        state["position"] = target_node
                        state["dwell_time"] = 0
                        expected = _resolve_expected_dwell(target_node, state["default_expected"])
                        state["expected_dwell"] = max(0, expected)
                        
                        # Simulate alighting
                        state["passenger_count"] = int(state["passenger_count"] * 0.8)
                        
                        if service_capacity > 0:
                            capacity = state["capacity"]
                            available = max(0, capacity - state["passenger_count"])
                            pickup = min(waiting_map.get(target_node, 0.0), available, service_capacity)
                            waiting_map[target_node] = max(0.0, waiting_map.get(target_node, 0.0) - pickup)
                            state["passenger_count"] += pickup
                    else:
                        state["dwell_time"] = min(state["expected_dwell"], state["dwell_time"] + 1)
                        if service_capacity > 0:
                            capacity = state["capacity"]
                            available = max(0, capacity - state["passenger_count"])
                            pickup = min(waiting_map.get(current_pos, 0.0), available, service_capacity)
                            waiting_map[current_pos] = max(0.0, waiting_map.get(current_pos, 0.0) - pickup)
                            state["passenger_count"] += pickup

                # Apply arrival growth (per-node rate + global bias)
                for node_id in list(waiting_map.keys()):
                    rate = node_arrival_rates.get(node_id, 0.0)
                    waiting_map[node_id] = max(0.0, waiting_map[node_id] + rate * tick_seconds + arrival_growth)

            return {"steps": plan_steps}
        finally:
            _restore_world()

    def _solve_queueing_analysis(self, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        base_env = self.policy.env.base_env
        world = base_env.scenario.world

        queue_threshold = float(config.get("queue_threshold") or 0.0)
        dwell_slack = int(config.get("dwell_slack") or 0)

        plan_horizon_raw = config.get("plan_horizon")
        plan_horizon = int(plan_horizon_raw)

        tick_seconds = int(getattr(world, "tick_seconds", 1) or 1)

        def _resolve_expected_dwell(node_id: int, fallback: int = 1) -> int:
            node = world.platform_nodes.get(node_id)
            if node is None:
                return 0
            weight = getattr(node, "weight", None)
            if weight is not None and weight > 0:
                return int(weight)
            dwell_seconds = getattr(node, "dwell", None) or getattr(node, "dwell_seconds", None)
            if dwell_seconds is None:
                return fallback
            return max(1, int(math.ceil(float(dwell_seconds) / tick_seconds)))

        sim_agent_states: List[Dict[str, Any]] = []
        for agent in world.agents:
            position = int(agent.state.position[0].item())
            dwell_time = int(agent.state.dwell_time[0].item())
            expected_dwell = int(agent.state.current_expected_dwell_time[0].item())
            passenger_count = int(agent.passenger_count[0].item())
            capacity = getattr(agent, "passenger_capacity", 2520)
            fallback_expected = _resolve_expected_dwell(position, fallback=1)
            resolved_expected = expected_dwell if expected_dwell > 0 else fallback_expected
            sim_agent_states.append({
                "position": position,
                "dwell_time": dwell_time,
                "expected_dwell": max(0, resolved_expected),
                "default_expected": max(1, fallback_expected),
                "passenger_count": passenger_count,
                "capacity": capacity,
            })

        waiting_map: Dict[int, float] = {
            node_id: float(node.waiting_passengers[0].item())
            for node_id, node in world.platform_nodes.items()
        }

        service_capacity = max(0, int(config.get("service_capacity", 80)))
        arrival_growth = float(config.get("arrival_growth", 0.0))

        # Capture per-node arrival rates
        node_arrival_rates: Dict[int, float] = {}
        for nid, node in world.platform_nodes.items():
            node_arrival_rates[nid] = float(getattr(node, "passenger_arrival_rate", 0.0))

        plan_steps: List[Dict[str, Dict[str, Any]]] = []

        def _lookahead_demand(start_node_id: int, limit: int = 100) -> float:
            """BFS lookahead to find downstream demand."""
            queue = deque([(start_node_id, 0)])
            visited = {start_node_id}
            while queue:
                curr, depth = queue.popleft()
                if curr in waiting_map and waiting_map[curr] > 0:
                    return waiting_map[curr] * (0.99 ** depth)
                if depth >= limit:
                    continue
                
                node_obj = world.nodes.get(curr)
                if node_obj:
                    next_map = node_obj.getnext() or {}
                    for next_id in next_map.values():
                        if next_id is not None and next_id not in visited:
                            visited.add(next_id)
                            queue.append((next_id, depth + 1))
            return 0.0

        def _queue_score(load: float) -> float:
            score = load
            if load < queue_threshold:
                score -= queue_threshold
            return score

        for step_idx in range(plan_horizon):
            step_plan: Dict[str, Dict[str, Any]] = {}

            for agent_id, agent_state in enumerate(sim_agent_states):
                current_pos = int(agent_state["position"])
                dwell_time = int(agent_state["dwell_time"])
                expected_dwell = int(agent_state["expected_dwell"])

                key = f"agent_{agent_id}"
                required_dwell = max(0, expected_dwell - dwell_slack)
                if current_pos in world.platform_nodes and dwell_time < required_dwell:
                    step_plan[key] = {
                        "action": 0,
                        "next_node": current_pos,
                        "reasoning": (
                            f"排队论: 停站时间不足, 继续服务当前站 | 计划步{step_idx + 1}"
                        ),
                    }
                    continue

                next_nodes = world.get_next_nodes(current_pos)
                best_next = current_pos
                best_score = -math.inf
                
                capacity = agent_state["capacity"]
                available = max(0, capacity - agent_state["passenger_count"])
                
                for node in next_nodes:
                    # Use lookahead for queueing analysis too
                    load = _lookahead_demand(node.id)
                    
                    # If full, reduce attraction to load
                    if available <= 0:
                        load *= 0.1
                        
                    score = _queue_score(load)
                    if score > best_score + 1e-9:
                        best_score = score
                        best_next = node.id

                if best_next == current_pos:
                    step_plan[key] = {
                        "action": 0,
                        "next_node": current_pos,
                        "reasoning": (
                            f"排队论: 邻近站点需求低, 保持当前位置 | 计划步{step_idx + 1}"
                        ),
                    }
                else:
                    step_plan[key] = {
                        "action": 1,
                        "next_node": int(best_next),
                        "reasoning": (
                            f"排队论: 优先前往乘客最多的下一站 | 计划步{step_idx + 1}"
                        ),
                    }

            if not step_plan:
                fallback_step: Dict[str, Dict[str, Any]] = {}
                for agent_id, agent_state in enumerate(sim_agent_states):
                    current_pos = int(agent_state["position"])
                    fallback_step[f"agent_{agent_id}"] = {
                        "action": 0,
                        "next_node": current_pos,
                        "reasoning": f"排队论: 缺省停留策略 | 计划步{step_idx + 1}",
                    }
                step_plan = fallback_step

            plan_steps.append(step_plan)

            for agent_id, agent_state in enumerate(sim_agent_states):
                decision = step_plan.get(f"agent_{agent_id}", {})
                action = int(decision.get("action", 0))
                current_pos = int(agent_state["position"])
                raw_next = decision.get("next_node", agent_state["position"])
                try:
                    target_node = int(raw_next)
                except (TypeError, ValueError):
                    target_node = current_pos

                if action == 1:
                    agent_state["position"] = target_node
                    agent_state["dwell_time"] = 0
                    expected = _resolve_expected_dwell(target_node, agent_state["default_expected"])
                    agent_state["expected_dwell"] = max(0, expected)
                    
                    # Simulate alighting
                    agent_state["passenger_count"] = int(agent_state["passenger_count"] * 0.8)
                    
                    if service_capacity > 0:
                        capacity = agent_state["capacity"]
                        available = max(0, capacity - agent_state["passenger_count"])
                        pickup = min(waiting_map.get(target_node, 0.0), available, service_capacity)
                        waiting_map[target_node] = max(0.0, waiting_map.get(target_node, 0.0) - pickup)
                        agent_state["passenger_count"] += pickup
                else:
                    agent_state["dwell_time"] = min(
                        agent_state["expected_dwell"],
                        agent_state["dwell_time"] + 1,
                    )
                    if service_capacity > 0:
                        capacity = agent_state["capacity"]
                        available = max(0, capacity - agent_state["passenger_count"])
                        pickup = min(waiting_map.get(current_pos, 0.0), available, service_capacity)
                        waiting_map[current_pos] = max(0.0, waiting_map.get(current_pos, 0.0) - pickup)
                        agent_state["passenger_count"] += pickup

            # Apply arrival growth (per-node rate + global bias)
            for node_id in list(waiting_map.keys()):
                rate = node_arrival_rates.get(node_id, 0.0)
                waiting_map[node_id] = max(0.0, waiting_map[node_id] + rate * tick_seconds + arrival_growth)

        return {"steps": plan_steps}

class LLMAgentPolicy:
    """
    一个基于工具的LLM Agent策略，用于地铁调度。
    Agent可以自主选择工具来查询环境、使用CPLEX解决局部问题或进行全局调度，
    并通过约束验证器确保决策的安全性。
    """
    def __init__(self, env, api_key: str, base_url: str = None, n_agents: int = None, device="cpu", model_path: str = None,
                 # 新增：为工具3（全局指令）配置独立的本地模型
                 global_model_path: str = None,
                 global_base_url: str = None,
                 global_api_key: str = None,
                 local_milp_time_horizon: int = 20,
                 local_milp_time_limit: int = 1000,
                 rag_db_path: str = None,
                 llm_max_input_tokens: int = DEFAULT_INPUT_TOKENS,
                 llm_max_output_tokens: int = DEFAULT_OUTPUT_TOKENS,
                 llm_request_timeout_s: float = 30.,
                 llm_thinking_mode: str = "disabled",
                 llm_reasoning_effort: str = "low"):
        if not api_key or api_key == "YOUR_OPENAI_API_KEY":
            raise ValueError("API Key未设置。请在'main_llm.py'中提供您的API Key。")
        if ChatOpenAI is None:
            raise ImportError("LLM inference requires langchain-openai and langchain-core; native no-LLM validation does not")

        self.env = env
        self.n_agents = n_agents
        self.device = device
        self.action_key = ("agents", "action")
        self.world = self.env.base_env.scenario.world if self.env else None
        self.local_milp_solver: Optional[MILPSolver] = None
        self.local_milp_time_horizon = local_milp_time_horizon
        self.local_milp_time_limit = local_milp_time_limit
        self.plan_horizon: int = max(int(local_milp_time_horizon), 1)
        check_messages([], llm_max_input_tokens)
        if type(llm_max_output_tokens) is not int or llm_max_output_tokens < 1:
            raise ValueError("LLM output limit must be a positive integer")
        self.llm_max_input_tokens = llm_max_input_tokens
        self.llm_max_output_tokens = llm_max_output_tokens
        if not math.isfinite(llm_request_timeout_s) or llm_request_timeout_s <= 0:
            raise ValueError("LLM request timeout must be finite and positive")
        if llm_thinking_mode not in ("auto", "disabled", "enabled") or llm_reasoning_effort not in ("low", "high", "max"):
            raise ValueError("Invalid LLM thinking configuration")
        self.llm_request_timeout_s = llm_request_timeout_s
        self.llm_thinking_mode = llm_thinking_mode
        self.llm_reasoning_effort = llm_reasoning_effort
        self.last_prompt_audit = None

        # 初始化 RAG 管理器
        # self.rag_manager = None
        # if rag_db_path:
        #     if os.path.exists(rag_db_path):
        #         self.rag_manager = RAGManager(rag_db_path)
        #         self.rag_manager.load_database()
        #         print(f"--- [RAG] 已加载知识库: {rag_db_path} ({len(self.rag_manager.cases)} 案例) ---")
        #     else:
        #         print(f"--- [RAG] 警告: 知识库路径不存在: {rag_db_path} ---")
        # else:
        #     # 尝试默认路径
        #     default_rag_path = os.path.join(os.path.dirname(__file__), "rag", "knowledge_base.jsonl")
        #     if os.path.exists(default_rag_path):
        #         self.rag_manager = RAGManager(default_rag_path)
        #         self.rag_manager.load_database()
        #         print(f"--- [RAG] 已加载默认知识库: {default_rag_path} ({len(self.rag_manager.cases)} 案例) ---")

        # 初始化 LangChain 组件（主Agent用 Qwen/Qwen3-30B-A3B-Instruct-2507）
        self.llm = ChatOpenAI(
            model=model_path,
            temperature=0.1,
            api_key=api_key,
            base_url=base_url,
            max_retries=2,
            request_timeout=120,
        )

        # 新增：工具3专用的本地LLM（使用OpenAI兼容服务加载本地权重）
        # - 如果未传 global_base_url/global_api_key，则复用外层 base_url/api_key
        # - model 支持直接给本地路径（服务端需支持以路径为model标识）
        self.global_llm = ChatOpenAI(
            model=global_model_path,
            temperature=0.1,
            api_key=global_api_key,
            base_url=global_base_url,
            max_retries=0,
            max_tokens=self.llm_max_output_tokens,
            request_timeout=120,
        )

        self.prompt_template = self._create_agent_prompt_template()
        self.output_parser = StrOutputParser()
        self.chain = self.prompt_template | self.llm | self.output_parser
        self.or_toolkit = OperationsResearchToolkit(self)
        self.cached_plan_steps: List[Dict[str, Dict[str, Any]]] = []
        self.cached_plan_index: int = 0
        self.cached_plan_env_signature: Optional[Tuple[Any, ...]] = None
        self._last_verified_step_plan: Dict[str, Dict[str, Any]] = {}
        self.last_decision_source: str = "MARL"
        self.sim_start_step: Optional[int] = None
        self._last_validation_error: str = ""


    def _create_agent_prompt_template(self):
        """定义指导LLM Agent的系统提示，包含工具使用的说明。"""
        system_template_str = """
        你是一名拥有高级工具的地铁调度AI指挥官。你的任务是确保整个地铁网络高效、安全地运行。
        你的核心优化目标是：
        1. 最小化平均乘客等待时间 (Average Passenger Waiting Time)。
        2. 最小化平均行车间隔标准差 (Average Headway Standard Deviation)，即保持列车间隔均匀。

        你将以一步一步的方式进行思考和行动。在每一步中，你都必须严格遵循以下格式：

        **思考**: 首先，简要分析当前情况，并决定下一步需要做什么。
        **行动**: 根据你的思考，选择并调用**一个**工具。你的输出必须以 `行动:` 开头，并且只包含工具调用本身。

        **重要规则:**
        1.  **单步执行**: 在你的每一次回复中，只能包含一次“思考”和一次“行动”。
        2.  **严禁假设**: **绝对不要**猜测或模拟工具的返回结果。在调用工具后，你的回复必须立即结束。系统会为你执行工具并提供真实结果。
        3.  **高效查询**: **首先分析初始状态摘要**。不要使用 `query_environment` 来获取已经在摘要中给出的全局信息（如完整的 `disruptions` 列表）。你的查询应该是为了获取更具体、更深入的局部信息。
        4.  **最终决策 (一次性规划__PLAN_HORIZON__步)**: 当你收集到足够的信息后，你的最后一步“行动”必须是调用 `verify_and_finalize` 工具。该工具的 `solution` 参数必须是如下结构:
            - 字典顶层包含键 `"steps"`。
            - `"steps"` 对应一个**长度为__PLAN_HORIZON__**的列表，列表中的每个元素代表未来一步的完整方案。
            - 列表内每一步都是一个完整字典，覆盖 `agent_0` 至 `agent_{{last_agent_index}}` 的所有列车。
            - 若某些列车在某一步保持原计划，也必须显式写出对应的 `action` 和 `next_node`。
            - 对缺乏详细分析的列车，同样要给出符合停站/邻接约束的默认动作。

    **可用工具:**
    1. `query_environment(query: str)`: 查询环境状态。可用查询包括: `disruptions`, `train_status(agent_id)`, `station_status(node_id)`. 例如: `query_environment("train_status(agent_3)")`
    2. `solve_local_problem_with_gurobi(agent_id: int, objective: str)`: 调用MILP模型为单个或少量列车求解局部最优行动。`objective` 可以是简单字符串（如 `"minimize_wait_time"`, `"minimize_headway_deviation"`），也可以是JSON，支持 `focus_nodes`, `focus_agents`, `time_horizon` (一般取20，最多不超过40), `time_limit` (最多取1000) 等字段。例如: `solve_local_problem_with_gurobi(agent_id=5, objective="{{\\"objective\\": \\"minimize_wait_time\\", \\"focus_nodes\\": [151, 152]}}")`
    3. `get_lagrangian_plan(config: str = None)`: 使用拉格朗日松弛方法生成全局调度方案，可通过 `config` 指定 `iterations`, `lambda_step`, `time_horizon` (一般取100，最多不超过200) 等参数。
    4. `get_column_generation_plan(config: str = None)`: 使用列生成框架生成调度方案，可通过 `config` 指定 `iterations`, `initial_columns`, `max_columns`, `time_limit`(最多取1000) 等参数。
    5. `get_benders_plan(config: str = None)`: 使用 Benders 分解（主问题 + 子问题校验）生成调度方案，可通过 `config` 指定 `focus_nodes`, `subproblem_horizon`(一般取100，最多不超过200) , `subproblem_limit`(最多取1000) 等参数。
    6. `get_queueing_plan(config: str = None)`: 使用排队论启发式生成调度方案，可通过 `config` 指定 `queue_threshold`, `dwell_slack` 等参数。
    7. `get_llm_global_plan()`: 调用全局LLM生成完整调度方案，作为运筹学方法的备用方案。
    8. `verify_and_finalize(solution: dict)`: 验证并提交**完整的**行动计划。`solution`字典必须包含从 `agent_0` 到 `agent_{{last_agent_index}}` 的每一个列车的决策。格式为：`verify_and_finalize({
        "steps": [
            {
              "agent_0": {"action": 1, "next_node": 102, "reasoning": "完成停站, 前往102"},
              "agent_1": {"action": 0, "next_node": 5, "reasoning": "停站未满, 继续服务"},
              "...": {...}
            },
            {
              "agent_0": {"action": 0, "next_node": 102, "reasoning": "已在途中, 保持"},
              "agent_1": {"action": 1, "next_node": 8, "reasoning": "停站完成, 前往高需求站"},
              "...": {...}
            },
            ... ← 共__PLAN_HORIZON__个连续时步
          ]
        })`

        **示例流程:**
        ---
        **初始状态:** ... 系统警告: 检测到中断! {{'4-5': ...}} ...
        ---
        **你的回复 1:**
        **思考**: 初始摘要显示区间 4-5 有限速。我需要检查是否有列车正驶向这个区间。根据拓扑，节点 4 是进入该区间的关键节点。我将查询 agent_8 的状态，因为它正位于节点 4。
        **行动**: `query_environment("train_status(agent_8)")`

        ---
        **(系统会执行工具并返回):** 环境查询返回结果: Train agent_8: pos=4, passengers=60, next_nodes=[5, 151]...
        ---
        **你的回复 2:**
        **思考**: agent_8 确实在节点 4，并且它的一个可能路径是前往节点 5（进入限速区）。为了避免延误，我需要为它规划最优路径，可能会让它前往节点 151。
        **行动**: `solve_local_problem_with_gurobi(agent_id=8, objective="minimize_wait_time")`

        ---
        **(系统会执行工具并返回):** CPLEX求解器返回结果: ...
        ---
        **你的回复 3:**
        **思考**: 我已经收集了所有必要信息，现在可以形成包含所有列车决策的最终方案了。
        **行动**: ``verify_and_finalize({
          "steps": [
            {
              "agent_0": {"action": 1, "next_node": 102, "reasoning": "完成停站, 前往102"},
              "agent_1": {"action": 0, "next_node": 5, "reasoning": "停站未满, 继续服务"},
              "...": {...}
            },
            {
              "agent_0": {"action": 0, "next_node": 102, "reasoning": "已在途中, 保持"},
              "agent_1": {"action": 1, "next_node": 8, "reasoning": "停站完成, 前往高需求站"},
              "...": {...}
            },
                        ... ← 共__PLAN_HORIZON__个连续时步
          ]
        })`

                如果无法保证__PLAN_HORIZON__步，请在思考中说明原因并继续完善方案，直到能够返回完整的__PLAN_HORIZON__步计划。

        """

        human_template_str = """
        ---
        **当前系统状态摘要:**
        {state_description}
        ---

        **你的任务:**
        现在，请严格按照上述规则，开始你的第一步思考和行动。
        """

        def _escape_except_placeholders(text: str, placeholders: List[str]) -> str:
                escaped = text.replace("{", "{{").replace("}", "}}")
                for name in placeholders:
                    escaped = escaped.replace(f"{{{{{name}}}}}", f"{{{name}}}")
                return escaped

        placeholders = ["state_description", "n_agents", "last_agent_index"]
        system_template_str = _escape_except_placeholders(system_template_str, placeholders)
        system_template_str = system_template_str.replace("__PLAN_HORIZON__", str(self.plan_horizon))
        human_template_str = _escape_except_placeholders(human_template_str, ["state_description"])

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template_str),
            HumanMessagePromptTemplate.from_template(human_template_str)
        ])

    def _get_segment_id(self, world, u: int, v: int) -> Optional[str]:
        """Helper to find segment ID for edge u->v."""
        if not hasattr(self, "_segment_edge_map") or self._segment_edge_map is None:
             self._segment_edge_map = {}
             segment_nodes = getattr(world, "segment2nodes", {}) or {}
             for segment, node_list in segment_nodes.items():
                if not node_list: continue
                for idx in range(len(node_list) - 1):
                    start_id = getattr(node_list[idx], "id", None)
                    end_id = getattr(node_list[idx + 1], "id", None)
                    if start_id is not None and end_id is not None:
                        self._segment_edge_map[(start_id, end_id)] = segment.id
                        self._segment_edge_map[(end_id, start_id)] = segment.id
        
        return self._segment_edge_map.get((u, v))

    # --- 工具实现 ---

    def _query_environment(self, query: str, batch_idx=0) -> str:
        """工具1: 解析LLM的查询并从环境中检索信息。"""
        self.world = self.env.base_env.scenario.world
        
        train_match = re.match(r"train_status\(agent_(\d+)\)", query)
        if train_match:
            agent_id = int(train_match.group(1))
            if 0 <= agent_id < len(self.world.agents):
                agent = self.world.agents[agent_id]
                pos = int(agent.state.position[batch_idx].item())
                next_nodes = [n.id for n in self.world.get_next_nodes(pos)]
                return f"Train agent_{agent_id}: pos={pos}, passengers={int(agent.passenger_count[batch_idx].item())}, dwell_time={int(agent.state.dwell_time[batch_idx].item())}, next_nodes={next_nodes}"
            return f"Error: Invalid agent_id {agent_id}."

        station_match = re.match(r"station_status\((\d+)\)", query)
        if station_match:
            node_id = int(station_match.group(1))
            if node_id in self.world.platform_nodes:
                node = self.world.platform_nodes[node_id]
                return f"Station {node_id}: waiting_passengers={int(node.waiting_passengers[batch_idx].item())}"
            return f"Error: Invalid station_id {node_id}."

        if query == 'disruptions':
            disruptions = self.world.get_disrupted_segments()
            return f"Disruptions: {disruptions if disruptions else 'None'}"
        
        return "Error: Unknown query."

    def _solve_local_problem_with_gurobi(self, agent_id: int, objective: str) -> dict:
        """Use CPLEX; retain the historical tool name for saved agent calls."""
        print(f"--- [Tool] CPLEX求解器被调用: agent_{agent_id}, 目标: {objective} ---")
        try:
            if not self.env or not hasattr(self.env, "base_env"):
                return {"action": 0, "next_node": 0, "reasoning": "CPLEX: Base environment unavailable."}

            base_env = self.env.base_env
            self.world = base_env.scenario.world

            if not (0 <= agent_id < len(self.world.agents)):
                return {"action": 0, "next_node": 0, "reasoning": f"CPLEX: Invalid agent_id {agent_id}."}

            agent = self.world.agents[agent_id]
            current_pos = int(agent.state.position[0].item())

            parsed_objective = self._parse_local_objective_config(objective)
            focus_nodes = parsed_objective.get("focus_nodes") or []
            extra_agents = parsed_objective.get("focus_agents") or []
            local_horizon = parsed_objective.get("time_horizon") or self.local_milp_time_horizon
            local_time_limit = parsed_objective.get("time_limit") or self.local_milp_time_limit

            focus_agent_ids: Set[int] = {agent_id}
            for extra in extra_agents:
                try:
                    idx = int(extra)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(self.world.agents):
                    focus_agent_ids.add(idx)

            if self.local_milp_solver is None:
                self.local_milp_solver = MILPSolver(
                    time_horizon=self.local_milp_time_horizon,
                    time_limit=self.local_milp_time_limit,
                )

            actions_dict = self.local_milp_solver.solve(
                base_env,
                focus_agent_ids=sorted(focus_agent_ids),
                focus_node_ids=focus_nodes if focus_nodes else None,
                time_horizon=local_horizon,
                time_limit=local_time_limit,
                objective_type=parsed_objective.get("objective", "minimize_wait_time"),
            )

            agent_name = agent.name
            action_value = actions_dict.get(agent_name, 0)
            solution_details = getattr(self.local_milp_solver, "last_solution", {}) or {}
            agent_solution = solution_details.get(agent_name, {})

            next_node = agent_solution.get("next_node")
            if next_node is None and action_value > 0:
                possible_nodes = [n.id for n in self.world.get_next_nodes(current_pos)]
                if action_value - 1 < len(possible_nodes):
                    next_node = possible_nodes[action_value - 1]

            if next_node is None:
                next_node = current_pos

            reasoning = agent_solution.get("reason")
            if not reasoning:
                reasoning = f"MILP: objective={parsed_objective.get('objective')}"

            return {"action": int(action_value), "next_node": int(next_node), "reasoning": reasoning}

        except Exception as e:
            print(f"!! CPLEX求解器异常: {e}")
            fallback_node = 0
            try:
                fallback_node = int(self.world.agents[agent_id].state.position[0].item())
            except Exception:
                pass
            return {"action": 0, "next_node": fallback_node, "reasoning": "CPLEX: Exception occurred."}

    def _parse_local_objective_config(self, objective: str) -> Dict[str, Any]:
        """解析LLM传入的objective字符串，提取焦点节点、额外列车以及时间参数。"""
        config: Dict[str, Any] = {
            "objective": "minimize_wait_time",
            "focus_nodes": [],
            "focus_agents": [],
            "time_horizon": None,
            "time_limit": None,
        }

        if not objective:
            return config

        raw = objective.strip()
        if not raw:
            return config

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                config["objective"] = parsed.get("objective", config["objective"])
                config["focus_nodes"] = self._parse_token_list_to_ints(parsed.get("focus_nodes"))
                config["focus_agents"] = self._parse_token_list_to_ints(
                    parsed.get("focus_agents") or parsed.get("agent_ids")
                )
                config["time_horizon"] = self._safe_int(parsed.get("time_horizon"))
                config["time_limit"] = self._safe_int(parsed.get("time_limit"))
                return config
            elif isinstance(parsed, list):
                config["focus_nodes"] = self._parse_token_list_to_ints(parsed)
                return config
            else:
                config["objective"] = raw
                return config
        except json.JSONDecodeError:
            pass

        # 处理非JSON的模式: 例如 "minimize_wait_time; nodes=[101,102]; horizon=6"
        parts = [segment.strip() for segment in raw.split(";") if segment.strip()]
        if parts:
            config["objective"] = parts[0]

        nodes_match = re.search(r"nodes?\s*=\s*\[([^\]]*)\]", raw)
        if nodes_match:
            config["focus_nodes"] = self._parse_token_list_to_ints(nodes_match.group(1))

        agents_match = re.search(r"agents?\s*=\s*\[([^\]]*)\]", raw)
        if agents_match:
            config["focus_agents"] = self._parse_token_list_to_ints(agents_match.group(1))

        horizon_match = re.search(r"time?_?horizon\s*=\s*(\d+)", raw)
        if horizon_match:
            config["time_horizon"] = self._safe_int(horizon_match.group(1))

        limit_match = re.search(r"time?_?limit\s*=\s*(\d+)", raw)
        if limit_match:
            config["time_limit"] = self._safe_int(limit_match.group(1))

        return config

    @staticmethod
    def _parse_token_list_to_ints(values: Any) -> List[int]:
        if values is None:
            return []
        tokens: List[int] = []
        if isinstance(values, (list, tuple)):
            for item in values:
                try:
                    tokens.append(int(item))
                except (TypeError, ValueError):
                    continue
            return tokens

        if isinstance(values, str):
            cleaned = values.strip().strip("[]")
            for item in re.split(r"[,\\s]+", cleaned):
                item = item.strip()
                if not item:
                    continue
                try:
                    tokens.append(int(item))
                except ValueError:
                    continue
        return tokens

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _parse_or_config(self, config_text: Optional[str]) -> Dict[str, Any]:
        if not config_text:
            return {}
        raw = config_text.strip()
        if not raw:
            return {}

        # 处理LLM为转义大括号而额外包裹的情形，例如 {{"key": 1}}
        while raw.startswith("{{") and raw.endswith("}}"):
            raw = raw[1:-1].strip()
            if not raw:
                return {}

        # 去除围绕整个配置的多余引号（如 ' {...}' 或 "{...}")
        if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
            raw = raw[1:-1].strip()
            if not raw:
                return {}

        # 首先尝试严格的 JSON 解析（期望双引号）
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"focus_nodes": parsed}
        except json.JSONDecodeError:
            # 再尝试解析为 Python 字面量（支持单引号的 dict/list）
            try:
                parsed_py = ast.literal_eval(raw)
                if isinstance(parsed_py, dict):
                    return parsed_py
                if isinstance(parsed_py, list):
                    return {"focus_nodes": parsed_py}
            except (ValueError, SyntaxError):
                pass

        config: Dict[str, Any] = {}
        segments = [seg.strip() for seg in re.split(r"[;\n]", raw) if seg.strip()]
        for segment in segments:
            if "=" not in segment:
                continue
            key, value = segment.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            config[key] = self._coerce_config_value(value)
        return config

    @staticmethod
    def _coerce_config_value(raw_value: str) -> Any:
        if not raw_value:
            return raw_value
        value = raw_value.strip().strip(",")
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
            return value[1:-1]
        if value.startswith(("[", "{")) and value.endswith(("]", "}")):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    @staticmethod
    def _parse_tool_kwargs(raw_args: str) -> Dict[str, str]:
        if raw_args is None:
            return {}
        stripped = raw_args.strip()
        if not stripped:
            return {}

        kwargs: Dict[str, str] = {}
        parts: List[str] = []
        current: List[str] = []
        depth_stack: List[str] = []
        in_quote: Optional[str] = None
        escape = False

        for ch in stripped:
            if escape:
                current.append(ch)
                escape = False
                continue

            if ch == "\\":
                current.append(ch)
                escape = True
                continue

            if in_quote:
                current.append(ch)
                if ch == in_quote:
                    in_quote = None
                continue

            if ch in {'"', "'"}:
                in_quote = ch
                current.append(ch)
                continue

            if ch in "({[":
                depth_stack.append(ch)
                current.append(ch)
                continue

            if ch in ")}]":
                if depth_stack:
                    depth_stack.pop()
                current.append(ch)
                continue

            if ch == "," and not depth_stack:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                continue

            current.append(ch)

        if current:
            part = "".join(current).strip()
            if part:
                parts.append(part)

        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value.startswith("`") and value.endswith("`"):
                value = value[1:-1]
            if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                value = value[1:-1]
            try:
                value = bytes(value, "utf-8").decode("unicode_escape")
            except UnicodeDecodeError:
                pass
            kwargs[key] = value

        return kwargs

    def _extract_kwarg_value(self, raw_args: str, key: str) -> Optional[str]:
        """从 raw_args 中安全提取 key 的值（支持 JSON 包含逗号与单/双引号）。"""
        if not raw_args or key not in raw_args:
            return None
        m = re.search(rf"{re.escape(key)}\s*=", raw_args)
        if not m:
            return None
        i = m.end()
        # 跳过空格
        while i < len(raw_args) and raw_args[i].isspace():
            i += 1
        if i >= len(raw_args):
            return None
        start_char = raw_args[i]
        # 值以单/双引号开头
        if start_char in {"'", '"'}:
            quote = start_char
            j = i + 1
            escaped = False
            while j < len(raw_args):
                ch = raw_args[j]
                if ch == "\\" and not escaped:
                    escaped = True
                    j += 1
                    continue
                if ch == quote and not escaped:
                    return raw_args[i + 1 : j]
                escaped = False
                j += 1
            return raw_args[i + 1 : j]  # 回退（不完整时尽可能返回）
        # 值以 { 开头，寻找匹配的 }
        if start_char == "{":
            depth = 0
            j = i
            while j < len(raw_args):
                ch = raw_args[j]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return raw_args[i : j + 1]
                j += 1
            return raw_args[i:]  # 回退
        # 否则取到下一个逗号或结尾
        j = i
        while j < len(raw_args) and raw_args[j] != ",":
            j += 1
        return raw_args[i:j].strip().strip("'\"")

    def _run_or_method(
        self,
        state_description: str,
        method_key: str,
        config: Optional[str] = None,
    ) -> dict:
        config_payload = self._parse_or_config(config)
        print(
            f"--- [Tool] {method_key} 规划调用: config={config_payload} ---"
        )

        plan_dict, diag = self.or_toolkit.solve(method_key, config_payload, state_description)
        if plan_dict:
            if isinstance(plan_dict, dict):
                if "steps" in plan_dict and isinstance(plan_dict["steps"], list):
                    normalized_steps = self._ensure_plan_horizon(plan_dict["steps"])
                else:
                    normalized_steps = self._ensure_plan_horizon([plan_dict])
                plan_dict = {"steps": normalized_steps}
            print(
                f"--- [Tool] {method_key} 规划成功: {json.dumps(diag, ensure_ascii=False)} ---"
            )
            return plan_dict

        error_msg = None
        if isinstance(diag, dict):
            error_msg = diag.get("error")
        if not error_msg:
            error_msg = f"{method_key} 规划失败" if diag is None else str(diag)
        print(f"!! {method_key} 规划失败: {error_msg}")
        return {"error": error_msg}

    def _get_lagrangian_plan(self, state_description: str, config: Optional[str] = None) -> dict:
        return self._run_or_method(state_description, "lagrangian_relaxation", config)

    def _get_column_generation_plan(self, state_description: str, config: Optional[str] = None) -> dict:
        return self._run_or_method(state_description, "column_generation", config)

    def _get_benders_plan(self, state_description: str, config: Optional[str] = None) -> dict:
        return self._run_or_method(state_description, "benders", config)

    def _get_queueing_plan(self, state_description: str, config: Optional[str] = None) -> dict:
        return self._run_or_method(state_description, "queueing", config)

    def _get_milp_plan(self, state_description: str, config: Optional[str] = None) -> dict:
        """使用 MILP 直接生成可供 hierarchical 执行层消费的初始计划。"""
        del state_description

        if not self.env or not hasattr(self.env, "base_env"):
            return {"error": "Base environment unavailable for MILP planning."}

        try:
            parsed_config = json.loads(config) if config else {}
            if not isinstance(parsed_config, dict):
                parsed_config = {}
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed_config = {}

        focus_agents = []
        for value in parsed_config.get("focus_agents", []) or []:
            try:
                focus_agents.append(int(value))
            except (TypeError, ValueError):
                continue

        focus_nodes = []
        for value in parsed_config.get("focus_nodes", []) or []:
            try:
                focus_nodes.append(int(value))
            except (TypeError, ValueError):
                continue

        try:
            local_horizon = int(parsed_config.get("time_horizon", self.local_milp_time_horizon))
        except (TypeError, ValueError):
            local_horizon = self.local_milp_time_horizon
        local_horizon = min(max(local_horizon, 1), self.local_milp_time_horizon)

        try:
            local_time_limit = int(parsed_config.get("time_limit", self.local_milp_time_limit))
        except (TypeError, ValueError):
            local_time_limit = self.local_milp_time_limit
        local_time_limit = min(max(local_time_limit, 1), self.local_milp_time_limit)

        if self.local_milp_solver is None:
            self.local_milp_solver = MILPSolver(
                time_horizon=self.local_milp_time_horizon,
                time_limit=self.local_milp_time_limit,
            )

        try:
            actions_dict = self.local_milp_solver.solve(
                self.env.base_env,
                focus_agent_ids=focus_agents or None,
                focus_node_ids=focus_nodes or None,
                time_horizon=local_horizon,
                time_limit=local_time_limit,
                objective_type="minimize_wait_time",
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"MILP global planning failed: {exc}"}

        world = self.env.base_env.scenario.world
        solution_details = getattr(self.local_milp_solver, "last_solution", {}) or {}
        if getattr(world, "dispatch", None) is not None:
            return {"dispatch_solution": copy.deepcopy(solution_details)}
        action_details = solution_details.get("actions", {}) if isinstance(solution_details.get("actions"), dict) else {}

        plan_step: Dict[str, Dict[str, Any]] = {}
        for agent_idx, agent in enumerate(world.agents):
            agent_name = getattr(agent, "name", f"agent_{agent_idx}")
            current_pos = int(agent.state.position[0].item())
            action_value = int(actions_dict.get(agent_name, 0) or 0)
            detail = action_details.get(agent_name, {}) if isinstance(action_details.get(agent_name), dict) else {}
            next_node = detail.get("next_node", current_pos)
            reason = str(detail.get("reason", "MILP 全局初始方案")).strip() or "MILP 全局初始方案"
            plan_step[agent_name] = {
                "action": action_value,
                "next_node": int(next_node),
                "reasoning": reason,
            }

        return {"steps": [plan_step]}

    def _get_llm_global_plan(self, state_description: str) -> dict:
        global_prompt_template_str = """
                你是一名经验丰富的地铁调度指挥官。你的职责是实时监控整个地铁网络，并为每一辆列车下达精确的行驶指令。

                核心目标：
                1. 最小化乘客等待时间，优先服务候车人数多的站点。
                2. 最大化运输效率，减少列车不必要停留。

                必须遵守的硬性约束：
                1. 列车只能移动到当前节点直接相连的可前往节点。
                2. 若列车仍在停站计时内，则必须停留。
                3. 若某条线路中断，则不能规划经过该中断区间。

                当前实时系统状态：
                {state_description}

                请严格返回一个 JSON 对象，不要输出 Markdown 代码块，不要输出额外解释。
                顶层必须只有一个字段：steps。
                steps 必须是一个数组，长度必须恰好等于 {plan_horizon}。
                steps 中的每个元素都是一个对象，键名使用 agent_0、agent_1 这种格式。
                每个 agent 对象必须包含 action、next_node、reasoning 三个字段。
                如果 action 为 0，则 next_node 必须等于该列车当前所在节点；如果 action 为 1，则 next_node 必须是当前节点的相邻可达节点。
                action 只能是 0 或 1。
        """
        
        # 2. 使用本地全局模型创建并调用临时 chain
        try:
            global_prompt = ChatPromptTemplate.from_template(global_prompt_template_str)
            global_chain = global_prompt | self.global_llm | StrOutputParser()

            response_str = global_chain.invoke({
                "state_description": state_description,
                "plan_horizon": self.plan_horizon,
            })

            # 3. 解析返回的JSON
            json_start = response_str.find('{')
            json_end = response_str.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise json.JSONDecodeError("未在回复中找到有效的JSON对象", response_str, 0)

            json_str = response_str[json_start:json_end]
            plan_dict = json.loads(json_str)
            raw_steps = plan_dict.get("steps") if isinstance(plan_dict, dict) else None
            candidate_steps = raw_steps if isinstance(raw_steps, list) else [plan_dict]
            normalized_steps = self._normalize_plan_steps(candidate_steps)
            normalized_steps = self._ensure_plan_horizon(normalized_steps)
            print(
                "--- [Tool] 全局LLM采用方案:\n"
                f"{json.dumps({'steps': normalized_steps}, ensure_ascii=False, indent=2)}"
            )
            return {"steps": normalized_steps}

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"!! 全局LLM方案解析失败: {exc}")
            return {"error": f"Failed to get or parse global instructions: {exc}"}
        except Exception as exc:  # noqa: BLE001
            print(f"!! 调用全局LLM发生未知异常: {exc}")
            return {"error": f"An unexpected error occurred: {exc}"}

    def _invoke_strategic(self, messages, *, timeout_s=None):
        from LLM.strategic_api import invoke
        return invoke(self.global_llm, messages,
            timeout_s=getattr(self, "llm_request_timeout_s", 30.) if timeout_s is None else timeout_s,
            max_tokens=self.llm_max_output_tokens,
            thinking_mode=getattr(self, "llm_thinking_mode", "disabled"),
            reasoning_effort=getattr(self, "llm_reasoning_effort", "low"))

    def _get_llm_global_guidance(self, state_description, *, retrieved_cases=None, request=None, timeout_s=None) -> dict:
        """让大模型仅输出紧凑的全局求解指导，而不是逐车多步动作。"""
        global_guidance_template_str = """
                你是一名地铁调度全局编排器。

                你的职责不是输出逐车 action，也不是输出多步 steps；
                你的职责是结合当前状态与 RAG 案例，选择并配置后续运筹学求解器。

                当前实时系统状态：
                {state_description}

                请严格返回一个 JSON 对象，不要输出 Markdown 代码块，不要输出额外解释。
                必须只包含以下字段：
                - retrieval_assessment: "useful_case" 或 "no_good_case"
                - recommended_solver: "milp"、"benders"、"lagrangian"、"column_generation"、"queueing"、"none" 之一
                - solver_config: 对象，可包含 focus_agents、focus_nodes、time_horizon、time_limit
                - strategy_summary: 极短摘要，不超过20个汉字

                约束：
                - 不要输出 steps
                - 不要输出 action / next_node / reasoning
                - 如果没有明显可复用案例或无需额外全局编排，recommended_solver 设为 "none"
                - focus_agents 和 focus_nodes 尽量简短，通常不超过 6 个
                - time_horizon 不超过 {plan_horizon}，time_limit 不超过 {time_limit}
        """

        if self.world is not None and getattr(self.world, "dispatch", None) is not None:
            global_guidance_template_str = """
            Configure one joint metro rescheduling problem using the published state
            and retrieved cases below. Return a JSON object with retrieval_assessment
            (useful_case or no_good_case), a brief retrieval_reason, solver_config (focus_agents, focus_nodes,
            time_horizon, time_limit), reused_case_ids, reuse_aspects, adaptations
            and strategy_summary. Judge transferability, not exact scenario identity.
            useful_case includes a supported partial analogy in shared infrastructure,
            vehicle/task competition, passenger service, or route/time coordination.
            A different line, fleet size, event combination, horizon or historical
            budget is not by itself a reason for no_good_case. Historical settings are
            examples to adapt to current limits, not mandatory solver requirements.
            No active incident is required: normal-service coordination and post-event
            queues can also share useful structure. Respect active versus released events.
            Cite supplied case IDs and concrete shared relationships, and explain needed
            adaptations. Do not invent past selected measures or benefits. Use no_good_case
            when no supported transferable aspect exists; do not force a positive assessment.
            Joint coordination may still be needed without a useful case.
            Use only current vehicle and facility identifiers. Do not output train
            actions, solver variables, or select a different mathematical model.
            Suggest the vehicles and platforms directly involved in one urgent
            coupled coordination need, not every train with a potential path overlap.
            Your focus is a preference used to rank active relations. The controller
            retains required joint participants and fixes checked background operations;
            shared capacity alone does not require making every user a decision variable.
            time_horizon must not exceed {plan_horizon}; time_limit must not exceed {time_limit}.
            Published inputs and retrieved cases: {state_description}
            """

        self.last_prompt_audit = None
        try:
            native = self.world is not None and getattr(self.world, "dispatch", None) is not None
            if native:
                global_guidance_template_str += "\n" + INSTRUCTIONS
            global_prompt = ChatPromptTemplate.from_template(global_guidance_template_str)
            input_limit = getattr(self, "llm_max_input_tokens", DEFAULT_INPUT_TOKENS)
            metadata = {}
            for detail in range(3):
                if native and isinstance(state_description, dict):
                    context = strategy_context(state_description, retrieved_cases, request,
                                               world=self.world, detail_level=detail)
                    description = context_dumps(context)
                    metadata = context_audit(state_description, retrieved_cases, context)
                else:
                    description = state_description if isinstance(state_description, str) else context_dumps(state_description)
                messages = global_prompt.format_messages(state_description=description,
                    plan_horizon=self.plan_horizon, time_limit=self.local_milp_time_limit)
                try:
                    audit = check_messages(messages, input_limit)
                    break
                except LLMInputTooLong:
                    if not (native and isinstance(state_description, dict)) or detail == 2:
                        raise
            self.last_prompt_audit = dict(metadata, **audit,
                max_output_tokens=getattr(self, "llm_max_output_tokens", DEFAULT_OUTPUT_TOKENS),
                request_timeout_s=getattr(self, "llm_request_timeout_s", 30.) if timeout_s is None else timeout_s,
                thinking_mode_requested=getattr(self, "llm_thinking_mode", "disabled"), status="ready")
            logger.info("llm_prompt_preflight %s", context_dumps(self.last_prompt_audit))
            self.last_prompt_audit["status"] = "sent"
            request_start = time.perf_counter()
            try:
                response = (self._invoke_strategic(messages, timeout_s=timeout_s) if native
                            else self.global_llm.invoke(messages))
            finally:
                self.last_prompt_audit["request_elapsed_s"] = time.perf_counter()-request_start
            self.last_prompt_audit["status"] = "received"
            usage = getattr(response, "usage_metadata", None) or {}
            response_metadata = getattr(response, "response_metadata", {})
            self.last_prompt_audit["thinking_mode_applied"] = response_metadata.get("thinking_mode_applied")
            self.last_prompt_audit["provider_output_tokens"] = (response_metadata.get("token_usage") or {}).get("completion_tokens")
            self.last_prompt_audit["reasoning_tokens"] = response_metadata.get("reasoning_tokens")
            self.last_prompt_audit["provider_input_tokens"] = usage.get("input_tokens",
                (response_metadata.get("token_usage") or {}).get("prompt_tokens"))
            if response_metadata.get("finish_reason") == "length":
                raise ValueError("Strategic LLM output reached the output token limit")
            response_str = StrOutputParser().invoke(response)

            json_start = response_str.find('{')
            json_end = response_str.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise json.JSONDecodeError("未在回复中找到有效的JSON对象", response_str, 0)

            guidance = json.loads(response_str[json_start:json_end])
            if not isinstance(guidance, dict):
                raise TypeError("global guidance must be a dict")
            assessment = str(guidance.get("retrieval_assessment", "not_reported")).strip().lower()
            if assessment not in ("useful_case", "no_good_case"):
                assessment = "not_reported"
            self.last_prompt_audit.update(returned_fields=sorted(guidance),
                retrieval_assessment_reported="retrieval_assessment" in guidance,
                retrieval_assessment=assessment)
            case_ids = set(self.last_prompt_audit.get("case_ids", []))
            reused = guidance.get("reused_case_ids", [])
            reused = list(dict.fromkeys(v for v in reused if isinstance(v, str) and v in case_ids)) if isinstance(reused, list) else []
            def brief_list(key):
                value = guidance.get(key, [])
                return [v[:160] for v in value[:3] if isinstance(v, str)] if isinstance(value, list) else []

            solver = str(guidance.get("recommended_solver", "milp" if native else "none")).strip().lower()
            if solver not in {"milp", "benders", "lagrangian", "column_generation", "queueing", "none"}:
                solver = "none"

            solver_config = guidance.get("solver_config") if isinstance(guidance.get("solver_config"), dict) else {}
            focus_agents = []
            for value in solver_config.get("focus_agents", []) or []:
                if native and isinstance(value, str) and value in {a.name for a in self.world.agents}:
                    value = next(i for i, a in enumerate(self.world.agents) if a.name == value)
                try:
                    focus_agents.append(int(value))
                except (TypeError, ValueError):
                    continue

            focus_nodes = []
            for value in solver_config.get("focus_nodes", []) or []:
                try:
                    focus_nodes.append(int(value))
                except (TypeError, ValueError):
                    continue

            time_horizon = solver_config.get("time_horizon")
            try:
                time_horizon = min(max(int(time_horizon), 1), self.plan_horizon) if time_horizon is not None else self.plan_horizon
            except (TypeError, ValueError):
                time_horizon = self.plan_horizon

            time_limit = solver_config.get("time_limit")
            try:
                time_limit = min(max(int(time_limit), 1), self.local_milp_time_limit) if time_limit is not None else self.local_milp_time_limit
            except (TypeError, ValueError):
                time_limit = self.local_milp_time_limit

            normalized = {
                "retrieval_assessment": assessment,
                "retrieval_reason": str(guidance.get("retrieval_reason", "")).strip()[:320],
                "reused_case_ids": reused,
                "reuse_aspects": brief_list("reuse_aspects"), "adaptations": brief_list("adaptations"),
                "recommended_solver": solver,
                "solver_config": {
                    "focus_agents": focus_agents if native else focus_agents[:6],
                    "focus_nodes": focus_nodes if native else focus_nodes[:6],
                    "time_horizon": time_horizon,
                    "time_limit": time_limit,
                },
                "strategy_summary": str(guidance.get("strategy_summary", "")).strip()[:160 if native else 20],
            }
            logger.info("llm_response_summary %s", context_dumps(self.last_prompt_audit))
            print(f"--- [Tool] 全局LLM指导摘要: {json.dumps(normalized, ensure_ascii=False)}")
            return normalized

        except LLMInputTooLong as exc:
            self.last_prompt_audit = dict(metadata, **exc.audit)
            logger.error("llm_prompt_rejected %s", context_dumps(self.last_prompt_audit))
            return {"error": str(exc), "prompt_check": self.last_prompt_audit}
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"!! 全局LLM指导解析失败: {exc}")
            if self.last_prompt_audit is not None:
                self.last_prompt_audit["status"] = "invalid_response"
            return {"error": f"Failed to get or parse global guidance: {exc}"}
        except Exception as exc:  # noqa: BLE001
            print(f"!! 调用全局LLM指导发生未知异常: {exc}")
            if self.last_prompt_audit is not None:
                self.last_prompt_audit["status"] = "timeout" if isinstance(exc, TimeoutError) else "failed"
                logger.error("llm_request_failed %s", context_dumps(self.last_prompt_audit))
            return {"error": f"An unexpected error occurred: {exc}"}

    def _get_global_instructions(
        self,
        state_description: str,
        method: Optional[str] = None,
        config: Optional[str] = None,
    ) -> dict:
        """兼容旧接口：根据 method 分派到对应的新工具。"""

        method_choice = (method or "llm").strip().lower()

        if method_choice == "milp":
            return self._get_milp_plan(state_description, config)
        if method_choice in {"lagrangian", "lagrangian_relaxation"}:
            return self._get_lagrangian_plan(state_description, config)
        if method_choice == "column_generation":
            return self._get_column_generation_plan(state_description, config)
        if method_choice in {"benders", "benders_decomposition"}:
            return self._get_benders_plan(state_description, config)
        if method_choice in {"queueing", "queueing_analysis"}:
            return self._get_queueing_plan(state_description, config)
        if method_choice in {"llm", "llm_only"}:
            return self._get_llm_global_plan(state_description)

        print(f"!! 未知的 get_global_instructions 方法: {method_choice}")
        return {"error": f"Unknown method '{method_choice}' for get_global_instructions."}

    def _capture_env_signature(self) -> Tuple[Any, ...]:
        """捕捉影响预案有效性的环境特征，这里主要关注线路中断。"""
        if not self.env or not hasattr(self.env, "base_env"):
            return tuple()
        world = self.env.base_env.scenario.world
        disruptions = world.get_disrupted_segments()
        return tuple(sorted((seg_id, json.dumps(info, sort_keys=True)) for seg_id, info in disruptions.items()))

    def _clear_cached_plan(self) -> None:
        if getattr(self, "_dispatch_runtime", None) is not None:
            self._dispatch_runtime.invalidate_cache("policy_cache_cleared")
        if self.cached_plan_steps:
            print("--- 预案失效，清空缓存 ---")
        self.cached_plan_steps = []
        self.cached_plan_index = 0
        self.cached_plan_env_signature = None
        self._last_verified_step_plan = {}

    def has_active_plan(self) -> bool:
        if getattr(self, "_dispatch_runtime", None) is not None:
            return self._dispatch_runtime.has_active_plan()
        return bool(self.cached_plan_steps) and self.cached_plan_index < len(self.cached_plan_steps)

    def notify_step_result(self, step_td: TensorDict) -> None:
        """在环境执行一步后调用，用于监控预案是否仍然有效。"""
        if getattr(self, "_dispatch_runtime", None) is not None:
            return self._dispatch_runtime.notify_step_result(step_td)
        try:
            if not self.has_active_plan():
                return
            world = self.env.base_env.scenario.world
            current_signature = self._capture_env_signature()
            if self.cached_plan_env_signature is not None and current_signature != self.cached_plan_env_signature:
                self._clear_cached_plan()
                return
        except Exception as exc:
            print(f"!! notify_step_result 异常: {exc}")

    def _plan_step_to_tensor(self, step_plan: Dict[str, Dict[str, Any]], marl_actions: torch.Tensor) -> torch.Tensor:
        final_actions = torch.zeros(1, self.n_agents, 1, device=self.device, dtype=torch.int64)
        world = self.env.base_env.scenario.world
        for i in range(self.n_agents):
            agent_name = f"agent_{i}"
            decision = step_plan.get(agent_name)
            if not isinstance(decision, dict) or "action" not in decision:
                final_actions[0, i, 0] = marl_actions[0, i]
                continue
            action = int(decision.get("action", 0))
            if action == 0:
                final_actions[0, i, 0] = 0
                continue
            next_node_id = decision.get("next_node")
            if next_node_id is None:
                next_node_id = int(world.agents[i].state.position[0].item())
            else:
                try:
                    next_node_id = int(next_node_id)
                except (TypeError, ValueError):
                    next_node_id = int(world.agents[i].state.position[0].item())
            current_pos = int(world.agents[i].state.position[0].item())
            candidate_nodes = [node.id for node in world.get_next_nodes(current_pos)]
            try:
                action_idx = candidate_nodes.index(int(next_node_id)) + 1
                final_actions[0, i, 0] = action_idx
            except (ValueError, TypeError):
                print(f"警告: 预案中 agent_{i} 的目标节点 {next_node_id} 无效，改用MARL动作。")
                final_actions[0, i, 0] = marl_actions[0, i]
        return final_actions

    def _normalize_plan_steps(self, plan_steps: List[Dict[str, Dict[str, Any]]]) -> List[Dict[str, Dict[str, Any]]]:
        """将计划中的停留动作对齐到列车真实位置，避免出现物理不一致的 next_node。"""
        steps = [copy.deepcopy(step) for step in plan_steps] if plan_steps else []
        world = self.world or (self.env.base_env.scenario.world if self.env else None)
        if world is None:
            return steps

        def _resolve_expected_dwell(node_id: int, fallback: int) -> int:
            node = world.platform_nodes.get(node_id)
            if node is None:
                return max(0, fallback)
            weight = getattr(node, "weight", None)
            if weight is not None and weight > 0:
                return int(weight)
            dwell_seconds = getattr(node, "dwell", None) or getattr(node, "dwell_seconds", None)
            if dwell_seconds is None:
                return max(0, fallback)
            return max(1, int(dwell_seconds))

        sim_agent_states: List[Dict[str, int]] = []
        for agent in world.agents:
            position = int(agent.state.position[0].item())
            dwell_time = int(agent.state.dwell_time[0].item())
            expected_dwell = int(agent.state.current_expected_dwell_time[0].item())
            if expected_dwell <= 0:
                expected_dwell = _resolve_expected_dwell(position, 0)
            sim_agent_states.append({
                "position": position,
                "dwell_time": max(0, dwell_time),
                "expected_dwell": max(0, expected_dwell),
            })

        disrupted_segments = world.get_disrupted_segments() if hasattr(world, "get_disrupted_segments") else {}
        normalized_steps: List[Dict[str, Dict[str, Any]]] = []

        for raw_step in steps:
            source_step = raw_step if isinstance(raw_step, dict) else {}
            normalized_step: Dict[str, Dict[str, Any]] = {}

            for i in range(self.n_agents):
                agent_name = f"agent_{i}"
                decision = source_step.get(agent_name) if isinstance(source_step.get(agent_name), dict) else {}
                agent_state = sim_agent_states[i]
                current_pos = int(agent_state["position"])
                raw_action = self._safe_int(decision.get("action"))
                raw_next_node = self._safe_int(decision.get("next_node"))
                reasoning = str(decision.get("reasoning", "")).strip()
                correction_note: Optional[str] = None

                action = 0
                next_node = current_pos

                if raw_action == 1:
                    can_move = True
                    if current_pos in world.platform_nodes and agent_state["dwell_time"] < agent_state["expected_dwell"]:
                        can_move = False
                    candidate_nodes = [node.id for node in world.get_next_nodes(current_pos)]
                    if raw_next_node is None or raw_next_node not in candidate_nodes:
                        can_move = False
                    else:
                        segment_id = self._get_segment_id(world, current_pos, raw_next_node)
                        if segment_id and segment_id in disrupted_segments:
                            segment_info = disrupted_segments.get(segment_id) or {}
                            if segment_info.get("type") == "BLOCKAGE":
                                can_move = False
                    if can_move:
                        action = 1
                        next_node = int(raw_next_node)
                    else:
                        correction_note = "非法移动已回退为原地停留"
                else:
                    if raw_action not in (0, None) or (raw_next_node is not None and raw_next_node != current_pos):
                        correction_note = "停留动作的 next_node 已对齐到当前位置"

                if not reasoning:
                    reasoning = "归一化: 保持当前位置" if action == 0 else "归一化: 保留可行移动"
                if correction_note:
                    reasoning = f"{reasoning} | {correction_note}"

                normalized_step[agent_name] = {
                    "action": int(action),
                    "next_node": int(next_node),
                    "reasoning": reasoning,
                }

            normalized_steps.append(normalized_step)

            for i in range(self.n_agents):
                normalized_decision = normalized_step.get(f"agent_{i}", {})
                agent_state = sim_agent_states[i]
                if normalized_decision.get("action", 0) == 1:
                    target_node = int(normalized_decision.get("next_node", agent_state["position"]))
                    agent_state["position"] = target_node
                    agent_state["dwell_time"] = 0
                    agent_state["expected_dwell"] = _resolve_expected_dwell(target_node, agent_state["expected_dwell"])
                elif agent_state["position"] in world.platform_nodes:
                    agent_state["dwell_time"] = min(
                        agent_state["expected_dwell"],
                        agent_state["dwell_time"] + 1,
                    )

        return normalized_steps

    def _ensure_plan_horizon(self, plan_steps: List[Dict[str, Dict[str, Any]]]) -> List[Dict[str, Dict[str, Any]]]:
        """仅在步数超过上限时截断；不再自动补齐，迫使LLM返回完整 plan_horizon 步。"""
        steps = [copy.deepcopy(step) for step in plan_steps] if plan_steps else []
        if len(steps) > self.plan_horizon:
            return steps[: self.plan_horizon]
        return steps

    def _verify_and_finalize(self, solution: dict) -> torch.Tensor:
        """
        工具4: 验证最终方案。
        返回一个张量，其中: 1=有效移动, 0=有效停留, -1=无效/回退MARL。
        如果发现无效决策，将自动修正为 -1 (MARL)，而不是拒绝整个方案。
        """
        print(f"--- [Tool] 约束验证器被调用，验证方案: {solution} ---")
        self._last_validation_error = ""
        if isinstance(solution, dict) and "steps" in solution and isinstance(solution["steps"], list):
            plan_steps = solution["steps"]
            if not plan_steps:
                print("!! 验证警告: steps 为空。将生成全MARL回退动作。")
                return torch.full((1, self.n_agents, 1), -1, device=self.device, dtype=torch.int64)
            # 自动截断或补齐
            if len(plan_steps) > self.plan_horizon:
                plan_steps = plan_steps[:self.plan_horizon]
            first_step = plan_steps[0]
        else:
            plan_steps = [solution]
            first_step = solution

        plan_steps = self._normalize_plan_steps(plan_steps)
        first_step = plan_steps[0]

        # 保存完整计划以供缓存
        self._last_verified_full_plan = plan_steps
        self._last_verified_step_plan = first_step

        actions = torch.full((1, self.n_agents, 1), -1, device=self.device, dtype=torch.int64)

        for i in range(self.n_agents):
            agent_name = f"agent_{i}"
            decision = first_step.get(agent_name)
            
            # 默认修正为 -1 (MARL)
            final_action = -1
            
            if not isinstance(decision, dict) or "action" not in decision:
                print(f"!! 验证修正: {agent_name} 决策格式错误。修正为MARL。")
                actions[0, i, 0] = -1
                continue
            
            raw_action = self._safe_int(decision.get("action"))
            if raw_action is None:
                print(f"!! 验证修正: {agent_name} action非法。修正为MARL。")
                actions[0, i, 0] = -1
                continue

            if raw_action == 0:
                final_action = 0
            elif raw_action == 1:
                agent = self.world.agents[i]
                position = int(agent.state.position[0].item())
                is_safe_to_move = True

                # 1. 停站时间约束
                if position in self.world.platform_nodes:
                    dwell_time = int(agent.state.dwell_time[0].item())
                    expected_dwell = int(agent.state.current_expected_dwell_time[0].item())
                    if dwell_time < expected_dwell:
                        print(f"!! 验证修正: {agent_name} 停站时间不足 ({dwell_time}/{expected_dwell})。修正为MARL。")
                        is_safe_to_move = False

                # 2. 邻接移动和中断约束
                if is_safe_to_move:
                    next_node = self._safe_int(decision.get("next_node"))
                    next_node_ids = [node.id for node in self.world.get_next_nodes(position)]

                    if next_node is None:
                        print(f"!! 验证修正: {agent_name} 缺少 next_node。修正为MARL。")
                        is_safe_to_move = False
                    elif next_node not in next_node_ids:
                        print(f"!! 验证修正: {agent_name} 目标非邻接 ({next_node})。修正为MARL。")
                        is_safe_to_move = False
                    else:
                        # 使用正确的 segment ID 查找
                        segment_id = self._get_segment_id(self.world, position, next_node)
                        disrupted_segments = self.world.get_disrupted_segments()
                        if segment_id and segment_id in disrupted_segments and disrupted_segments[segment_id]['type'] == 'BLOCKAGE':
                            print(f"!! 验证修正: {agent_name} 路径中断 ({segment_id})。修正为MARL。")
                            is_safe_to_move = False

                if is_safe_to_move:
                    final_action = 1
                else:
                    final_action = -1
            else:
                final_action = -1

            actions[0, i, 0] = final_action

        # 缓存计划
        self._last_verified_step_plan = first_step
        prepared_steps = plan_steps if plan_steps else [first_step]
        normalized_steps = self._ensure_plan_horizon(prepared_steps)
        self.cached_plan_steps = normalized_steps
        self.cached_plan_index = 1 if len(self.cached_plan_steps) > 1 else 0
        self.cached_plan_env_signature = self._capture_env_signature()

        print(f"验证后的最终动作为 (1=有效移动, 0=有效停留): {actions.squeeze().tolist()}")
        return actions

    def _format_observation(self, tensordict_obs: TensorDict) -> str:
        """将环境观察TensorDict格式化为人类可读的文本描述，包含拓扑信息。"""
        base_env = self.env.base_env
        world = base_env.scenario.world
        batch_idx = 0
        current_step = int(base_env.steps[batch_idx].item())
        if self.sim_start_step is None:
            self.sim_start_step = current_step

        state_parts = [
            f"**全局信息:** 当前时间步: {current_step} (仿真从第 {self.sim_start_step} 步开始)"
        ]
        
        disruptions = world.get_disrupted_segments()
        if disruptions:
            # 创建一个副本以进行修改，避免影响原始环境数据
            adjusted_disruptions = copy.deepcopy(disruptions)
            # 将相对时间转换为绝对仿真时间步
            for seg_id, details in adjusted_disruptions.items():
                for field in ('reported_at', 'predicted_end_seconds'):
                    if isinstance(details.get(field), (int, float)):
                        details[field] += self.sim_start_step
            
            state_parts.append(f"**系统警告: 检测到中断或限速!** {adjusted_disruptions}")

        # 1. 精炼拓扑信息：显示站台间的连接关系和距离
        state_parts.append("\n**线路摘要 (站台-区间-站台):**")
        topology_info = []
        # 使用一个集合来避免重复处理线路
        processed_pairs = set()

        def _format_path(nodes: List[int]) -> str:
            if len(nodes) <= 8:
                return " -> ".join(str(n) for n in nodes)
            head = " -> ".join(str(n) for n in nodes[:4])
            tail = " -> ".join(str(n) for n in nodes[-3:])
            return f"{head} -> ... -> {tail}"

        for start_node_id in sorted(world.platform_nodes.keys()):
            next_nodes = world.get_next_nodes(start_node_id)
            for node in next_nodes:
                path = [start_node_id, node.id]
                current_node = node
                visited: Set[int] = {start_node_id}
                while current_node.id not in world.platform_nodes:
                    successors = [
                        successor for successor in world.get_next_nodes(current_node.id)
                        if successor.id not in visited
                    ]
                    if len(successors) != 1:
                        break
                    visited.add(current_node.id)
                    current_node = successors[0]
                    path.append(current_node.id)
                
                # 如果找到了一个站台到站台的路径
                if path[-1] in world.platform_nodes and len(path) > 1:
                    end_node_id = path[-1]
                    pair = tuple(sorted((start_node_id, end_node_id)))
                    if pair not in processed_pairs:
                        topology_info.append(f"- {_format_path(path)}")
                        processed_pairs.add(pair)
        state_parts.append("\n".join(topology_info) if topology_info else "- 无法生成线路摘要。")

        # 2. 站台状态：只显示有人的
        state_parts.append("\n**站台状态:**")
        waiting_passengers_info = [f"{nid}({int(n.waiting_passengers[batch_idx].item())}人)" 
                                   for nid, n in world.platform_nodes.items() 
                                   if n.waiting_passengers[batch_idx].item() > 0]
        state_parts.append(", ".join(waiting_passengers_info) if waiting_passengers_info else "所有站台无人等待。")

        # 列车信息
        state_parts.append("\n**列车详细状态:**")
        for i, agent in enumerate(world.agents):
            agent_name = agent.name
            
            # 直接从agent状态获取信息（更可靠）
            position = int(agent.state.position[batch_idx].item())
            dwell_time = int(agent.state.dwell_time[batch_idx].item())
            expected_dwell = int(agent.state.current_expected_dwell_time[batch_idx].item())
            passenger_count = int(agent.passenger_count[batch_idx].item())
            
            # 获取可前往的节点
            next_nodes = world.get_next_nodes(position)
            next_node_ids = [node.id for node in next_nodes]

            desc = (
                f"- **{agent_name}**:\n"
                f"  - **位置**: 节点 {position} {'(站台)' if position in world.platform_nodes else '(区间)'}\n"
                f"  - **载客量**: {passenger_count} 人\n"
                f"  - **停站状态**: {'停留 ' + str(dwell_time) + 's / 预计 ' + str(expected_dwell) + 's' if position in world.platform_nodes else '在途'}\n"
                f"  - **可前往节点**: {next_node_ids}"
            )
            state_parts.append(desc)
            
        return "\n".join(state_parts)

    def get_action(self, tensordict_obs: TensorDict, marl_actions: torch.Tensor) -> TensorDict:
        """
        执行一个ReAct (Reason-Act)循环来让LLM Agent决策。
        Agent会实际调用工具并根据返回结果进行下一步操作。
        如果LLM对某个agent的决策无效，则会采用传入的MARL策略作为备用。
        """
        world = self.env.base_env.scenario.world
        if getattr(world, "dispatch", None) is not None:
            from LLM.dispatch_coordination import direct_dispatch_action
            if self.local_milp_solver is None:
                self.local_milp_solver = MILPSolver(time_horizon=self.local_milp_time_horizon,
                                                   time_limit=self.local_milp_time_limit)
            return direct_dispatch_action(self, tensordict_obs, marl_actions, solver=self.local_milp_solver,
                horizon=self.local_milp_time_horizon, time_limit=self.local_milp_time_limit, use_cache=True,
                specify=lambda state, context, trigger: self._get_llm_global_guidance(
                    state, retrieved_cases=context, request=trigger))
        print("\n--- 正在激活LLM Agent进行决策 ---")
        self.last_decision_source = "LLM新方案"

        if self.cached_plan_steps:
            current_signature = self._capture_env_signature()
            if self.cached_plan_env_signature is not None and current_signature != self.cached_plan_env_signature:
                self._clear_cached_plan()

        if self.has_active_plan():
            plan_step = self.cached_plan_steps[self.cached_plan_index]
            final_actions = self._plan_step_to_tensor(plan_step, marl_actions)
            self.cached_plan_index += 1
            if self.cached_plan_index >= len(self.cached_plan_steps):
                self._clear_cached_plan()
            self.last_decision_source = "LLM缓存预案"
            print(f"--- 使用缓存预案第 {self.cached_plan_index} 步 ---")
            return TensorDict({self.action_key: final_actions}, batch_size=[1], device=self.device)

        # 初始化ReAct循环
        max_steps = 15 # 限制Agent的思考步数，防止无限循环
        state_description = self._format_observation(tensordict_obs)

        # --- RAG: 检索相似历史案例 ---
        # if self.rag_manager:
        #     try:
        #         similar_cases = self.rag_manager.retrieve(state_description, top_k=2)
        #         if similar_cases:
        #             rag_context = "\n\n**参考历史相似案例 (专家决策):**\n"
        #             for idx, case in enumerate(similar_cases):
        #                 rag_context += f"案例 {idx+1}:\n- 场景摘要: ...{case['state']}\n- 专家建议: {case['action']}\n"
        #             state_description += rag_context
        #             print(f"--- [RAG] 已检索并注入 {len(similar_cases)} 个相似案例 ---")
        #     except Exception as e:
        #         print(f"!! [RAG] 检索失败: {e}")
        # ---------------------------
        
        # 在函数作用域内初始化，以确保循环结束后可以访问
        final_solution_dict = {}
        actions = torch.zeros(1, self.n_agents, 1, device=self.device, dtype=torch.int64)

        # 使用LangChain的消息格式来构建对话历史
        initial_prompt = self.prompt_template.format_prompt(
            state_description=state_description,
            n_agents=self.n_agents,
            last_agent_index=self.n_agents - 1
        ).to_messages()
        
        conversation_history = initial_prompt
        
        print("--- AGENT 对话历史 (初始) ---")
        for msg in conversation_history:
            print(f"[{msg.type.upper()}] {msg.content}\n")
        print("-" * 20)

        for step in range(max_steps):
            print(f"\n--- Agent 思考步骤 {step + 1}/{max_steps} ---")
            
            # 1. 调用LLM进行思考和工具选择
            response = self.llm.invoke(conversation_history)
            response_str = response.content
            
            # 将LLM的回复添加到历史记录中
            conversation_history.append(AIMessage(content=response_str))
            print(f"[{response.type.upper()}] {response_str}\n")

            # 2. 解析LLM的回复，检查是否有工具调用
            
            # 从LLM的回复中提取“行动:”部分
            action_match = re.search(r"行动\s*(.*)", response_str, re.DOTALL)
            if not action_match:
                print("--- [状态] 未检测到'行动:'指令，将当前思考作为上下文继续 ---")
                conversation_history.append(HumanMessage(content="你的回复格式不正确。请使用'思考:'和'行动:'格式。"))
                continue

            action_str = action_match.group(1).strip()
            # 修复：去除LLM可能添加的反引号
            action_str = action_str.strip('`')

            # 最终决策工具
            # 使用非贪婪匹配 (.+?) 来防止匹配到多余的括号
            finalize_match = re.search(r"verify_and_finalize\((.+?)\)$", action_str, re.DOTALL)
            if finalize_match:
                print("--- [决策] Agent请求最终方案验证 ---")
                try:
                    solution_str = finalize_match.group(1).strip()
                    
                    # 尝试解析JSON或Python字典
                    try:
                        solution_dict = json.loads(solution_str)
                    except json.JSONDecodeError:
                        try:
                            # 尝试作为Python字典解析 (支持单引号)
                            solution_dict = ast.literal_eval(solution_str)
                        except (ValueError, SyntaxError):
                            # 最后尝试简单的引号替换
                            solution_str_fixed = solution_str.replace("'", '"')
                            # 尝试修复末尾可能缺失的括号
                            open_braces = solution_str_fixed.count('{')
                            close_braces = solution_str_fixed.count('}')
                            if open_braces > close_braces:
                                solution_str_fixed += '}' * (open_braces - close_braces)
                            solution_dict = json.loads(solution_str_fixed)
                    
                    # 修复：将成功解析的字典存储在函数作用域的变量中
                    final_solution_dict = solution_dict
                    
                    actions = self._verify_and_finalize(solution_dict)
                    
                    # 移除强制重试逻辑，允许部分回退到MARL
                    break 
                except (json.JSONDecodeError, TypeError) as e:
                    tool_output = f"工具调用失败: JSON解析错误 - {e}。请检查你的字典格式。"
                    print(f"!! {tool_output}")
                    conversation_history.append(HumanMessage(content=tool_output))
                    continue

            # Gurobi求解工具
            gurobi_match = re.search(r"solve_local_problem_with_gurobi\(agent_id=(\d+),\s*objective=\"(.*?)\"\)", action_str)
            if gurobi_match:
                agent_id = int(gurobi_match.group(1))
                objective = gurobi_match.group(2)
                print(f"--- [执行工具] 调用 CPLEX: agent_id={agent_id}, objective='{objective}' ---")
                tool_output = self._solve_local_problem_with_gurobi(agent_id, objective)
                tool_output_str = f"CPLEX求解器返回结果: {json.dumps(tool_output)}"
                print(f"   [工具返回] {tool_output_str}")
                conversation_history.append(HumanMessage(content=tool_output_str))
                continue

            # 拉格朗日分解工具
            lagrangian_match = re.search(r"get_lagrangian_plan\((.*?)\)\s*$", action_str)
            if lagrangian_match:
                raw_args = lagrangian_match.group(1)
                kwargs = self._parse_tool_kwargs(raw_args)
                config_arg = kwargs.get("config")
                if config_arg is None and raw_args and "=" not in raw_args:
                    config_arg = raw_args.strip()

                print(f"--- [执行工具] 拉格朗日分解 (config={config_arg}) ---")
                solution_dict = self._get_lagrangian_plan(state_description, config=config_arg)
                if isinstance(solution_dict, dict) and "error" in solution_dict:
                    tool_output_str = f"拉格朗日分解失败: {solution_dict['error']}"
                    print(f"   [工具返回] {tool_output_str}")
                    conversation_history.append(HumanMessage(content=tool_output_str))
                    continue

                print("--- [决策] 拉格朗日方案生成成功，准备验证 ---")
                final_solution_dict = solution_dict
                actions = self._verify_and_finalize(final_solution_dict)
                # 移除强制重试逻辑，允许部分回退到MARL
                break

            # 列生成工具
            column_match = re.search(r"get_column_generation_plan\((.*?)\)\s*$", action_str)
            if column_match:
                raw_args = column_match.group(1)
                kwargs = self._parse_tool_kwargs(raw_args)
                config_arg = kwargs.get("config")
                if config_arg is None and raw_args and "=" not in raw_args:
                    config_arg = raw_args.strip()

                print(f"--- [执行工具] 列生成 (config={config_arg}) ---")
                solution_dict = self._get_column_generation_plan(state_description, config=config_arg)
                if isinstance(solution_dict, dict) and "error" in solution_dict:
                    tool_output_str = f"列生成失败: {solution_dict['error']}"
                    print(f"   [工具返回] {tool_output_str}")
                    conversation_history.append(HumanMessage(content=tool_output_str))
                    continue

                print("--- [决策] 列生成方案生成成功，准备验证 ---")
                final_solution_dict = solution_dict
                actions = self._verify_and_finalize(final_solution_dict)
                # 移除强制重试逻辑，允许部分回退到MARL
                break

            # Benders 分解工具
            benders_match = re.search(r"get_benders_plan\((.*?)\)\s*$", action_str)
            if benders_match:
                raw_args = benders_match.group(1)
                kwargs = self._parse_tool_kwargs(raw_args)
                config_arg = kwargs.get("config")
                if config_arg is None and raw_args and "=" not in raw_args:
                    config_arg = raw_args.strip()

                print(f"--- [执行工具] Benders 分解 (config={config_arg}) ---")
                solution_dict = self._get_benders_plan(state_description, config=config_arg)
                if isinstance(solution_dict, dict) and "error" in solution_dict:
                    tool_output_str = f"Benders 分解失败: {solution_dict['error']}"
                    print(f"   [工具返回] {tool_output_str}")
                    conversation_history.append(HumanMessage(content=tool_output_str))
                    continue

                print("--- [决策] Benders 方案生成成功，准备验证 ---")
                final_solution_dict = solution_dict
                actions = self._verify_and_finalize(final_solution_dict)
                # 移除强制重试逻辑，允许部分回退到MARL
                break

            # 排队论分析工具
            queue_match = re.search(r"get_queueing_plan\((.*?)\)\s*$", action_str)
            if queue_match:
                raw_args = queue_match.group(1)
                kwargs = self._parse_tool_kwargs(raw_args)
                config_arg = kwargs.get("config")
                if config_arg is None and raw_args and "=" not in raw_args:
                    config_arg = raw_args.strip()

                print(f"--- [执行工具] 排队论分析 (config={config_arg}) ---")
                solution_dict = self._get_queueing_plan(state_description, config=config_arg)
                if isinstance(solution_dict, dict) and "error" in solution_dict:
                    tool_output_str = f"排队论分析失败: {solution_dict['error']}"
                    print(f"   [工具返回] {tool_output_str}")
                    conversation_history.append(HumanMessage(content=tool_output_str))
                    continue

                print("--- [决策] 排队论方案生成成功，准备验证 ---")
                final_solution_dict = solution_dict
                actions = self._verify_and_finalize(final_solution_dict)
                # 移除强制重试逻辑，允许部分回退到MARL
                break

            # LLM 全局方案工具
            llm_global_match = re.search(r"get_llm_global_plan\((.*?)\)\s*$", action_str)
            if llm_global_match:
                print("--- [执行工具] 调用全局LLM生成方案 ---")
                solution_dict = self._get_llm_global_plan(state_description)
                if isinstance(solution_dict, dict) and "error" in solution_dict:
                    tool_output_str = f"全局LLM方案生成失败: {solution_dict['error']}"
                    print(f"   [工具返回] {tool_output_str}")
                    conversation_history.append(HumanMessage(content=tool_output_str))
                    continue

                print("--- [决策] 全局LLM方案生成成功，准备验证并结束 ---")
                final_solution_dict = solution_dict
                actions = self._verify_and_finalize(final_solution_dict)
                # 移除强制重试逻辑，允许部分回退到MARL
                break

            # 全局指令工具
            global_match = re.search(r"get_global_instructions\((.*?)\)$", action_str)
            if global_match:
                raw_args = global_match.group(1)
                kwargs = self._parse_tool_kwargs(raw_args)
                method_arg = kwargs.get("method")
                config_arg = kwargs.get("config")

                print(
                    f"--- [执行工具] 调用全局指令获取 (method={method_arg}, config={config_arg}) ---"
                )
                solution_dict = self._get_global_instructions(
                    state_description,
                    method=method_arg,
                    config=config_arg,
                )

                if isinstance(solution_dict, dict) and "error" in solution_dict:
                    tool_output_str = f"全局指令获取失败: {solution_dict['error']}"
                    print(f"   [工具返回] {tool_output_str}")
                    conversation_history.append(HumanMessage(content=tool_output_str))
                    continue

                print("--- [决策] 已通过全局指令获取完整方案，准备验证并结束 ---")
                final_solution_dict = solution_dict
                actions = self._verify_and_finalize(final_solution_dict)
                # 移除强制重试逻辑，允许部分回退到MARL
                break

            # 环境查询工具
            query_match = re.search(r"query_environment\(\"(.*?)\"\)", action_str)
            if query_match:
                query = query_match.group(1)
                print(f"--- [执行工具] 调用环境查询: query='{query}' ---")
                tool_output = self._query_environment(query)
                tool_output_str = f"环境查询返回结果: {tool_output}"
                print(f"   [工具返回] {tool_output_str}")
                conversation_history.append(HumanMessage(content=tool_output_str))
                continue

            # 如果没有找到工具调用，可能是LLM在总结或出错，让它再试一次
            print(f"--- [状态] 在'行动:'指令中未检测到有效工具调用: '{action_str}' ---")
            # 可以选择在这里添加一条消息提示LLM必须调用工具
            conversation_history.append(HumanMessage(content="你在'行动:'中没有调用任何已知工具。请调用一个工具。"))
        
        else: # for循环正常结束 (未被break)
            print("!! Agent在达到最大步数后仍未做出最终决策。将回退到完整的MARL策略。")
            # 直接返回完整的 MARL 决策作为最终方案
            return TensorDict({self.action_key: marl_actions}, batch_size=[1], device=self.device)

        # 将action从(0,1)映射到环境期望的格式
        final_actions = torch.zeros(1, self.n_agents, 1, device=self.device, dtype=torch.int64)

        # 选取已经通过验证的首个时步计划，用于立即执行
        if self._last_verified_step_plan:
            step_plan = self._last_verified_step_plan
        elif isinstance(final_solution_dict, dict) and "steps" in final_solution_dict and final_solution_dict["steps"]:
            step_plan = final_solution_dict["steps"][0]
        elif isinstance(final_solution_dict, dict):
            step_plan = final_solution_dict
        else:
            step_plan = {}

        for i in range(self.n_agents):
            verified_action = actions[0, i, 0].item()

            if verified_action == 1: # 如果验证后的动作是有效移动
                agent_name = f"agent_{i}"
                decision = step_plan.get(agent_name, {})
                next_node_id = decision.get("next_node")
                
                current_pos = int(self.world.agents[i].state.position[0].item())
                next_nodes_list = self.world.get_next_nodes(current_pos)
                
                try:
                    # 找到目标节点在可行动画列表中的索引，动作值为索引+1
                    target_node_id = int(next_node_id)
                    action_idx = [node.id for node in next_nodes_list].index(target_node_id) + 1
                    final_actions[0, i, 0] = action_idx
                except (ValueError, TypeError):
                    # next_node 无效或未提供 (虽然验证器应该已经捕获，但作为双重保险)
                    print(f"警告: Agent {i} 的目标节点 {next_node_id} 无法映射，将采用MARL策略。")
                    final_actions[0, i, 0] = marl_actions[0, i]

            elif verified_action == 0: # 如果验证后的动作是有效停留
                final_actions[0, i, 0] = 0

            else: # 如果动作被验证器标记为无效 (verified_action == -1)
                print(f"警告: Agent {i} 的决策被验证器否决，将采用MARL策略。")
                # 修复：使用二维索引访问marl_actions
                final_actions[0, i, 0] = marl_actions[0, i]


        print(f"解析后的最终动作为: {final_actions.squeeze().tolist()}")
        self.last_decision_source = "LLM新方案"
        return TensorDict({self.action_key: final_actions}, batch_size=[1], device=self.device)
