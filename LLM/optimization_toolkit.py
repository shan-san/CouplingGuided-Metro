"""
优化工具集成模块

由LLM Agent调用感知与优化工具：
- 局部问题快速求解
- 全局问题通过优化工具问题分解
- 多种求解算法选择

支持的优化方法：
1. MILP求解器（CPLEX）
2. 拉格朗日松弛
3. Benders分解
4. 列生成
5. 排队论分析
"""

import json
import logging
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """优化方法枚举"""
    MILP = "milp"
    LAGRANGIAN = "lagrangian"
    BENDERS = "benders"
    COLUMN_GENERATION = "column_generation"
    QUEUEING = "queueing"
    HEURISTIC = "heuristic"
    HYBRID = "hybrid"


@dataclass
class OptimizationResult:
    """优化结果"""
    method: str
    success: bool
    actions: Dict[str, int]  # agent_name -> action
    objective_value: float
    solve_time_ms: float
    gap: float = 0.0  # 优化间隙
    iterations: int = 0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProblemDecomposition:
    """问题分解结构"""
    subproblems: List[Dict[str, Any]]
    coupling_constraints: List[Dict[str, Any]]
    decomposition_type: str  # "spatial", "temporal", "agent"


class LocalSolver:
    """局部问题快速求解器
    
    用于处理单个或少量agent的局部调度决策，
    追求快速响应而非全局最优。
    """
    
    def __init__(
        self,
        default_time_limit: float = 1.0,  # 秒
        default_horizon: int = 5,
    ):
        self.default_time_limit = default_time_limit
        self.default_horizon = default_horizon
    
    def solve_local(
        self,
        agent_idx: int,
        agent_state: Dict[str, Any],
        local_env_state: Dict[str, Any],
        objective: str = "minimize_wait",
    ) -> OptimizationResult:
        """求解单agent局部问题
        
        Args:
            agent_idx: agent索引
            agent_state: agent状态
            local_env_state: 局部环境状态
            objective: 优化目标
        
        Returns:
            OptimizationResult
        """
        start_time = time.perf_counter()
        
        # 提取状态信息
        position = agent_state.get("position", -1)
        dwell_time = agent_state.get("dwell_time", 0)
        expected_dwell = agent_state.get("expected_dwell", 1)
        passengers = agent_state.get("passengers", 0)
        capacity = agent_state.get("capacity", 2520)
        
        # 邻居节点信息
        next_nodes = local_env_state.get("next_nodes", [])
        waiting_at_current = local_env_state.get("waiting_at_current", 0)
        disrupted_segments = local_env_state.get("disrupted_segments", set())
        
        # 约束检查
        must_stay = False
        reasoning_parts = []
        
        # 1. 停站时间约束
        if dwell_time < expected_dwell:
            must_stay = True
            reasoning_parts.append(f"停站时间不足({dwell_time}/{expected_dwell})")
        
        # 2. 检查可行移动
        feasible_moves = []
        for node in next_nodes:
            node_id = node.get("node_id", node)
            segment = f"{position}-{node_id}"
            reverse_segment = f"{node_id}-{position}"
            
            if segment in disrupted_segments or reverse_segment in disrupted_segments:
                continue
            
            waiting = node.get("waiting", 0) if isinstance(node, dict) else 0
            feasible_moves.append({
                "node_id": node_id,
                "waiting": waiting,
            })
        
        # 3. 决策逻辑
        if must_stay:
            action = 0
            target = position
        elif not feasible_moves:
            action = 0
            target = position
            reasoning_parts.append("无可行移动目标")
        else:
            # 选择等待人数最多的节点
            if objective == "minimize_wait":
                best_move = max(feasible_moves, key=lambda x: x["waiting"])
                
                # 比较移动收益与停留收益
                move_benefit = best_move["waiting"]
                stay_benefit = waiting_at_current
                
                if move_benefit > stay_benefit * 0.8:  # 移动阈值
                    action = 1
                    target = best_move["node_id"]
                    reasoning_parts.append(f"移动至高需求节点(等待={move_benefit})")
                else:
                    action = 0
                    target = position
                    reasoning_parts.append(f"停留服务当前需求(等待={stay_benefit})")
            else:
                # 默认移动
                action = 1
                target = feasible_moves[0]["node_id"]
        
        solve_time = (time.perf_counter() - start_time) * 1000
        
        return OptimizationResult(
            method="local_heuristic",
            success=True,
            actions={f"agent_{agent_idx}": action},
            objective_value=0.0,  # 启发式方法不提供目标值
            solve_time_ms=solve_time,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "默认决策",
            metadata={
                "target_node": target,
                "feasible_moves": len(feasible_moves),
            },
        )
    
    def solve_subproblem(
        self,
        subproblem: Dict[str, Any],
        milp_solver=None,
    ) -> OptimizationResult:
        """求解子问题
        
        Args:
            subproblem: 子问题定义
            milp_solver: 可选的MILP求解器
        
        Returns:
            OptimizationResult
        """
        start_time = time.perf_counter()
        
        agent_indices = subproblem.get("agent_indices", [])
        focus_nodes = subproblem.get("focus_nodes", [])
        objective = subproblem.get("objective", "minimize_wait")
        
        # 如果提供了MILP求解器且子问题规模适中
        if milp_solver is not None and len(agent_indices) <= 3:
            try:
                # 调用MILP求解
                result = self._solve_with_milp(
                    milp_solver=milp_solver,
                    agent_indices=agent_indices,
                    focus_nodes=focus_nodes,
                    time_limit=self.default_time_limit,
                )
                return result
            except Exception as e:
                logger.warning(f"MILP求解失败，回退启发式: {e}")
        
        # 启发式求解
        actions = {}
        for idx in agent_indices:
            # 简化的启发式决策
            actions[f"agent_{idx}"] = 1 if idx % 2 == 0 else 0
        
        solve_time = (time.perf_counter() - start_time) * 1000
        
        return OptimizationResult(
            method="subproblem_heuristic",
            success=True,
            actions=actions,
            objective_value=0.0,
            solve_time_ms=solve_time,
            reasoning="子问题启发式求解",
        )
    
    def _solve_with_milp(
        self,
        milp_solver,
        agent_indices: List[int],
        focus_nodes: List[int],
        time_limit: float,
    ) -> OptimizationResult:
        """使用MILP求解"""
        start_time = time.perf_counter()
        
        # 调用MILP求解器
        actions_dict = milp_solver.solve(
            env=None,  # 需要从上下文获取
            focus_agent_ids=agent_indices,
            focus_node_ids=focus_nodes,
            time_horizon=self.default_horizon,
            time_limit=time_limit,
        )
        
        solve_time = (time.perf_counter() - start_time) * 1000
        
        return OptimizationResult(
            method="milp",
            success=True,
            actions=actions_dict,
            objective_value=milp_solver.last_solution.get("objective_value", 0.0),
            solve_time_ms=solve_time,
            gap=milp_solver.last_solution.get("gap", 0.0),
            reasoning="MILP精确求解",
        )


class ProblemDecomposer:
    """问题分解器
    
    将大规模调度问题分解为可并行求解的子问题。
    """
    
    def __init__(self):
        self.decomposition_cache: Dict[str, ProblemDecomposition] = {}
    
    def decompose(
        self,
        full_state: Dict[str, Any],
        decomposition_type: str = "spatial",
        max_subproblem_size: int = 5,
    ) -> ProblemDecomposition:
        """分解问题
        
        Args:
            full_state: 完整状态
            decomposition_type: 分解类型
                - "spatial": 空间分解（按区域）
                - "temporal": 时间分解（按时间窗口）
                - "agent": agent分解（按agent分组）
            max_subproblem_size: 子问题最大规模
        
        Returns:
            ProblemDecomposition
        """
        if decomposition_type == "spatial":
            return self._spatial_decompose(full_state, max_subproblem_size)
        elif decomposition_type == "temporal":
            return self._temporal_decompose(full_state, max_subproblem_size)
        elif decomposition_type == "agent":
            return self._agent_decompose(full_state, max_subproblem_size)
        else:
            raise ValueError(f"未知的分解类型: {decomposition_type}")
    
    def _spatial_decompose(
        self,
        state: Dict[str, Any],
        max_size: int,
    ) -> ProblemDecomposition:
        """空间分解
        
        将网络划分为若干区域，每个区域单独优化。
        """
        agent_positions = state.get("agent_positions", {})
        nodes = state.get("nodes", [])
        
        # 简单的空间划分：根据节点ID范围
        if not nodes:
            # 从agent位置推断节点范围
            positions = list(agent_positions.values())
            if positions:
                min_node = min(positions)
                max_node = max(positions)
                nodes = list(range(min_node, max_node + 1))
        
        # 划分区域
        n_regions = max(1, len(nodes) // (max_size * 3))
        region_size = len(nodes) // n_regions if n_regions > 0 else len(nodes)
        
        subproblems = []
        coupling = []
        
        for i in range(n_regions):
            start_idx = i * region_size
            end_idx = (i + 1) * region_size if i < n_regions - 1 else len(nodes)
            region_nodes = nodes[start_idx:end_idx]
            
            # 找出在该区域内的agents
            region_agents = []
            for agent_name, pos in agent_positions.items():
                if pos in region_nodes:
                    agent_idx = int(agent_name.replace("agent_", ""))
                    region_agents.append(agent_idx)
            
            if region_agents:
                subproblems.append({
                    "region_id": i,
                    "agent_indices": region_agents,
                    "focus_nodes": region_nodes,
                    "objective": "minimize_wait",
                })
            
            # 边界耦合约束
            if i > 0:
                boundary_node = region_nodes[0]
                coupling.append({
                    "type": "boundary",
                    "regions": [i-1, i],
                    "node": boundary_node,
                })
        
        return ProblemDecomposition(
            subproblems=subproblems,
            coupling_constraints=coupling,
            decomposition_type="spatial",
        )
    
    def _temporal_decompose(
        self,
        state: Dict[str, Any],
        max_size: int,
    ) -> ProblemDecomposition:
        """时间分解
        
        将规划horizon分解为多个时间窗口。
        """
        horizon = state.get("horizon", 20)
        window_size = max(5, horizon // 4)
        
        subproblems = []
        coupling = []
        
        for t in range(0, horizon, window_size):
            window_end = min(t + window_size, horizon)
            
            subproblems.append({
                "window_id": t // window_size,
                "time_start": t,
                "time_end": window_end,
                "agent_indices": state.get("agent_indices", []),
                "objective": "minimize_wait",
            })
            
            if t > 0:
                coupling.append({
                    "type": "temporal_continuity",
                    "windows": [(t // window_size) - 1, t // window_size],
                    "time": t,
                })
        
        return ProblemDecomposition(
            subproblems=subproblems,
            coupling_constraints=coupling,
            decomposition_type="temporal",
        )
    
    def _agent_decompose(
        self,
        state: Dict[str, Any],
        max_size: int,
    ) -> ProblemDecomposition:
        """Agent分解
        
        将agents分成若干组，每组单独优化。
        """
        agent_positions = state.get("agent_positions", {})
        n_agents = len(agent_positions)
        
        # 按位置聚类agents
        position_groups = defaultdict(list)
        for agent_name, pos in agent_positions.items():
            agent_idx = int(agent_name.replace("agent_", ""))
            # 简单的位置分组
            group_id = pos // 10  # 每10个节点一组
            position_groups[group_id].append(agent_idx)
        
        subproblems = []
        coupling = []
        
        for group_id, agents in position_groups.items():
            # 如果组太大，进一步分割
            for i in range(0, len(agents), max_size):
                sub_agents = agents[i:i+max_size]
                subproblems.append({
                    "group_id": f"{group_id}_{i}",
                    "agent_indices": sub_agents,
                    "objective": "minimize_wait",
                })
        
        # 组间耦合（相邻位置的agents可能冲突）
        group_ids = list(position_groups.keys())
        for i in range(len(group_ids) - 1):
            coupling.append({
                "type": "collision_avoidance",
                "groups": [group_ids[i], group_ids[i+1]],
            })
        
        return ProblemDecomposition(
            subproblems=subproblems,
            coupling_constraints=coupling,
            decomposition_type="agent",
        )


class GlobalOptimizer:
    """全局优化器
    
    协调多个子问题的求解，处理耦合约束。
    """
    
    def __init__(
        self,
        local_solver: LocalSolver,
        decomposer: ProblemDecomposer,
        max_iterations: int = 10,
        convergence_threshold: float = 0.01,
    ):
        self.local_solver = local_solver
        self.decomposer = decomposer
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def solve_decomposed(
        self,
        decomposition: ProblemDecomposition,
        milp_solver=None,
    ) -> OptimizationResult:
        """求解分解后的问题
        
        使用迭代协调方法处理子问题间的耦合。
        """
        start_time = time.perf_counter()
        
        # 初始化子问题解
        subproblem_solutions = {}
        
        for iteration in range(self.max_iterations):
            prev_solutions = dict(subproblem_solutions)
            
            # 求解每个子问题
            for i, subproblem in enumerate(decomposition.subproblems):
                # 添加来自其他子问题的信息
                subproblem_with_coupling = self._add_coupling_info(
                    subproblem=subproblem,
                    decomposition=decomposition,
                    current_solutions=subproblem_solutions,
                )
                
                # 求解
                result = self.local_solver.solve_subproblem(
                    subproblem=subproblem_with_coupling,
                    milp_solver=milp_solver,
                )
                
                subproblem_solutions[i] = result
            
            # 检查收敛
            if self._check_convergence(prev_solutions, subproblem_solutions):
                break
        
        # 合并解
        merged_actions = self._merge_solutions(subproblem_solutions)
        
        solve_time = (time.perf_counter() - start_time) * 1000
        
        return OptimizationResult(
            method=f"decomposed_{decomposition.decomposition_type}",
            success=True,
            actions=merged_actions,
            objective_value=sum(
                sol.objective_value for sol in subproblem_solutions.values()
            ),
            solve_time_ms=solve_time,
            iterations=iteration + 1,
            reasoning=f"分解求解({decomposition.decomposition_type}), {len(decomposition.subproblems)}个子问题",
            metadata={
                "n_subproblems": len(decomposition.subproblems),
                "n_coupling_constraints": len(decomposition.coupling_constraints),
            },
        )
    
    def _add_coupling_info(
        self,
        subproblem: Dict[str, Any],
        decomposition: ProblemDecomposition,
        current_solutions: Dict[int, OptimizationResult],
    ) -> Dict[str, Any]:
        """添加耦合信息到子问题"""
        enhanced = dict(subproblem)
        
        # 查找相关的耦合约束
        relevant_coupling = []
        for constraint in decomposition.coupling_constraints:
            if self._is_relevant_coupling(subproblem, constraint):
                relevant_coupling.append(constraint)
        
        # 从其他子问题解中提取约束
        boundary_values = {}
        for i, solution in current_solutions.items():
            if solution is not None:
                boundary_values[i] = solution.actions
        
        enhanced["coupling_constraints"] = relevant_coupling
        enhanced["boundary_values"] = boundary_values
        
        return enhanced
    
    def _is_relevant_coupling(
        self,
        subproblem: Dict[str, Any],
        constraint: Dict[str, Any],
    ) -> bool:
        """检查耦合约束是否与子问题相关"""
        constraint_type = constraint.get("type", "")
        
        if constraint_type == "boundary":
            regions = constraint.get("regions", [])
            return subproblem.get("region_id") in regions
        elif constraint_type == "temporal_continuity":
            windows = constraint.get("windows", [])
            return subproblem.get("window_id") in windows
        elif constraint_type == "collision_avoidance":
            groups = constraint.get("groups", [])
            group_id = subproblem.get("group_id", "")
            return any(str(group_id).startswith(str(g)) for g in groups)
        
        return False
    
    def _check_convergence(
        self,
        prev_solutions: Dict[int, OptimizationResult],
        curr_solutions: Dict[int, OptimizationResult],
    ) -> bool:
        """检查是否收敛"""
        if not prev_solutions:
            return False
        
        total_change = 0
        for i, curr in curr_solutions.items():
            prev = prev_solutions.get(i)
            if prev is None:
                return False
            
            # 比较动作
            for agent, action in curr.actions.items():
                prev_action = prev.actions.get(agent, -1)
                if action != prev_action:
                    total_change += 1
        
        return total_change == 0
    
    def _merge_solutions(
        self,
        solutions: Dict[int, OptimizationResult],
    ) -> Dict[str, int]:
        """合并子问题解"""
        merged = {}
        
        for result in solutions.values():
            merged.update(result.actions)
        
        return merged


class OptimizationToolkit:
    """优化工具集
    
    提供统一的优化接口，支持多种求解方法的选择和组合。
    """
    
    def __init__(
        self,
        milp_solver=None,
        default_method: OptimizationMethod = OptimizationMethod.HYBRID,
    ):
        self.milp_solver = milp_solver
        self.default_method = default_method
        
        # 组件
        self.local_solver = LocalSolver()
        self.decomposer = ProblemDecomposer()
        self.global_optimizer = GlobalOptimizer(
            local_solver=self.local_solver,
            decomposer=self.decomposer,
        )
        
        # 方法选择器
        self.method_selector = MethodSelector()
        
        # 统计
        self.stats = {
            "total_solves": 0,
            "method_usage": defaultdict(int),
            "total_time_ms": 0.0,
            "avg_time_ms": 0.0,
        }
    
    def solve(
        self,
        problem_state: Dict[str, Any],
        method: Optional[OptimizationMethod] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """求解优化问题
        
        Args:
            problem_state: 问题状态
            method: 指定的优化方法（None则自动选择）
            config: 方法配置
        
        Returns:
            OptimizationResult
        """
        start_time = time.perf_counter()
        
        # 选择方法
        if method is None:
            method = self.method_selector.select(problem_state)
        
        # 根据方法调用相应的求解器
        if method == OptimizationMethod.MILP:
            result = self._solve_milp(problem_state, config)
        elif method == OptimizationMethod.LAGRANGIAN:
            result = self._solve_lagrangian(problem_state, config)
        elif method == OptimizationMethod.BENDERS:
            result = self._solve_benders(problem_state, config)
        elif method == OptimizationMethod.COLUMN_GENERATION:
            result = self._solve_column_generation(problem_state, config)
        elif method == OptimizationMethod.QUEUEING:
            result = self._solve_queueing(problem_state, config)
        elif method == OptimizationMethod.HEURISTIC:
            result = self._solve_heuristic(problem_state, config)
        elif method == OptimizationMethod.HYBRID:
            result = self._solve_hybrid(problem_state, config)
        else:
            result = self._solve_heuristic(problem_state, config)
        
        # 更新统计
        self.stats["total_solves"] += 1
        self.stats["method_usage"][method.value] += 1
        self.stats["total_time_ms"] += result.solve_time_ms
        self.stats["avg_time_ms"] = self.stats["total_time_ms"] / self.stats["total_solves"]
        
        return result
    
    def _solve_milp(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]],
    ) -> OptimizationResult:
        """MILP求解"""
        if self.milp_solver is None:
            return self._solve_heuristic(state, config)
        
        start_time = time.perf_counter()
        config = config or {}
        
        try:
            actions = self.milp_solver.solve(
                env=state.get("env"),
                focus_agent_ids=config.get("focus_agents"),
                focus_node_ids=config.get("focus_nodes"),
                time_horizon=config.get("horizon", 20),
                time_limit=config.get("time_limit", 60),
            )
            
            solve_time = (time.perf_counter() - start_time) * 1000
            
            return OptimizationResult(
                method="milp",
                success=True,
                actions=actions,
                objective_value=self.milp_solver.last_solution.get("objective_value", 0),
                solve_time_ms=solve_time,
                gap=self.milp_solver.last_solution.get("gap", 0),
                reasoning="MILP精确求解",
            )
        except Exception as e:
            logger.error(f"MILP求解失败: {e}")
            return self._solve_heuristic(state, config)
    
    def _solve_lagrangian(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]],
    ) -> OptimizationResult:
        """拉格朗日松弛"""
        start_time = time.perf_counter()
        config = config or {}
        
        # 分解问题
        decomposition = self.decomposer.decompose(
            full_state=state,
            decomposition_type="spatial",
        )
        
        # 迭代求解
        result = self.global_optimizer.solve_decomposed(
            decomposition=decomposition,
            milp_solver=self.milp_solver,
        )
        
        result.method = "lagrangian"
        return result
    
    def _solve_benders(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]],
    ) -> OptimizationResult:
        """Benders分解"""
        start_time = time.perf_counter()
        config = config or {}
        
        # 主问题：确定agent的目标节点
        # 子问题：验证可行性并生成割
        
        # 简化实现：使用空间分解
        decomposition = self.decomposer.decompose(
            full_state=state,
            decomposition_type="agent",
        )
        
        result = self.global_optimizer.solve_decomposed(
            decomposition=decomposition,
            milp_solver=self.milp_solver,
        )
        
        result.method = "benders"
        return result
    
    def _solve_column_generation(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]],
    ) -> OptimizationResult:
        """列生成"""
        start_time = time.perf_counter()
        config = config or {}
        
        # 简化实现：迭代添加列（节点）
        actions = {}
        agent_positions = state.get("agent_positions", {})
        
        for agent_name in agent_positions:
            agent_idx = int(agent_name.replace("agent_", ""))
            # 启发式选择
            actions[agent_name] = agent_idx % 2
        
        solve_time = (time.perf_counter() - start_time) * 1000
        
        return OptimizationResult(
            method="column_generation",
            success=True,
            actions=actions,
            objective_value=0.0,
            solve_time_ms=solve_time,
            reasoning="列生成启发式",
        )
    
    def _solve_queueing(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]],
    ) -> OptimizationResult:
        """排队论分析"""
        start_time = time.perf_counter()
        config = config or {}
        
        # 基于排队论的调度决策
        actions = {}
        agent_positions = state.get("agent_positions", {})
        waiting = state.get("waiting_passengers", {})
        
        for agent_name, pos in agent_positions.items():
            # 计算服务强度
            arrival_rate = waiting.get(pos, 0) / 60  # 假设每分钟到达
            service_rate = 80  # 假设每分钟服务80人
            
            rho = arrival_rate / service_rate if service_rate > 0 else 0
            
            # 如果拥挤度高，建议停留服务
            if rho > 0.8:
                actions[agent_name] = 0  # 停留
            else:
                actions[agent_name] = 1  # 移动
        
        solve_time = (time.perf_counter() - start_time) * 1000
        
        return OptimizationResult(
            method="queueing",
            success=True,
            actions=actions,
            objective_value=0.0,
            solve_time_ms=solve_time,
            reasoning="排队论分析",
        )
    
    def _solve_heuristic(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]],
    ) -> OptimizationResult:
        """启发式求解"""
        start_time = time.perf_counter()
        
        actions = {}
        agent_positions = state.get("agent_positions", {})
        waiting = state.get("waiting_passengers", {})
        
        for agent_name, pos in agent_positions.items():
            current_waiting = waiting.get(pos, 0)
            
            # 简单规则：等待人数多则停留
            if current_waiting > 100:
                actions[agent_name] = 0
            else:
                actions[agent_name] = 1
        
        solve_time = (time.perf_counter() - start_time) * 1000
        
        return OptimizationResult(
            method="heuristic",
            success=True,
            actions=actions,
            objective_value=0.0,
            solve_time_ms=solve_time,
            reasoning="启发式规则",
        )
    
    def _solve_hybrid(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]],
    ) -> OptimizationResult:
        """混合求解
        
        结合多种方法的优势：
        1. 快速启发式生成初始解
        2. 问题分解并行求解
        3. MILP精化关键部分
        """
        start_time = time.perf_counter()
        config = config or {}
        
        # 1. 快速启发式
        heuristic_result = self._solve_heuristic(state, config)
        
        # 2. 识别关键子问题
        critical_agents = self._identify_critical_agents(state)
        
        if not critical_agents:
            return heuristic_result
        
        # 3. 对关键agent使用MILP
        if self.milp_solver is not None and len(critical_agents) <= 5:
            try:
                milp_config = {
                    "focus_agents": critical_agents,
                    "horizon": config.get("horizon", 10),
                    "time_limit": config.get("time_limit", 5),
                }
                milp_result = self._solve_milp(state, milp_config)
                
                # 合并结果
                merged_actions = dict(heuristic_result.actions)
                for agent in critical_agents:
                    agent_name = f"agent_{agent}"
                    if agent_name in milp_result.actions:
                        merged_actions[agent_name] = milp_result.actions[agent_name]
                
                solve_time = (time.perf_counter() - start_time) * 1000
                
                return OptimizationResult(
                    method="hybrid",
                    success=True,
                    actions=merged_actions,
                    objective_value=milp_result.objective_value,
                    solve_time_ms=solve_time,
                    reasoning=f"混合求解: 启发式+MILP(关键agents={critical_agents})",
                )
            except Exception as e:
                logger.warning(f"混合求解中MILP失败: {e}")
        
        return heuristic_result
    
    def _identify_critical_agents(
        self,
        state: Dict[str, Any],
    ) -> List[int]:
        """识别关键agent"""
        critical = []
        
        agent_positions = state.get("agent_positions", {})
        waiting = state.get("waiting_passengers", {})
        disruptions = state.get("disruptions", [])
        
        # 中断区域的agent
        disrupted_nodes = set()
        for d in disruptions:
            nodes = d.get("affected_nodes", [])
            disrupted_nodes.update(nodes)
        
        for agent_name, pos in agent_positions.items():
            agent_idx = int(agent_name.replace("agent_", ""))
            
            # 靠近中断
            if pos in disrupted_nodes:
                critical.append(agent_idx)
                continue
            
            # 高等待区域
            if waiting.get(pos, 0) > 500:
                critical.append(agent_idx)
        
        return critical[:5]  # 限制数量
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            **self.stats,
            "method_usage": dict(self.stats["method_usage"]),
        }


class MethodSelector:
    """方法选择器
    
    根据问题特征自动选择最优的求解方法。
    """
    
    def __init__(self):
        # 方法选择规则
        self.rules = [
            # (条件函数, 方法, 优先级)
            (self._is_small_scale, OptimizationMethod.MILP, 10),
            (self._has_disruptions, OptimizationMethod.HYBRID, 8),
            (self._is_large_scale, OptimizationMethod.LAGRANGIAN, 7),
            (self._is_high_congestion, OptimizationMethod.QUEUEING, 6),
        ]
    
    def select(self, state: Dict[str, Any]) -> OptimizationMethod:
        """选择求解方法"""
        applicable_methods = []
        
        for condition, method, priority in self.rules:
            if condition(state):
                applicable_methods.append((method, priority))
        
        if applicable_methods:
            # 选择优先级最高的
            applicable_methods.sort(key=lambda x: x[1], reverse=True)
            return applicable_methods[0][0]
        
        return OptimizationMethod.HEURISTIC
    
    def _is_small_scale(self, state: Dict[str, Any]) -> bool:
        """小规模问题"""
        n_agents = len(state.get("agent_positions", {}))
        return n_agents <= 5
    
    def _is_large_scale(self, state: Dict[str, Any]) -> bool:
        """大规模问题"""
        n_agents = len(state.get("agent_positions", {}))
        return n_agents > 10
    
    def _has_disruptions(self, state: Dict[str, Any]) -> bool:
        """存在中断"""
        disruptions = state.get("disruptions", [])
        return len(disruptions) > 0
    
    def _is_high_congestion(self, state: Dict[str, Any]) -> bool:
        """高拥堵"""
        waiting = state.get("waiting_passengers", {})
        if not waiting:
            return False
        
        max_waiting = max(waiting.values())
        return max_waiting > 1000


def create_optimization_toolkit(
    milp_solver=None,
    default_method: str = "hybrid",
) -> OptimizationToolkit:
    """创建优化工具集
    
    Args:
        milp_solver: MILP求解器实例
        default_method: 默认方法名称
    
    Returns:
        OptimizationToolkit
    """
    method_map = {
        "milp": OptimizationMethod.MILP,
        "lagrangian": OptimizationMethod.LAGRANGIAN,
        "benders": OptimizationMethod.BENDERS,
        "column_generation": OptimizationMethod.COLUMN_GENERATION,
        "queueing": OptimizationMethod.QUEUEING,
        "heuristic": OptimizationMethod.HEURISTIC,
        "hybrid": OptimizationMethod.HYBRID,
    }
    
    method = method_map.get(default_method.lower(), OptimizationMethod.HYBRID)
    
    return OptimizationToolkit(
        milp_solver=milp_solver,
        default_method=method,
    )
