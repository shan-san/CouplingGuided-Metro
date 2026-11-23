"""
分层协同框架核心模块

以强化学习作为执行层，实现高频、低延时在线决策；
以LLM Agent作为策略层，在策略不确定性升高或出现扰动时按需介入。

核心设计：
1. 执行层（RL）- 高频决策，毫秒级响应
2. 策略层（LLM Agent）- 按需介入，提供策略指导
3. 触发机制 - 多维度触发条件判断
4. 协同接口 - 层间通信与决策融合
"""

import time
import logging
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import numpy as np
import torch

logger = logging.getLogger(__name__)


class InterventionTrigger(Enum):
    """介入触发类型"""
    NONE = auto()              # 无需介入
    DISRUPTION = auto()        # 中断事件
    HIGH_ENTROPY = auto()      # 高不确定性
    LOW_PERFORMANCE = auto()   # 低性能
    HIGH_WAITING = auto()      # 高等待人数
    PERIODIC = auto()          # 周期性检查
    MANUAL = auto()            # 手动触发
    CONFLICT = auto()          # 冲突检测
    SCHEDULE_DRIFT = auto()    # 时刻表偏离


# Backward compatibility for older imports.
TriggerType = InterventionTrigger


@dataclass
class TriggerConfig:
    """触发条件配置"""
    # 不确定性阈值
    entropy_threshold: float = 1.5  # 动作熵阈值
    
    # 性能阈值
    reward_threshold: float = -500.0  # 累积奖励阈值
    reward_window: int = 50  # 奖励观察窗口
    
    # 等待阈值
    max_waiting_threshold: int = 1000  # 单站最大等待人数
    total_waiting_threshold: int = 5000  # 总等待人数
    
    # 周期触发
    periodic_interval: int = 100  # 周期触发间隔（步）
    
    # 时刻表偏离
    schedule_drift_threshold: float = 60.0  # 偏离阈值（秒）
    
    # 冷却时间
    cooldown_steps: int = 10  # 介入后的冷却步数


@dataclass
class LayerState:
    """层状态"""
    name: str
    is_active: bool = True
    last_decision_time: float = 0.0
    decision_count: int = 0
    total_time: float = 0.0
    avg_decision_time: float = 0.0


@dataclass
class CoordinationResult:
    """协同结果"""
    actions: Dict[str, int]  # agent_name -> action
    source: str  # 决策来源
    trigger: InterventionTrigger
    confidence: float
    reasoning: str
    decision_time_ms: float
    layer_contributions: Dict[str, float] = field(default_factory=dict)


class ExecutionLayer:
    """执行层：基于强化学习的高频决策
    
    特点：
    - 毫秒级响应
    - 预训练策略
    - 适合常规场景
    """
    
    def __init__(
        self,
        marl_policy,
        device: str = "cpu",
    ):
        self.policy = marl_policy
        self.device = torch.device(device)
        self.state = LayerState(name="execution_layer")
        
        # 性能监控
        self.recent_rewards = deque(maxlen=100)
        self.recent_entropies = deque(maxlen=50)
    
    def decide(
        self,
        observation: torch.Tensor,
        return_entropy: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """执行层决策
        
        Args:
            observation: 观测tensor
            return_entropy: 是否返回动作熵
        
        Returns:
            (actions, entropy, decision_time_ms)
        """
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output = self.policy(observation.to(self.device))
        
        actions = output.get(("agents", "action"))
        
        # 计算动作熵（用于触发判断）
        entropy = None
        if return_entropy:
            logits = output.get(("agents", "logits"))
            if logits is not None:
                probs = torch.softmax(logits, dim=-1)
                entropy = torch.distributions.Categorical(probs=probs).entropy()
                self.recent_entropies.append(entropy.mean().item())
        
        decision_time = (time.perf_counter() - start_time) * 1000
        
        # 更新状态
        self.state.decision_count += 1
        self.state.total_time += decision_time
        self.state.last_decision_time = decision_time
        self.state.avg_decision_time = self.state.total_time / self.state.decision_count
        
        return actions, entropy, decision_time
    
    def record_reward(self, reward: float):
        """记录奖励（用于性能监控）"""
        self.recent_rewards.append(reward)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        return {
            "avg_reward": np.mean(self.recent_rewards) if self.recent_rewards else 0.0,
            "reward_trend": self._compute_trend(list(self.recent_rewards)),
            "avg_entropy": np.mean(self.recent_entropies) if self.recent_entropies else 0.0,
            "avg_decision_time_ms": self.state.avg_decision_time,
        }
    
    @staticmethod
    def _compute_trend(values: List[float], window: int = 20) -> float:
        """计算趋势（正=上升，负=下降）"""
        if len(values) < window:
            return 0.0
        
        recent = values[-window:]
        older = values[-2*window:-window] if len(values) >= 2*window else values[:-window]
        
        if not older:
            return 0.0
        
        return np.mean(recent) - np.mean(older)


class StrategyLayer:
    """策略层：基于LLM Agent的按需决策
    
    特点：
    - 处理复杂场景
    - 提供策略指导
    - 调用优化工具
    - 生成可解释决策
    """
    
    def __init__(
        self,
        llm_policy,
        rag_manager=None,
        decision_analyzer=None,
    ):
        self.policy = llm_policy
        self.rag_manager = rag_manager
        self.decision_analyzer = decision_analyzer
        self.state = LayerState(name="strategy_layer")
        
        # 缓存
        self.cached_guidance: Optional[Dict[str, Any]] = None
        self.cache_valid_until: int = -1
    
    def decide(
        self,
        env,
        current_state: Dict[str, Any],
        trigger: InterventionTrigger,
        execution_layer_output: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, int], Dict[str, Any], float]:
        """策略层决策
        
        Args:
            env: 环境实例
            current_state: 当前状态字典
            trigger: 触发原因
            execution_layer_output: 执行层输出（用于融合）
        
        Returns:
            (actions, guidance_info, decision_time_ms)
        """
        start_time = time.perf_counter()
        
        # 1. 检索历史经验（RAG）
        rag_context = None
        if self.rag_manager is not None:
            try:
                rag_context = self.rag_manager.retrieve(
                    current_state=current_state,
                    k=5,
                )
            except Exception as e:
                logger.warning(f"RAG检索失败: {e}")
        
        # 2. 调用LLM策略
        try:
            guidance = self._get_llm_guidance(
                env=env,
                current_state=current_state,
                trigger=trigger,
                rag_context=rag_context,
            )
        except Exception as e:
            logger.error(f"LLM策略调用失败: {e}")
            guidance = None
        
        # 3. 融合执行层输出
        actions = self._fuse_decisions(
            llm_guidance=guidance,
            execution_output=execution_layer_output,
            current_state=current_state,
        )
        
        decision_time = (time.perf_counter() - start_time) * 1000
        
        # 更新状态
        self.state.decision_count += 1
        self.state.total_time += decision_time
        self.state.last_decision_time = decision_time
        self.state.avg_decision_time = self.state.total_time / self.state.decision_count
        
        # 构建指导信息
        guidance_info = {
            "trigger": trigger.name,
            "llm_guidance": guidance,
            "rag_context": {
                "num_similar_cases": len(rag_context.get("similar_cases", [])) if rag_context else 0,
                "constraints": rag_context.get("constraints", []) if rag_context else [],
                "recommendations": rag_context.get("strategy_recommendations", []) if rag_context else [],
            } if rag_context else None,
        }
        
        return actions, guidance_info, decision_time
    
    def _get_llm_guidance(
        self,
        env,
        current_state: Dict[str, Any],
        trigger: InterventionTrigger,
        rag_context: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """获取LLM指导"""
        # 检查缓存
        if self.cached_guidance is not None:
            step = current_state.get("step", 0)
            if step < self.cache_valid_until:
                return self.cached_guidance
        
        # 调用LLM策略
        if hasattr(self.policy, "get_llm_guidance"):
            guidance = self.policy.get_llm_guidance()
            
            # 缓存结果
            if guidance:
                self.cached_guidance = guidance
                horizon = getattr(self.policy, "milp_time_horizon", 20)
                self.cache_valid_until = current_state.get("step", 0) + horizon
            
            return guidance
        
        return None
    
    def _fuse_decisions(
        self,
        llm_guidance: Optional[Dict[str, Any]],
        execution_output: Optional[torch.Tensor],
        current_state: Dict[str, Any],
    ) -> Dict[str, int]:
        """融合LLM指导和执行层输出"""
        actions = {}
        
        # 获取focus agents
        focus_agents = set()
        if llm_guidance:
            focus_agents = set(llm_guidance.get("focus_agents", []))
        
        # 从状态获取agent列表
        agent_positions = current_state.get("agent_positions", {})
        
        for idx, (agent_name, pos) in enumerate(agent_positions.items()):
            if idx in focus_agents and llm_guidance:
                # 使用LLM指导
                warm_start = llm_guidance.get("warm_start", {})
                action = self._extract_action_from_guidance(
                    agent_idx=idx,
                    warm_start=warm_start,
                )
                actions[agent_name] = action
            elif execution_output is not None:
                # 使用执行层输出
                try:
                    if execution_output.ndim == 3:
                        action = int(execution_output[0, idx, 0].item())
                    elif execution_output.ndim == 2:
                        action = int(execution_output[0, idx].item())
                    else:
                        action = int(execution_output[idx].item())
                    actions[agent_name] = action
                except (IndexError, RuntimeError):
                    actions[agent_name] = 0  # 默认停留
            else:
                actions[agent_name] = 0  # 默认停留
        
        return actions
    
    def _extract_action_from_guidance(
        self,
        agent_idx: int,
        warm_start: Dict[str, int],
    ) -> int:
        """从指导中提取动作"""
        # 查找 stay[agent_idx, *, 0] 变量
        for var_name, value in warm_start.items():
            if var_name.startswith(f"stay[{agent_idx},") and var_name.endswith(",0]"):
                # 如果 stay=1，则动作=0（停留）
                return 0 if value == 1 else 1
        
        # 默认移动
        return 1
    
    def invalidate_cache(self):
        """清除缓存"""
        self.cached_guidance = None
        self.cache_valid_until = -1


class TriggerEvaluator:
    """触发条件评估器"""
    
    def __init__(self, config: TriggerConfig = None):
        self.config = config or TriggerConfig()
        
        # 状态追踪
        self.last_intervention_step: int = -999
        self.cumulative_rewards: Dict[str, float] = {}
    
    def evaluate(
        self,
        step: int,
        env_state: Dict[str, Any],
        execution_layer: ExecutionLayer,
        entropy: Optional[torch.Tensor] = None,
    ) -> Tuple[bool, InterventionTrigger, str]:
        """评估是否需要策略层介入
        
        Returns:
            (should_intervene, trigger_type, reason)
        """
        # 冷却期检查
        if step - self.last_intervention_step < self.config.cooldown_steps:
            return False, InterventionTrigger.NONE, ""
        
        # 1. 中断事件检查
        disruptions = env_state.get("disruptions", [])
        active_blocks = [
            d for d in disruptions
            if d.get("type") in ("BLOCK", "BLOCKAGE")
        ]
        if active_blocks:
            return True, InterventionTrigger.DISRUPTION, f"检测到 {len(active_blocks)} 处区间中断"
        
        # 2. 高不确定性检查
        if entropy is not None:
            max_entropy = entropy.max().item()
            if max_entropy > self.config.entropy_threshold:
                return True, InterventionTrigger.HIGH_ENTROPY, f"动作熵过高: {max_entropy:.3f}"
        
        # 3. 低性能检查
        perf_stats = execution_layer.get_performance_stats()
        if perf_stats["avg_reward"] < self.config.reward_threshold:
            return True, InterventionTrigger.LOW_PERFORMANCE, f"平均奖励过低: {perf_stats['avg_reward']:.2f}"
        
        # 4. 高等待人数检查
        waiting = env_state.get("waiting_passengers", {})
        if waiting:
            max_waiting = max(waiting.values())
            total_waiting = sum(waiting.values())
            
            if max_waiting > self.config.max_waiting_threshold:
                return True, InterventionTrigger.HIGH_WAITING, f"单站等待过多: {max_waiting}"
            
            if total_waiting > self.config.total_waiting_threshold:
                return True, InterventionTrigger.HIGH_WAITING, f"总等待过多: {total_waiting}"
        
        # 5. 周期性检查
        if step > 0 and step % self.config.periodic_interval == 0:
            return True, InterventionTrigger.PERIODIC, f"周期性检查 (step={step})"
        
        # 6. 时刻表偏离检查
        schedule_drift = env_state.get("schedule_drift", 0.0)
        if abs(schedule_drift) > self.config.schedule_drift_threshold:
            return True, InterventionTrigger.SCHEDULE_DRIFT, f"时刻表偏离: {schedule_drift:.1f}秒"
        
        return False, InterventionTrigger.NONE, ""
    
    def record_intervention(self, step: int):
        """记录介入事件"""
        self.last_intervention_step = step


class HierarchicalCoordinator:
    """分层协同协调器
    
    核心职责：
    1. 管理执行层和策略层
    2. 评估触发条件
    3. 协调层间决策
    4. 维护系统状态
    """
    
    def __init__(
        self,
        execution_layer: ExecutionLayer,
        strategy_layer: StrategyLayer,
        trigger_config: TriggerConfig = None,
        decision_analyzer = None,
    ):
        self.execution_layer = execution_layer
        self.strategy_layer = strategy_layer
        self.trigger_evaluator = TriggerEvaluator(trigger_config)
        self.decision_analyzer = decision_analyzer
        
        # 统计信息
        self.stats = {
            "total_steps": 0,
            "execution_layer_decisions": 0,
            "strategy_layer_interventions": 0,
            "intervention_triggers": {},
            "avg_decision_time_ms": 0.0,
        }
        
        # 决策历史（用于分析）
        self.decision_history = deque(maxlen=1000)
    
    def step(
        self,
        env,
        observation: torch.Tensor,
        env_state: Dict[str, Any],
        force_strategy_layer: bool = False,
    ) -> CoordinationResult:
        """执行一步协同决策
        
        Args:
            env: 环境实例
            observation: 观测tensor
            env_state: 环境状态字典
            force_strategy_layer: 是否强制使用策略层
        
        Returns:
            CoordinationResult
        """
        step = env_state.get("step", self.stats["total_steps"])
        self.stats["total_steps"] = step + 1
        
        start_time = time.perf_counter()
        
        # 1. 执行层快速决策
        exec_actions, entropy, exec_time = self.execution_layer.decide(
            observation=observation,
            return_entropy=True,
        )
        
        # 2. 评估是否需要策略层介入
        should_intervene, trigger, reason = self.trigger_evaluator.evaluate(
            step=step,
            env_state=env_state,
            execution_layer=self.execution_layer,
            entropy=entropy,
        )
        
        if force_strategy_layer:
            should_intervene = True
            trigger = InterventionTrigger.MANUAL
            reason = "手动触发策略层"
        
        # 3. 根据评估结果选择决策来源
        if should_intervene:
            # 策略层介入
            strategy_actions, guidance_info, strategy_time = self.strategy_layer.decide(
                env=env,
                current_state=env_state,
                trigger=trigger,
                execution_layer_output=exec_actions,
            )
            
            self.trigger_evaluator.record_intervention(step)
            self.stats["strategy_layer_interventions"] += 1
            
            # 更新触发统计
            trigger_name = trigger.name
            self.stats["intervention_triggers"][trigger_name] = \
                self.stats["intervention_triggers"].get(trigger_name, 0) + 1
            
            # 构建结果
            result = CoordinationResult(
                actions=strategy_actions,
                source="strategy_layer",
                trigger=trigger,
                confidence=0.9,
                reasoning=reason,
                decision_time_ms=exec_time + strategy_time,
                layer_contributions={
                    "execution_layer": 0.3,
                    "strategy_layer": 0.7,
                },
            )
        else:
            # 仅使用执行层
            self.stats["execution_layer_decisions"] += 1
            
            # 转换为字典格式
            actions = self._tensor_to_actions(exec_actions, env_state)
            
            result = CoordinationResult(
                actions=actions,
                source="execution_layer",
                trigger=InterventionTrigger.NONE,
                confidence=0.7,
                reasoning="常规执行层决策",
                decision_time_ms=exec_time,
                layer_contributions={
                    "execution_layer": 1.0,
                    "strategy_layer": 0.0,
                },
            )
        
        # 更新统计
        total_time = (time.perf_counter() - start_time) * 1000
        self._update_stats(total_time)
        
        # 记录决策历史
        self.decision_history.append({
            "step": step,
            "source": result.source,
            "trigger": result.trigger.name,
            "decision_time_ms": result.decision_time_ms,
        })
        
        # 记录到决策分析器
        if self.decision_analyzer is not None:
            for agent_name, action in result.actions.items():
                pos = env_state.get("agent_positions", {}).get(agent_name, -1)
                self.decision_analyzer.analyze_and_record_decision(
                    step=step,
                    agent_name=agent_name,
                    action=action,
                    target_node=pos,  # 简化，实际应从环境获取
                    source=result.source.upper(),
                    reasoning=result.reasoning,
                    decision_time_ms=result.decision_time_ms,
                )
        
        return result
    
    def _tensor_to_actions(
        self,
        tensor: torch.Tensor,
        env_state: Dict[str, Any],
    ) -> Dict[str, int]:
        """将tensor转换为动作字典"""
        actions = {}
        agent_positions = env_state.get("agent_positions", {})
        
        for idx, agent_name in enumerate(agent_positions.keys()):
            try:
                if tensor.ndim == 3:
                    action = int(tensor[0, idx, 0].item())
                elif tensor.ndim == 2:
                    action = int(tensor[0, idx].item())
                else:
                    action = int(tensor[idx].item())
                actions[agent_name] = action
            except (IndexError, RuntimeError):
                actions[agent_name] = 0
        
        return actions
    
    def _update_stats(self, decision_time_ms: float):
        """更新统计信息"""
        total = self.stats["total_steps"]
        if total > 0:
            # 指数移动平均
            alpha = 0.05
            self.stats["avg_decision_time_ms"] = (
                self.stats["avg_decision_time_ms"] * (1 - alpha) +
                decision_time_ms * alpha
            )
    
    def record_reward(self, reward: float):
        """记录奖励"""
        self.execution_layer.record_reward(reward)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.stats["total_steps"]
        exec_count = self.stats["execution_layer_decisions"]
        strategy_count = self.stats["strategy_layer_interventions"]
        
        return {
            **self.stats,
            "execution_layer_ratio": exec_count / max(total, 1),
            "strategy_layer_ratio": strategy_count / max(total, 1),
            "execution_layer_stats": self.execution_layer.get_performance_stats(),
            "strategy_layer_stats": {
                "decision_count": self.strategy_layer.state.decision_count,
                "avg_decision_time_ms": self.strategy_layer.state.avg_decision_time,
            },
        }
    
    def reset_stats(self):
        """重置统计"""
        self.stats = {
            "total_steps": 0,
            "execution_layer_decisions": 0,
            "strategy_layer_interventions": 0,
            "intervention_triggers": {},
            "avg_decision_time_ms": 0.0,
        }
        self.decision_history.clear()


def create_hierarchical_coordinator(
    marl_policy,
    llm_policy,
    rag_manager=None,
    decision_analyzer=None,
    trigger_config: TriggerConfig = None,
    device: str = "cpu",
) -> HierarchicalCoordinator:
    """创建分层协同协调器
    
    Args:
        marl_policy: 预训练的MARL策略
        llm_policy: LLM策略实例
        rag_manager: RAG管理器（可选）
        decision_analyzer: 决策分析器（可选）
        trigger_config: 触发配置（可选）
        device: 计算设备
    
    Returns:
        HierarchicalCoordinator实例
    """
    # 创建执行层
    execution_layer = ExecutionLayer(
        marl_policy=marl_policy,
        device=device,
    )
    
    # 创建策略层
    strategy_layer = StrategyLayer(
        llm_policy=llm_policy,
        rag_manager=rag_manager,
        decision_analyzer=decision_analyzer,
    )
    
    # 创建协调器
    coordinator = HierarchicalCoordinator(
        execution_layer=execution_layer,
        strategy_layer=strategy_layer,
        trigger_config=trigger_config,
        decision_analyzer=decision_analyzer,
    )
    
    return coordinator
