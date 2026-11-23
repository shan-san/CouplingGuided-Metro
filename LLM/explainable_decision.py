"""
可解释决策分析模块

结合注意力机制回溯决策路径中的关键影响因素，刻画系统状态变化、策略干预与调度结果之间的
关联关系，构建可解释与可信的决策分析方法。

核心功能：
1. 决策路径追踪 - 记录完整的决策链路
2. 注意力权重计算 - 识别关键影响因素
3. 因果关系分析 - 状态-动作-结果关联
4. 可视化解释 - 生成人可读的决策解释
"""

import json
import logging
import math
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StateSnapshot:
    """状态快照"""
    timestamp: float  # 模拟时间
    step: int  # 步骤编号
    
    # 列车状态
    agent_positions: Dict[str, int] = field(default_factory=dict)  # agent -> node_id
    agent_passengers: Dict[str, int] = field(default_factory=dict)
    agent_dwell_times: Dict[str, int] = field(default_factory=dict)
    
    # 环境状态
    waiting_passengers: Dict[int, int] = field(default_factory=dict)  # node_id -> count
    disruptions: List[Dict[str, Any]] = field(default_factory=list)
    
    # 系统指标
    total_waiting: int = 0
    total_boarded: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Decision:
    """决策记录"""
    step: int
    agent_name: str
    
    # 决策内容
    action: int  # 0=停留, 正数=第N个可用方向
    target_node: int
    
    # 决策来源
    source: str  # "MARL", "LLM", "MILP", "RAG", "Heuristic"
    
    # 影响因素（注意力权重）
    attention_weights: Dict[str, float] = field(default_factory=dict)
    
    # 决策理由
    reasoning: str = ""
    
    # 时间信息
    decision_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionOutcome:
    """决策结果"""
    step: int
    agent_name: str
    
    # 执行结果
    executed: bool  # 是否成功执行
    actual_action: int
    actual_target: int
    
    # 即时效果
    reward: float = 0.0
    passengers_boarded: int = 0
    passengers_alighted: int = 0
    
    # 失败原因（如果有）
    failure_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AttentionMechanism:
    """注意力机制：计算决策过程中各因素的重要性权重"""
    
    def __init__(
        self,
        feature_dim: int = 32,
        num_heads: int = 4,
        temperature: float = 1.0,
    ):
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.temperature = temperature
        
        # 特征名称映射
        self.feature_names = [
            # 位置特征
            "current_position_waiting",
            "next_position_waiting",
            "position_is_platform",
            "position_near_disruption",
            
            # 列车特征
            "dwell_time_ratio",
            "passenger_load_ratio",
            "time_since_departure",
            
            # 环境特征
            "total_waiting_ratio",
            "max_waiting_ratio",
            "disruption_severity",
            "disruption_distance",
            
            # 邻居特征
            "neighbor_train_count",
            "neighbor_avg_waiting",
            "headway_variance",
            
            # 历史特征
            "recent_reward_trend",
            "action_entropy",
        ]
    
    def compute_attention(
        self,
        state: StateSnapshot,
        agent_name: str,
        candidate_actions: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """计算注意力权重
        
        Args:
            state: 当前状态快照
            agent_name: 智能体名称
            candidate_actions: 候选动作列表
            context: 额外上下文（如RAG检索结果）
        
        Returns:
            特征名称 -> 注意力权重
        """
        # 提取特征向量
        features = self._extract_features(state, agent_name, candidate_actions, context)
        
        # 计算自注意力分数
        attention_scores = self._self_attention(features)
        
        # 应用softmax
        attention_weights = self._softmax_with_temperature(attention_scores)
        
        # 映射到特征名称
        result = {}
        for i, name in enumerate(self.feature_names):
            if i < len(attention_weights):
                result[name] = float(attention_weights[i])
        
        return result
    
    def _extract_features(
        self,
        state: StateSnapshot,
        agent_name: str,
        candidate_actions: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """提取特征向量"""
        features = []
        
        # === 位置特征 ===
        current_pos = state.agent_positions.get(agent_name, -1)
        current_waiting = state.waiting_passengers.get(current_pos, 0)
        
        # 当前位置等待人数（归一化）
        max_waiting = max(state.waiting_passengers.values()) if state.waiting_passengers else 1
        features.append(current_waiting / max(max_waiting, 1))
        
        # 下一位置等待人数（取候选动作的平均）
        next_waitings = []
        for action in candidate_actions:
            target = action.get("next_node", current_pos)
            next_waitings.append(state.waiting_passengers.get(target, 0))
        features.append(np.mean(next_waitings) / max(max_waiting, 1) if next_waitings else 0)
        
        # 是否在站台
        features.append(1.0 if current_waiting > 0 else 0.0)
        
        # 是否靠近中断
        near_disruption = 0.0
        for d in state.disruptions:
            affected = d.get("affected_nodes", [])
            if current_pos in affected:
                near_disruption = 1.0
                break
        features.append(near_disruption)
        
        # === 列车特征 ===
        dwell_time = state.agent_dwell_times.get(agent_name, 0)
        expected_dwell = 5  # 默认期望停站时间
        features.append(min(dwell_time / max(expected_dwell, 1), 1.0))
        
        passengers = state.agent_passengers.get(agent_name, 0)
        capacity = 2520  # 默认容量
        features.append(passengers / capacity)
        
        features.append(0.5)  # 离站时间（需要更多上下文）
        
        # === 环境特征 ===
        total_waiting = state.total_waiting
        features.append(min(total_waiting / 10000, 1.0))
        features.append(max_waiting / 2000 if max_waiting else 0)
        
        # 中断严重程度
        disruption_severity = len(state.disruptions) / 10
        features.append(min(disruption_severity, 1.0))
        features.append(near_disruption)  # 复用中断距离
        
        # === 邻居特征 ===
        neighbor_count = sum(
            1 for pos in state.agent_positions.values()
            if pos == current_pos and pos != state.agent_positions.get(agent_name)
        )
        features.append(neighbor_count / 5)
        
        neighbor_waiting = []
        for node_id in state.waiting_passengers:
            if abs(node_id - current_pos) < 5:  # 简化的邻居判断
                neighbor_waiting.append(state.waiting_passengers[node_id])
        features.append(np.mean(neighbor_waiting) / max(max_waiting, 1) if neighbor_waiting else 0)
        features.append(0.5)  # 间隔方差（需要历史数据）
        
        # === 历史特征 ===
        features.append(0.5)  # 奖励趋势
        features.append(0.5)  # 动作熵
        
        return np.array(features, dtype=np.float32)
    
    def _self_attention(self, features: np.ndarray) -> np.ndarray:
        """计算自注意力分数"""
        # 简化的自注意力：特征值的绝对值 + 偏差项
        # 实际应用中可以使用学习的Query/Key/Value矩阵
        
        # 添加位置编码
        position_encoding = np.sin(np.arange(len(features)) * 0.1)
        
        # 计算注意力分数
        scores = np.abs(features) + 0.1 * position_encoding
        
        # 强调异常值（高或低的特征值更重要）
        mean_feature = np.mean(features)
        deviation = np.abs(features - mean_feature)
        scores += 0.5 * deviation
        
        return scores
    
    def _softmax_with_temperature(self, scores: np.ndarray) -> np.ndarray:
        """带温度的softmax"""
        scores = scores / self.temperature
        exp_scores = np.exp(scores - np.max(scores))  # 数值稳定性
        return exp_scores / np.sum(exp_scores)


class CausalAnalyzer:
    """因果关系分析器：分析状态-动作-结果的关联"""
    
    def __init__(self):
        # 因果关系统计
        self.action_outcome_stats = defaultdict(lambda: {
            "count": 0,
            "success_count": 0,
            "total_reward": 0.0,
            "avg_passengers_boarded": 0.0,
        })
        
        # 状态-动作关联
        self.state_action_correlations = defaultdict(list)
    
    def record_transition(
        self,
        state_before: StateSnapshot,
        decision: Decision,
        outcome: DecisionOutcome,
        state_after: StateSnapshot,
    ):
        """记录状态转移"""
        # 构建状态特征键
        state_key = self._get_state_key(state_before, decision.agent_name)
        action_key = f"{decision.source}_{decision.action}"
        
        # 更新统计
        stats = self.action_outcome_stats[(state_key, action_key)]
        stats["count"] += 1
        if outcome.executed:
            stats["success_count"] += 1
        stats["total_reward"] += outcome.reward
        
        # 记录详细关联
        self.state_action_correlations[state_key].append({
            "action": decision.action,
            "source": decision.source,
            "reward": outcome.reward,
            "executed": outcome.executed,
        })
    
    def _get_state_key(self, state: StateSnapshot, agent_name: str) -> str:
        """生成状态键（用于分组统计）"""
        # 简化的状态分类
        pos = state.agent_positions.get(agent_name, -1)
        waiting = state.waiting_passengers.get(pos, 0)
        
        waiting_level = "high" if waiting > 500 else "medium" if waiting > 100 else "low"
        disruption_level = "disrupted" if state.disruptions else "normal"
        
        return f"{waiting_level}_{disruption_level}"
    
    def get_action_recommendation(
        self,
        state: StateSnapshot,
        agent_name: str,
    ) -> Dict[str, Any]:
        """基于因果分析推荐动作"""
        state_key = self._get_state_key(state, agent_name)
        
        correlations = self.state_action_correlations.get(state_key, [])
        if not correlations:
            return {"recommendation": None, "confidence": 0.0}
        
        # 统计各动作的平均奖励
        action_rewards = defaultdict(list)
        for c in correlations:
            action_rewards[c["action"]].append(c["reward"])
        
        # 选择平均奖励最高的动作
        best_action = None
        best_reward = float("-inf")
        
        for action, rewards in action_rewards.items():
            avg_reward = np.mean(rewards)
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_action = action
        
        confidence = min(len(correlations) / 100, 1.0)  # 样本量置信度
        
        return {
            "recommendation": best_action,
            "expected_reward": best_reward,
            "confidence": confidence,
            "sample_size": len(correlations),
        }
    
    def analyze_failure_patterns(self) -> List[Dict[str, Any]]:
        """分析失败模式"""
        patterns = []
        
        for (state_key, action_key), stats in self.action_outcome_stats.items():
            if stats["count"] < 5:
                continue
            
            success_rate = stats["success_count"] / stats["count"]
            
            if success_rate < 0.5:  # 失败率高
                patterns.append({
                    "state_pattern": state_key,
                    "action": action_key,
                    "success_rate": success_rate,
                    "count": stats["count"],
                    "avg_reward": stats["total_reward"] / stats["count"],
                })
        
        return sorted(patterns, key=lambda x: x["success_rate"])


class DecisionExplainer:
    """决策解释器：生成人可读的决策解释"""
    
    def __init__(self):
        # 解释模板
        self.templates = {
            "stay_dwell": "列车 {agent} 在站台 {node} 停站中（{dwell}/{expected}），继续等待完成上下客",
            "stay_waiting": "列车 {agent} 在站台 {node} 服务 {waiting} 位等待乘客",
            "move_demand": "列车 {agent} 前往站台 {target}，该站有 {waiting} 位乘客等待",
            "move_avoid": "列车 {agent} 避让中断区段，改道前往 {target}",
            "move_balance": "列车 {agent} 前往 {target} 以平衡线路服务",
            "llm_guided": "LLM策略层建议：{reasoning}",
            "marl_decision": "RL执行层决策：基于当前状态选择动作 {action}",
            "rag_similar": "参考历史相似案例（相似度 {similarity:.2f}）：{case_summary}",
        }

    @staticmethod
    def _describe_action(action: int) -> str:
        """按环境/MARL动作语义返回动作描述。"""
        if action == 0:
            return "停留"
        if action > 0:
            return f"方向{action}"
        return f"动作{action}"
    
    def explain_decision(
        self,
        decision: Decision,
        state: StateSnapshot,
        attention_weights: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """生成决策解释
        
        Returns:
            包含以下字段:
            - summary: 简短摘要
            - detailed_reasoning: 详细推理过程
            - key_factors: 关键影响因素（按重要性排序）
            - confidence_indicators: 置信度指标
            - alternative_actions: 备选方案分析
        """
        # 识别关键因素
        key_factors = self._identify_key_factors(attention_weights)
        
        # 生成摘要
        summary = self._generate_summary(decision, state, key_factors)
        
        # 详细推理
        detailed_reasoning = self._generate_detailed_reasoning(
            decision, state, attention_weights, context
        )
        
        # 置信度评估
        confidence_indicators = self._assess_confidence(decision, attention_weights)
        
        # 备选方案
        alternative_actions = self._analyze_alternatives(decision, state, context)
        
        return {
            "summary": summary,
            "detailed_reasoning": detailed_reasoning,
            "key_factors": key_factors,
            "confidence_indicators": confidence_indicators,
            "alternative_actions": alternative_actions,
            "decision_source": decision.source,
            "timestamp": datetime.now().isoformat(),
        }
    
    def _identify_key_factors(
        self,
        attention_weights: Dict[str, float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """识别关键影响因素"""
        # 按权重排序
        sorted_factors = sorted(
            attention_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        key_factors = []
        for name, weight in sorted_factors[:top_k]:
            # 生成因素描述
            description = self._get_factor_description(name)
            
            key_factors.append({
                "name": name,
                "weight": weight,
                "importance": "high" if weight > 0.15 else "medium" if weight > 0.08 else "low",
                "description": description,
            })
        
        return key_factors
    
    def _get_factor_description(self, factor_name: str) -> str:
        """获取因素描述"""
        descriptions = {
            "current_position_waiting": "当前位置等待乘客数",
            "next_position_waiting": "目标位置等待乘客数",
            "position_is_platform": "当前位置是否为站台",
            "position_near_disruption": "是否靠近中断区域",
            "dwell_time_ratio": "停站时间完成度",
            "passenger_load_ratio": "列车载客率",
            "time_since_departure": "距上次发车时间",
            "total_waiting_ratio": "系统总等待人数比例",
            "max_waiting_ratio": "最大站点等待比例",
            "disruption_severity": "中断严重程度",
            "disruption_distance": "中断距离",
            "neighbor_train_count": "邻近列车数量",
            "neighbor_avg_waiting": "邻近站点平均等待",
            "headway_variance": "发车间隔方差",
            "recent_reward_trend": "近期奖励趋势",
            "action_entropy": "动作不确定性",
        }
        return descriptions.get(factor_name, factor_name)
    
    def _generate_summary(
        self,
        decision: Decision,
        state: StateSnapshot,
        key_factors: List[Dict[str, Any]],
    ) -> str:
        """生成决策摘要"""
        action_desc = self._describe_action(decision.action)
        source_desc = {
            "MARL": "强化学习",
            "LLM": "LLM策略层",
            "MILP": "优化求解器",
            "RAG": "历史经验",
            "Heuristic": "启发式规则",
        }.get(decision.source, decision.source)
        
        top_factor = key_factors[0] if key_factors else {"description": "综合判断"}

        if decision.action == 0:
            action_summary = f"选择{action_desc}在节点 {decision.target_node}"
        else:
            action_summary = f"选择{action_desc}，目标节点 {decision.target_node}"

        return f"列车 {decision.agent_name} {action_summary}，由{source_desc}决策，主要依据：{top_factor['description']}"
    
    def _generate_detailed_reasoning(
        self,
        decision: Decision,
        state: StateSnapshot,
        attention_weights: Dict[str, float],
        context: Optional[Dict[str, Any]],
    ) -> List[str]:
        """生成详细推理过程"""
        reasoning = []
        
        # 1. 状态分析
        current_pos = state.agent_positions.get(decision.agent_name, -1)
        waiting = state.waiting_passengers.get(current_pos, 0)
        
        reasoning.append(f"当前状态：位于节点 {current_pos}，等待乘客 {waiting} 人")
        
        # 2. 中断信息
        if state.disruptions:
            disruption_info = ", ".join([
                f"{d.get('type', 'UNKNOWN')}@{d.get('segment_id', '?')}"
                for d in state.disruptions[:3]
            ])
            reasoning.append(f"活动中断：{disruption_info}")
        
        # 3. 决策依据
        if decision.reasoning:
            reasoning.append(f"决策理由：{decision.reasoning}")
        
        # 4. 关键因素权重
        top_weights = sorted(attention_weights.items(), key=lambda x: x[1], reverse=True)[:3]
        weight_str = ", ".join([f"{k}={v:.3f}" for k, v in top_weights])
        reasoning.append(f"关键因素权重：{weight_str}")
        
        # 5. RAG上下文
        if context and context.get("similar_cases"):
            cases = context["similar_cases"][:2]
            for i, case in enumerate(cases):
                sim = case.get("similarity", 0)
                outcome = case.get("outcome", {}).get("total_reward", 0)
                reasoning.append(f"相似案例{i+1}：相似度{sim:.2f}，历史奖励{outcome:.2f}")
        
        return reasoning
    
    def _assess_confidence(
        self,
        decision: Decision,
        attention_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """评估决策置信度"""
        # 基于注意力分布的置信度
        weights = list(attention_weights.values())
        entropy = -sum(w * np.log(w + 1e-8) for w in weights if w > 0)
        max_entropy = np.log(len(weights))
        concentration = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
        
        # 基于决策来源的置信度
        source_confidence = {
            "MARL": 0.7,
            "LLM": 0.8,
            "MILP": 0.9,
            "RAG": 0.6,
            "Heuristic": 0.5,
        }.get(decision.source, 0.5)
        
        # 综合置信度
        overall_confidence = 0.6 * concentration + 0.4 * source_confidence
        
        return {
            "overall": overall_confidence,
            "attention_concentration": concentration,
            "source_reliability": source_confidence,
            "decision_time_factor": 1.0 if decision.decision_time_ms < 1000 else 0.8,
        }
    
    def _analyze_alternatives(
        self,
        decision: Decision,
        state: StateSnapshot,
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """分析备选方案"""
        alternatives = []
        
        # 备选方案1：在停留和默认发车方向之间切换，保持与环境动作语义一致。
        alt_action = 1 if decision.action == 0 else 0
        alternatives.append({
            "action": alt_action,
            "description": self._describe_action(alt_action),
            "reason": "未被选择，可能因为收益较低",
            "estimated_value": -0.1,  # 估计值
        })
        
        # 从RAG建议中提取备选
        if context and context.get("strategy_recommendations"):
            for rec in context["strategy_recommendations"][:2]:
                alternatives.append({
                    "action": -1,
                    "description": rec,
                    "reason": "RAG系统建议",
                    "estimated_value": 0.0,
                })
        
        return alternatives


class ExplainableDecisionAnalyzer:
    """可解释决策分析器（主类）
    
    集成注意力机制、因果分析和决策解释，提供完整的决策分析能力。
    """
    
    def __init__(
        self,
        max_history: int = 10000,
        attention_temperature: float = 1.0,
    ):
        # 组件
        self.attention = AttentionMechanism(temperature=attention_temperature)
        self.causal_analyzer = CausalAnalyzer()
        self.explainer = DecisionExplainer()
        
        # 决策历史
        self.decision_history: List[Decision] = []
        self.outcome_history: List[DecisionOutcome] = []
        self.state_history: List[StateSnapshot] = []
        
        self.max_history = max_history
        
        # 统计
        self.stats = {
            "total_decisions": 0,
            "successful_executions": 0,
            "total_reward": 0.0,
            "avg_decision_time": 0.0,
        }
    
    def record_state(
        self,
        step: int,
        sim_time: float,
        agent_positions: Dict[str, int],
        agent_passengers: Dict[str, int],
        agent_dwell_times: Dict[str, int],
        waiting_passengers: Dict[int, int],
        disruptions: List[Dict[str, Any]],
    ) -> StateSnapshot:
        """记录状态快照"""
        snapshot = StateSnapshot(
            timestamp=sim_time,
            step=step,
            agent_positions=agent_positions.copy(),
            agent_passengers=agent_passengers.copy(),
            agent_dwell_times=agent_dwell_times.copy(),
            waiting_passengers=waiting_passengers.copy(),
            disruptions=disruptions.copy(),
            total_waiting=sum(waiting_passengers.values()),
            total_boarded=sum(agent_passengers.values()),
        )
        
        self.state_history.append(snapshot)
        
        # 限制历史长度
        if len(self.state_history) > self.max_history:
            self.state_history = self.state_history[-self.max_history:]
        
        return snapshot
    
    def analyze_and_record_decision(
        self,
        step: int,
        agent_name: str,
        action: int,
        target_node: int,
        source: str,
        reasoning: str = "",
        candidate_actions: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        decision_time_ms: float = 0.0,
    ) -> Tuple[Decision, Dict[str, Any]]:
        """分析并记录决策
        
        Returns:
            (决策对象, 解释结果)
        """
        # 获取最新状态
        state = self.state_history[-1] if self.state_history else StateSnapshot(
            timestamp=0, step=step
        )
        
        # 计算注意力权重
        attention_weights = self.attention.compute_attention(
            state=state,
            agent_name=agent_name,
            candidate_actions=candidate_actions or [{"action": action, "next_node": target_node}],
            context=context,
        )
        
        # 创建决策记录
        decision = Decision(
            step=step,
            agent_name=agent_name,
            action=action,
            target_node=target_node,
            source=source,
            attention_weights=attention_weights,
            reasoning=reasoning,
            decision_time_ms=decision_time_ms,
        )
        
        self.decision_history.append(decision)
        self.stats["total_decisions"] += 1
        
        # 生成解释
        explanation = self.explainer.explain_decision(
            decision=decision,
            state=state,
            attention_weights=attention_weights,
            context=context,
        )
        
        # 限制历史长度
        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-self.max_history:]
        
        return decision, explanation
    
    def record_outcome(
        self,
        step: int,
        agent_name: str,
        executed: bool,
        actual_action: int,
        actual_target: int,
        reward: float,
        passengers_boarded: int = 0,
        passengers_alighted: int = 0,
        failure_reason: str = "",
    ) -> DecisionOutcome:
        """记录决策结果"""
        outcome = DecisionOutcome(
            step=step,
            agent_name=agent_name,
            executed=executed,
            actual_action=actual_action,
            actual_target=actual_target,
            reward=reward,
            passengers_boarded=passengers_boarded,
            passengers_alighted=passengers_alighted,
            failure_reason=failure_reason,
        )
        
        self.outcome_history.append(outcome)
        
        # 更新统计
        if executed:
            self.stats["successful_executions"] += 1
        self.stats["total_reward"] += reward
        
        # 更新因果分析
        if len(self.state_history) >= 2 and len(self.decision_history) >= 1:
            state_before = self.state_history[-2]
            state_after = self.state_history[-1]
            decision = self.decision_history[-1]
            
            self.causal_analyzer.record_transition(
                state_before=state_before,
                decision=decision,
                outcome=outcome,
                state_after=state_after,
            )
        
        # 限制历史长度
        if len(self.outcome_history) > self.max_history:
            self.outcome_history = self.outcome_history[-self.max_history:]
        
        return outcome
    
    def get_decision_path(
        self,
        agent_name: str,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """获取决策路径（用于回溯分析）"""
        path = []
        
        for decision, outcome in zip(self.decision_history, self.outcome_history):
            if decision.agent_name != agent_name:
                continue
            
            if start_step is not None and decision.step < start_step:
                continue
            if end_step is not None and decision.step > end_step:
                continue
            
            path.append({
                "step": decision.step,
                "decision": decision.to_dict(),
                "outcome": outcome.to_dict() if outcome.step == decision.step else None,
                "key_factors": sorted(
                    decision.attention_weights.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
            })
        
        return path
    
    def get_causal_insights(self) -> Dict[str, Any]:
        """获取因果洞察"""
        failure_patterns = self.causal_analyzer.analyze_failure_patterns()
        
        return {
            "failure_patterns": failure_patterns,
            "action_outcome_summary": dict(self.causal_analyzer.action_outcome_stats),
            "total_transitions": sum(
                stats["count"]
                for stats in self.causal_analyzer.action_outcome_stats.values()
            ),
        }
    
    def generate_report(
        self,
        agent_name: Optional[str] = None,
        last_n_steps: int = 100,
    ) -> Dict[str, Any]:
        """生成分析报告"""
        # 筛选相关历史
        relevant_decisions = self.decision_history[-last_n_steps:]
        if agent_name:
            relevant_decisions = [d for d in relevant_decisions if d.agent_name == agent_name]
        
        # 统计决策来源分布
        source_distribution = defaultdict(int)
        for d in relevant_decisions:
            source_distribution[d.source] += 1
        
        # 统计动作分布
        action_distribution = defaultdict(int)
        for d in relevant_decisions:
            action_distribution[d.action] += 1
        
        # 关键因素频率
        factor_frequency = defaultdict(float)
        for d in relevant_decisions:
            for factor, weight in d.attention_weights.items():
                if weight > 0.1:  # 只统计显著因素
                    factor_frequency[factor] += weight
        
        # 因果洞察
        causal_insights = self.get_causal_insights()
        
        return {
            "summary": {
                "total_decisions": len(relevant_decisions),
                "success_rate": self.stats["successful_executions"] / max(self.stats["total_decisions"], 1),
                "avg_reward": self.stats["total_reward"] / max(self.stats["total_decisions"], 1),
            },
            "source_distribution": dict(source_distribution),
            "action_distribution": dict(action_distribution),
            "top_influential_factors": sorted(
                factor_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "causal_insights": causal_insights,
            "stats": self.stats.copy(),
        }
    
    def export_history(self, path: str):
        """导出分析历史"""
        data = {
            "decisions": [d.to_dict() for d in self.decision_history],
            "outcomes": [o.to_dict() for o in self.outcome_history],
            "states": [s.to_dict() for s in self.state_history[-1000:]],  # 限制状态数量
            "stats": self.stats,
            "export_time": datetime.now().isoformat(),
        }
        
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"决策历史已导出: {path}")
