"""
Hierarchical RL-LLM Agent Policy for Metro Dispatching

分层协同框架：
- 强化学习执行层：高频、低延时在线决策
- LLM Agent策略层：在策略不确定性升高或出现扰动时按需介入

    2. LLM生成可缓存的紧凑全局求解指导
    3. 按LLM指导选择并配置可选OR方法生成初始方案
    4. SLM基于当前步方案做执行细化
3. 优化工具集成 - 局部问题快速求解，全局问题通过问题分解
4. 小模型引导 - SFT微调的小规模语言模型进行求解引导

Author: MetroRL Team
Date: 2025
"""

import torch
import json
import copy
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np

from tensordict import TensorDict

# 导入现有组件

from LLM.llm_policy import LLMAgentPolicy, OperationsResearchToolkit
from LLM.rag.enhanced_rag_manager import EnhancedRAGManager
from LLM.explainable_decision import ExplainableDecisionAnalyzer
from LLM.hierarchical_coordinator import HierarchicalCoordinator, TriggerType
from LLM.optimization_toolkit import OptimizationToolkit, OptimizationMethod
from LLM.small_model_guidance import SmallModelGuidance, MultiModelFusion
from MILP.milp_model import MILPSolver
from LLM.dispatch_coordination import DispatchCoordination, policy_entropy


logger = logging.getLogger(__name__)


class DecisionLayer(Enum):
    """决策层级枚举"""
    RL_EXECUTION = "rl_execution"           # RL执行层：高频低延时
    RULE_EXECUTION = "rule_execution"
    LLM_STRATEGY = "llm_strategy"           # LLM策略层：复杂决策
    HYBRID = "hybrid"                       # 混合决策：两层协同


@dataclass
class HierarchicalConfig:
    """分层框架配置"""
    # 执行层配置
    rl_model_path: str = None
    rl_inference_timeout_ms: int = 10  # RL推理超时(毫秒)
    
    # 策略层配置
    llm_api_key: str = ""
    llm_base_url: str = None
    llm_model_path: str = "gpt-4"
    llm_temperature: float = 0.1
    llm_max_input_tokens: int = 32768
    llm_max_output_tokens: int = 1024
    llm_request_timeout_s: float = 30.
    llm_thinking_mode: str = "disabled"
    llm_reasoning_effort: str = "low"
    solver_reserve_fraction: float = .5
    
    # 小模型配置
    small_model_path: str = None
    small_model_base_url: str = None
    small_model_api_key: str = None
    small_model_model: str = "qwen2.5-7b-instruct"
    small_model_backend: str = "auto"
    small_model_tokenizer_path: Optional[str] = None
    small_model_request_timeout: float = 30.0
    use_small_model_guidance: bool = True
    
    # RAG配置
    rag_db_path: str = None
    rag_use_embeddings: bool = False
    rag_top_k: int = 5
    use_rag: bool = True
    
    # 触发条件配置
    entropy_threshold: float = 0.8          # 策略熵阈值
    performance_threshold: float = 0.7      # 性能阈值
    waiting_threshold: float = 50.0         # 等待人数阈值
    waiting_total_threshold: Optional[float] = None
    waiting_max_threshold: Optional[float] = None
    trigger_mode: str = "coupling"
    coupling_check_horizon: int = 3
    coupling_service_gap: float = 20.0
    coupling_persistence: int = 2
    intervention_cooldown: int = 10         # 介入冷却时间
    no_op_intervention_cooldown: int = 120  # 策略层判定无需编排后的退避步数
    always_hierarchical: bool = False       # 消融When2Ask：跳过触发器，保留plan cache
    
    # 优化配置
    local_solver_timeout_ms: int = 100      # 局部求解超时(毫秒)
    global_decomposition: bool = True       # 是否使用问题分解
    use_milp_execution: bool = True         # 由MILP执行最终动作
    milp_time_horizon: int = 20             # MILP规划范围
    milp_time_limit: int = 30               # MILP求解时限(秒)
    max_milp_focus_agents: int = 12         # 在线局部MILP最多重优化的critical trains
    cached_plan_max_steps: int = 20         # 单次高层计划最多回放步数
    cached_plan_waiting_drift_ratio: float = 0.15  # 等待人数漂移超过该比例时强制重规划
    milp_acceptance_gate: bool = True       # MILP首步弱于RL基线时不采用
    milp_terminal_focus_bonus: bool = True  # MILP目标中加入软热点终端势能
    default_station_dwell_cap: int = 60     # 对齐MILP ground truth的默认最大停站步数
    cached_target_tracking: bool = True     # cache回放时按目标节点追踪，而不是机械照抄动作
    coordination_time_budget_s: Optional[float] = None
    allowed_decision_delay_s: Optional[float] = None
    joint_guidance_mode: str = "full"
    reference_quality_guard: bool = True
    reference_policy: str = "marl"
    coordination_period: int = 10
    coordination_scope: str = "coupling"
    nearest_vehicles: int = 6
    optimization_mode: str = "joint"
    fixed_planning_horizon: bool = False
    coordination_horizon_policy: str = "adaptive"
    coordination_max_horizon: int = 24
    coordination_service_tail: int = 3
    scope_expansion_agents: int = 2
    scope_window_closure: bool = False
    scope_window_vehicles: int = 12
    
    # 可解释性配置
    enable_explanation: bool = True
    explanation_detail_level: str = "medium"  # low/medium/high


@dataclass
class DecisionRecord:
    """决策记录"""
    timestamp: float
    step: int
    layer: DecisionLayer
    state_hash: str
    actions: Dict[str, int]
    reasoning: Dict[str, str]
    confidence: float
    trigger_type: Optional[str] = None
    operational_metrics: Dict[str, float] = field(default_factory=dict)
    rag_retrieved_cases: int = 0
    attention_weights: Optional[Dict] = None
    optimization_method: Optional[str] = None
    latency_ms: float = 0.0
    coordination: Dict[str, Any] = field(default_factory=dict)


class HierarchicalLLMPolicy:
    """
    分层RL-LLM Agent策略
    
    核心设计思想：
    1. 强化学习作为执行层，实现高频、低延时在线决策
    2. LLM Agent作为策略层，在策略不确定性升高或出现扰动时按需介入
    3. RAG检索增强生成，匹配历史相似状态，生成初始解与经验约束
    4. 注意力机制回溯决策路径中的关键影响因素
    5. 小规模语言模型进行快速求解引导
    """
    
    def __init__(
        self,
        env,
        config: HierarchicalConfig = None,
        n_agents: int = None,
        device: str = "cpu",
    ):
        self.env = env
        self.config = config or HierarchicalConfig()
        from LLM.experiment_controls import validate_experiment_controls
        validate_experiment_controls(self.config)
        self.n_agents = n_agents
        self.device = device
        self.action_key = ("agents", "action")
        
        # 初始化世界引用
        self.world = None
        if env and hasattr(env, "base_env"):
            self.world = env.base_env.scenario.world
        
        # 决策历史
        self.decision_history: deque = deque(maxlen=1000)
        self.current_step = 0
        self.last_intervention_step = -float('inf')
        self.suppress_strategy_until_step = 0
        
        # 缓存的计划
        self.cached_plan_steps: List[Dict[str, Dict[str, Any]]] = []
        self.cached_plan_index: int = 0
        self.cached_plan_trigger_type: Optional[str] = None
        self.cached_plan_signature: Optional[Tuple[Tuple[str, str, float], ...]] = None
        self.cached_focus_agent_ids: Optional[List[int]] = None
        self.cached_global_guidance: Optional[Dict[str, Any]] = None
        self.cached_plan_metrics: Optional[Dict[str, float]] = None
        self.cached_agent_targets: Dict[int, int] = {}
        self.last_small_model_guidance = None
        self._segment_edge_map: Optional[Dict[Tuple[int, int], str]] = None
        
        # 初始化各个子系统
        self._init_execution_layer()
        self._init_strategy_layer()
        self._init_rag_system()
        self._init_explainability()
        self._init_milp_solver()
        self._init_optimization_toolkit()
        self._init_small_model()
        
        # 统计信息
        self.stats = {
            "total_decisions": 0,
            "rl_decisions": 0,
            "rule_decisions": 0,
            "llm_interventions": 0,
            "small_model_uses": 0,
            "rag_retrievals": 0,
            "optimization_calls": 0,
            "milp_solves": 0,
            "rl_safety_overrides": 0,
            "default_dispatch_overrides": 0,
            "target_tracking_overrides": 0,
            "total_latency_ms": 0.0,
        }

        # IO日志 (默认关闭，调用 set_io_log_file() 启用)
        self._io_log_file = None
        self._rag_dim_mismatch_warned = False
        self._dispatch_runtime = None
        self._policy_logits = None

        logger.info("分层RL-LLM策略初始化完成")
    
    def _init_execution_layer(self):
        """初始化RL执行层"""
        self.rl_policy = None
        self.native_rl_policy = None
        self.rl_available = False

        if self.config.reference_policy == "rule":
            if getattr(self.world, "dispatch", None) is None:
                raise ValueError("Rule-reference experiments require the native dispatch world")
            logger.info("Rule-reference ablation: no MARL checkpoint is loaded or called")
            return
        
        if self.config.rl_model_path:
            try:
                payload = torch.load(self.config.rl_model_path, map_location="cpu", weights_only=False)
                if isinstance(payload, dict) and payload.get("kind") == "native_dispatch_mappo_v1":
                    from RL.dispatch_training import NativeDispatchPolicy
                    self.native_rl_policy = NativeDispatchPolicy.load(self.config.rl_model_path, self.device)
                    self.rl_available = True
                    return
                from main_marl import load_rl_policy
                from Params import rl_parser, apply_topology_defaults

                rl_args = rl_parser.parse_args([])
                topology = getattr(self.env, "topology", None)
                if topology is None and hasattr(self.env, "base_env"):
                    topology = getattr(self.env.base_env, "topology", None)
                if topology is not None:
                    apply_topology_defaults(rl_args, topology)

                self.rl_policy = load_rl_policy(
                    self.config.rl_model_path,
                    self.env,
                    torch.device(self.device),
                    rl_args,
                )
                self.rl_available = True
                logger.info(f"RL执行层加载成功: {self.config.rl_model_path}")
            except Exception as e:
                logger.warning(f"RL执行层加载失败: {e}")
                if getattr(self.world, "dispatch", None) is not None and self.config.trigger_mode == "coupling":
                    raise RuntimeError("Requested dispatch RL checkpoint could not be loaded; refusing silent rule fallback") from e
    
    def _init_strategy_layer(self):
        """初始化LLM策略层"""
        self.llm_policy = None
        self.llm_available = False

        if self.config.joint_guidance_mode not in ("full", "no_slm"):
            return
        
        if self.config.llm_api_key and LLMAgentPolicy:
            try:
                self.llm_policy = LLMAgentPolicy(
                    env=self.env,
                    api_key=self.config.llm_api_key,
                    base_url=self.config.llm_base_url,
                    n_agents=self.n_agents,
                    device=self.device,
                    model_path=self.config.llm_model_path,
                    local_milp_time_horizon=self.config.milp_time_horizon,
                    local_milp_time_limit=self.config.milp_time_limit,
                    # global_model 复用主模型配置，避免 model=None 的 pydantic 报错
                    global_model_path=self.config.llm_model_path,
                    global_base_url=self.config.llm_base_url,
                    global_api_key=self.config.llm_api_key,
                    llm_max_input_tokens=self.config.llm_max_input_tokens,
                    llm_max_output_tokens=self.config.llm_max_output_tokens,
                    llm_request_timeout_s=self.config.llm_request_timeout_s,
                    llm_thinking_mode=self.config.llm_thinking_mode,
                    llm_reasoning_effort=self.config.llm_reasoning_effort,
                )
                self.llm_available = True
                logger.info("LLM策略层初始化成功")
            except Exception as e:
                logger.warning(f"LLM策略层初始化失败: {e}")
    
    def _init_rag_system(self):
        """初始化RAG检索系统"""
        self.rag_manager = None

        if not self.config.use_rag:
            logger.info("RAG检索系统已关闭")
            return
        
        if EnhancedRAGManager:
            try:
                # __init__ 内部已根据 db_path 自动调用 load()
                self.rag_manager = EnhancedRAGManager(
                    db_path=self.config.rag_db_path,
                    use_embeddings=self.config.rag_use_embeddings,
                )
                logger.info("RAG检索系统初始化成功")
            except Exception as e:
                logger.warning(f"RAG系统初始化失败: {e}")
    
    def _init_explainability(self):
        """初始化可解释性分析器"""
        self.explainer = None
        
        if self.config.enable_explanation and ExplainableDecisionAnalyzer:
            try:
                self.explainer = ExplainableDecisionAnalyzer()
                logger.info("可解释性分析器初始化成功")
            except Exception as e:
                logger.warning(f"可解释性分析器初始化失败: {e}")
    
    def _init_optimization_toolkit(self):
        """初始化优化工具集"""
        self.optimizer = None
        
        if OptimizationToolkit:
            try:
                self.optimizer = OptimizationToolkit(milp_solver=getattr(self, "milp_solver", None))
                logger.info("优化工具集初始化成功")
            except Exception as e:
                logger.warning(f"优化工具集初始化失败: {e}")

    def _init_milp_solver(self):
        """初始化MILP执行层。"""
        self.milp_solver = None

        if not self.config.use_milp_execution or MILPSolver is None:
            return

        try:
            self.milp_solver = MILPSolver(
                time_horizon=self.config.milp_time_horizon,
                time_limit=self.config.milp_time_limit,
            )
            logger.info("MILP执行层初始化成功")
        except Exception as e:
            logger.warning(f"MILP执行层初始化失败: {e}")
    
    def _init_small_model(self):
        """初始化小模型引导"""
        self.small_model = None
        self.multi_model_fusion = None
        
        if self.config.use_small_model_guidance and SmallModelGuidance:
            try:
                use_small_model_api = bool(
                    self.config.small_model_base_url or self.config.small_model_api_key
                )
                self.small_model = SmallModelGuidance(
                    model_path=self.config.small_model_path,
                    base_url=self.config.small_model_base_url,
                    api_key=self.config.small_model_api_key,
                    model_name=self.config.small_model_model,
                    use_local=bool(self.config.small_model_path) and not use_small_model_api,
                    backend=self.config.small_model_backend,
                    tokenizer_path=self.config.small_model_tokenizer_path,
                    request_timeout=self.config.small_model_request_timeout,
                )
                
                if MultiModelFusion and self.llm_policy:
                    self.multi_model_fusion = MultiModelFusion(
                        small_model=self.small_model,
                        large_model_callback=getattr(self.llm_policy, "_get_llm_global_plan", None),
                    )
                logger.info("小模型引导初始化成功")
            except Exception as e:
                logger.warning(f"小模型引导初始化失败: {e}")
                if self.config.small_model_backend == "vllm":
                    raise RuntimeError("Requested vLLM adapter service could not be initialized") from e
    
    # =========================================================================
    # IO 日志工具
    # =========================================================================

    def set_io_log_file(self, path: str) -> None:
        """打开指定路径的 JSONL 文件，后续大模型/小模型调用均会追加记录。"""
        import io as _io
        import os as _os
        if self._io_log_file is not None:
            try:
                self._io_log_file.close()
            except Exception:
                pass
        directory = _os.path.dirname(path)
        if directory:
            _os.makedirs(directory, exist_ok=True)
        self._io_log_file = _io.open(path, "a", encoding="utf-8", buffering=1)
        logger.info(f"LLM IO 日志已开启 → {path}")

    def _write_io_log(
        self,
        model: str,
        input_text: str,
        output: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """向 IO 日志文件写一行 JSONL 记录（失败时静默跳过）。"""
        if self._io_log_file is None:
            return
        record = {
            "timestamp": time.time(),
            "step": self.current_step,
            "model": model,
            "input": (input_text[:4000] if isinstance(input_text, str) else str(input_text)[:4000]),
            "output": (output if not isinstance(output, str) else output[:4000]),
        }
        if metadata:
            record["metadata"] = metadata
        try:
            self._io_log_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning(f"写 IO 日志失败: {e}")

    # =========================================================================
    # 核心决策接口
    # =========================================================================

    def __call__(self, td: TensorDict, **kwargs) -> TensorDict:
        """主决策入口"""
        return self.decide(td, **kwargs)
    
    def decide(self, td: TensorDict, **kwargs) -> TensorDict:
        """
        分层决策主函数
        
        流程：
        1. 提取当前状态
        2. 评估是否需要LLM介入
        3. 执行相应层级的决策
        4. 记录决策并返回
        """
        start_time = time.time()
        self.current_step += 1
        
        # 更新世界引用
        if self.env and hasattr(self.env, "base_env"):
            self.world = self.env.base_env.scenario.world

        if self.world is not None and getattr(self.world, "dispatch", None) is not None:
            return self._decide_dispatch(td, **kwargs)
        
        # 提取状态信息
        state_info = self._extract_state_info(td)
        
        # 检查是否有缓存的计划
        trigger_type = None
        if self._has_valid_cached_plan(state_info):
            actions, reasoning = self._get_cached_action(td, state_info)
            layer = DecisionLayer.HYBRID if self.small_model else DecisionLayer.LLM_STRATEGY
        else:
            # 评估触发条件，决定使用哪一层
            layer, trigger_type = self._evaluate_intervention(state_info)
            
            if layer == DecisionLayer.LLM_STRATEGY:
                actions, reasoning = self._llm_strategy_decide(td, state_info, trigger_type)
                self.last_intervention_step = self.current_step
                self.stats["llm_interventions"] += 1
            else:
                actions, reasoning = self._rl_execution_decide(td, state_info)
                self.stats["rl_decisions"] += 1

        actions, reasoning = self._enforce_default_dispatch_rules(
            actions=actions,
            reasoning=reasoning,
            state_info=state_info,
        )
        
        # 记录决策
        latency_ms = (time.time() - start_time) * 1000
        self._record_decision(
            state_info=state_info,
            layer=layer,
            actions=actions,
            reasoning=reasoning,
            trigger_type=trigger_type if layer == DecisionLayer.LLM_STRATEGY else None,
            latency_ms=latency_ms,
        )
        
        self.stats["total_decisions"] += 1
        self.stats["total_latency_ms"] += latency_ms
        
        # 构建返回的TensorDict
        return self._build_action_tensordict(td, actions)

    def _decide_dispatch(self, td, **kwargs):
        started = time.perf_counter()
        state = self._extract_state_info(td)
        from dispatch_candidates import public_dispatch_state
        state.update(public_dispatch_state(self.world))
        self._policy_logits = None
        native_reference = None
        if self.config.reference_policy == "rule":
            from LLM.dispatch_coordination import checked_reference
            native_reference = checked_reference(self.world, None, 1)
            actions = dict(native_reference["dispatch_packets"][0]["actions"])
            state["reference_policy"] = "published_route_rule"
        elif self.native_rl_policy:
            native_reference, _, native_info = self.native_rl_policy.propose(self.world, deterministic=True,
                quality_guard=getattr(self.config, "reference_quality_guard", True), defer_service_review=True)
            actions = dict(native_reference["dispatch_packets"][0]["actions"])
            state["native_marl_proposal"] = native_info
        else:
            actions, _ = self._rl_execution_decide(td, state)
        reference_seconds = time.perf_counter() - started
        if self._dispatch_runtime is None:
            if self.milp_solver is None:
                raise RuntimeError("Configured dispatch requires the common joint MILP backend")
            self._dispatch_runtime = DispatchCoordination(self.env, self.config, self.milp_solver)

        def specify(current, context, trigger, *, timeout_s=None):
            if not self.llm_policy:
                return {}
            return self.llm_policy._get_llm_global_guidance(
                current, retrieved_cases=context, request=trigger, timeout_s=timeout_s)

        def guide(current, context, specification, *, timeout_s=None):
            if not self.small_model:
                return {}
            self.stats["small_model_uses"] += 1
            payload = dict(state={k: v for k, v in current.items() if k != "solver_variables"},
                retrieved_cases=context, specification=specification, solver_variables=current["solver_variables"])
            if getattr(self.small_model, "native_input_encoding", None) == "metro_guidance_group_v2":
                from LLM.finetune.guidance_view import inference_payload
                payload = inference_payload(payload)
            prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
            return self.small_model.generate_guidance(state_description=prompt, state_dict=current,
                max_tokens=2048, time_budget_s=timeout_s)

        slm_ready = self.small_model is not None and (
            getattr(self.small_model, "_model", None) is not None or getattr(self.small_model, "_llm_client", None) is not None)
        if self.llm_policy:
            self.llm_policy.last_prompt_audit = None
        actions, record = self._dispatch_runtime.decide(actions, state, logits=self._policy_logits,
            reference_solution=native_reference,
            retrieve=self._rag_retrieve if self.config.use_rag else None,
            specify=specify if self.llm_policy else None,
            guide=guide if self.config.use_small_model_guidance and slm_ready else None,
            candidates=kwargs.get("dispatch_candidates"), passenger_inputs=kwargs.get("passenger_inputs"),
            max_evaluations=kwargs.get("max_evaluations"), started_at=started, budgeted_callbacks=True)
        record["timings_s"]["state_and_marl"] = reference_seconds
        record["reference_policy"] = self.config.reference_policy
        record["reference_quality_guard"] = state.get("native_marl_proposal", {}).get("quality_guard", {})
        record["model_sources"] = dict(rl_loaded=self.rl_available, rl_checkpoint=self.config.rl_model_path,
            native_rl_metadata=getattr(self.native_rl_policy, "metadata", None), rag_path=self.config.rag_db_path,
            llm_available=self.llm_available, llm_model=self.config.llm_model_path,
            slm_path=self.config.small_model_path,
            slm_backend=("vllm" if self.small_model and getattr(self.small_model, "backend", None) == "vllm"
                         else "local" if self.small_model and getattr(self.small_model, "_model", None) is not None
                         else "api" if self.small_model and getattr(self.small_model, "_llm_client", None) is not None
                         else "no_loaded_model"),
            slm_model=self.config.small_model_model, slm_url=getattr(self.small_model, "base_url", None),
            slm_input_encoding=getattr(self.small_model, "native_input_encoding", None))
        record["slm_stats_cumulative"] = dict(getattr(self.small_model, "stats", {}))
        record["llm_prompt_audit"] = getattr(self.llm_policy, "last_prompt_audit", None)
        record["budget_exceeded"] = record["total_decision_s"] > record["budget_s"]
        record["allowed_delay_exceeded"] = record["total_decision_s"] > record["allowed_delay_s"]
        execution_layer = (DecisionLayer.RULE_EXECUTION if self.config.reference_policy == "rule"
                           else DecisionLayer.RL_EXECUTION)
        layer = DecisionLayer.LLM_STRATEGY if record["requested"] else (
            DecisionLayer.HYBRID if record["source"] == "cache" and
            record.get("cache_kind") != "checked_reference" else execution_layer)
        self._record_decision(state, layer, actions, {name: record["source"] for name in actions},
                              record["trigger"], record["total_decision_s"] * 1000)
        self.decision_history[-1].coordination = record
        output = self._build_action_tensordict(td, actions)
        record["total_decision_s"] = time.perf_counter() - started
        record["budget_exceeded"] = record["total_decision_s"] > record["budget_s"]
        record["allowed_delay_exceeded"] = record["total_decision_s"] > record["allowed_delay_s"]
        self.decision_history[-1].latency_ms = record["total_decision_s"] * 1000
        logger.info("dispatch_decision_summary %s", json.dumps(dict(time=record["time"],
            source=record["source"], total_s=record["total_decision_s"], timings_s=record["timings_s"],
            cache_kind=record.get("cache_kind"),
            strictly_improved=record["strictly_improved"], reference_quality_guard=record["reference_quality_guard"],
            best_known_cost=record.get("best_known_cost"), incumbent_history=record.get("incumbent_history", []),
            incumbent_scope=record.get("incumbent_scope"),
            coordination_reference_quality=record.get("coordination_reference_quality"),
            regional_request=record.get("regional_request"), horizon_policy=record.get("horizon_policy"),
            scope_expansion=record.get("scope_expansion"),
            scope_window_closure=record.get("scope_window_closure"),
            pending_service_review=record.get("pending_service_review", False),
            reference_service_review_required=record.get("reference_service_review_required", False),
            skipped_stages=record.get("skipped_stages", []), errors=record["errors"],
            solver_attempts=len(record["attempts"]), solver_results=[
                {k:a.get(k) for k in ("status", "termination", "cost", "reference_cost", "acceptance", "physical_evaluations",
                    "novel_candidate_evaluations", "feasible_candidate_evaluations", "shared_reference_evaluations",
                    "infeasible_candidate_reasons", "blocking_background_vehicles", "timings")}
                for a in record["attempts"]], experiment=record["experiment"],
            requested=record["requested"], trigger=record["trigger"],
            focus_vehicles=len(record.get("focus", {}).get("resolved", []))), ensure_ascii=False))
        self.stats["total_decisions"] += 1
        self.stats["llm_interventions"] += int(record["requested"])
        reference_key = "rule_decisions" if self.config.reference_policy == "rule" else "rl_decisions"
        self.stats[reference_key] += int(not record["requested"] and (
            record["source"] != "cache" or record.get("cache_kind") == "checked_reference"))
        self.stats["milp_solves"] += len(record["attempts"])
        self.stats["total_latency_ms"] += record["total_decision_s"] * 1000
        return output

    def notify_step_result(self, *_):
        if self._dispatch_runtime is not None:
            return self._dispatch_runtime.notify_step_result()

    def has_active_plan(self):
        if self._dispatch_runtime is not None:
            return self._dispatch_runtime.has_active_plan()
        return bool(self.cached_plan_steps and self.cached_plan_index < len(self.cached_plan_steps))
    
    def _extract_state_info(self, td: TensorDict) -> Dict[str, Any]:
        """提取状态信息"""
        state_info = {
            "step": self.current_step,
            "n_agents": self.n_agents,
            "agent_positions": [],
            "agent_dwell_times": [],
            "agent_passengers": [],
            "waiting_passengers": {},
            "total_waiting_passengers": 0.0,
            "max_waiting_passengers": 0.0,
            "avg_waiting_passengers": 0.0,
            "top_wait_nodes": [],
            "disruptions": [],
            "high_wait_nodes": [],
            "over_dwell_agents": [],
        }
        
        if self.world is None:
            return state_info
        
        # 提取列车状态
        for i, agent in enumerate(self.world.agents):
            pos = int(agent.state.position[0].item())
            dwell = int(agent.state.dwell_time[0].item())
            passengers = int(agent.passenger_count[0].item())

            expected_dwell = 0
            expected_dwell_tensor = getattr(agent.state, "current_expected_dwell_time", None)
            if expected_dwell_tensor is None:
                expected_dwell_tensor = getattr(agent.state, "expected_dwell_time", None)
            if expected_dwell_tensor is not None:
                try:
                    expected_dwell = int(expected_dwell_tensor[0].item())
                except Exception:
                    expected_dwell = 0

            state_info["agent_positions"].append({
                "node_id": pos,
                "position": pos,
                "dwell_time": dwell,
                "expected_dwell": expected_dwell,
                "passengers": passengers,
                "capacity": 2520,
                "at_platform": pos in self.world.platform_nodes,
            })
            state_info["agent_dwell_times"].append(dwell)
            state_info["agent_passengers"].append(passengers)

            if pos in self.world.platform_nodes and expected_dwell > 0 and dwell > expected_dwell + 2:
                state_info["over_dwell_agents"].append({
                    "agent_idx": i,
                    "node_id": pos,
                    "dwell_time": dwell,
                    "expected_dwell": expected_dwell,
                    "passengers": passengers,
                })
        
        # 提取站台等待人数
        waiting_entries = []
        for node_id, node in self.world.platform_nodes.items():
            waiting = int(self.world.waiting_at(node_id)) if hasattr(self.world, "waiting_at") else int(node.waiting_passengers[0].item())
            state_info["waiting_passengers"][node_id] = waiting
            waiting_entries.append((node_id, waiting))
            if waiting > self.config.waiting_threshold:
                state_info["high_wait_nodes"].append({
                    "node_idx": node_id,
                    "node_id": node_id,
                    "waiting": waiting,
                })

        if waiting_entries:
            total_waiting = float(sum(waiting for _, waiting in waiting_entries))
            max_waiting = float(max(waiting for _, waiting in waiting_entries))
            avg_waiting = total_waiting / max(len(waiting_entries), 1)
            top_wait_nodes = [
                {"node_idx": node_id, "node_id": node_id, "waiting": waiting}
                for node_id, waiting in sorted(waiting_entries, key=lambda item: item[1], reverse=True)[:8]
                if waiting > 0
            ]
            state_info["total_waiting_passengers"] = total_waiting
            state_info["max_waiting_passengers"] = max_waiting
            state_info["avg_waiting_passengers"] = avg_waiting
            state_info["top_wait_nodes"] = top_wait_nodes

            # waiting_threshold 在命令行中表示“乘客积压”阈值。Topology16 站点多，
            # 单站很少早早超过该阈值；总等待超阈值时仍应把热点站交给 RAG/SLM。
            if total_waiting >= float(self.config.waiting_threshold) and len(state_info["high_wait_nodes"]) < 3:
                existing_nodes = {
                    int(item.get("node_id", item.get("node_idx", -1)))
                    for item in state_info["high_wait_nodes"]
                }
                for item in top_wait_nodes[:6]:
                    node_id = int(item.get("node_id", item.get("node_idx", -1)))
                    if node_id in existing_nodes:
                        continue
                    state_info["high_wait_nodes"].append(dict(item))
                    existing_nodes.add(node_id)
        
        # 提取中断信息
        disruptions = self.world.get_disrupted_segments()
        if disruptions:
            for seg_id, info in disruptions.items():
                state_info["disruptions"].append({
                    "segment": seg_id,
                    "type": info.get("type", "UNKNOWN"),
                    "severity": info.get("severity", 1.0),
                })
        
        state_info["published_events"] = self.world.get_published_events() if hasattr(self.world, "get_published_events") else []
        return state_info
    
    def _evaluate_intervention(self, state_info: Dict[str, Any]) -> Tuple[DecisionLayer, Optional[str]]:
        """
        评估是否需要LLM介入
        
        触发条件：
        1. 检测到中断事件
        2. 策略熵过高（不确定性大）
        3. 站台或系统等待人数过多
        """
        if self.config.always_hierarchical:
            return DecisionLayer.LLM_STRATEGY, "ALWAYS_HIERARCHICAL"

        if self.current_step < self.suppress_strategy_until_step:
            return DecisionLayer.RL_EXECUTION, None

        # 冷却检查
        if self.current_step - self.last_intervention_step < self.config.intervention_cooldown:
            return DecisionLayer.RL_EXECUTION, None
        
        # 1. 中断触发
        if state_info["disruptions"] or any(e["active"] for e in state_info.get("published_events", [])):
            return DecisionLayer.LLM_STRATEGY, "DISRUPTION"
        
        # 2. 高等待人数触发
        total_waiting = float(state_info.get("total_waiting_passengers", 0.0) or 0.0)
        max_waiting = float(state_info.get("max_waiting_passengers", 0.0) or 0.0)
        total_threshold = self.config.waiting_total_threshold
        local_threshold = self.config.waiting_max_threshold
        if (
            total_waiting >= (self.config.waiting_threshold if total_threshold is None else total_threshold)
            or max_waiting >= (self.config.waiting_threshold if local_threshold is None else local_threshold)
        ):
            return DecisionLayer.LLM_STRATEGY, "HIGH_WAITING"
        
        # 3. 策略熵触发（如果RL可用）
        if self.rl_available and self.rl_policy:
            entropy = self._compute_policy_entropy(state_info)
            if entropy > self.config.entropy_threshold:
                return DecisionLayer.LLM_STRATEGY, "HIGH_ENTROPY"
        
        # 默认使用RL执行层
        return DecisionLayer.RL_EXECUTION, None
    
    def _compute_policy_entropy(self, state_info: Dict[str, Any]) -> float:
        """Use current policy probabilities, never pooled action frequencies."""
        return policy_entropy(self._policy_logits)
    
    def _check_performance_decline(self, state_info: Dict[str, Any]) -> bool:
        """检查真实运行指标是否出现持续恶化。"""
        minimum_steps = max(60, self.config.intervention_cooldown * 3)
        if self.current_step < minimum_steps or len(self.decision_history) < 20:
            return False

        recent_history = [
            record.operational_metrics
            for record in list(self.decision_history)[-20:]
            if record.operational_metrics
        ]
        if len(recent_history) < 20:
            return False

        previous_window = recent_history[:10]
        recent_window = recent_history[10:]
        previous_waiting = float(np.mean([item["total_waiting"] for item in previous_window]))
        recent_waiting = float(np.mean([item["total_waiting"] for item in recent_window]))
        recent_max_waiting = float(np.mean([item["max_waiting"] for item in recent_window]))
        recent_dwell_overruns = float(np.mean([item["agents_over_expected_dwell"] for item in recent_window]))

        if previous_waiting <= 0.0:
            waiting_worsened = recent_waiting > (self.config.waiting_threshold * 2.0)
        else:
            waiting_worsened = recent_waiting > previous_waiting * (1.0 + self.config.performance_threshold)

        if not waiting_worsened:
            return False

        current_metrics = self._build_operational_metrics_snapshot(state_info)
        return (
            recent_dwell_overruns >= 1.0
            or recent_max_waiting > self.config.waiting_threshold * 1.5
            or current_metrics["agents_over_expected_dwell"] >= 2.0
        )

    def _build_operational_metrics_snapshot(self, state_info: Dict[str, Any]) -> Dict[str, float]:
        """构建可用于性能回退检测的运行指标快照。"""
        waiting_values = list(state_info.get("waiting_passengers", {}).values())
        agent_positions = state_info.get("agent_positions", [])

        agents_over_expected_dwell = 0
        moving_agents = 0
        for agent in agent_positions:
            dwell_time = float(agent.get("dwell_time", 0.0))
            expected_dwell = float(agent.get("expected_dwell", 0.0))
            at_platform = bool(agent.get("at_platform", False))

            if expected_dwell > 0 and dwell_time > expected_dwell + 2:
                agents_over_expected_dwell += 1
            if not at_platform:
                moving_agents += 1

        return {
            "total_waiting": float(sum(waiting_values)),
            "max_waiting": float(max(waiting_values) if waiting_values else 0.0),
            "agents_over_expected_dwell": float(agents_over_expected_dwell),
            "moving_agents": float(moving_agents),
        }

    def _capture_plan_signature(
        self,
        state_info: Dict[str, Any],
    ) -> Tuple[Tuple[str, str, float], ...]:
        """提取足以判断缓存高层计划是否失效的扰动签名。"""
        disruptions = []
        for item in state_info.get("disruptions", []):
            disruptions.append(
                (
                    str(item.get("segment", "")),
                    str(item.get("type", "UNKNOWN")),
                    float(item.get("severity", 1.0)),
                )
            )
        return tuple(sorted(disruptions))

    def _clear_cached_plan(self) -> None:
        if getattr(self, "_dispatch_runtime", None) is not None:
            self._dispatch_runtime.invalidate_cache("policy_cache_cleared")
        self.cached_plan_steps = []
        self.cached_plan_index = 0
        self.cached_plan_trigger_type = None
        self.cached_plan_signature = None
        self.cached_focus_agent_ids = None
        self.cached_global_guidance = None
        self.cached_plan_metrics = None
        self.cached_agent_targets = {}

    def _get_segment_id(self, u: int, v: int) -> Optional[str]:
        """根据边 (u, v) 查找所属区段。"""
        if self.world is None:
            return None
        if self._segment_edge_map is None:
            self._segment_edge_map = {}
            segment_nodes = getattr(self.world, "segment2nodes", {}) or {}
            for segment, node_list in segment_nodes.items():
                if not node_list:
                    continue
                for idx in range(len(node_list) - 1):
                    start_id = getattr(node_list[idx], "id", None)
                    end_id = getattr(node_list[idx + 1], "id", None)
                    if start_id is None or end_id is None:
                        continue
                    self._segment_edge_map[(start_id, end_id)] = segment.id
                    self._segment_edge_map[(end_id, start_id)] = segment.id
        return self._segment_edge_map.get((u, v))

    # =========================================================================
    # RL执行层决策
    # =========================================================================

    def _is_blocked_transition(self, current_node_id: int, next_node_id: int) -> bool:
        """Return whether the immediate transition is blocked by a disruption."""
        if self.world is None or current_node_id == next_node_id:
            return False
        segment_id = self._get_segment_id(current_node_id, next_node_id)
        if not segment_id:
            return False
        disruptions = self.world.get_disrupted_segments() if hasattr(self.world, "get_disrupted_segments") else {}
        disruption_info = disruptions.get(segment_id) if isinstance(disruptions, dict) else None
        return bool(disruption_info and disruption_info.get("type") in {"BLOCK", "BLOCKAGE"})

    def _select_safe_move_action(
        self,
        agent_idx: int,
        current_node_id: int,
        claimed_next_nodes: Set[int],
    ) -> int:
        """Pick the first executable move action for RL safety fallback."""
        if self.world is None:
            return 0

        try:
            candidate_nodes = [int(node.id) for node in self.world.get_next_nodes(current_node_id)]
        except Exception:
            candidate_nodes = []
        if not candidate_nodes:
            return 0

        tick_seconds = float(getattr(self.world, "tick_seconds", 1.0) or 1.0)
        conflict_distance_limit = max(1, int(np.ceil(30.0 / max(tick_seconds, 1e-6))))

        def _violates_forward_safety(next_node_id: int) -> bool:
            if not hasattr(self.world, "_get_forward_distance"):
                return False
            for other_idx, other_agent in enumerate(getattr(self.world, "agents", [])):
                if other_idx == agent_idx:
                    continue
                other_pos = int(other_agent.state.position[0].item())
                forward_distance = self.world._get_forward_distance(
                    next_node_id,
                    other_pos,
                    conflict_distance_limit,
                )
                if forward_distance < conflict_distance_limit:
                    return True
            return False

        for next_node_id in candidate_nodes:
            if next_node_id in claimed_next_nodes:
                continue
            if self._is_blocked_transition(current_node_id, next_node_id):
                continue
            if _violates_forward_safety(next_node_id):
                continue
            action = self._resolve_next_node_action(current_node_id, next_node_id)
            if action <= 0:
                continue
            claimed_next_nodes.add(next_node_id)
            return action
        return 0

    def _sanitize_rl_actions(
        self,
        actions: Dict[str, int],
        reasoning: Dict[str, str],
    ) -> Tuple[Dict[str, int], Dict[str, str]]:
        """Correct pathological RL stops while preserving normal dwell holds.

        The MARL policy is still the source of the action. This layer only applies
        real-time fallback rules for cases where a stop is clearly harmful or
        inconsistent with the simulator semantics.
        """
        if self.world is None:
            return actions, reasoning

        sanitized_actions = dict(actions)
        sanitized_reasoning = dict(reasoning)
        claimed_next_nodes: Set[int] = set()
        override_count = 0

        for agent_key, action in sorted(actions.items(), key=lambda item: int(item[0].split("_")[1])):
            try:
                agent_idx = int(agent_key.split("_")[1])
            except (IndexError, ValueError):
                continue
            if agent_idx >= len(self.world.agents):
                continue

            agent = self.world.agents[agent_idx]
            current_node_id = int(agent.state.position[0].item())
            action_value = int(action)
            if action_value > 0:
                next_node_id = self._resolve_action_next_node(current_node_id, action_value)
                if next_node_id is not None and next_node_id != current_node_id:
                    claimed_next_nodes.add(int(next_node_id))
                continue

            is_platform = current_node_id in getattr(self.world, "platform_nodes", {})
            dwell_time = float(getattr(agent.state, "dwell_time", torch.tensor([0.0]))[0].item())
            time_since_arrival_tensor = getattr(agent.state, "time_since_arrival", None)
            time_since_arrival = (
                float(time_since_arrival_tensor[0].item())
                if time_since_arrival_tensor is not None
                else dwell_time
            )
            expected_tensor = getattr(agent.state, "current_expected_dwell_time", None)
            expected_dwell = float(expected_tensor[0].item()) if expected_tensor is not None else 0.0
            dwell_min_tensor = getattr(agent.state, "dwell_min_limit", None)
            dwell_min = float(dwell_min_tensor[0].item()) if dwell_min_tensor is not None else 0.0
            dwell_max_tensor = getattr(agent.state, "dwell_max_limit", None)
            dwell_max = float(dwell_max_tensor[0].item()) if dwell_max_tensor is not None else 0.0

            must_hold = is_platform and max(dwell_time, time_since_arrival) + 1e-6 < max(dwell_min, expected_dwell)
            if must_hold:
                continue

            should_release = (
                not is_platform
                or (dwell_max > 0.0 and time_since_arrival >= dwell_max)
                or (expected_dwell > 0.0 and dwell_time >= expected_dwell + 5.0)
            )
            if not should_release:
                continue

            fallback_action = self._select_safe_move_action(agent_idx, current_node_id, claimed_next_nodes)
            if fallback_action <= 0:
                continue

            sanitized_actions[agent_key] = int(fallback_action)
            override_count += 1
            base_reason = sanitized_reasoning.get(agent_key, "RL策略决策")
            sanitized_reasoning[agent_key] = (
                f"{base_reason} | RL安全过滤: 避免"
                f"{'区间停车' if not is_platform else '过度停站'}，改为可执行移动"
            )

        if override_count:
            self.stats["rl_safety_overrides"] = self.stats.get("rl_safety_overrides", 0) + override_count

        return sanitized_actions, sanitized_reasoning

    def _enforce_default_dispatch_rules(
        self,
        actions: Dict[str, int],
        reasoning: Dict[str, str],
        state_info: Dict[str, Any],
    ) -> Tuple[Dict[str, int], Dict[str, str]]:
        """Apply cheap layer-agnostic operating rules learned from MILP event traces."""
        if self.world is None or not actions:
            return actions, reasoning

        dwell_cap = int(getattr(self.config, "default_station_dwell_cap", 60) or 0)
        if dwell_cap <= 0:
            return actions, reasoning

        adjusted_actions = dict(actions)
        adjusted_reasoning = dict(reasoning)
        claimed_next_nodes: Set[int] = set()
        agent_positions = state_info.get("agent_positions", []) if isinstance(state_info, dict) else []

        for agent_key, action in sorted(actions.items(), key=lambda item: int(item[0].split("_")[1])):
            try:
                agent_idx = int(agent_key.split("_")[1])
            except (IndexError, ValueError):
                continue
            if agent_idx >= len(getattr(self.world, "agents", [])):
                continue

            current_node_id = int(self.world.agents[agent_idx].state.position[0].item())
            try:
                action_value = int(action)
            except (TypeError, ValueError):
                action_value = 0
            if action_value > 0:
                next_node_id = self._resolve_action_next_node(current_node_id, action_value)
                if next_node_id is not None and next_node_id != current_node_id:
                    claimed_next_nodes.add(int(next_node_id))
                continue

            if current_node_id not in getattr(self.world, "platform_nodes", {}):
                fallback_action = self._select_safe_move_action(agent_idx, current_node_id, claimed_next_nodes)
                if fallback_action <= 0:
                    continue
                adjusted_actions[agent_key] = int(fallback_action)
                base_reason = adjusted_reasoning.get(agent_key, "默认运行规则")
                adjusted_reasoning[agent_key] = (
                    f"{base_reason} | 默认运行规则: 区间停车释放"
                )
                self.stats["default_dispatch_overrides"] = self.stats.get("default_dispatch_overrides", 0) + 1
                continue

            if 0 <= agent_idx < len(agent_positions):
                agent_info = agent_positions[agent_idx]
                dwell_time = float(agent_info.get("dwell_time", 0.0) or 0.0)
            else:
                agent = self.world.agents[agent_idx]
                dwell_time = float(getattr(agent.state, "dwell_time", torch.tensor([0.0]))[0].item())

            agent = self.world.agents[agent_idx]
            time_since_arrival_tensor = getattr(agent.state, "time_since_arrival", None)
            time_since_arrival = (
                float(time_since_arrival_tensor[0].item())
                if time_since_arrival_tensor is not None
                else dwell_time
            )
            dwell_min_tensor = getattr(agent.state, "dwell_min_limit", None)
            dwell_min = float(dwell_min_tensor[0].item()) if dwell_min_tensor is not None else 0.0
            dwell_max_tensor = getattr(agent.state, "dwell_max_limit", None)
            dwell_max = float(dwell_max_tensor[0].item()) if dwell_max_tensor is not None else 0.0

            effective_cap = float(dwell_cap)
            if dwell_max > 0.0:
                effective_cap = min(effective_cap, dwell_max)
            if max(dwell_time, time_since_arrival) + 1e-6 < max(dwell_min, effective_cap):
                continue

            fallback_action = self._select_safe_move_action(agent_idx, current_node_id, claimed_next_nodes)
            if fallback_action <= 0:
                continue

            adjusted_actions[agent_key] = int(fallback_action)
            base_reason = adjusted_reasoning.get(agent_key, "默认运行规则")
            adjusted_reasoning[agent_key] = (
                f"{base_reason} | 默认发车规则: 停站达到{int(effective_cap)}步后释放"
            )
            self.stats["default_dispatch_overrides"] = self.stats.get("default_dispatch_overrides", 0) + 1

        return adjusted_actions, adjusted_reasoning
    
    def _rl_execution_decide(self, td: TensorDict, state_info: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, str]]:
        """
        RL执行层决策
        
        特点：高频、低延时
        """
        actions = {}
        reasoning = {}
        
        if self.rl_available and self.rl_policy:
            # 使用训练好的RL策略
            try:
                start = time.time()
                with torch.no_grad():
                    # 按 benchmark_marl 的路径直接调用 policy(tensordict) 获取动作。
                    policy_device = getattr(self.rl_policy, "device", torch.device(self.device))
                    policy_output = self.rl_policy(td.to(policy_device).to(torch.float32))
                    logits_tensor = policy_output.get(("agents", "logits"))
                    self._policy_logits = logits_tensor.detach().cpu() if logits_tensor is not None else None
                    if logits_tensor is not None:
                        # Evaluation should be deterministic. ProbabilisticActor's
                        # sampled action can over-sample high-probability stop
                        # actions, which is exactly what the history showed.
                        action_tensor = logits_tensor.detach().argmax(dim=-1).to("cpu")
                    else:
                        action_tensor = policy_output.get(self.action_key)
                    if action_tensor is None:
                        raise KeyError(f"策略输出缺少动作键: {self.action_key}")
                    action_tensor = action_tensor.detach().to("cpu")
                    if action_tensor.ndim == 3 and action_tensor.shape[-1] == 1:
                        action_tensor = action_tensor.squeeze(-1)
                    if action_tensor.ndim == 1:
                        action_tensor = action_tensor.unsqueeze(0)
                    first_batch_actions = action_tensor[0]
                latency = (time.time() - start) * 1000
                
                for i in range(self.n_agents or len(state_info["agent_positions"])):
                    agent_key = f"agent_{i}"
                    if i < first_batch_actions.shape[0]:
                        actions[agent_key] = int(first_batch_actions[i].item())
                    else:
                        actions[agent_key] = 0
                    reasoning[agent_key] = f"RL策略决策 (延迟{latency:.1f}ms)"
                
                return self._sanitize_rl_actions(actions, reasoning)
            except Exception as e:
                logger.warning(f"RL决策失败，使用默认策略: {e}")
        
        # 默认策略：保持当前状态
        for i in range(self.n_agents or len(state_info["agent_positions"])):
            agent_key = f"agent_{i}"
            actions[agent_key] = 0  # 停留
            reasoning[agent_key] = "默认停留策略"
        
        return actions, reasoning
    
    # =========================================================================
    # LLM策略层决策
    # =========================================================================
    
    def _llm_strategy_decide(
        self,
        td: TensorDict,
        state_info: Dict[str, Any],
        trigger_type: str,
    ) -> Tuple[Dict[str, int], Dict[str, str]]:
        """
        LLM策略层决策
        
        流程：
        1. RAG检索相似案例
        2. LLM生成可缓存的高层多步计划
        3. SLM基于当前步高层计划做执行细化
        4. 可选优化器做本地补充
        5. 可解释性分析
        """
        logger.info(f"LLM策略层介入 [触发类型: {trigger_type}]")
        
        # 步骤1: RAG检索
        rag_context = self._rag_retrieve(state_info)

        # 步骤2: LLM生成紧凑的全局指导；必要时再交给选定的OR方法产出初始方案。
        global_guidance = self._llm_high_level_plan(
            state_info=state_info,
            rag_context=rag_context,
            trigger_type=trigger_type,
        )
        guided_solution = self._execute_global_guidance(
            state_info=state_info,
            rag_context=rag_context,
            guidance=global_guidance,
            trigger_type=trigger_type,
        )
        normalized_steps = self._normalize_solution_steps(guided_solution)
        self.cached_global_guidance = copy.deepcopy(global_guidance) if isinstance(global_guidance, dict) else None

        current_step_plan: Dict[str, Any] = copy.deepcopy(normalized_steps[0]) if normalized_steps else {}
        current_step_plan = self._align_plan_step_with_state(state_info, current_step_plan)
        current_step_plan = self._fill_plan_with_rl_baseline(td, state_info, current_step_plan)

        local_focus_agent_ids, local_focus_node_ids = self._derive_local_focus_targets(
            state_info=state_info,
            fallback_solution=current_step_plan,
        )

        recommended_solver = ""
        retrieval_assessment = ""
        if isinstance(global_guidance, dict):
            recommended_solver = str(global_guidance.get("recommended_solver", "")).strip().lower()
            retrieval_assessment = str(global_guidance.get("retrieval_assessment", "")).strip().lower()
        no_solver_requested = recommended_solver in {"", "none"} or retrieval_assessment == "no_good_case"
        if no_solver_requested and not local_focus_agent_ids and not local_focus_node_ids:
            logger.info("策略层未找到可用RAG案例或指定求解器，继续调用MILP生成可缓存计划")

        # 步骤3: 小模型基于高层计划做执行细化，并可重新识别关键列车子集。
        refined_solution = self._small_model_guide(
            state_info,
            self._merge_rag_and_guidance(rag_context, global_guidance),
            base_plan=current_step_plan,
            trigger_type=trigger_type,
            focus_agent_ids=None,
        )

        focus_agent_ids, focus_node_ids = self._resolve_milp_focus_targets(
            state_info=state_info,
            fallback_solution=current_step_plan,
            guidance=self.last_small_model_guidance or rag_context,
        )
        focus_agent_ids = self._cap_milp_focus_agents(
            state_info=state_info,
            focus_agent_ids=focus_agent_ids,
            focus_node_ids=focus_node_ids,
        )

        initial_solution = self._filter_plan_to_agents(refined_solution or current_step_plan, focus_agent_ids)

        # 步骤4: LLM/SLM 只提供引导，最终动作由MILP执行层产出。
        final_solution = self._solve_with_milp_guidance(
            td=td,
            state_info=state_info,
            initial_solution=initial_solution,
            guidance=self.last_small_model_guidance,
            trigger_type=trigger_type,
            focus_agent_ids=focus_agent_ids,
            focus_node_ids=focus_node_ids,
        )

        cached_steps = self._build_cached_plan_from_milp(
            trigger_type=trigger_type,
            focus_agent_ids=focus_agent_ids,
        )
        if cached_steps:
            self.cached_plan_steps = copy.deepcopy(cached_steps)
            self.cached_plan_index = 0
            self.cached_plan_trigger_type = trigger_type
            self.cached_plan_signature = self._capture_plan_signature(state_info)
            self.cached_focus_agent_ids = copy.deepcopy(focus_agent_ids) if focus_agent_ids else None
            self.cached_plan_metrics = self._build_operational_metrics_snapshot(state_info)
            return self._get_cached_action(td, state_info)

        final_solution = self._fill_plan_with_rl_baseline(td, state_info, final_solution)
        actions, reasoning = self._extract_actions_and_reasoning(
            final_solution,
            default_reason=f"LLM策略 [{trigger_type}]",
        )

        # 步骤5: 可解释性分析
        if self.config.enable_explanation and self.explainer:
            self._analyze_decision(state_info, actions, reasoning)
        
        return actions, reasoning
    
    def _rag_retrieve(self, state_info: Dict[str, Any]) -> Dict[str, Any]:
        """RAG检索相似案例"""
        if not self.rag_manager:
            return {"cases": [], "constraints": []}
        
        try:
            self.stats["rag_retrievals"] += 1

            rag_state = dict(state_info)
            rag_state["disruptions"] = (state_info["dispatch_state"].get("events", []) if "dispatch_state" in state_info else [
                {
                    "segment_id": item.get("segment", item.get("segment_id", "")),
                    "type": item.get("type", "UNKNOWN"),
                    "severity": item.get("severity", 1.0),
                }
                for item in state_info.get("disruptions", [])
            ])

            result = self.rag_manager.retrieve(
                current_state=rag_state,
                k=self.config.rag_top_k,
                min_similarity=0.35,
            )
            return {
                "cases": result.get("similar_cases", []),
                "constraints": result.get("constraints", []),
                "warm_start_hints": result.get("warm_start_hints", {}),
                "strategy_recommendations": result.get("strategy_recommendations", []),
                "count": len(result.get("similar_cases", [])),
                "retrieval_diagnostics": result.get("retrieval_diagnostics", {}),
            }
        except Exception as e:
            logger.warning(f"RAG检索失败: {e}")
            return {"cases": [], "constraints": []}
    
    def _small_model_guide(
        self,
        state_info: Dict[str, Any],
        rag_context: Dict[str, Any],
        base_plan: Optional[Dict[str, Any]] = None,
        trigger_type: Optional[str] = None,
        focus_agent_ids: Optional[List[int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """小模型基于高层计划生成执行细化解"""
        normalized_base_plan = self._normalize_plan_step(base_plan)
        self.last_small_model_guidance = None
        if not self.small_model:
            return copy.deepcopy(normalized_base_plan) if normalized_base_plan else None
        
        try:
            self.stats["small_model_uses"] += 1
            
            # 构建提示
            prompt = self._build_small_model_prompt(
                state_info,
                rag_context,
                base_plan=normalized_base_plan,
                trigger_type=trigger_type,
            )
            small_model_state = self._filter_state_info_to_agents(
                state_info,
                focus_agent_ids,
            )
            
            # 获取引导
            guidance = self.small_model.generate_guidance(
                state_description=prompt,
                state_dict=small_model_state,
                max_tokens=512,
            )
            self.last_small_model_guidance = guidance

            # 记录小模型 IO
            self._write_io_log(
                model="small_model",
                input_text=prompt,
                output=guidance.__dict__ if hasattr(guidance, "__dict__") else str(guidance),
                metadata={
                    "backend": (
                        "local" if getattr(self.small_model, "_model", None) is not None
                        else "api" if getattr(self.small_model, "_llm_client", None) is not None
                        else "default"
                    ),
                    "model_name": getattr(self.small_model, "model_name", None),
                    "base_url": getattr(self.small_model, "base_url", None),
                    "model_path": getattr(self.small_model, "model_path", None),
                },
            )

            # 解析为细化后的执行解
            return self._parse_guidance_to_solution(guidance, base_plan=normalized_base_plan)
        except Exception as e:
            logger.warning(f"小模型引导失败: {e}")
            return copy.deepcopy(normalized_base_plan) if normalized_base_plan else None
    
    def _optimize_solution(
        self,
        state_info: Dict[str, Any],
        initial_solution: Optional[Dict[str, Any]],
        trigger_type: str,
    ) -> Optional[Dict[str, Any]]:
        """使用优化工具精细化解"""
        if not self.optimizer:
            return initial_solution
        if initial_solution:
            return initial_solution
        
        try:
            self.stats["optimization_calls"] += 1

            from LLM.optimization_toolkit import OptimizationMethod

            problem_state = {
                "agent_positions": {
                    f"agent_{i}": agent_info.get("node_id", -1)
                    for i, agent_info in enumerate(state_info.get("agent_positions", []))
                },
                "waiting_passengers": state_info.get("waiting_passengers", {}),
                "high_wait_nodes": state_info.get("high_wait_nodes", []),
                "disruptions": state_info.get("disruptions", []),
                "env": self.env,
            }

            result = self.optimizer.solve(
                problem_state=problem_state,
                method=OptimizationMethod.HEURISTIC,
                config={
                    "trigger_type": trigger_type,
                    "time_limit": max(1, int(self.config.local_solver_timeout_ms / 1000)),
                },
            )

            if result and result.success and result.actions:
                return {
                    agent_key: {
                        "action": int(action),
                        "reasoning": result.reasoning or "优化器建议",
                    }
                    for agent_key, action in result.actions.items()
                }
            return initial_solution
        except Exception as e:
            logger.warning(f"优化求解失败: {e}")
            return initial_solution
    
    def _select_optimization_method(
        self,
        state_info: Dict[str, Any],
        trigger_type: str,
    ) -> str:
        """根据场景选择优化方法"""
        # 中断场景：使用Benders分解
        if trigger_type == "DISRUPTION":
            return "benders"
        
        # 高等待：使用排队论
        if trigger_type == "HIGH_WAITING":
            return "queueing"
        
        # 大规模问题：使用列生成
        if self.n_agents and self.n_agents > 15:
            return "column_generation"
        
        # 默认：拉格朗日松弛
        return "lagrangian"

    def _get_execution_action_horizon(self) -> int:
        """Return the number of execution actions to cache and replay."""
        return max(1, min(int(self.config.milp_time_horizon), 20))

    def _get_cached_plan_horizon(self, trigger_type: Optional[str] = None) -> int:
        """Separate long MILP lookahead from short execution replay to avoid stale plans."""
        try:
            max_cached_steps = int(self.config.cached_plan_max_steps)
        except (TypeError, ValueError):
            max_cached_steps = 20

        return max(1, min(self._get_execution_action_horizon(), max_cached_steps))

    def _get_milp_state_horizon(self) -> int:
        """Online MILP state horizon used by the strategic execution layer."""
        return max(2, min(int(self.config.milp_time_horizon), 20))

    def _normalize_global_guidance(
        self,
        guidance: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Force global guidance horizon semantics to match the fixed execution horizon."""
        if not isinstance(guidance, dict):
            return {}

        normalized = copy.deepcopy(guidance)
        solver_config = normalized.get("solver_config")
        if not isinstance(solver_config, dict):
            solver_config = {}
            normalized["solver_config"] = solver_config

        solver_config["time_horizon"] = self._get_execution_action_horizon()
        solver_config["time_limit"] = self.config.milp_time_limit

        return normalized

    def _llm_high_level_plan(
        self,
        state_info: Dict[str, Any],
        rag_context: Dict[str, Any],
        trigger_type: str,
    ) -> Dict[str, Any]:
        """LLM生成紧凑的全局求解指导，而不是逐车多步动作。"""
        if not self.llm_policy:
            return {}
        
        try:
            # 构建增强的状态描述
            enhanced_description = self._build_enhanced_description(
                state_info=state_info,
                rag_context=rag_context,
                optimized_solution=None,
                trigger_type=trigger_type,
            )
            
            # 调用LLM全局指导接口
            result = self.llm_policy._get_llm_global_guidance(enhanced_description)
            result = self._normalize_global_guidance(result)
            result = self._apply_rag_guidance_fallback(
                rag_context=rag_context,
                llm_guidance=result,
                trigger_type=trigger_type,
            )
            self._write_io_log(
                model="llm",
                input_text=enhanced_description,
                output=result,
                metadata={
                    "model_name": self.config.llm_model_path,
                    "base_url": self.config.llm_base_url,
                },
            )
            return result
        except Exception as e:
            logger.warning(f"LLM全局指导失败: {e}")
            return {}

    def _map_case_method_to_solver(self, method: Any, trigger_type: str) -> str:
        method_name = str(method or "").strip().lower()
        if method_name == "milp":
            return "milp"
        if method_name == "hybrid":
            return "milp" if trigger_type == "DISRUPTION" else "lagrangian"
        if method_name in {"lagrangian", "column_generation", "benders", "queueing"}:
            return method_name
        if method_name == "heuristic":
            if trigger_type == "HIGH_WAITING":
                return "queueing"
            return "lagrangian"
        if trigger_type == "DISRUPTION":
            return "benders"
        if trigger_type == "HIGH_WAITING":
            return "queueing"
        return "lagrangian"

    def _apply_rag_guidance_fallback(
        self,
        rag_context: Dict[str, Any],
        llm_guidance: Optional[Dict[str, Any]],
        trigger_type: str,
    ) -> Dict[str, Any]:
        """当 RAG 已命中强匹配案例时，不接受 no_good_case/none 的保守退化。"""
        normalized = self._normalize_global_guidance(llm_guidance)
        cases = rag_context.get("cases", []) if isinstance(rag_context, dict) else []
        if not cases:
            return normalized

        top_case = cases[0]
        exact_match = bool(top_case.get("exact_disruption_match", False))
        match_score = float(top_case.get("scenario_match_score", 0.0) or 0.0)
        similarity = float(top_case.get("similarity", 0.0) or 0.0)
        strong_match = exact_match or match_score >= 4.0 or (match_score >= 3.0 and similarity >= 0.85)
        if not strong_match:
            return normalized

        retrieval_assessment = str(normalized.get("retrieval_assessment", "")).strip().lower()
        recommended_solver = str(normalized.get("recommended_solver", "none")).strip().lower()
        if retrieval_assessment != "no_good_case" and recommended_solver != "none":
            return normalized

        action_plan = top_case.get("action_plan", {}) if isinstance(top_case.get("action_plan"), dict) else {}
        selected_plan = action_plan.get("selected_plan") if isinstance(action_plan.get("selected_plan"), dict) else {}
        metadata = top_case.get("metadata", {}) if isinstance(top_case.get("metadata"), dict) else {}
        warm_start_hints = rag_context.get("warm_start_hints", {}) if isinstance(rag_context.get("warm_start_hints"), dict) else {}

        focus_agents: List[int] = []
        focus_nodes: List[int] = []

        def _append_unique(target: List[int], value: Any) -> None:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return
            if parsed not in target:
                target.append(parsed)

        for value in warm_start_hints.get("focus_agents", []) or []:
            _append_unique(focus_agents, value)
        for value in warm_start_hints.get("focus_nodes", []) or []:
            _append_unique(focus_nodes, value)
        for value in selected_plan.get("focus_agents", []) or []:
            _append_unique(focus_agents, value)
        for value in selected_plan.get("focus_nodes", []) or []:
            _append_unique(focus_nodes, value)

        case_method = (
            selected_plan.get("method")
            or metadata.get("selected_method")
            or recommended_solver
        )

        fallback_guidance = copy.deepcopy(normalized)
        fallback_guidance["retrieval_assessment"] = "useful_case"
        fallback_guidance["recommended_solver"] = self._map_case_method_to_solver(case_method, trigger_type)
        solver_config = fallback_guidance.get("solver_config") if isinstance(fallback_guidance.get("solver_config"), dict) else {}
        solver_config = dict(solver_config)
        if focus_agents:
            solver_config["focus_agents"] = focus_agents[:6]
        if focus_nodes:
            solver_config["focus_nodes"] = focus_nodes[:6]
        fallback_guidance["solver_config"] = solver_config

        summary = str(top_case.get("summary", "")).strip()
        if summary:
            fallback_guidance["strategy_summary"] = summary[:20]
        return self._normalize_global_guidance(fallback_guidance)

    def _llm_final_decide(
        self,
        state_info: Dict[str, Any],
        rag_context: Dict[str, Any],
        optimized_solution: Optional[Dict[str, Any]],
        trigger_type: str,
    ) -> Dict[str, Any]:
        """兼容旧接口，当前直接复用高层规划。"""
        return self._llm_high_level_plan(
            state_info=state_info,
            rag_context=rag_context,
            trigger_type=trigger_type,
        ) or (optimized_solution or {})

    def _merge_rag_and_guidance(
        self,
        rag_context: Dict[str, Any],
        global_guidance: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged = dict(rag_context or {})
        if isinstance(global_guidance, dict) and not global_guidance.get("error"):
            merged["global_guidance"] = copy.deepcopy(global_guidance)
        return merged

    def _execute_global_guidance(
        self,
        state_info: Dict[str, Any],
        rag_context: Dict[str, Any],
        guidance: Optional[Dict[str, Any]],
        trigger_type: str,
    ) -> Dict[str, Any]:
        """按大模型给出的全局指导选择并配置 OR 方法，生成可选初始方案。"""
        if not guidance or not isinstance(guidance, dict) or guidance.get("error"):
            return {}
        if not self.llm_policy:
            return {}

        solver = str(guidance.get("recommended_solver", "none")).strip().lower()
        if solver not in {"milp", "lagrangian", "column_generation", "benders", "queueing"}:
            return {}

        solver_config = guidance.get("solver_config") if isinstance(guidance.get("solver_config"), dict) else {}
        solver_config = dict(solver_config)
        solver_config["time_horizon"] = self._get_execution_action_horizon()
        solver_config["time_limit"] = self.config.milp_time_limit

        try:
            enhanced_description = self._build_enhanced_description(
                state_info=state_info,
                rag_context=rag_context,
                optimized_solution=None,
                trigger_type=trigger_type,
            )
            return self.llm_policy._get_global_instructions(
                enhanced_description,
                method=solver,
                config=json.dumps(solver_config, ensure_ascii=False),
            )
        except Exception as exc:
            logger.warning(f"执行全局指导指定的 {solver} 方案失败: {exc}")
            return {}
    
    def _analyze_decision(
        self,
        state_info: Dict[str, Any],
        actions: Dict[str, int],
        reasoning: Dict[str, str],
    ):
        """可解释性分析"""
        if not self.explainer:
            return
        
        try:
            self.explainer.record_state(
                step=self.current_step,
                sim_time=float(self.current_step),
                agent_positions={
                    f"agent_{i}": agent_info.get("node_id", -1)
                    for i, agent_info in enumerate(state_info.get("agent_positions", []))
                },
                agent_passengers={
                    f"agent_{i}": int(value)
                    for i, value in enumerate(state_info.get("agent_passengers", []))
                },
                agent_dwell_times={
                    f"agent_{i}": int(value)
                    for i, value in enumerate(state_info.get("agent_dwell_times", []))
                },
                waiting_passengers=state_info.get("waiting_passengers", {}),
                disruptions=state_info.get("disruptions", []),
            )

            for agent_key, action in actions.items():
                try:
                    agent_idx = int(agent_key.split("_")[1])
                except (IndexError, ValueError):
                    agent_idx = -1
                target_node = -1
                if 0 <= agent_idx < len(state_info.get("agent_positions", [])):
                    target_node = state_info["agent_positions"][agent_idx].get("node_id", -1)
                self.explainer.analyze_and_record_decision(
                    step=self.current_step,
                    agent_name=agent_key,
                    action=int(action),
                    target_node=target_node,
                    source="hierarchical_llm_policy",
                    reasoning=reasoning.get(agent_key, ""),
                    context={"detail_level": self.config.explanation_detail_level},
                    decision_time_ms=0.0,
                )
        except Exception as e:
            logger.warning(f"可解释性分析失败: {e}")
    
    # =========================================================================
    # 辅助方法
    # =========================================================================

    def _normalize_plan_step(self, plan: Any) -> Dict[str, Dict[str, Any]]:
        """将嵌套计划结构归一化为单步 agent->decision 字典。"""
        normalized: Dict[str, Dict[str, Any]] = {}

        if isinstance(plan, dict):
            for key, value in plan.items():
                if str(key).startswith("agent_"):
                    if isinstance(value, dict):
                        normalized[str(key)] = copy.deepcopy(value)
                    else:
                        normalized[str(key)] = {"action": int(value)}

            nested_steps = plan.get("steps")
            if nested_steps is not None:
                nested_plan = self._normalize_plan_step(nested_steps)
                if nested_plan:
                    normalized.update(nested_plan)
            return normalized

        if isinstance(plan, list):
            for item in plan:
                nested_plan = self._normalize_plan_step(item)
                if nested_plan:
                    normalized.update(nested_plan)
            return normalized

        return normalized

    def _normalize_solution_steps(self, solution: Any) -> List[Dict[str, Dict[str, Any]]]:
        """将模型返回的多种 steps 结构统一为标准多步计划。"""
        if not solution:
            return []

        raw_steps = solution.get("steps") if isinstance(solution, dict) and "steps" in solution else solution

        if isinstance(raw_steps, list):
            normalized_steps = []
            for item in raw_steps:
                normalized_step = self._normalize_plan_step(item)
                if normalized_step:
                    normalized_steps.append(normalized_step)

            if not normalized_steps:
                return []

            # 某些输出会把单步拆成多个 {agent_i: ...} 片段，这里合并回单步。
            if len(normalized_steps) > 1 and all(len(step) <= 2 for step in normalized_steps):
                merged_step: Dict[str, Dict[str, Any]] = {}
                for step in normalized_steps:
                    merged_step.update(step)
                normalized_steps = [merged_step]

            target_horizon = max(int(self.config.milp_time_horizon), 1)
            if len(normalized_steps) > target_horizon:
                return normalized_steps[:target_horizon]
            if len(normalized_steps) < target_horizon:
                template_step = copy.deepcopy(normalized_steps[-1])
                while len(normalized_steps) < target_horizon:
                    step_no = len(normalized_steps) + 1
                    extended_step = copy.deepcopy(template_step)
                    for decision in extended_step.values():
                        if not isinstance(decision, dict):
                            continue
                        reason = str(decision.get("reasoning", "")).strip()
                        suffix = f"高层计划自动延展到第{step_no}步"
                        decision["reasoning"] = f"{reason} | {suffix}" if reason else suffix
                    normalized_steps.append(extended_step)

            return normalized_steps

        normalized_step = self._normalize_plan_step(raw_steps)
        if not normalized_step:
            return []
        return self._normalize_solution_steps([normalized_step])

    def _resolve_action_next_node(self, current_node_id: int, action: int) -> Optional[int]:
        """按环境动作语义解析动作对应的下一节点。0 表示停留，正数表示第 N 个方向。"""
        if current_node_id is None or current_node_id < 0:
            return None

        if action <= 0:
            return int(current_node_id)

        if self.world is None:
            return None

        node_obj = self.world.get_node(current_node_id) if hasattr(self.world, "get_node") else None
        route_map = node_obj.getnext() if node_obj is not None and hasattr(node_obj, "getnext") else None
        route_map = route_map or {}
        available = {key: value for key, value in route_map.items() if value is not None}

        if available:
            route_keys = sorted(available.keys(), key=lambda x: str(x))
            direction_idx = action - 1
            if direction_idx < 0 or direction_idx >= len(route_keys):
                direction_idx = 0
            chosen_next = available.get(route_keys[direction_idx])
            return int(chosen_next) if chosen_next is not None else int(current_node_id)

        try:
            candidate_nodes = [node.id for node in self.world.get_next_nodes(current_node_id)]
        except Exception:
            candidate_nodes = []

        if not candidate_nodes:
            return int(current_node_id)

        direction_idx = action - 1
        if direction_idx < 0 or direction_idx >= len(candidate_nodes):
            direction_idx = 0
        return int(candidate_nodes[direction_idx])

    def _resolve_next_node_action(self, current_node_id: int, next_node_id: Optional[int]) -> int:
        """按环境真实拓扑，将 next_node 反解为合法动作编号。"""
        if current_node_id is None or current_node_id < 0:
            return 0

        try:
            parsed_next_node = int(next_node_id)
        except (TypeError, ValueError):
            return 0

        if parsed_next_node == int(current_node_id):
            return 0

        if self.world is None:
            return 0

        node_obj = self.world.get_node(current_node_id) if hasattr(self.world, "get_node") else None
        route_map = node_obj.getnext() if node_obj is not None and hasattr(node_obj, "getnext") else None
        route_map = route_map or {}
        available = {key: value for key, value in route_map.items() if value is not None}

        if available:
            route_keys = sorted(available.keys(), key=lambda x: str(x))
            for direction_idx, route_key in enumerate(route_keys):
                try:
                    candidate_next = int(available[route_key])
                except (TypeError, ValueError):
                    continue
                if candidate_next == parsed_next_node:
                    return direction_idx + 1
            return 0

        try:
            candidate_nodes = [int(node.id) for node in self.world.get_next_nodes(current_node_id)]
        except Exception:
            candidate_nodes = []

        for direction_idx, candidate_next in enumerate(candidate_nodes):
            if candidate_next == parsed_next_node:
                return direction_idx + 1

        return 0

    def _find_best_action_to_target(self, current_node_id: int, target_node_id: Optional[int]) -> int:
        """Match MILP ground-truth replay: BFS over world._next_nodes_id_map and return 1-based next-hop."""
        if current_node_id is None or current_node_id < 0:
            return 0

        try:
            parsed_target = int(target_node_id)
        except (TypeError, ValueError):
            return 0

        if parsed_target == int(current_node_id):
            return 0

        if self.world is None:
            return 0

        adjacency = getattr(self.world, "_next_nodes_id_map", None)
        if isinstance(adjacency, dict):
            neighbors = [int(node_id) for node_id in adjacency.get(int(current_node_id), [])]
        else:
            try:
                neighbors = [int(node.id) for node in self.world.get_next_nodes(current_node_id)]
            except Exception:
                neighbors = []

        if not neighbors:
            return 1

        best_action = 0
        best_dist = float("inf")
        max_depth = 200

        for direction_idx, next_node_id in enumerate(neighbors, start=1):
            if next_node_id == parsed_target:
                return direction_idx

            visited = {current_node_id, next_node_id}
            queue = deque([(next_node_id, 1)])
            found_dist = float("inf")

            while queue and len(visited) < max_depth:
                node_id, dist = queue.popleft()
                if dist >= best_dist:
                    break
                if isinstance(adjacency, dict):
                    successors = [int(successor) for successor in adjacency.get(node_id, [])]
                else:
                    try:
                        successors = [int(node.id) for node in self.world.get_next_nodes(node_id)]
                    except Exception:
                        successors = []
                for successor_id in successors:
                    if successor_id == parsed_target:
                        found_dist = dist + 1
                        break
                    if successor_id not in visited:
                        visited.add(successor_id)
                        queue.append((successor_id, dist + 1))
                if found_dist < float("inf"):
                    break

            if found_dist < best_dist:
                best_dist = found_dist
                best_action = direction_idx

        return best_action + 1 if best_action == 0 else best_action

    def _nearest_forward_platform(self, start_node_id: Optional[int], max_depth: int = 240) -> Optional[int]:
        """Map an internal route/segment node to the next platform along the directed topology."""
        if self.world is None:
            return None
        try:
            start_id = int(start_node_id)
        except (TypeError, ValueError):
            return None
        if start_id < 0:
            return None

        platform_nodes = getattr(self.world, "platform_nodes", {}) or {}
        if start_id in platform_nodes:
            return start_id

        visited = {start_id}
        queue = deque([(start_id, 0)])
        while queue:
            node_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            try:
                successors = [int(node.id) for node in self.world.get_next_nodes(node_id)]
            except Exception:
                successors = []
            for successor_id in successors:
                if successor_id in visited:
                    continue
                if successor_id in platform_nodes:
                    return successor_id
                visited.add(successor_id)
                queue.append((successor_id, depth + 1))
        return None

    def _node_corridor_key(self, node_id: Any) -> str:
        """Group topology16-style platforms by direction and coarse station zone."""
        try:
            parsed_node = int(node_id)
        except (TypeError, ValueError):
            return "unknown"
        if self.world is None:
            return "unknown"

        node_obj = (getattr(self.world, "platform_nodes", {}) or {}).get(parsed_node)
        if node_obj is None and hasattr(self.world, "get_node"):
            try:
                node_obj = self.world.get_node(parsed_node)
            except Exception:
                node_obj = None

        label = ""
        for attr in ("name", "station", "station_name"):
            value = getattr(node_obj, attr, None) if node_obj is not None else None
            if value:
                label = str(value)
                break
        if not label:
            label = str(parsed_node)

        match = re.search(r"s([12])(\d+)$", label)
        if match:
            direction = match.group(1)
            station_idx = int(match.group(2))
            zone = min(3, max(0, (station_idx - 1) // 10))
            return f"dir{direction}_zone{zone}"
        if label.startswith("d"):
            return "terminal"
        return "unknown"

    def _align_plan_step_with_state(
        self,
        state_info: Dict[str, Any],
        plan: Optional[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Replay MILP/cache trajectories with the same controller as MILP ground truth."""
        normalized_plan = self._normalize_plan_step(plan)
        if not normalized_plan or self.world is None:
            return normalized_plan

        agent_positions = state_info.get("agent_positions", [])
        aligned: Dict[str, Dict[str, Any]] = {}

        def _parse_optional_node(value: Any) -> Optional[int]:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            return parsed if parsed >= 0 else None

        for i in range(self.n_agents or len(agent_positions)):
            agent_key = f"agent_{i}"
            if agent_key not in normalized_plan:
                continue
            decision = copy.deepcopy(normalized_plan.get(agent_key, {}))
            current_pos = -1
            at_platform = False
            time_since_arrival = 0.0
            dwell_min = 0.0
            dwell_max = 0.0

            if 0 <= i < len(agent_positions):
                current_pos = int(agent_positions[i].get("node_id", agent_positions[i].get("position", -1)))
                at_platform = bool(agent_positions[i].get("at_platform", False))
            if 0 <= i < len(self.world.agents):
                agent = self.world.agents[i]
                current_pos = int(agent.state.position[0].item())
                at_platform = current_pos in self.world.platform_nodes
                time_since_arrival_tensor = getattr(agent.state, "time_since_arrival", None)
                time_since_arrival = (
                    float(time_since_arrival_tensor[0].item())
                    if time_since_arrival_tensor is not None
                    else float(getattr(agent.state, "dwell_time", torch.tensor([0]))[0].item())
                )
                dwell_min_tensor = getattr(agent.state, "dwell_min_limit", None)
                dwell_min = float(dwell_min_tensor[0].item()) if dwell_min_tensor is not None else 0.0
                dwell_max_tensor = getattr(agent.state, "dwell_max_limit", None)
                dwell_max = float(dwell_max_tensor[0].item()) if dwell_max_tensor is not None else 0.0

            try:
                raw_action = int(decision.get("action", 0))
            except (TypeError, ValueError):
                raw_action = 0

            reasoning = str(decision.get("reasoning", "")).strip()
            correction_note: Optional[str] = None

            requested_next_node = _parse_optional_node(decision.get("next_node"))
            requested_target_node = _parse_optional_node(decision.get("target_node"))
            milp_target = requested_target_node if requested_target_node is not None else requested_next_node
            if bool(self.config.cached_target_tracking):
                platform_target = self._nearest_forward_platform(milp_target)
                if platform_target is not None:
                    milp_target = platform_target

            if raw_action > 0 and milp_target is not None and milp_target > 0:
                self.cached_agent_targets[i] = int(milp_target)

            target_node = self.cached_agent_targets.get(i, -1)

            if at_platform:
                if dwell_min > 0.0 and time_since_arrival < dwell_min:
                    aligned_action = 0
                    correction_note = "MILP轨迹回放: 未满足最小停站时间"
                elif dwell_max > 0.0 and time_since_arrival >= dwell_max:
                    if target_node and target_node > 0:
                        aligned_action = self._find_best_action_to_target(current_pos, target_node)
                    else:
                        aligned_action = min(raw_action, 3) if raw_action > 0 else 1
                    correction_note = "MILP轨迹回放: 达到最大停站时间，按目标发车"
                elif raw_action == 0:
                    aligned_action = 0
                    correction_note = "MILP轨迹回放: MILP指示停留"
                else:
                    if target_node and target_node > 0:
                        aligned_action = self._find_best_action_to_target(current_pos, target_node)
                    else:
                        aligned_action = min(raw_action, 3)
                    correction_note = "MILP轨迹回放: 满足停站约束，按目标移动"
            else:
                if target_node and target_node > 0:
                    aligned_action = self._find_best_action_to_target(current_pos, target_node)
                    correction_note = "MILP轨迹回放: 非站台按目标持续移动"
                else:
                    aligned_action = 1
                    correction_note = "MILP轨迹回放: 非站台默认移动"

            aligned_next_node = self._resolve_action_next_node(current_pos, aligned_action)
            if aligned_next_node is None:
                aligned_next_node = current_pos

            if not reasoning:
                reasoning = "高层计划对齐: 保持当前位置" if aligned_action == 0 else "高层计划对齐: 保留移动"
            if correction_note:
                reasoning = f"{reasoning} | {correction_note}"

            aligned[agent_key] = {
                "action": int(aligned_action),
                "next_node": int(aligned_next_node),
                "target_node": int(target_node) if target_node and target_node > 0 else None,
                "reasoning": reasoning,
            }

        return aligned

    def _fill_plan_with_rl_baseline(
        self,
        td: TensorDict,
        state_info: Dict[str, Any],
        plan: Optional[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """对缺失 agent 的计划用 RL 基线补齐，避免未覆盖列车退化为默认停留。"""
        normalized_plan = self._normalize_plan_step(plan)
        rl_actions, rl_reasoning = self._rl_execution_decide(td, state_info)

        for agent_key, action in rl_actions.items():
            if agent_key in normalized_plan:
                continue
            try:
                agent_idx = int(agent_key.split("_")[1])
            except (IndexError, TypeError, ValueError):
                agent_idx = -1

            current_pos = -1
            agent_positions = state_info.get("agent_positions", []) if isinstance(state_info, dict) else []
            if 0 <= agent_idx < len(agent_positions):
                current_pos = int(agent_positions[agent_idx].get("node_id", agent_positions[agent_idx].get("position", -1)))
            elif self.world is not None and 0 <= agent_idx < len(self.world.agents):
                current_pos = int(self.world.agents[agent_idx].state.position[0].item())

            action_value = int(action)
            normalized_plan[agent_key] = {
                "action": action_value,
                "next_node": self._resolve_action_next_node(current_pos, action_value),
                "reasoning": rl_reasoning.get(agent_key, "RL基线补齐"),
            }

        return normalized_plan

    def _filter_plan_to_agents(
        self,
        plan: Optional[Dict[str, Any]],
        focus_agent_ids: Optional[List[int]],
    ) -> Dict[str, Dict[str, Any]]:
        """Keep only the requested focus agents in a normalized plan."""
        normalized_plan = self._normalize_plan_step(plan)
        if not normalized_plan or not focus_agent_ids:
            return normalized_plan

        focus_keys = {f"agent_{int(agent_idx)}" for agent_idx in focus_agent_ids}
        return {
            agent_key: copy.deepcopy(decision)
            for agent_key, decision in normalized_plan.items()
            if agent_key in focus_keys
        }

    def _filter_state_info_to_agents(
        self,
        state_info: Dict[str, Any],
        focus_agent_ids: Optional[List[int]],
    ) -> Dict[str, Any]:
        """Keep only focus-agent slices in the structured state passed to the small model."""
        filtered_state = copy.deepcopy(state_info)
        if not focus_agent_ids:
            return filtered_state

        selected_agent_ids = [int(agent_idx) for agent_idx in focus_agent_ids]
        agent_positions = state_info.get("agent_positions", []) if isinstance(state_info, dict) else []
        agent_dwell_times = state_info.get("agent_dwell_times", []) if isinstance(state_info, dict) else []
        agent_passengers = state_info.get("agent_passengers", []) if isinstance(state_info, dict) else []

        filtered_state["agent_positions"] = [
            copy.deepcopy(agent_positions[agent_idx])
            for agent_idx in selected_agent_ids
            if 0 <= agent_idx < len(agent_positions)
        ]
        filtered_state["agent_dwell_times"] = [
            copy.deepcopy(agent_dwell_times[agent_idx])
            for agent_idx in selected_agent_ids
            if 0 <= agent_idx < len(agent_dwell_times)
        ]
        filtered_state["agent_passengers"] = [
            copy.deepcopy(agent_passengers[agent_idx])
            for agent_idx in selected_agent_ids
            if 0 <= agent_idx < len(agent_passengers)
        ]
        filtered_state["n_agents"] = len(filtered_state["agent_positions"])
        filtered_state["focus_agent_ids"] = selected_agent_ids
        return filtered_state

    def _build_soft_focus_node_weights(
        self,
        state_info: Dict[str, Any],
        focus_node_ids: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        """Build weighted hotspot hints for focus ranking and MILP soft guidance."""
        weights: Dict[int, float] = {}

        def _add(node_id: Any, weight: Any) -> None:
            try:
                parsed_node = int(node_id)
                parsed_weight = float(weight)
            except (TypeError, ValueError):
                return
            if parsed_node < 0:
                return
            weights[parsed_node] = max(weights.get(parsed_node, 0.0), parsed_weight)

        for item in state_info.get("top_wait_nodes", []) or []:
            if isinstance(item, dict):
                _add(item.get("node_id", item.get("node_idx")), item.get("waiting", 0.0))
        for item in state_info.get("high_wait_nodes", []) or []:
            if isinstance(item, dict):
                _add(item.get("node_id", item.get("node_idx")), item.get("waiting", 0.0))

        waiting_by_node = state_info.get("waiting_passengers", {}) if isinstance(state_info, dict) else {}
        for node_id in focus_node_ids or []:
            try:
                waiting = float(waiting_by_node.get(int(node_id), 0.0))
            except (TypeError, ValueError, AttributeError):
                waiting = 0.0
            _add(node_id, max(waiting, float(self.config.waiting_threshold) * 0.5, 1.0))

        if isinstance(waiting_by_node, dict):
            for node_id, waiting in waiting_by_node.items():
                try:
                    waiting_value = float(waiting)
                except (TypeError, ValueError):
                    continue
                if waiting_value <= 0.0:
                    continue
                _add(node_id, waiting_value * 0.75)

        ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        selected: List[Tuple[int, float]] = []
        selected_nodes: Set[int] = set()
        group_counts: Dict[str, int] = defaultdict(int)
        per_group_first_pass = 3

        for node_id, weight in ordered:
            group_key = self._node_corridor_key(node_id)
            if group_counts[group_key] >= per_group_first_pass:
                continue
            selected.append((node_id, weight))
            selected_nodes.add(node_id)
            group_counts[group_key] += 1
            if len(selected) >= 12:
                break

        if len(selected) < 12:
            for node_id, weight in ordered:
                if node_id in selected_nodes:
                    continue
                selected.append((node_id, weight))
                selected_nodes.add(node_id)
                if len(selected) >= 12:
                    break

        return dict(selected)

    def _forward_distance_to_targets(
        self,
        start_node_id: int,
        target_weights: Dict[int, float],
        max_depth: Optional[int] = None,
    ) -> Tuple[float, Optional[int], float]:
        """Return distance, best target, and weighted reachability score."""
        if not target_weights:
            return float("inf"), None, 0.0
        if start_node_id in target_weights:
            return 0.0, int(start_node_id), float(target_weights[start_node_id])
        if self.world is None:
            return float("inf"), None, 0.0

        depth_limit = int(max_depth if max_depth is not None else self._get_milp_state_horizon())
        visited = {int(start_node_id)}
        queue = deque([(int(start_node_id), 0)])
        best_distance = float("inf")
        best_target: Optional[int] = None
        best_score = 0.0

        while queue:
            node_id, distance = queue.popleft()
            if distance >= depth_limit:
                continue
            try:
                successors = [int(node.id) for node in self.world.get_next_nodes(node_id)]
            except Exception:
                successors = []
            for successor_id in successors:
                if successor_id in visited:
                    continue
                next_distance = distance + 1
                if successor_id in target_weights:
                    weight = float(target_weights[successor_id])
                    score = weight / max(1.0, float(next_distance))
                    if score > best_score:
                        best_score = score
                        best_distance = float(next_distance)
                        best_target = successor_id
                visited.add(successor_id)
                if next_distance < depth_limit:
                    queue.append((successor_id, next_distance))

        return best_distance, best_target, best_score

    def _rank_focus_agents_by_service_potential(
        self,
        state_info: Dict[str, Any],
        focus_agent_ids: List[int],
        focus_node_ids: Optional[List[int]] = None,
        max_focus: Optional[int] = None,
    ) -> List[int]:
        """Rank critical trains by 20-step service potential and corridor diversity."""
        if not focus_agent_ids:
            return []

        agent_positions = state_info.get("agent_positions", []) if isinstance(state_info, dict) else []
        target_weights = self._build_soft_focus_node_weights(state_info, focus_node_ids)
        horizon = max(2, self._get_milp_state_horizon())
        max_focus = max(1, int(max_focus if max_focus is not None else self.config.max_milp_focus_agents))

        unique_agents: List[int] = []
        seen: Set[int] = set()
        for value in focus_agent_ids:
            try:
                agent_idx = int(value)
            except (TypeError, ValueError):
                continue
            if agent_idx in seen:
                continue
            seen.add(agent_idx)
            unique_agents.append(agent_idx)

        scored: List[Tuple[float, Optional[int], int, float]] = []
        for order, agent_idx in enumerate(unique_agents):
            if 0 <= agent_idx < len(agent_positions):
                agent_info = agent_positions[agent_idx]
                current_node_id = int(agent_info.get("node_id", agent_info.get("position", -1)))
                dwell_time = float(agent_info.get("dwell_time", 0.0) or 0.0)
                expected_dwell = float(agent_info.get("expected_dwell", 0.0) or 0.0)
                passengers = float(agent_info.get("passengers", 0.0) or 0.0)
                at_platform = bool(agent_info.get("at_platform", False))
            elif self.world is not None and 0 <= agent_idx < len(self.world.agents):
                agent = self.world.agents[agent_idx]
                current_node_id = int(agent.state.position[0].item())
                dwell_time = float(getattr(agent.state, "dwell_time", torch.tensor([0.0]))[0].item())
                expected_tensor = getattr(agent.state, "current_expected_dwell_time", None)
                expected_dwell = float(expected_tensor[0].item()) if expected_tensor is not None else 0.0
                passengers = float(getattr(agent, "passenger_count", torch.tensor([0.0]))[0].item())
                at_platform = current_node_id in getattr(self.world, "platform_nodes", {})
            else:
                continue

            distance, best_target, reach_score = self._forward_distance_to_targets(
                current_node_id,
                target_weights,
                max_depth=horizon,
            )
            score = 0.0
            if distance < float("inf"):
                score += reach_score * 3.0
                score += max(0.0, float(horizon + 1) - distance) * 0.5
            if current_node_id in target_weights:
                score += target_weights[current_node_id] * 4.0
            if at_platform and expected_dwell > 0.0 and dwell_time > expected_dwell:
                score += min(dwell_time - expected_dwell, 300.0) * 2.0
            score += min(passengers, 2520.0) * 0.01

            for disruption in state_info.get("disruptions", []) or []:
                segment = str(disruption.get("segment", ""))
                match = re.match(r"^(\d+)-(\d+)$", segment)
                if match and current_node_id in {int(match.group(1)), int(match.group(2))}:
                    score += 200.0

            scored.append((score, best_target, agent_idx, float(order)))

        if not scored:
            return unique_agents[:max_focus]

        scored.sort(key=lambda item: (-item[0], item[3]))
        positive = [item for item in scored if item[0] >= 1.0]
        if not positive:
            return [item[2] for item in scored[: min(max_focus, max(4, min(8, len(scored))))]]

        selected: List[int] = []
        target_counts: Dict[str, int] = defaultdict(int)
        reserve = max(4, min(8, len(positive), max_focus))

        for score, target, agent_idx, _order in positive:
            if len(selected) >= max_focus:
                break
            target_key = self._node_corridor_key(target) if target is not None else "unknown"
            if len(selected) >= reserve and target_counts[target_key] >= 3:
                continue
            selected.append(agent_idx)
            target_counts[target_key] += 1

        if len(selected) < reserve:
            for _score, _target, agent_idx, _order in positive:
                if agent_idx not in selected:
                    selected.append(agent_idx)
                if len(selected) >= reserve:
                    break

        return selected[:max_focus]

    def _derive_focus_agents_from_nodes(
        self,
        state_info: Dict[str, Any],
        fallback_solution: Optional[Dict[str, Any]],
        focus_node_ids: List[int],
    ) -> List[int]:
        """Map focus nodes to nearby critical agents."""
        if not focus_node_ids:
            return []

        normalized_solution = self._normalize_plan_step(fallback_solution)
        agent_positions = state_info.get("agent_positions", []) if isinstance(state_info, dict) else []
        focus_node_set = {int(node_id) for node_id in focus_node_ids}
        derived_agent_ids: List[int] = []
        agent_current_nodes: Dict[int, int] = {}

        for agent_idx in range(self.n_agents or len(agent_positions)):
            current_node_id = None
            if 0 <= agent_idx < len(agent_positions):
                current_node_id = int(agent_positions[agent_idx].get("node_id", agent_positions[agent_idx].get("position", -1)))
            elif self.world is not None and 0 <= agent_idx < len(self.world.agents):
                current_node_id = int(self.world.agents[agent_idx].state.position[0].item())
            if current_node_id is not None:
                agent_current_nodes[agent_idx] = current_node_id

            next_node_id = None
            decision = normalized_solution.get(f"agent_{agent_idx}", {})
            if isinstance(decision, dict):
                next_node_id = decision.get("next_node")
                if next_node_id is None:
                    try:
                        action = int(decision.get("action", 0))
                    except (TypeError, ValueError):
                        action = 0
                    if current_node_id is not None and action > 0:
                        next_node_id = self._resolve_action_next_node(current_node_id, action)

            if current_node_id in focus_node_ids or next_node_id in focus_node_ids:
                derived_agent_ids.append(agent_idx)

        if derived_agent_ids or self.world is None:
            return self._rank_focus_agents_by_service_potential(
                state_info,
                derived_agent_ids,
                focus_node_ids,
                max_focus=int(self.config.max_milp_focus_agents),
            )

        def _distance_to_focus(start_node_id: int, max_depth: int = 240) -> float:
            if start_node_id in focus_node_set:
                return 0.0
            visited = {start_node_id}
            queue = deque([(start_node_id, 0)])
            while queue:
                node_id, distance = queue.popleft()
                if distance >= max_depth:
                    continue
                try:
                    successors = [int(node.id) for node in self.world.get_next_nodes(node_id)]
                except Exception:
                    successors = []
                for successor_id in successors:
                    if successor_id in visited:
                        continue
                    if successor_id in focus_node_set:
                        return float(distance + 1)
                    visited.add(successor_id)
                    queue.append((successor_id, distance + 1))
            return float("inf")

        nearest_agents = [
            (_distance_to_focus(current_node_id), agent_idx)
            for agent_idx, current_node_id in agent_current_nodes.items()
        ]
        nearest_agents = [item for item in nearest_agents if item[0] < float("inf")]
        nearest_agents.sort(key=lambda item: (item[0], item[1]))
        max_focus_agents = min(
            self.n_agents or len(agent_positions),
            max(4, min(int(self.config.max_milp_focus_agents), len(focus_node_set) * 3)),
        )
        derived_agent_ids.extend(agent_idx for _, agent_idx in nearest_agents[:max_focus_agents])
        return self._rank_focus_agents_by_service_potential(
            state_info,
            derived_agent_ids,
            focus_node_ids,
            max_focus=max_focus_agents,
        )

    def _cap_milp_focus_agents(
        self,
        state_info: Dict[str, Any],
        focus_agent_ids: Optional[List[int]],
        focus_node_ids: Optional[List[int]] = None,
    ) -> Optional[List[int]]:
        """Keep the online MILP subproblem small enough for real-time use."""
        if not focus_agent_ids:
            return focus_agent_ids

        max_focus = max(1, int(self.config.max_milp_focus_agents))
        unique_agents: List[int] = []
        seen: Set[int] = set()
        for value in focus_agent_ids:
            try:
                agent_idx = int(value)
            except (TypeError, ValueError):
                continue
            if agent_idx in seen:
                continue
            seen.add(agent_idx)
            unique_agents.append(agent_idx)

        priority: Dict[int, float] = {agent_idx: 0.0 for agent_idx in unique_agents}
        for item in state_info.get("over_dwell_agents", []) or []:
            try:
                agent_idx = int(item.get("agent_idx", -1))
            except (TypeError, ValueError):
                continue
            if agent_idx in priority:
                priority[agent_idx] += 1000.0 + float(item.get("dwell_time", 0.0) or 0.0)

        agent_positions = state_info.get("agent_positions", []) if isinstance(state_info, dict) else []
        for agent_idx in unique_agents:
            if 0 <= agent_idx < len(agent_positions):
                agent_info = agent_positions[agent_idx]
                priority[agent_idx] += float(agent_info.get("passengers", 0.0) or 0.0) * 0.01
                dwell_time = float(agent_info.get("dwell_time", 0.0) or 0.0)
                expected_dwell = float(agent_info.get("expected_dwell", 0.0) or 0.0)
                if expected_dwell > 0.0 and dwell_time > expected_dwell:
                    priority[agent_idx] += min(dwell_time - expected_dwell, 300.0)

        focus_node_set: Set[int] = set()
        for node_id in focus_node_ids or []:
            try:
                focus_node_set.add(int(node_id))
            except (TypeError, ValueError):
                continue
        if focus_node_set and self.world is not None:
            for agent_idx in unique_agents:
                current_node_id = None
                if 0 <= agent_idx < len(agent_positions):
                    current_node_id = int(agent_positions[agent_idx].get("node_id", agent_positions[agent_idx].get("position", -1)))
                elif 0 <= agent_idx < len(self.world.agents):
                    current_node_id = int(self.world.agents[agent_idx].state.position[0].item())
                if current_node_id is None:
                    continue
                if current_node_id in focus_node_set:
                    priority[agent_idx] += 500.0

        service_ranked = self._rank_focus_agents_by_service_potential(
            state_info,
            unique_agents,
            focus_node_ids,
            max_focus=max_focus,
        )
        service_order = {agent_idx: rank for rank, agent_idx in enumerate(service_ranked)}
        ranked_agents = sorted(
            unique_agents,
            key=lambda agent_idx: (
                service_order.get(agent_idx, len(unique_agents)),
                -priority.get(agent_idx, 0.0),
                unique_agents.index(agent_idx),
            ),
        )
        capped = [agent_idx for agent_idx in ranked_agents if agent_idx in set(service_ranked)][:max_focus]
        if not capped:
            capped = ranked_agents[: min(max_focus, len(ranked_agents))]
        logger.info(
            "MILP focus agents capped: %s -> %s (max=%s)",
            len(unique_agents),
            len(capped),
            max_focus,
        )
        return capped

    def _derive_local_focus_targets(
        self,
        state_info: Dict[str, Any],
        fallback_solution: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """Derive a minimal local focus slice when global guidance leaves it empty.

        We still need intervention on obviously pathological local states, such as
        trains stuck far beyond dwell or trains sitting on disrupted endpoints.
        """
        normalized_solution = self._normalize_plan_step(fallback_solution)
        agent_positions = state_info.get("agent_positions", []) if isinstance(state_info, dict) else []
        disruptions = state_info.get("disruptions", []) if isinstance(state_info, dict) else []

        disrupted_node_ids: Set[int] = set()
        for item in disruptions:
            segment = str(item.get("segment", ""))
            match = re.match(r"^(\d+)-(\d+)$", segment)
            if match:
                disrupted_node_ids.add(int(match.group(1)))
                disrupted_node_ids.add(int(match.group(2)))

        focus_agent_ids: List[int] = []
        focus_node_ids: List[int] = []

        def _append_unique(target: List[int], value: Any) -> None:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return
            if parsed not in target:
                target.append(parsed)

        def _distance_to_targets(start_node_id: int, target_node_ids: Set[int], max_depth: int = 240) -> float:
            if not target_node_ids:
                return float("inf")
            if start_node_id in target_node_ids:
                return 0.0
            if self.world is None:
                return float("inf")

            visited = {start_node_id}
            queue = deque([(start_node_id, 0)])
            while queue:
                node_id, distance = queue.popleft()
                if distance >= max_depth:
                    continue
                try:
                    successors = [int(node.id) for node in self.world.get_next_nodes(node_id)]
                except Exception:
                    successors = []
                for successor_id in successors:
                    if successor_id in visited:
                        continue
                    if successor_id in target_node_ids:
                        return float(distance + 1)
                    visited.add(successor_id)
                    queue.append((successor_id, distance + 1))
            return float("inf")

        waiting_threshold = float(self.config.waiting_threshold)
        total_waiting = float(state_info.get("total_waiting_passengers", 0.0) or 0.0)
        max_waiting = float(state_info.get("max_waiting_passengers", 0.0) or 0.0)
        waiting_pressure = total_waiting >= waiting_threshold or max_waiting >= waiting_threshold
        waiting_target_nodes: Set[int] = set()
        if waiting_pressure:
            for item in state_info.get("high_wait_nodes", []) or state_info.get("top_wait_nodes", []):
                if not isinstance(item, dict):
                    continue
                try:
                    node_id = int(item.get("node_id", item.get("node_idx", -1)))
                except (TypeError, ValueError):
                    continue
                if node_id >= 0:
                    waiting_target_nodes.add(node_id)
                    _append_unique(focus_node_ids, node_id)

            nearest_agents: List[Tuple[float, int]] = []
            for agent_idx in range(self.n_agents or len(agent_positions)):
                if 0 <= agent_idx < len(agent_positions):
                    current_node_id = int(agent_positions[agent_idx].get("node_id", agent_positions[agent_idx].get("position", -1)))
                elif self.world is not None and 0 <= agent_idx < len(self.world.agents):
                    current_node_id = int(self.world.agents[agent_idx].state.position[0].item())
                else:
                    continue
                distance = _distance_to_targets(current_node_id, waiting_target_nodes)
                if distance < float("inf"):
                    nearest_agents.append((distance, agent_idx))

            nearest_agents.sort(key=lambda item: (item[0], item[1]))
            max_waiting_focus_agents = min(
                self.n_agents or len(agent_positions),
                max(6, min(int(self.config.max_milp_focus_agents), len(waiting_target_nodes) * 3)),
            )
            for _, agent_idx in nearest_agents[:max_waiting_focus_agents]:
                _append_unique(focus_agent_ids, agent_idx)

        for agent_idx in range(self.n_agents or len(agent_positions)):
            if 0 <= agent_idx < len(agent_positions):
                agent_info = agent_positions[agent_idx]
                current_node_id = int(agent_info.get("node_id", agent_info.get("position", -1)))
                dwell_time = float(agent_info.get("dwell_time", 0.0))
                expected_dwell = float(agent_info.get("expected_dwell", 0.0))
                at_platform = bool(agent_info.get("at_platform", False))
            elif self.world is not None and 0 <= agent_idx < len(self.world.agents):
                agent = self.world.agents[agent_idx]
                current_node_id = int(agent.state.position[0].item())
                dwell_time = float(agent.state.dwell_time[0].item())
                expected_dwell = float(getattr(agent.state, "current_expected_dwell_time", agent.state.dwell_time)[0].item())
                at_platform = current_node_id in self.world.platform_nodes
            else:
                continue

            decision = normalized_solution.get(f"agent_{agent_idx}", {})
            next_node_id = decision.get("next_node") if isinstance(decision, dict) else None
            if next_node_id is None and isinstance(decision, dict):
                try:
                    action = int(decision.get("action", 0))
                except (TypeError, ValueError):
                    action = 0
                if action > 0:
                    next_node_id = self._resolve_action_next_node(current_node_id, action)

            is_over_dwell = (
                at_platform
                and expected_dwell > 0
                and dwell_time >= expected_dwell + 5.0
            )
            touches_disruption = current_node_id in disrupted_node_ids or next_node_id in disrupted_node_ids

            if is_over_dwell or touches_disruption:
                _append_unique(focus_agent_ids, agent_idx)
                _append_unique(focus_node_ids, current_node_id)
                _append_unique(focus_node_ids, next_node_id)

        ranked_focus_agents = self._rank_focus_agents_by_service_potential(
            state_info,
            focus_agent_ids,
            focus_node_ids,
            max_focus=int(self.config.max_milp_focus_agents),
        )
        return ranked_focus_agents or None, focus_node_ids or None

    def _resolve_milp_focus_targets(
        self,
        state_info: Dict[str, Any],
        fallback_solution: Optional[Dict[str, Any]],
        guidance: Any,
    ) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """Resolve the focus slice that should be handed to SLM-guided MILP.

        Only the global guidance may choose the focus slice. Small-model outputs can
        refine that slice, but must not widen it to all agents.
        """
        focus_agent_ids: List[int] = []
        focus_node_ids: List[int] = []

        def _append_unique(target: List[int], value: Any) -> None:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return
            if parsed not in target:
                target.append(parsed)

        if isinstance(self.cached_global_guidance, dict):
            solver_config = self.cached_global_guidance.get("solver_config")
            if isinstance(solver_config, dict):
                for value in solver_config.get("focus_agents", []) or []:
                    _append_unique(focus_agent_ids, value)
                for value in solver_config.get("focus_nodes", []) or []:
                    _append_unique(focus_node_ids, value)

        extracted_agent_ids, extracted_node_ids = self._extract_focus_targets_from_guidance(guidance)
        for value in extracted_agent_ids:
            _append_unique(focus_agent_ids, value)
        for value in extracted_node_ids:
            _append_unique(focus_node_ids, value)

        if focus_node_ids and not focus_agent_ids:
            for agent_idx in self._derive_focus_agents_from_nodes(state_info, fallback_solution, focus_node_ids):
                _append_unique(focus_agent_ids, agent_idx)

        if not focus_agent_ids and not focus_node_ids:
            local_focus_agents, local_focus_nodes = self._derive_local_focus_targets(
                state_info,
                fallback_solution,
            )
            for agent_idx in local_focus_agents or []:
                _append_unique(focus_agent_ids, agent_idx)
            for node_id in local_focus_nodes or []:
                _append_unique(focus_node_ids, node_id)

        return focus_agent_ids or None, focus_node_ids or None

    def _extract_focus_targets_from_guidance(
        self,
        guidance: Any,
    ) -> Tuple[List[int], List[int]]:
        """从不同来源的 guidance 中提取 focus_agents / focus_nodes。"""
        focus_agent_ids: List[int] = []
        focus_node_ids: List[int] = []

        def _append_unique(target: List[int], value: Any) -> None:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return
            if parsed not in target:
                target.append(parsed)

        def _consume(payload: Any) -> None:
            if payload is None:
                return

            if hasattr(payload, "focus_agents") or hasattr(payload, "focus_nodes"):
                for value in getattr(payload, "focus_agents", []) or []:
                    _append_unique(focus_agent_ids, value)
                for value in getattr(payload, "focus_nodes", []) or []:
                    _append_unique(focus_node_ids, value)
                return

            if not isinstance(payload, dict):
                return

            for value in payload.get("focus_agents", []) or []:
                _append_unique(focus_agent_ids, value)
            for value in payload.get("focus_nodes", []) or []:
                _append_unique(focus_node_ids, value)

            solver_config = payload.get("solver_config")
            if isinstance(solver_config, dict):
                _consume(solver_config)

            warm_start_hints = payload.get("warm_start_hints")
            if isinstance(warm_start_hints, dict):
                _consume(warm_start_hints)

            global_guidance = payload.get("global_guidance")
            if isinstance(global_guidance, dict):
                _consume(global_guidance)

        _consume(guidance)
        return focus_agent_ids, focus_node_ids

    def _build_cached_plan_from_milp(
        self,
        trigger_type: str,
        focus_agent_ids: Optional[List[int]],
    ) -> List[Dict[str, Dict[str, Any]]]:
        """Build rolling-horizon cached steps from the most recent MILP solution."""
        if self.milp_solver is None:
            return []

        solution = getattr(self.milp_solver, "last_solution", {}) or {}
        if solution.get("status") == "heuristic_fallback" or solution.get("accepted") is False:
            return []
        all_step_actions = solution.get("all_step_actions")
        all_step_targets = solution.get("all_step_targets")
        if not isinstance(all_step_actions, list) or not all_step_actions:
            return []

        if focus_agent_ids:
            selected_agents = sorted({int(agent_idx) for agent_idx in focus_agent_ids})
        else:
            selected_agents = list(range(self.n_agents or 0))

        simulated_positions: Dict[int, Optional[int]] = {}
        if self.world is not None:
            for agent_idx in selected_agents:
                if 0 <= agent_idx < len(getattr(self.world, "agents", [])):
                    simulated_positions[agent_idx] = int(self.world.agents[agent_idx].state.position[0].item())
                else:
                    simulated_positions[agent_idx] = None

        cached_steps: List[Dict[str, Dict[str, Any]]] = []
        cached_horizon = self._get_cached_plan_horizon(trigger_type)

        for step_idx, step_actions in enumerate(all_step_actions[:cached_horizon]):
            if not isinstance(step_actions, dict):
                continue
            step_targets = all_step_targets[step_idx] if isinstance(all_step_targets, list) and step_idx < len(all_step_targets) else {}
            if not isinstance(step_targets, dict):
                step_targets = {}
            step_plan: Dict[str, Dict[str, Any]] = {}
            for agent_idx in selected_agents:
                agent_key = f"agent_{agent_idx}"
                if agent_key not in step_actions:
                    continue
                action_value = int(step_actions.get(agent_key, 0))
                planned_current = simulated_positions.get(agent_idx)
                planned_next = step_targets.get(agent_key, planned_current)
                if planned_next is None and planned_current is not None and action_value > 0:
                    resolved_next = self._resolve_action_next_node(planned_current, action_value)
                    planned_next = int(resolved_next) if resolved_next is not None else planned_current
                elif planned_next is not None:
                    try:
                        planned_next = int(planned_next)
                    except (TypeError, ValueError):
                        planned_next = planned_current
                tracking_target = self._nearest_forward_platform(planned_next) or planned_next
                simulated_positions[agent_idx] = planned_next
                step_plan[agent_key] = {
                    "action": action_value,
                    "next_node": int(planned_next) if planned_next is not None else None,
                    "target_node": int(tracking_target) if tracking_target is not None else None,
                    "track_target": True,
                    "reasoning": f"MILP滚动规划[{trigger_type}] step={step_idx + 1}",
                }
            if step_plan:
                cached_steps.append(step_plan)

        return cached_steps

    def _has_valid_cached_plan(self, state_info: Dict[str, Any]) -> bool:
        """检查是否有有效的缓存计划"""
        if not self.cached_plan_steps:
            return False
        if self.cached_plan_index >= len(self.cached_plan_steps):
            self._clear_cached_plan()
            return False
        current_signature = self._capture_plan_signature(state_info)
        if self.cached_plan_signature is not None and current_signature != self.cached_plan_signature:
            self._clear_cached_plan()
            return False
        if isinstance(self.cached_plan_metrics, dict):
            current_metrics = self._build_operational_metrics_snapshot(state_info)
            cached_waiting = float(self.cached_plan_metrics.get("total_waiting", 0.0))
            current_waiting = float(current_metrics.get("total_waiting", 0.0))
            waiting_ratio = min(max(float(self.config.cached_plan_waiting_drift_ratio), 0.0), 0.04)
            waiting_drift_abs = max(float(self.config.waiting_threshold) * 0.25, 200.0)
            waiting_threshold = cached_waiting + max(cached_waiting * waiting_ratio, waiting_drift_abs)
            if current_waiting > waiting_threshold:
                self._clear_cached_plan()
                return False

            cached_over_dwell = float(self.cached_plan_metrics.get("agents_over_expected_dwell", 0.0))
            current_over_dwell = float(current_metrics.get("agents_over_expected_dwell", 0.0))
            if current_over_dwell > cached_over_dwell + 1.0:
                self._clear_cached_plan()
                return False
        return True
    
    def _get_cached_action(
        self,
        td: TensorDict,
        state_info: Dict[str, Any],
    ) -> Tuple[Dict[str, int], Dict[str, str]]:
        """获取缓存的滚动计划；当前步只补 RL，不再重复调用 SLM/MILP。"""
        step_plan = copy.deepcopy(self.cached_plan_steps[self.cached_plan_index])
        self.cached_plan_index += 1
        step_plan = self._align_plan_step_with_state(state_info, step_plan)
        step_plan = self._fill_plan_with_rl_baseline(td, state_info, step_plan)
        step_plan = self._guard_cached_plan_step(td, state_info, step_plan)
        return self._extract_actions_and_reasoning(
            step_plan,
            default_reason="缓存高层计划",
        )

    def _build_rl_solution(
        self,
        td: TensorDict,
        state_info: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Build a normalized one-step RL baseline solution for cache gating."""
        rl_actions, rl_reasoning = self._rl_execution_decide(td, state_info)
        rl_solution: Dict[str, Dict[str, Any]] = {}
        agent_positions = state_info.get("agent_positions", []) if isinstance(state_info, dict) else []

        for agent_key, action in rl_actions.items():
            try:
                agent_idx = int(agent_key.split("_")[1])
            except (IndexError, TypeError, ValueError):
                continue
            current_pos = -1
            if 0 <= agent_idx < len(agent_positions):
                current_pos = int(agent_positions[agent_idx].get("node_id", agent_positions[agent_idx].get("position", -1)))
            elif self.world is not None and 0 <= agent_idx < len(getattr(self.world, "agents", [])):
                current_pos = int(self.world.agents[agent_idx].state.position[0].item())

            action_value = int(action)
            rl_solution[agent_key] = {
                "action": action_value,
                "next_node": self._resolve_action_next_node(current_pos, action_value),
                "reasoning": rl_reasoning.get(agent_key, "RL缓存保护基线"),
            }
        return rl_solution

    def _build_default_dispatch_solution(
        self,
        state_info: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Build the greedy-style baseline: request forward movement everywhere.

        This mirrors the strongest simple baseline in benchmark_heuristic.py: the
        simulator still enforces mandatory dwell, blockages, and safety holds.
        """
        solution: Dict[str, Dict[str, Any]] = {}
        agent_positions = state_info.get("agent_positions", []) if isinstance(state_info, dict) else []
        n_agents = self.n_agents or len(agent_positions)

        for agent_idx in range(n_agents):
            current_pos = -1
            if 0 <= agent_idx < len(agent_positions):
                current_pos = int(agent_positions[agent_idx].get("node_id", agent_positions[agent_idx].get("position", -1)))
            elif self.world is not None and 0 <= agent_idx < len(getattr(self.world, "agents", [])):
                current_pos = int(self.world.agents[agent_idx].state.position[0].item())

            action_value = 1
            solution[f"agent_{agent_idx}"] = {
                "action": action_value,
                "next_node": self._resolve_action_next_node(current_pos, action_value),
                "reasoning": "默认前进派车基线",
            }
        return solution

    def _select_best_guard_baseline(
        self,
        td: TensorDict,
        state_info: Dict[str, Any],
    ) -> Tuple[Dict[str, Dict[str, Any]], float, Dict[str, float], str]:
        """Pick the better one-step safety baseline between RL and greedy dispatch."""
        rl_solution = self._build_rl_solution(td, state_info)
        rl_score, rl_metrics = self._score_action_plan_proxy(state_info, rl_solution, None)

        dispatch_solution = self._build_default_dispatch_solution(state_info)
        dispatch_score, dispatch_metrics = self._score_action_plan_proxy(state_info, dispatch_solution, None)

        if dispatch_score > rl_score:
            return dispatch_solution, dispatch_score, dispatch_metrics, "default_dispatch"
        return rl_solution, rl_score, rl_metrics, "rl_execution"

    def _guard_cached_plan_step(
        self,
        td: TensorDict,
        state_info: Dict[str, Any],
        step_plan: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Keep 20-step cache, but do not blindly execute a stale harmful step.

        The cached plan remains the primary source. This guard compares the current
        cached step against the online RL safety baseline with the same proxy used
        by MILP acceptance. If the cached step is clearly worse under the current
        state, only this step is replaced; the remaining cached horizon is kept.
        """
        normalized_plan = self._normalize_plan_step(step_plan)
        if not normalized_plan:
            return normalized_plan

        cached_score, cached_metrics = self._score_action_plan_proxy(
            state_info,
            normalized_plan,
            None,
        )
        baseline_solution, baseline_score, baseline_metrics, baseline_name = self._select_best_guard_baseline(td, state_info)

        harmful_stay = (
            cached_metrics.get("stay_ratio", 0.0) >= 0.85
            and cached_metrics.get("hotspot_touch", 0.0) <= baseline_metrics.get("hotspot_touch", 0.0)
            and baseline_metrics.get("move_count", 0.0) > cached_metrics.get("move_count", 0.0)
        )
        total_waiting = float(state_info.get("total_waiting_passengers", 0.0) or 0.0)
        waiting_pressure = total_waiting >= float(self.config.waiting_threshold)
        clearly_worse = cached_score + 1e-6 < baseline_score
        weak_under_pressure = (
            waiting_pressure
            and cached_score < baseline_score + 0.5
            and baseline_metrics.get("move_count", 0.0) >= cached_metrics.get("move_count", 0.0)
        )
        if not (clearly_worse or harmful_stay or weak_under_pressure):
            return normalized_plan

        guarded_plan = copy.deepcopy(normalized_plan)
        replacement_keys = set(baseline_solution.keys())

        replaced = 0
        for agent_key in sorted(replacement_keys):
            if agent_key not in baseline_solution:
                continue
            guarded_plan[agent_key] = copy.deepcopy(baseline_solution[agent_key])
            base_reason = guarded_plan[agent_key].get("reasoning", "RL缓存保护基线")
            guarded_plan[agent_key]["reasoning"] = (
                f"{base_reason} | 缓存保护: 当前缓存步代理评分较弱 "
                f"(cache={cached_score:.2f}, {baseline_name}={baseline_score:.2f})"
            )
            replaced += 1

        if replaced:
            self.stats["cached_plan_guard_replacements"] = (
                self.stats.get("cached_plan_guard_replacements", 0) + replaced
            )
            logger.info(
                "缓存保护替换当前步: agents=%s cache_score=%.2f baseline_score=%.2f cache_metrics=%s baseline_metrics=%s",
                replaced,
                cached_score,
                baseline_score,
                cached_metrics,
                baseline_metrics,
            )
        return guarded_plan

    def _extract_actions_and_reasoning(
        self,
        solution: Optional[Dict[str, Any]],
        default_reason: str,
    ) -> Tuple[Dict[str, int], Dict[str, str]]:
        """从单步计划中提取动作与推理文本。"""
        solution = self._normalize_plan_step(solution)
        actions: Dict[str, int] = {}
        reasoning: Dict[str, str] = {}
        n_agents = self.n_agents or 0

        if not solution:
            for i in range(n_agents):
                agent_key = f"agent_{i}"
                actions[agent_key] = 0
                reasoning[agent_key] = default_reason
            return actions, reasoning

        for i in range(n_agents):
            agent_key = f"agent_{i}"
            decision = solution.get(agent_key, {}) if isinstance(solution, dict) else {}
            if isinstance(decision, dict):
                actions[agent_key] = int(decision.get("action", 0))
                reasoning[agent_key] = str(decision.get("reasoning", default_reason))
            else:
                actions[agent_key] = int(decision) if decision is not None else 0
                reasoning[agent_key] = default_reason
        return actions, reasoning
    
    def _build_action_tensordict(
        self,
        td: TensorDict,
        actions: Dict[str, int],
    ) -> TensorDict:
        """构建动作TensorDict"""
        n_agents = len(actions)
        action_tensor = torch.zeros(1, n_agents, dtype=torch.long, device=self.device)
        
        for agent_key, action in actions.items():
            idx = int(agent_key.split("_")[1])
            if 0 <= idx < n_agents:
                action_tensor[0, idx] = action
        
        out_td = td.clone()
        out_td[self.action_key] = action_tensor
        return out_td
    
    def _record_decision(
        self,
        state_info: Dict[str, Any],
        layer: DecisionLayer,
        actions: Dict[str, int],
        reasoning: Dict[str, str],
        trigger_type: Optional[str],
        latency_ms: float,
    ):
        """记录决策"""
        import hashlib
        state_str = json.dumps(state_info, sort_keys=True, default=str)
        state_hash = hashlib.md5(state_str.encode()).hexdigest()[:8]
        
        record = DecisionRecord(
            timestamp=time.time(),
            step=self.current_step,
            layer=layer,
            state_hash=state_hash,
            actions=actions,
            reasoning=reasoning,
            confidence=self._compute_confidence(actions, reasoning),
            trigger_type=trigger_type,
            operational_metrics=self._build_operational_metrics_snapshot(state_info),
            latency_ms=latency_ms,
        )
        
        self.decision_history.append(record)
    
    def _compute_confidence(
        self,
        actions: Dict[str, int],
        reasoning: Dict[str, str],
    ) -> float:
        """计算决策置信度"""
        # 简化实现：基于推理长度和一致性
        total_confidence = 0.0
        count = 0
        
        for agent_key, reason in reasoning.items():
            # 较长的推理通常意味着更深入的分析
            length_score = min(len(reason) / 100, 1.0)
            # 检查关键词
            keyword_score = 0.0
            keywords = ["优化", "分析", "考虑", "因此", "建议"]
            for kw in keywords:
                if kw in reason:
                    keyword_score += 0.2
            keyword_score = min(keyword_score, 1.0)
            
            total_confidence += (length_score + keyword_score) / 2
            count += 1
        
        return total_confidence / count if count > 0 else 0.5
    
    def _build_small_model_prompt(
        self,
        state_info: Dict[str, Any],
        rag_context: Dict[str, Any],
        base_plan: Optional[Dict[str, Any]] = None,
        trigger_type: Optional[str] = None,
    ) -> str:
        """构建小模型提示"""
        base_plan = self._normalize_plan_step(base_plan)
        lines = [
            "地铁调度状态:",
            f"- 列车数: {len(state_info['agent_positions'])}",
            f"- 中断数: {len(state_info['disruptions'])}",
            (
                f"- 乘客积压: total={state_info.get('total_waiting_passengers', 0):.0f}, "
                f"max_station={state_info.get('max_waiting_passengers', 0):.0f}, "
                f"avg_station={state_info.get('avg_waiting_passengers', 0):.1f}"
            ),
            f"- 高等待站点: {state_info['high_wait_nodes'][:5]}",
            f"- 热点等待站点: {state_info.get('top_wait_nodes', [])[:5]}",
            f"- 过停站列车: {state_info.get('over_dwell_agents', [])[:5]}",
        ]

        if trigger_type:
            lines.append(f"- 触发类型: {trigger_type}")
        
        if rag_context.get("cases"):
            lines.append(f"\n历史相似案例: {len(rag_context['cases'])} 个")
            for i, case in enumerate(rag_context["cases"][:2]):
                lines.append(f"  案例{i+1}: {case.get('summary', '无摘要')[:100]}")
        
        if rag_context.get("constraints"):
            lines.append(f"\n经验约束: {len(rag_context['constraints'])} 条")

        global_guidance = rag_context.get("global_guidance") if isinstance(rag_context, dict) else None
        if isinstance(global_guidance, dict) and not global_guidance.get("error"):
            lines.append("\n全局求解指导:")
            lines.append(f"- recommended_solver: {global_guidance.get('recommended_solver', 'none')}")
            solver_config = global_guidance.get("solver_config") if isinstance(global_guidance.get("solver_config"), dict) else {}
            lines.append(f"- focus_agents: {solver_config.get('focus_agents', [])}")
            lines.append(f"- focus_nodes: {solver_config.get('focus_nodes', [])}")
            lines.append(f"- strategy_summary: {global_guidance.get('strategy_summary', '')}")

        if base_plan:
            focus_agent_keys = set(base_plan.keys())
            lines.append("\n需要细化执行的高层计划:")
            for i, agent_info in enumerate(state_info.get("agent_positions", [])):
                if focus_agent_keys and len(focus_agent_keys) < len(state_info.get("agent_positions", [])) and f"agent_{i}" not in focus_agent_keys:
                    continue
                current_pos = agent_info.get("node_id", agent_info.get("position", -1))
                next_nodes = []
                if self.world is not None and current_pos in getattr(self.world, "nodes", {}):
                    next_nodes = [node.id for node in self.world.get_next_nodes(current_pos)]
                lines.append(
                    f"  agent_{i}: current_node={current_pos}, "
                    f"dwell={agent_info.get('dwell_time', 0)}/{agent_info.get('expected_dwell', 0)}, "
                    f"next_nodes={next_nodes[:6]}"
                )
            lines.append("")
            for agent_key in sorted(base_plan.keys()):
                decision = base_plan.get(agent_key)
                if not isinstance(decision, dict):
                    continue
                try:
                    action = int(decision.get("action", 0))
                except (TypeError, ValueError):
                    action = 0
                lines.append(
                    f"  {agent_key}: action={action}, "
                    f"next_node={decision.get('next_node', 'STAY' if action == 0 else 'NA')}"
                )
            if focus_agent_keys and len(focus_agent_keys) < len(state_info.get("agent_positions", [])):
                lines.append("请只对上述 focus 列车生成执行细化引导，其他列车保持 MARL 执行。")
            lines.append("请保持高层计划目标不变，仅输出执行细化引导。")
        
        lines.append("\n请为每个列车生成简短的调度建议:")
        
        return "\n".join(lines)
    
    def _parse_guidance_to_solution(
        self,
        guidance,
        base_plan: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """解析小模型输出为解（兼容 GuidanceOutput 对象和字符串）"""
        base_plan = self._normalize_plan_step(base_plan)
        if guidance is None:
            return None

        # --- GuidanceOutput 对象 ---
        if hasattr(guidance, "focus_agents"):
            focus_set = set(guidance.focus_agents or [])
            n = self.n_agents or 10
            solution = {}
            hint_text = "；".join([hint for hint in guidance.strategy_hints[:2] if hint])
            for i in range(n):
                base_decision = {}
                if base_plan and isinstance(base_plan.get(f"agent_{i}"), dict):
                    base_decision = copy.deepcopy(base_plan[f"agent_{i}"])

                action = int(base_decision.get("action", 0))
                if not base_decision and i in focus_set:
                    action = 1

                reasoning_parts = []
                base_reason = str(base_decision.get("reasoning", "")).strip()
                if base_reason:
                    reasoning_parts.append(f"高层计划: {base_reason}")
                if i in focus_set:
                    reasoning_parts.append(hint_text or f"SLM细化: 重点关注列车{i}")
                elif not reasoning_parts:
                    reasoning_parts.append(hint_text or "SLM细化: 沿用高层计划")

                solution[f"agent_{i}"] = {
                    "action": action,
                    "reasoning": " | ".join(reasoning_parts),
                }
                if "next_node" in base_decision:
                    solution[f"agent_{i}"]["next_node"] = base_decision["next_node"]
            return solution

        # --- 字符串形式 ---
        if not isinstance(guidance, str) or not guidance:
            return None
        try:
            if "{" in guidance:
                start = guidance.find("{")
                end = guidance.rfind("}") + 1
                return self._normalize_plan_step(json.loads(guidance[start:end]))
        except Exception:
            pass
        return None

    def _build_milp_guidance_payload(
        self,
        guidance: Any,
        base_plan: Optional[Dict[str, Any]] = None,
        focus_agent_ids: Optional[List[int]] = None,
        focus_node_ids: Optional[List[int]] = None,
        state_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """将小模型输出转换成MILP可消费的求解引导。

        SLM 仍然提供 warm start / variable fixing / branching priority，但变量名
        必须先按当前真实状态校验。这样保留方法里的 solver guidance，同时避免把
        环境 node_id 和 MILP dense node_idx 混用后错误固定变量。
        """
        if self.world is None or base_plan is None:
            return None

        normalized_plan = self._normalize_plan_step(base_plan)
        if not normalized_plan:
            return None

        def _extract_guidance_dict(payload: Any) -> Dict[str, Any]:
            if payload is None:
                return {}
            if isinstance(payload, dict):
                return payload
            extracted: Dict[str, Any] = {}
            for key in ("warm_start", "variable_fixing", "branching_priority"):
                value = getattr(payload, key, None)
                if isinstance(value, dict):
                    extracted[key] = value
            return extracted

        def _is_milp_var_name(name: Any) -> bool:
            return isinstance(name, str) and (
                name.startswith("stay[") or name.startswith("move[") or name.startswith("x[")
            )

        def _parse_var_name(name: Any) -> Optional[Tuple[str, Tuple[int, ...]]]:
            if not isinstance(name, str):
                return None
            match = re.match(r"^(stay|x|move)\[([0-9,\s]+)\]$", name.strip())
            if not match:
                return None
            try:
                indices = tuple(int(part.strip()) for part in match.group(2).split(","))
            except ValueError:
                return None
            return match.group(1), indices

        node_ids = sorted(self.world.nodes.keys())
        node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        warm_start: Dict[str, int] = {}
        variable_fixing: Dict[str, int] = {}
        branching_priority: Dict[str, int] = {}

        if focus_agent_ids:
            candidate_agent_indices = [int(agent_idx) for agent_idx in focus_agent_ids]
        else:
            candidate_agent_indices = list(range(self.n_agents or len(self.world.agents)))

        for agent_idx in candidate_agent_indices:
            agent_key = f"agent_{agent_idx}"
            if agent_key not in normalized_plan or agent_idx >= len(self.world.agents):
                continue

            decision = normalized_plan[agent_key]
            current_node_id = int(self.world.agents[agent_idx].state.position[0].item())
            current_idx = node_index.get(current_node_id)
            if current_idx is None:
                continue

            try:
                action = int(decision.get("action", 0))
            except (TypeError, ValueError):
                action = 0

            if action > 0:
                try:
                    target_node_id = int(decision.get("next_node"))
                except (TypeError, ValueError):
                    target_node_id = self._resolve_action_next_node(current_node_id, action)

                candidate_nodes = [node.id for node in self.world.get_next_nodes(current_node_id)]
                target_idx = node_index.get(target_node_id) if target_node_id is not None else None
                if target_node_id in candidate_nodes and target_idx is not None:
                    warm_start[f"move[{agent_idx},{current_idx},{target_idx},0]"] = 1
                    continue

            warm_start[f"stay[{agent_idx},{current_idx},0]"] = 1

        guidance_dict = _extract_guidance_dict(guidance)
        candidate_agent_set = set(candidate_agent_indices)

        for var_name, value in (guidance_dict.get("variable_fixing") or {}).items():
            parsed = _parse_var_name(var_name)
            if parsed is None:
                continue
            var_type, indices = parsed
            try:
                fixed_value = int(value)
            except (TypeError, ValueError):
                continue
            if fixed_value not in (0, 1):
                continue

            # 只接受 t=0 且能和当前真实位置一致的 x/stay 固定建议。
            # 未来时刻或非当前位置的固定容易把滚动 MILP 锁死到错误轨迹。
            if var_type not in {"stay", "x"} or len(indices) != 3:
                continue
            agent_idx, raw_node_idx, t_step = indices
            if agent_idx not in candidate_agent_set or agent_idx >= len(self.world.agents) or t_step != 0:
                continue

            current_node_id = int(self.world.agents[agent_idx].state.position[0].item())
            current_idx = node_index.get(current_node_id)
            if current_idx is None:
                continue

            raw_as_node_id_idx = node_index.get(raw_node_idx)
            raw_matches_current = raw_node_idx == current_idx or raw_as_node_id_idx == current_idx
            if not raw_matches_current:
                continue

            canonical_name = f"{var_type}[{agent_idx},{current_idx},0]"
            if var_type == "x":
                if fixed_value == 1:
                    variable_fixing[canonical_name] = 1
                continue

            if fixed_value == 1:
                variable_fixing[canonical_name] = 1
                continue

            agent = self.world.agents[agent_idx]
            current_is_platform = current_node_id in getattr(self.world, "platform_nodes", {})
            dwell_time = float(getattr(agent.state, "dwell_time", torch.tensor([0.0]))[0].item())
            expected_dwell_tensor = getattr(agent.state, "current_expected_dwell_time", None)
            expected_dwell = float(expected_dwell_tensor[0].item()) if expected_dwell_tensor is not None else 0.0
            if (not current_is_platform) or dwell_time + 1e-6 >= expected_dwell:
                variable_fixing[canonical_name] = 0

        for var_name, value in (guidance_dict.get("branching_priority") or {}).items():
            if not _is_milp_var_name(var_name):
                continue
            try:
                branching_priority[str(var_name)] = int(value)
            except (TypeError, ValueError):
                continue

        payload: Dict[str, Any] = {}
        if warm_start:
            payload["warm_start"] = warm_start
        if variable_fixing:
            payload["variable_fixing"] = variable_fixing
        if branching_priority:
            payload["branching_priority"] = branching_priority
        if state_info is not None:
            soft_weights = self._build_soft_focus_node_weights(state_info, focus_node_ids)
            if soft_weights:
                payload["soft_focus_node_ids"] = list(soft_weights.keys())
                payload["soft_focus_node_weights"] = {
                    int(node_id): float(weight)
                    for node_id, weight in soft_weights.items()
                }
        payload["max_search_nodes_per_agent"] = 80
        payload["terminal_focus_bonus"] = bool(self.config.milp_terminal_focus_bonus)

        if not payload:
            return None
        return payload

    def _expand_milp_focus_node_ids(
        self,
        focus_node_ids: Optional[List[int]],
        fallback_solution: Optional[Dict[str, Any]],
        focus_agent_ids: Optional[List[int]] = None,
    ) -> Optional[List[int]]:
        """Expand focus nodes so MILP hints do not freeze trains outside the focus area.

        The global LLM's focus nodes are solver hints, not a hard instruction to ban
        all other immediate moves. Include each train's current node and the current
        step plan's next node so one-step execution remains feasible.
        """
        if self.world is None:
            return focus_node_ids

        expanded: List[int] = []
        seen: Set[int] = set()

        def _append_node(node_id: Optional[Any]) -> None:
            try:
                parsed = int(node_id)
            except (TypeError, ValueError):
                return
            if parsed not in seen:
                seen.add(parsed)
                expanded.append(parsed)

        for node_id in focus_node_ids or []:
            _append_node(node_id)

        normalized_solution = self._normalize_plan_step(fallback_solution)
        agent_indices = focus_agent_ids or list(range(min(self.n_agents or 0, len(getattr(self.world, "agents", [])))))
        for agent_idx in agent_indices:
            current_node_id = int(self.world.agents[agent_idx].state.position[0].item())
            _append_node(current_node_id)

            decision = normalized_solution.get(f"agent_{agent_idx}", {})
            if not isinstance(decision, dict):
                continue

            try:
                action = int(decision.get("action", 0))
            except (TypeError, ValueError):
                action = 0

            next_node_id = decision.get("next_node")
            if action > 0 and next_node_id is None:
                next_node_id = self._resolve_action_next_node(current_node_id, action)
            _append_node(next_node_id)

        return expanded or None

    def _convert_milp_actions_to_solution(
        self,
        actions_dict: Dict[str, int],
        fallback_solution: Optional[Dict[str, Any]],
        trigger_type: str,
        focus_agent_ids: Optional[List[int]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """将MILP输出动作转换为单步计划结构。"""
        fallback_solution = self._normalize_plan_step(fallback_solution)
        solution: Dict[str, Dict[str, Any]] = {}
        world = self.world
        solution_details = getattr(self.milp_solver, "last_solution", {}) if self.milp_solver else {}
        focus_agent_set = {int(agent_idx) for agent_idx in focus_agent_ids or []}

        for i in range(self.n_agents or 0):
            agent_key = f"agent_{i}"
            base_decision = copy.deepcopy(fallback_solution.get(agent_key, {}))
            if focus_agent_set and i not in focus_agent_set:
                solution[agent_key] = base_decision if isinstance(base_decision, dict) else {"action": int(base_decision or 0)}
                if not solution[agent_key].get("reasoning"):
                    solution[agent_key]["reasoning"] = "RL基线执行"
                continue

            detail = solution_details.get(agent_key, {}) if isinstance(solution_details, dict) else {}
            action = int(actions_dict.get(agent_key, base_decision.get("action", 0)))

            next_node = base_decision.get("next_node")
            if isinstance(detail, dict) and detail.get("next_node") is not None:
                next_node = int(detail.get("next_node"))
            elif world is not None and 0 <= i < len(world.agents):
                current_pos = int(world.agents[i].state.position[0].item())
                next_node = current_pos
                if action > 0:
                    candidate_nodes = [node.id for node in world.get_next_nodes(current_pos)]
                    direction_idx = action - 1
                    if 0 <= direction_idx < len(candidate_nodes):
                        next_node = int(candidate_nodes[direction_idx])

            reasoning = base_decision.get("reasoning") or f"MILP执行 [{trigger_type}]"
            if isinstance(detail, dict):
                reasoning = str(detail.get("reason") or detail.get("reasoning") or reasoning)

            solution[agent_key] = {
                "action": action,
                "reasoning": reasoning,
            }
            if next_node is not None:
                solution[agent_key]["next_node"] = next_node
                target_node = self._nearest_forward_platform(next_node) or next_node
                solution[agent_key]["target_node"] = target_node

        return solution

    def _score_action_plan_proxy(
        self,
        state_info: Dict[str, Any],
        solution: Dict[str, Dict[str, Any]],
        focus_agent_ids: Optional[List[int]],
    ) -> Tuple[float, Dict[str, float]]:
        """Cheap one-step proxy for hotspot service and harmful holds."""
        if self.world is None:
            return 0.0, {}

        target_weights = self._build_soft_focus_node_weights(state_info)
        agent_positions = state_info.get("agent_positions", []) if isinstance(state_info, dict) else []
        focus_set = {int(agent_idx) for agent_idx in focus_agent_ids or []}
        if not focus_set:
            focus_set = set(range(self.n_agents or len(agent_positions)))

        score = 0.0
        stay_count = 0
        move_count = 0
        hotspot_touch = 0
        harmful_hold = 0

        for agent_idx in sorted(focus_set):
            if 0 <= agent_idx < len(agent_positions):
                agent_info = agent_positions[agent_idx]
                current_node_id = int(agent_info.get("node_id", agent_info.get("position", -1)))
                dwell_time = float(agent_info.get("dwell_time", 0.0) or 0.0)
                expected_dwell = float(agent_info.get("expected_dwell", 0.0) or 0.0)
                at_platform = bool(agent_info.get("at_platform", False))
                passengers = float(agent_info.get("passengers", 0.0) or 0.0)
            elif 0 <= agent_idx < len(self.world.agents):
                agent = self.world.agents[agent_idx]
                current_node_id = int(agent.state.position[0].item())
                dwell_time = float(getattr(agent.state, "dwell_time", torch.tensor([0.0]))[0].item())
                expected_tensor = getattr(agent.state, "current_expected_dwell_time", None)
                expected_dwell = float(expected_tensor[0].item()) if expected_tensor is not None else 0.0
                at_platform = current_node_id in getattr(self.world, "platform_nodes", {})
                passengers = float(getattr(agent, "passenger_count", torch.tensor([0.0]))[0].item())
            else:
                continue

            decision = solution.get(f"agent_{agent_idx}", {}) if isinstance(solution, dict) else {}
            try:
                action = int(decision.get("action", 0))
            except (TypeError, ValueError):
                action = 0

            next_node_id = decision.get("next_node")
            if next_node_id is None and action > 0:
                next_node_id = self._resolve_action_next_node(current_node_id, action)
            try:
                next_node_id = int(next_node_id) if next_node_id is not None else current_node_id
            except (TypeError, ValueError):
                next_node_id = current_node_id

            current_distance, current_target, current_reach = self._forward_distance_to_targets(
                current_node_id,
                target_weights,
            )
            next_distance, next_target, next_reach = self._forward_distance_to_targets(
                next_node_id,
                target_weights,
            )

            if action > 0 and next_node_id != current_node_id:
                move_count += 1
                score += 0.5 + min(passengers, 2520.0) * 0.002
                if current_distance < float("inf") and next_distance < current_distance:
                    target = next_target if next_target is not None else current_target
                    target_weight = target_weights.get(int(target), 1.0) if target is not None else 1.0
                    score += max(0.0, current_distance - next_distance) * max(1.0, target_weight) * 0.02
            else:
                stay_count += 1
                local_wait = target_weights.get(current_node_id, 0.0)
                if at_platform and local_wait > 0.0:
                    hotspot_touch += 1
                    score += local_wait * 0.03
                if (not at_platform) or (expected_dwell > 0.0 and dwell_time >= expected_dwell + 5.0 and local_wait <= 0.0):
                    harmful_hold += 1
                    score -= 5.0

            if next_node_id in target_weights:
                hotspot_touch += 1
                score += target_weights[next_node_id] * 0.04
            if next_reach > current_reach:
                score += (next_reach - current_reach) * 0.5

        total = max(1, len(focus_set))
        metrics = {
            "stay_ratio": stay_count / total,
            "move_count": float(move_count),
            "hotspot_touch": float(hotspot_touch),
            "harmful_hold": float(harmful_hold),
        }
        return score, metrics

    def _accept_milp_solution(
        self,
        td: TensorDict,
        state_info: Dict[str, Any],
        milp_solution: Dict[str, Dict[str, Any]],
        fallback_solution: Dict[str, Dict[str, Any]],
        focus_agent_ids: Optional[List[int]],
    ) -> Tuple[bool, str, Optional[Dict[str, Dict[str, Any]]]]:
        """Gate weak local MILP steps against the RL execution baseline."""
        if not bool(self.config.milp_acceptance_gate):
            return True, "gate_disabled", None

        solution_details = getattr(self.milp_solver, "last_solution", {}) if self.milp_solver else {}
        if isinstance(solution_details, dict) and solution_details.get("status") == "heuristic_fallback":
            return False, "heuristic_fallback_not_cached", None

        baseline_solution, baseline_score, baseline_metrics, baseline_name = self._select_best_guard_baseline(td, state_info)

        milp_score, milp_metrics = self._score_action_plan_proxy(state_info, milp_solution, focus_agent_ids)
        focus_baseline_score, focus_baseline_metrics = self._score_action_plan_proxy(
            state_info,
            baseline_solution,
            focus_agent_ids,
        )
        if milp_score + 1e-6 < focus_baseline_score:
            return (
                False,
                (
                    f"proxy_milp<{focus_baseline_score:.2f} "
                    f"(milp={milp_score:.2f}, {baseline_name}={focus_baseline_score:.2f})"
                ),
                baseline_solution,
            )

        if (
            milp_metrics.get("stay_ratio", 0.0) >= 0.85
            and milp_metrics.get("hotspot_touch", 0.0) <= 0.0
            and focus_baseline_metrics.get("move_count", 0.0) > milp_metrics.get("move_count", 0.0)
        ):
            return False, "low_service_stay_heavy_milp", baseline_solution

        return True, f"accepted_proxy milp={milp_score:.2f} {baseline_name}={focus_baseline_score:.2f}", None

    def _solve_with_milp_guidance(
        self,
        td: TensorDict,
        state_info: Dict[str, Any],
        initial_solution: Optional[Dict[str, Any]],
        guidance: Any,
        trigger_type: str,
        focus_agent_ids: Optional[List[int]] = None,
        focus_node_ids: Optional[List[int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """使用MILP执行层生成最终动作；失败时回退到当前计划。"""
        fallback_solution = self._normalize_plan_step(initial_solution)
        if not fallback_solution:
            return fallback_solution
        if self.milp_solver is None or self.env is None or not hasattr(self.env, "base_env"):
            return fallback_solution

        if self.milp_solver is not None:
            self.milp_solver.last_solution = {}

        if focus_agent_ids is None and focus_node_ids is None:
            focus_agent_ids, focus_node_ids = self._resolve_milp_focus_targets(
                state_info=state_info,
                fallback_solution=fallback_solution,
                guidance=guidance,
            )
        focus_agent_ids = self._cap_milp_focus_agents(
            state_info=state_info,
            focus_agent_ids=focus_agent_ids,
            focus_node_ids=focus_node_ids,
        )

        if focus_node_ids and not focus_agent_ids:
            return fallback_solution

        soft_focus_node_ids = copy.deepcopy(focus_node_ids) if focus_node_ids else None
        focus_node_ids = self._expand_milp_focus_node_ids(
            focus_node_ids=focus_node_ids,
            fallback_solution=fallback_solution,
            focus_agent_ids=focus_agent_ids,
        )
        # focus_nodes 来自 RAG/LLM/SLM，是关键区域提示；真正的硬裁剪应只作用在
        # critical trains 上。若把远处热点站作为 node_allowed 硬约束，MILP 会被禁止
        # 经过中间节点，导致大量 focus trains 只能停留，行为退化到 MARL/全停。
        milp_focus_node_ids = None if focus_agent_ids else focus_node_ids

        llm_guidance = self._build_milp_guidance_payload(
            guidance,
            fallback_solution,
            focus_agent_ids=focus_agent_ids,
            focus_node_ids=soft_focus_node_ids,
            state_info=state_info,
        )

        try:
            self.stats["optimization_calls"] += 1
            time_horizon = self._get_milp_state_horizon()
            time_limit = max(1, min(int(self.config.milp_time_limit), 30))
            solve_start = time.perf_counter()
            logger.info(
                "MILP执行层开始: focus_agents=%s, focus_nodes=%s, soft_focus_nodes=%s, horizon=%s, time_limit=%ss",
                len(focus_agent_ids or []),
                len(milp_focus_node_ids or []),
                len((llm_guidance or {}).get("soft_focus_node_ids", []) if isinstance(llm_guidance, dict) else []),
                time_horizon,
                time_limit,
            )
            actions_dict = self.milp_solver.solve(
                self.env.base_env,
                focus_agent_ids=focus_agent_ids,
                focus_node_ids=milp_focus_node_ids,
                llm_guidance=llm_guidance,
                time_horizon=time_horizon,
                time_limit=time_limit,
            )
            self.stats["milp_solves"] += 1
            solve_elapsed = time.perf_counter() - solve_start
            solution_details = getattr(self.milp_solver, "last_solution", {}) or {}
            logger.info(
                "MILP执行层结束: elapsed=%.2fs, status=%s, solve_time=%.2fs",
                solve_elapsed,
                solution_details.get("status"),
                float(solution_details.get("solve_time", solve_elapsed) or solve_elapsed),
            )
            milp_solution = self._convert_milp_actions_to_solution(
                actions_dict=actions_dict,
                fallback_solution=fallback_solution,
                trigger_type=trigger_type,
                focus_agent_ids=focus_agent_ids,
            )
            accepted, accept_reason, replacement_solution = self._accept_milp_solution(
                td=td,
                state_info=state_info,
                milp_solution=milp_solution,
                fallback_solution=fallback_solution,
                focus_agent_ids=focus_agent_ids,
            )
            if not accepted:
                if isinstance(solution_details, dict):
                    solution_details["accepted"] = False
                    solution_details["accept_reason"] = accept_reason
                    solution_details["all_step_actions"] = []
                    solution_details["all_step_targets"] = []
                logger.info("MILP接受门控拒绝: %s", accept_reason)
                return replacement_solution or fallback_solution
            if isinstance(solution_details, dict):
                solution_details["accepted"] = True
                solution_details["accept_reason"] = accept_reason
            logger.info("MILP接受门控通过: %s", accept_reason)
            return milp_solution
        except Exception as e:
            logger.warning(f"MILP执行层求解失败，回退到当前计划: {e}")
            return fallback_solution
    
    def _build_enhanced_description(
        self,
        state_info: Dict[str, Any],
        rag_context: Dict[str, Any],
        optimized_solution: Optional[Dict[str, Any]],
        trigger_type: str,
    ) -> str:
        """构建增强的状态描述"""
        lines = [
            f"=== 当前状态 (触发类型: {trigger_type}) ===",
            f"列车数: {len(state_info['agent_positions'])}",
            f"中断: {state_info['disruptions']}",
            (
                "乘客积压: "
                f"total={state_info.get('total_waiting_passengers', 0):.0f}, "
                f"max_station={state_info.get('max_waiting_passengers', 0):.0f}, "
                f"avg_station={state_info.get('avg_waiting_passengers', 0):.1f}"
            ),
            f"高等待站点: {state_info['high_wait_nodes']}",
            f"热点等待站点: {state_info.get('top_wait_nodes', [])[:8]}",
            f"过停站列车: {state_info.get('over_dwell_agents', [])[:8]}",
        ]

        lines.append("\n=== 列车实时状态 ===")
        for i, agent_info in enumerate(state_info.get("agent_positions", [])):
            current_pos = agent_info.get("node_id", agent_info.get("position", -1))
            next_nodes = []
            if self.world is not None and current_pos in getattr(self.world, "nodes", {}):
                next_nodes = [node.id for node in self.world.get_next_nodes(current_pos)]
            lines.append(
                f"- agent_{i}: current_node={current_pos}, "
                f"at_platform={bool(agent_info.get('at_platform', False))}, "
                f"dwell={agent_info.get('dwell_time', 0)}/{agent_info.get('expected_dwell', 0)}, "
                f"passengers={agent_info.get('passengers', 0)}, "
                f"next_nodes={next_nodes[:6]}"
            )
        
        if rag_context.get("cases"):
            lines.append(f"\n=== RAG检索结果 ({len(rag_context['cases'])} 案例) ===")
            for case in rag_context["cases"][:3]:
                lines.append(f"- {case.get('summary', '无摘要')[:150]}")
        
        if rag_context.get("constraints"):
            lines.append(f"\n=== 经验约束 ({len(rag_context['constraints'])} 条) ===")
            for constraint in rag_context["constraints"][:5]:
                lines.append(f"- {constraint}")
        
        if optimized_solution:
            lines.append("\n=== 优化器初始解 ===")
            lines.append(json.dumps(optimized_solution, ensure_ascii=False, indent=2)[:500])
        
        return "\n".join(lines)
    
    # =========================================================================
    # 统计和导出
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.stats["total_decisions"]
        return {
            **self.stats,
            "rl_ratio": self.stats["rl_decisions"] / total if total > 0 else 0.0,
            "rule_ratio": self.stats.get("rule_decisions", 0) / total if total > 0 else 0.0,
            "llm_ratio": self.stats["llm_interventions"] / total if total > 0 else 0.0,
            "avg_latency_ms": self.stats["total_latency_ms"] / total if total > 0 else 0.0,
        }
    
    def export_decision_history(self, path: str):
        """导出决策历史"""
        records = []
        for record in self.decision_history:
            records.append({
                "timestamp": record.timestamp,
                "step": record.step,
                "layer": record.layer.value,
                "state_hash": record.state_hash,
                "actions": record.actions,
                "reasoning": record.reasoning,
                "confidence": record.confidence,
                "trigger_type": record.trigger_type,
                "latency_ms": record.latency_ms,
                "coordination": record.coordination,
            })
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        logger.info(f"决策历史已导出: {path}")
    
    def save_to_rag(self, case_quality: float = 0.8):
        """将优质决策保存到RAG库"""
        if not self.rag_manager:
            return
        
        for record in self.decision_history:
            if record.confidence >= case_quality:
                self.rag_manager.add_case({
                    "state_hash": record.state_hash,
                    "actions": record.actions,
                    "reasoning": record.reasoning,
                    "confidence": record.confidence,
                    "layer": record.layer.value,
                    "trigger_type": record.trigger_type,
                })


# =============================================================================
# 工厂函数
# =============================================================================

def create_hierarchical_policy(
    env,
    config_dict: Dict[str, Any] = None,
    **kwargs,
) -> HierarchicalLLMPolicy:
    """
    创建分层策略的工厂函数
    
    Args:
        env: TorchRL环境
        config_dict: 配置字典
        **kwargs: 额外参数
    
    Returns:
        HierarchicalLLMPolicy实例
    """
    config_dict = config_dict or {}
    config = HierarchicalConfig(**config_dict)
    
    return HierarchicalLLMPolicy(
        env=env,
        config=config,
        **kwargs,
    )


# =============================================================================
# 向后兼容
# =============================================================================

# 别名，保持向后兼容
HierarchicalPolicy = HierarchicalLLMPolicy


if __name__ == "__main__":
    # 简单测试
    print("分层RL-LLM策略模块加载成功")
    print(f"可用组件:")
    print(f"  - EnhancedRAGManager: {EnhancedRAGManager is not None}")
    print(f"  - ExplainableDecisionAnalyzer: {ExplainableDecisionAnalyzer is not None}")
    print(f"  - HierarchicalCoordinator: {HierarchicalCoordinator is not None}")
    print(f"  - OptimizationToolkit: {OptimizationToolkit is not None}")
    print(f"  - SmallModelGuidance: {SmallModelGuidance is not None}")
