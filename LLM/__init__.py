"""
LLM Module for Metro Dispatching

分层RL-LLM Agent地铁调度系统

模块组成:
- hierarchical_llm_policy: 分层策略主类
- enhanced_rag_manager: RAG检索增强生成
- explainable_decision: 可解释决策分析
- hierarchical_coordinator: 分层协调框架
- optimization_toolkit: 优化工具集
- small_model_guidance: 小模型引导

核心功能:
1. 强化学习执行层 - 高频、低延时在线决策
2. LLM Agent策略层 - 在策略不确定性升高或出现扰动时按需介入
3. RAG增强检索 - 匹配历史相似状态、维护策略与调度案例
4. 可解释决策分析 - 注意力机制回溯决策路径中的关键影响因素
5. 优化工具集成 - 局部问题快速求解，全局问题通过问题分解
6. 小模型引导 - SFT+RL微调的小规模语言模型进行求解引导

使用示例:
    from LLM import HierarchicalLLMPolicy, HierarchicalConfig
    
    config = HierarchicalConfig(
        llm_api_key="your-api-key",
        rl_model_path="path/to/rl/model.pt",
        use_small_model_guidance=True,
    )
    
    policy = HierarchicalLLMPolicy(
        env=env,
        config=config,
        n_agents=10,
    )
    
    # 运行决策
    td = policy(td)

Author: MetroRL Team
Date: 2025
"""

from importlib import import_module as _import_module
from pathlib import Path as _Path
import warnings as _warnings


# 版本信息
__version__ = "2.0.0"
__author__ = "MetroRL Team"

# Keep package import independent of the dispatch/training dependency stacks.
_COMPONENTS = {
    "hierarchical_policy": ("LLM.hierarchical_llm_policy", (
        "HierarchicalLLMPolicy", "HierarchicalConfig", "DecisionLayer",
        "DecisionRecord", "create_hierarchical_policy",
    )),
    "llm_policy": ("LLM.llm_policy", (
        "LLMAgentPolicy", "OperationsResearchToolkit",
    )),
    "enhanced_rag": ("LLM.rag.enhanced_rag_manager", (
        "EnhancedRAGManager", "StateEncoder", "CaseLibrary", "ConstraintExtractor",
    )),
    "explainability": ("LLM.explainable_decision", (
        "ExplainableDecisionAnalyzer", "AttentionMechanism", "CausalAnalyzer",
        "DecisionExplainer",
    )),
    "coordinator": ("LLM.hierarchical_coordinator", (
        "HierarchicalCoordinator", "ExecutionLayer", "StrategyLayer",
        "TriggerEvaluator", "TriggerType",
    )),
    "optimization": ("LLM.optimization_toolkit", (
        "OptimizationToolkit", "LocalSolver", "GlobalOptimizer",
        "ProblemDecomposer", "OptimizationMethod",
    )),
    "small_model": ("LLM.small_model_guidance", (
        "SmallModelGuidance", "KnowledgeDistillation", "OnlineLearner",
        "MultiModelFusion",
    )),
}
_SYMBOL_MODULES = {
    symbol: module
    for module, symbols in _COMPONENTS.values()
    for symbol in symbols
}
__all__ = ["__version__", "__author__", *_SYMBOL_MODULES]


def __getattr__(name):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


def get_available_components() -> dict:
    """Load components on request and return the legacy boolean status mapping.

    Each unavailable component emits a RuntimeWarning naming its first missing
    external dependency. Missing project modules and other code errors propagate.
    Direct access to a public symbol also preserves the original import error;
    failed imports are never cached as None.
    """
    status = {}
    for component, (module_name, symbols) in _COMPONENTS.items():
        try:
            module = _import_module(module_name)
            exports = [getattr(module, symbol) for symbol in symbols]
        except ModuleNotFoundError as exc:
            missing_root = (exc.name or "").split(".")[0]
            project_root = _Path(__file__).resolve().parent.parent
            if (not missing_root or missing_root == __name__
                    or (project_root / missing_root).is_dir()
                    or (project_root / (missing_root + ".py")).is_file()):
                raise
            _warnings.warn(
                f"{component}: unavailable (missing dependency {exc.name!r})",
                RuntimeWarning,
                stacklevel=2,
            )
            status[component] = False
        else:
            status[component] = all(value is not None for value in exports)
    return status


def print_component_status():
    """打印组件状态"""
    status = get_available_components()
    print("=" * 50)
    print("LLM Module Component Status")
    print("=" * 50)
    for name, available in status.items():
        icon = "✓" if available else "✗"
        print(f"  [{icon}] {name}")
    print("=" * 50)
