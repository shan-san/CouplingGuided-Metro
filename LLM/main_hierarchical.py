"""
主入口文件：分层RL-LLM Agent地铁调度系统

使用方法:
    python main_hierarchical.py --mode train    # 训练模式
    python main_hierarchical.py --mode eval     # 评估模式
    python main_hierarchical.py --mode demo     # 演示模式

Author: MetroRL Team
Date: 2025
"""

import argparse
import hashlib
import os
import sys
import json
import time
import logging
import random
from typing import Dict, Any, Optional


def _normalize_thread_env() -> None:
    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        value = os.environ.get(env_name)
        if value is None:
            continue
        try:
            if int(value) <= 0:
                os.environ[env_name] = "1"
        except ValueError:
            os.environ[env_name] = "1"


_normalize_thread_env()

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import pandas as pd
from tensordict import TensorDict
from torchrl.envs import RewardSum, TransformedEnv
from tqdm import tqdm

# 导入环境与拓扑配置
from Params import TOPOLOGY_DEFAULTS, apply_topology_defaults, get_default_n_agents, get_default_tick_seconds, rl_parser
from linedata import load_topology
from metro_rl_env import MetroRLEnvV2
from train_run_config import TrainRunConfig
from benchmark_marl import AgentLoadRecorder, GlobalMetricRecorder, StationEventRecorder
from experiment_scenarios import (
    add_scenario_args,
    scenario_config_from_args,
    scenario_kwargs,
    scenario_suffix,
)

# 导入分层策略
from LLM.hierarchical_llm_policy import (
    HierarchicalLLMPolicy,
    HierarchicalConfig,
)
from LLM.experiment_controls import component_checks, experiment_configuration
from experiment_results import EpisodeResults

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hierarchical_agent.log', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)


TOPOLOGY_ALIASES = {
    "shenzhenline1": 3,
    "topology4": 4,
    "topology16": 16,
}
from topology.metro_lines import LINE_SPECS, dispatch_configuration, event_resources
from scenarios.prepare_dispatch import runtime_scenario, scenario_names, SPLIT_OFFSETS
TOPOLOGY_ALIASES.update({spec["key"]: tid for tid, spec in LINE_SPECS.items()})

DEFAULT_RL_CHECKPOINTS = {
    4: os.path.join(PROJECT_ROOT, "MARL_model", "metro_topology_4_mappo_best.pt"),
    16: os.path.join(PROJECT_ROOT, "MARL_model", "metro_topology_16_mappo_best.pt"),
}

DEFAULT_RAG_CASES = {
    4: os.path.join(PROJECT_ROOT, "LLM", "rag", "topology4_gold_cases.json"),
    16: os.path.join(PROJECT_ROOT, "LLM", "rag", "topology16_gold_cases.json"),
}

DEFAULT_MILP_TIME_LIMITS = {
    16: 360,
}

DEFAULT_MILP_HORIZONS = {
    4: 40,
    16: 20,
}

DEFAULT_CACHED_PLAN_MAX_STEPS = {
    4: 8,
    16: 20,
}

DEFAULT_MILP_ACCEPTANCE_GATES = {
    4: False,
    16: True,
}


def parse_topology(value: str | int) -> int:
    if isinstance(value, int):
        topology = value
    else:
        normalized = str(value).strip().lower()
        if normalized in TOPOLOGY_ALIASES:
            topology = TOPOLOGY_ALIASES[normalized]
        else:
            try:
                topology = int(normalized)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"不支持的拓扑: {value}") from exc

    if topology not in TOPOLOGY_DEFAULTS:
        supported = ", ".join(str(key) for key in sorted(TOPOLOGY_DEFAULTS))
        raise argparse.ArgumentTypeError(f"不支持的拓扑: {value}。可选值: {supported}")
    return topology


def resolve_default_rl_model_path(topology: int) -> Optional[str]:
    candidate = DEFAULT_RL_CHECKPOINTS.get(topology)
    if candidate and os.path.exists(candidate):
        return candidate
    return None


def resolve_default_rag_db_path(topology: int) -> Optional[str]:
    candidate = DEFAULT_RAG_CASES.get(topology)
    if candidate and os.path.exists(candidate):
        return candidate
    return None


def resolve_default_milp_time_limit(topology: int) -> int:
    """Choose a topology-aware hard-coded MILP time limit."""
    return int(DEFAULT_MILP_TIME_LIMITS.get(topology, 30))


def resolve_default_milp_horizon(topology: int) -> int:
    """Use the same scale as the topology-specific teacher/RAG decision interval."""
    return int(DEFAULT_MILP_HORIZONS.get(topology, 20))


def resolve_default_cached_plan_max_steps(topology: int) -> int:
    """Choose the topology-specific cache replay length."""
    return int(DEFAULT_CACHED_PLAN_MAX_STEPS.get(topology, 20))


def resolve_default_milp_acceptance_gate(topology: int) -> bool:
    """Topology4's historical RAG benchmark used multi-step replay without the strict online gate."""
    return bool(DEFAULT_MILP_ACCEPTANCE_GATES.get(topology, True))


def build_env(
    topology: int,
    n_steps: int,
    seed: int,
    n_agents: Optional[int] = None,
    scenario_config: Optional[Dict[str, Any]] = None,
):
    resolved_rl_args = rl_parser.parse_args([])
    apply_topology_defaults(resolved_rl_args, topology)

    tick_seconds = (scenario_config or {}).get("tick_seconds", get_default_tick_seconds(topology) or 1)
    if not isinstance(tick_seconds, (int, float)) or tick_seconds <= 0:
        raise ValueError("tick_seconds must be positive")
    resolved_n_agents = n_agents or get_default_n_agents(topology)
    if resolved_n_agents is None:
        raise ValueError(f"未找到 topology {topology} 对应的默认列车数")

    os.environ["METRO_TOPOLOGY"] = str(topology)
    load_topology(topology)

    start_steps = int(getattr(resolved_rl_args, "start_steps", 0))
    max_steps = start_steps + int(n_steps)
    episode_length = int(getattr(resolved_rl_args, "max_steps", n_steps))

    train_run_config = TrainRunConfig(
        route_id=None,
        headway=getattr(resolved_rl_args, "train_headway", 240),
        loops_per_run=getattr(resolved_rl_args, "train_loops_per_run", 7),
        first_start_time=getattr(resolved_rl_args, "train_first_start_time", 0),
        assignments=getattr(resolved_rl_args, "train_assignments", None),
    )

    native_recipe = (scenario_config or {}).get("scenario_recipe")
    device = torch.device("cpu" if native_recipe else ("cuda" if torch.cuda.is_available() else "cpu"))
    extra_kwargs = scenario_kwargs(scenario_config)
    if topology in LINE_SPECS:
        extra_kwargs.setdefault("dispatch_config", dispatch_configuration(topology, resolved_n_agents))
        extra_kwargs.setdefault("event_resources", event_resources(topology))
    if native_recipe:
        unsupported = {"blockages", "speed_limits", "passenger_demand_scale", "passenger_od_variant",
                       "boarding_rate_persons_per_second", "safe_platform_queue_capacity", "events_file"}
        if unsupported & set(scenario_config):
            raise ValueError("Canonical scenario recipes cannot silently override physical inputs")
        if native_recipe["topology"] != topology:
            raise ValueError("Scenario recipe topology differs from the benchmark topology")
        if extra_kwargs["dispatch_config"] != dispatch_configuration(topology, resolved_n_agents):
            raise ValueError("Canonical recipe uses the declared fleet and dispatch configuration")
        if extra_kwargs["event_resources"] != event_resources(topology):
            raise ValueError("Canonical recipe uses the declared infrastructure resources")
        extra_kwargs["canonical_dispatch_seed"] = seed
    if extra_kwargs.get("dispatch_config") is not None:
        # Timed operations start from the declared 06:00-relative origin.
        start_steps, max_steps = 0, int(n_steps)

    base_env = MetroRLEnvV2(
        num_envs=1,
        max_steps=max_steps,
        device=device,
        n_agents=resolved_n_agents,
        train_run_config=train_run_config,
        topology=topology,
        start_step=start_steps,
        episode_length=episode_length,
        enable_random_blockages=False,
        enable_random_speed_limits=False,
        enable_simulation_blockages=not bool(native_recipe),
        enable_simulation_speed_limits=not bool(native_recipe),
        default_interval=tick_seconds,
        seed=seed,
        use_float_boarding=True,
        **extra_kwargs,
    )
    if not native_recipe and scenario_config and ("events_file" in scenario_config or "events" in scenario_config):
        base_env.scenario.world.load_events(scenario_config.get("events_file", scenario_config.get("events")),
                                           resources=extra_kwargs.get("event_resources"))
    env = TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    return env, resolved_n_agents


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='分层RL-LLM Agent地铁调度系统'
    )
    
    # 运行模式
    parser.add_argument(
        '--mode', type=str, default='eval',
        choices=['train', 'eval', 'demo', 'benchmark'],
        help='运行模式'
    )
    
    # 环境配置
    parser.add_argument('--n_agents', type=int, default=None, help='列车数量，留空时按topology自动设置')
    parser.add_argument('--n_steps', type=int, default=500, help='仿真步数')
    parser.add_argument('--topology', type=parse_topology, default=4, help='拓扑结构，例如 4、16 或 shenzhenline1')
    parser.add_argument('--seed', type=int, default=111, help='随机种子')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='MARL设备；vLLM模式下auto使用CPU，模型权重仅由独立服务加载')
    parser.add_argument('--dispatch-config', type=str, default=None, help='持续作业与车辆服务配置JSON')
    parser.add_argument('--events-file', type=str, default=None, help='独立编号的多事件JSON输入')
    parser.add_argument('--dispatch-scenario', choices=scenario_names('test'), default=None,
                        help='从Python定义即时生成多事件情形，不读取场景文件')
    parser.add_argument('--scenario-seed', type=int, default=111)
    parser.add_argument('--scenario-split', choices=['train', 'validation', 'test'], default='test')
    parser.add_argument('--scenario-repeat', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=1, help='benchmark episode 数，默认 1')
    
    # RL配置
    parser.add_argument(
        '--rl_model_path', type=str, default=None,
        help='RL模型路径'
    )
    
    # LLM配置
    parser.add_argument('--llm_api_key', type=str, default=None, help='LLM API密钥')
    parser.add_argument('--llm_base_url', type=str, default=None, help='LLM API基础URL')
    parser.add_argument('--llm_model', type=str, default='gpt-4', help='LLM模型名称')
    parser.add_argument('--llm-max-input-tokens', type=int, default=32768,
                        help='战略LLM输入保守预算：UTF-8字节数加1024聊天封装预留，不是供应商tokenizer实测值')
    parser.add_argument('--llm-max-output-tokens', type=int, default=1024,
                        help='战略LLM最大输出token数；与输入预算分开预留')
    parser.add_argument('--llm-request-timeout', type=float, default=30., help='原生战略LLM完整请求的硬墙钟时限，秒')
    parser.add_argument('--solver-reserve-fraction', type=float, default=.5,
                        help='为MILP保留触发检查后剩余协调预算的比例，默认0.5；不改变总预算')
    parser.add_argument('--llm-thinking-mode', choices=['disabled', 'enabled', 'auto'], default='disabled',
                        help='DeepSeek战略请求的思考模式；默认关闭；auto保留供应商默认值')
    parser.add_argument('--llm-reasoning-effort', choices=['low', 'high', 'max'], default='low',
                        help='显式启用DeepSeek思考模式时的强度')
    
    # 小模型配置
    parser.add_argument('--small_model_path', type=str, default=None, help='小模型路径')
    parser.add_argument('--small_model_url', type=str, default=None, help='小模型服务URL')
    parser.add_argument('--small_model_api_key', type=str, default=None, help='小模型 API 密钥')
    parser.add_argument('--small_model_model', type=str, default='metro-slm-v9', help='小模型服务中的LoRA名称，不是基础模型名称')
    parser.add_argument('--use_small_model', action='store_true', help='启用小模型引导')
    parser.add_argument('--small-model-backend', '--small_model_backend', choices=['auto', 'local', 'vllm'],
                        default='vllm', help='默认使用独立vLLM服务；auto/local仅用于旧接口兼容')
    parser.add_argument('--small-model-tokenizer', default=None, help='本地文本tokenizer目录，默认使用适配器目录')
    parser.add_argument('--small-model-timeout', type=float, default=30.0, help='vLLM单次请求及本次协调中SLM分组指导的总时限（秒），不自动重试')
    
    # RAG配置
    parser.add_argument('--rag_db_path', type=str, default=None, help='RAG知识库路径')
    parser.add_argument('--use_embeddings', action='store_true', help='使用向量嵌入')
    parser.add_argument(
        '--ablate_rag',
        '--disable_rag',
        dest='ablate_rag',
        action='store_true',
        help='消融RAG模块：不初始化、不检索RAG案例库',
    )
    
    # 触发条件
    parser.add_argument('--entropy_threshold', type=float, default=0.99, help='熵阈值')
    parser.add_argument('--waiting_threshold', type=float, default=2000.0, help='等待人数阈值')
    parser.add_argument('--waiting-total-threshold', type=float, default=None, help='系统总候车人数阈值')
    parser.add_argument('--waiting-max-threshold', type=float, default=None, help='单站候车人数阈值')
    parser.add_argument('--reference-policy', choices=['marl', 'rule'], default='marl',
                        help='Checked reference provider; rule never loads or calls MARL')
    parser.add_argument('--trigger-mode', choices=['coupling', 'legacy', 'periodic', 'event'], default='coupling')
    parser.add_argument('--coordination-period', type=int, default=10,
                        help='Periodic requests every N decision intervals, including stage zero')
    parser.add_argument('--coordination-scope', choices=['coupling', 'full_fleet', 'nearest'], default='coupling')
    parser.add_argument('--nearest-vehicles', type=int, default=6,
                        help='Nearest-scope seed size; mandatory operation partners remain included')
    parser.add_argument('--optimization-mode', choices=['joint', 'staged'], default='joint')
    parser.add_argument('--fixed-planning-horizon', action='store_true',
                        help='Use exactly --milp_horizon instead of the LLM horizon recommendation')
    parser.add_argument('--coordination-horizon-policy', choices=['adaptive', 'fixed'], default='adaptive',
                        help='Use published operation durations and a service tail, or keep the requested window fixed')
    parser.add_argument('--coordination-max-horizon', type=int, default=24,
                        help='Maximum adaptive coordination window in decision stages; episode/input limits still apply')
    parser.add_argument('--coordination-service-tail', type=int, default=3,
                        help='Service observation stages after the estimated operation duration')
    parser.add_argument('--scope-expansion-agents', type=int, default=2,
                        help='Maximum directly blocking background trains released across request-local scope repairs')
    parser.add_argument('--scope-window-closure', action='store_true',
                        help='Opt-in reference-window motion closure; validate against the unchanged method before deployment')
    parser.add_argument('--scope-window-vehicles', type=int, default=12,
                        help='Maximum window-overlap decision owners before mandatory operation closure')
    parser.add_argument('--coupling-check-horizon', type=int, default=3)
    parser.add_argument('--coupling-service-gap', type=float, default=20.)
    parser.add_argument('--coupling-persistence', type=int, default=2)
    parser.add_argument('--coordination-budget', type=float, default=None, help='含检索、生成、检查和求解的协作式总预算秒数')
    parser.add_argument('--allowed-decision-delay', type=float, default=None, help='用于如实统计超时的允许决策延迟秒数')
    parser.add_argument('--joint-guidance-mode', choices=['full', 'no_guidance', 'rules', 'retrieval', 'no_llm', 'no_slm'], default='full')
    parser.add_argument('--no-reference-quality-guard', action='store_true', help='Ablate the same-state MARL/route cost guard')
    parser.add_argument('--torch-threads', type=int, default=1, help='CPU threads for single-world tensor operations')
    parser.add_argument('--cooldown', type=int, default=10, help='耦合模式为每个区域的冷却步数，其他模式保留全局冷却')
    parser.add_argument(
        '--ablate_when2ask',
        '--always_hierarchical',
        dest='ablate_when2ask',
        action='store_true',
        help='消融When2Ask trigger：不使用触发器、冷却或退避判断，保留hierarchical内部plan cache',
    )

    # MILP配置
    parser.add_argument(
        '--milp_horizon',
        type=int,
        default=None,
        help='MILP规划时间域。留空时按 topology 使用写死默认值',
    )
    parser.add_argument(
        '--milp_time_limit',
        type=int,
        default=None,
        help='MILP求解时限(秒)。留空时按 topology 使用写死默认值',
    )
    parser.add_argument(
        '--cached-plan-max-steps',
        '--cached_plan_max_steps',
        dest='cached_plan_max_steps',
        type=int,
        default=None,
        help='高层/MILP缓存计划最大连续回放步数。留空时按 topology 使用默认值',
    )
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--output', type=str, default=None, help='结果CSV输出路径')
    parser.add_argument('--station-output', type=str, default=None, help='站点事件CSV输出路径')
    parser.add_argument('--agent-load-output', type=str, default=None, help='Agent载客数CSV输出路径')
    parser.add_argument('--save_history', action='store_true', help='保存决策历史')
    parser.add_argument('--require-components', action='store_true', help='完整方法实验时禁止静默缺少MARL、LLM、SLM或案例库')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    parser.add_argument(
        '--log_llm_io', type=str, default=None,
        help='将大模型/小模型的输入输出记录到指定 JSONL 文件（追加模式）'
    )
    add_scenario_args(parser)

    return parser.parse_args()


def resolve_policy_device(args):
    if args.device == "auto":
        if args.small_model_backend == "vllm":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return args.device


def build_config(args) -> HierarchicalConfig:
    """从命令行参数构建配置"""
    rag_db_path = None if args.ablate_rag else (args.rag_db_path or resolve_default_rag_db_path(args.topology))
    milp_time_limit = (
        int(args.milp_time_limit)
        if args.milp_time_limit is not None
        else resolve_default_milp_time_limit(args.topology)
    )
    milp_horizon = (
        int(args.milp_horizon)
        if args.milp_horizon is not None
        else resolve_default_milp_horizon(args.topology)
    )
    return HierarchicalConfig(
        # RL配置
        rl_model_path=None if args.reference_policy == 'rule' else args.rl_model_path,
        rl_inference_timeout_ms=10,
        
        # LLM配置
        llm_api_key=args.llm_api_key or os.environ.get('OPENAI_API_KEY', ''),
        llm_base_url=args.llm_base_url,
        llm_model_path=args.llm_model,
        llm_temperature=0.1,
        llm_max_input_tokens=args.llm_max_input_tokens,
        llm_max_output_tokens=args.llm_max_output_tokens,
        llm_request_timeout_s=args.llm_request_timeout,
        llm_thinking_mode=args.llm_thinking_mode,
        llm_reasoning_effort=args.llm_reasoning_effort,
        solver_reserve_fraction=args.solver_reserve_fraction,
        
        # 小模型配置
        small_model_path=args.small_model_path,
        small_model_base_url=args.small_model_url,
        small_model_api_key=args.small_model_api_key,
        small_model_model=args.small_model_model,
        small_model_backend=args.small_model_backend,
        small_model_tokenizer_path=args.small_model_tokenizer,
        small_model_request_timeout=args.small_model_timeout,
        use_small_model_guidance=args.use_small_model,
        
        # RAG配置
        rag_db_path=rag_db_path,
        rag_use_embeddings=args.use_embeddings,
        rag_top_k=5,
        use_rag=not args.ablate_rag,
        
        # 触发条件
        entropy_threshold=args.entropy_threshold,
        waiting_threshold=args.waiting_threshold,
        waiting_total_threshold=args.waiting_total_threshold,
        waiting_max_threshold=args.waiting_max_threshold,
        trigger_mode=args.trigger_mode,
        coupling_check_horizon=args.coupling_check_horizon,
        coupling_service_gap=args.coupling_service_gap,
        coupling_persistence=args.coupling_persistence,
        coordination_time_budget_s=args.coordination_budget,
        allowed_decision_delay_s=args.allowed_decision_delay,
        joint_guidance_mode=args.joint_guidance_mode,
        reference_quality_guard=not getattr(args, 'no_reference_quality_guard', False),
        reference_policy=args.reference_policy,
        coordination_period=args.coordination_period,
        coordination_scope=args.coordination_scope,
        nearest_vehicles=args.nearest_vehicles,
        optimization_mode=args.optimization_mode,
        fixed_planning_horizon=args.fixed_planning_horizon,
        coordination_horizon_policy=args.coordination_horizon_policy,
        coordination_max_horizon=args.coordination_max_horizon,
        coordination_service_tail=args.coordination_service_tail,
        scope_expansion_agents=args.scope_expansion_agents,
        scope_window_closure=args.scope_window_closure,
        scope_window_vehicles=args.scope_window_vehicles,
        intervention_cooldown=args.cooldown,
        no_op_intervention_cooldown=120,
        always_hierarchical=args.ablate_when2ask,
        
        # 优化配置
        local_solver_timeout_ms=100,
        global_decomposition=True,
        milp_time_horizon=milp_horizon,
        milp_time_limit=milp_time_limit,
        max_milp_focus_agents=12,
        cached_plan_max_steps=(
            int(args.cached_plan_max_steps)
            if args.cached_plan_max_steps is not None
            else resolve_default_cached_plan_max_steps(args.topology)
        ),
        milp_acceptance_gate=resolve_default_milp_acceptance_gate(args.topology),
        
        # 可解释性
        enable_explanation=True,
        explanation_detail_level="medium",
    )


def run_evaluation(
    env,
    policy: HierarchicalLLMPolicy,
    n_steps: int,
    verbose: bool = False,
    execution_output: str | None = None,
) -> Dict[str, Any]:
    """
    运行评估
    
    Returns:
        评估结果字典
    """
    logger.info("开始评估运行...")
    
    # 重置环境
    td = env.reset()
    from dispatch_records import enable_execution_records
    native_world = env.base_env.scenario.world
    if getattr(native_world, "second_precision", False):
        enable_execution_records(native_world)
    execution_episode = 0
    
    # 记录指标
    total_wait_time = 0.0
    total_transported = 0
    step_results = []
    
    start_time = time.time()
    
    for step in range(n_steps):
        # 获取决策
        td = policy(td)
        
        # 执行动作
        td = env.step(td)
        policy.notify_step_result(td)
        
        # 收集指标
        if hasattr(env, 'base_env'):
            world = env.base_env.scenario.world
            step_wait = sum(
                node.waiting_passengers[0].item()
                for node in world.platform_nodes.values()
            )
            total_wait_time += step_wait
            
            step_transported = sum(
                agent.passenger_count[0].item()
                for agent in world.agents
            )
        else:
            step_wait = 0
            step_transported = 0
        
        step_results.append({
            "step": step,
            "waiting": step_wait,
            "transported": step_transported,
        })
        
        if verbose and step % 50 == 0:
            logger.info(f"Step {step}: 等待={step_wait:.0f}, 运输={step_transported}")
        
        # 准备下一步
        done = bool(torch.as_tensor(td.get(("next", "done"), td.get("done", False))).any())
        if execution_output and (done or step == n_steps - 1) and getattr(native_world, "execution_records", None) is not None:
            native_world.execution_records.write(os.path.join(execution_output, f"episode_{execution_episode}_execution"), native_world)
        if done and step < n_steps - 1:
            td = env.reset()
            execution_episode += 1
            if getattr(native_world, "second_precision", False):
                enable_execution_records(native_world)
    
    elapsed_time = time.time() - start_time
    
    # 计算最终指标
    stats = policy.get_stats()
    
    results = {
        "total_steps": n_steps,
        "elapsed_time_s": elapsed_time,
        "avg_wait_time": total_wait_time / n_steps,
        "total_transported": total_transported,
        "policy_stats": stats,
        "step_results": step_results,
    }
    
    logger.info("=" * 60)
    logger.info("评估完成!")
    logger.info(f"  总步数: {n_steps}")
    logger.info(f"  运行时间: {elapsed_time:.2f}s")
    logger.info(f"  平均等待: {results['avg_wait_time']:.2f}")
    logger.info(f"  RL决策比例: {stats['rl_ratio']:.2%}")
    logger.info(f"  LLM介入次数: {stats['llm_interventions']}")
    logger.info(f"  平均延迟: {stats['avg_latency_ms']:.2f}ms")
    logger.info("=" * 60)
    
    return results


def reset_policy_state(policy: HierarchicalLLMPolicy, reset_stats: bool = False) -> None:
    policy.decision_history.clear()
    policy._clear_cached_plan()
    policy._policy_logits = None
    if policy._dispatch_runtime is not None:
        policy._dispatch_runtime.reset()
    policy.cached_plan_steps = []
    policy.cached_plan_index = 0
    if hasattr(policy, "cached_plan_trigger_type"):
        policy.cached_plan_trigger_type = None
    policy.current_step = 0
    policy.last_intervention_step = -float('inf')
    if hasattr(policy, "suppress_strategy_until_step"):
        policy.suppress_strategy_until_step = 0

    if reset_stats:
        for key, value in policy.stats.items():
            if isinstance(value, (int, float)):
                policy.stats[key] = type(value)()


def run_hierarchical_benchmark(
    *,
    env,
    policy: HierarchicalLLMPolicy,
    num_steps: int,
    n_episodes: int,
    policy_label: str,
    execution_output: str | None = None,
    result_context: Dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    all_results = []
    all_station_frames = []
    all_agent_load_frames = []

    total_waiting_time = 0.0
    total_boarded = 0.0
    headway_stds = []
    rl_ratios = []
    llm_interventions = []
    latencies_ms = []
    physical_episodes = []
    if execution_output:
        os.makedirs(execution_output, exist_ok=True)

    for episode in range(n_episodes):
        logger.info("\n=== Episode %s/%s ===", episode + 1, n_episodes)
        reset_policy_state(policy, reset_stats=True)

        tensordict_obs = env.reset()
        episode_writer = None
        current_world = env.base_env.scenario.world
        if execution_output and getattr(current_world, "second_precision", False):
            context = result_context or {}
            directory = execution_output if n_episodes == 1 else os.path.join(execution_output, f"episode_{episode:03d}")
            episode_writer = EpisodeResults(directory, current_world, method=context.get("method", "when2ask"),
                case_id=context.get("case_id", f"topology{getattr(current_world, 'topology', '')}"),
                requested_stages=num_steps, episode=episode, entry=context.get("entry"), recipe=context.get("recipe"),
                configuration=experiment_configuration(policy.config), source_sha256=context.get("source_sha256"))
        initial_state_sha256 = None
        if getattr(env.base_env.scenario.world, "dispatch", None) is not None:
            import hashlib
            from MILP.operation_model import state_guard
            initial_state_sha256 = hashlib.sha256(json.dumps(
                state_guard(env.base_env.scenario.world), sort_keys=True).encode()).hexdigest()
        station_recorder = StationEventRecorder(env.base_env, policy_label)
        agent_load_recorder = AgentLoadRecorder(env, policy_label)
        metric_recorder = GlobalMetricRecorder(env)
        episode_results = []
        physical_cost = physical_waiting = physical_onboard = physical_headway = physical_operations = 0.

        pbar = tqdm(range(num_steps), desc=f"Benchmark {policy_label} E{episode + 1}")
        try:
            for step in pbar:
                decision_start = time.perf_counter()
                with torch.no_grad():
                    action_td = policy(tensordict_obs)
                decision_time = time.perf_counter() - decision_start

                tensordict_result = env.step(action_td)
                policy.notify_step_result(tensordict_result)
                reward = tensordict_result.get(("next", "agents", "reward")).mean().item()
                done = tensordict_result.get(("next", "done"), torch.tensor(False)).any().item()

                step_metrics = metric_recorder.update()
                station_recorder.record(step)
                agent_load_recorder.record(step)

                last_record = policy.decision_history[-1] if policy.decision_history else None
                decision_source = last_record.layer.value if last_record is not None else "unknown"

                row = {
                    "policy": policy_label,
                    "step": step,
                    "reward": reward,
                    "decision_time_sec": decision_time,
                    "decision_source": decision_source,
                    "waiting_passengers": step_metrics["waiting_count"],
                    "boarded_passengers": step_metrics["boarded_count"],
                }
                current_world = env.base_env.scenario.world
                if getattr(current_world, "dispatch", None) is not None:
                    objective = current_world.dispatch.objective
                    interval = current_world.dispatch.interval_cost
                    terminal = objective.terminal(current_world)["total"] if done or step == num_steps-1 else 0.
                    gamma = objective.weights["discount"]
                    physical_cost += gamma**step*interval["total"] + gamma**(step+1)*terminal
                    physical_waiting += interval["waiting_person_seconds"]
                    physical_onboard += interval["onboard_person_seconds"]
                    physical_headway += interval["headway_deviation_seconds"]
                    physical_operations += interval["operation_units"]
                    row.update(interval_cost=interval["total"], terminal_cost=terminal,
                        waiting_person_seconds=interval["waiting_person_seconds"],
                        onboard_person_seconds=interval["onboard_person_seconds"],
                        passenger_person_seconds=interval["passenger_person_seconds"],
                        headway_deviation_seconds=interval["headway_deviation_seconds"],
                        operation_units=interval["operation_units"])
                if last_record is not None and last_record.coordination:
                    native = last_record.coordination
                    row.update(trigger_type=last_record.trigger_type,
                        coordination_source=native["source"], budget_exceeded=native["budget_exceeded"],
                        allowed_delay_exceeded=native["allowed_delay_exceeded"],
                        coordination_timings_json=json.dumps(native["timings_s"], sort_keys=True),
                        execution_audit_json=json.dumps(native.get("execution_audit", {}), sort_keys=True),
                        reference_policy=native.get("reference_policy", "marl"),
                        trigger_mode=native["experiment"]["trigger_mode"],
                        coordination_scope=native["experiment"]["coordination_scope"],
                        optimization_mode=native["experiment"]["optimization_mode"],
                        coordination_requested=native["requested"], strictly_improved=native["strictly_improved"],
                        safety_override=native.get("safety_override", False),
                        focus_vehicles=len(native.get("focus", {}).get("resolved", [])),
                        planning_horizon=native.get("horizon"), solver_attempts=len(native["attempts"]),
                        pending_service_review=native.get("pending_service_review", False),
                        reference_service_review_required=native.get("reference_service_review_required", False),
                        regional_request_json=json.dumps(native.get("regional_request", {}), sort_keys=True),
                        horizon_policy_json=json.dumps(native.get("horizon_policy", {}), sort_keys=True),
                        scope_expansion_json=json.dumps(native.get("scope_expansion", {}), sort_keys=True),
                        coordination_error_count=len(native["errors"]),
                        skipped_stage_count=len(native.get("skipped_stages", [])),
                        physical_candidate_evaluations=sum(a.get("physical_evaluations", 0) or 0 for a in native["attempts"]))
                if n_episodes > 1:
                    row["episode"] = episode
                episode_results.append(row)
                if episode_writer is not None:
                    native = last_record.coordination if last_record is not None and last_record.coordination else {}
                    episode_writer.annotate(decision_s=decision_time,
                        budget_s=getattr(policy.config, "coordination_time_budget_s", None),
                        metadata=dict(decision_source=decision_source, coordination=native),
                        audit=native.get("execution_audit"),
                        fallback=bool(native.get("requested")) and native.get("source") == "checked_fallback",
                        strictly_improved=native.get("strictly_improved", False))
                if execution_output and getattr(current_world, "dispatch", None) is not None:
                    # Expose executed physical progress, not overlapping forecast gains.
                    from pathlib import Path
                    from LLM.run_experiment_suite import _atomic_json
                    interval_total = physical_cost-gamma**(step+1)*terminal
                    terminal_now = objective.terminal(current_world)["total"]
                    progress_directory = Path(execution_output)/"diagnostics"
                    progress_directory.mkdir(exist_ok=True)
                    _atomic_json(progress_directory/"benchmark_progress.json", dict(
                        episode=episode, completed_stages=step+1, target_stages=num_steps,
                        simulation_seconds=float(current_world.sim_time[0]),
                        realized_interval_cost=interval_total,
                        terminal_if_stopped_now=terminal_now,
                        cost_if_stopped_now=interval_total+gamma**(step+1)*terminal_now,
                        waiting_person_seconds=physical_waiting,
                        onboard_person_seconds=physical_onboard,
                        passenger_person_seconds=physical_waiting+physical_onboard,
                        headway_deviation_seconds=physical_headway, operation_units=physical_operations,
                        strict_improvements=sum(bool(r.get("strictly_improved")) for r in episode_results),
                        full_episode_complete=(step+1==240 and num_steps==240),
                        scope="executed prefix; partial cost is not a complete-episode benchmark result"))

                tensordict_obs = tensordict_result.get("next").clone()
                if done:
                    logger.info("回合在第 %s 步结束。", step)
                    break
        except Exception as exc:
            if episode_writer is not None:
                episode_writer.finish(status="error", error=repr(exc), diagnostics=dict(steps=episode_results))
            raise
        finally:
            pbar.close()

        metrics = metric_recorder.get_metrics()
        metrics["avg_headway_std"] = station_recorder.get_headway_std()
        stats = policy.get_stats()

        total_waiting_time += float(metrics["total_waiting_time"])
        total_boarded += float(metrics["total_boarded"])
        headway_stds.append(float(metrics["avg_headway_std"]))
        rl_ratios.append(float(stats.get("rl_ratio", 0.0)))
        llm_interventions.append(float(stats.get("llm_interventions", 0.0)))
        latencies_ms.append(float(stats.get("avg_latency_ms", 0.0)))

        result_df = pd.DataFrame(episode_results)
        world = env.base_env.scenario.world
        if getattr(world, "dispatch", None) is not None:
            physical_episodes.append(dict(episode=episode, cost=physical_cost, completed_stages=len(episode_results),
                initial_state_sha256=initial_state_sha256,
                waiting_person_seconds=physical_waiting, headway_deviation_seconds=physical_headway,
                onboard_person_seconds=physical_onboard, passenger_person_seconds=physical_waiting+physical_onboard,
                operation_units=physical_operations, terminal=world.dispatch.objective.terminal(world),
                objective_definition=world.dispatch.objective.manifest(),
                metrics_scope="same realized physical episode; direct interval and terminal costs, not reconstructed rewards"))
        if episode_writer is not None:
            episode_writer.finish(diagnostics=dict(physical=physical_episodes[-1], steps=episode_results, policy_stats=stats),
                                  computation_time_s=sum(r["decision_time_sec"] for r in episode_results))
        elif execution_output and getattr(world, "execution_records", None) is not None:
            world.execution_records.write(os.path.join(execution_output, f"episode_{episode}_execution"), world)
        station_df = station_recorder.to_frame()
        agent_load_df = agent_load_recorder.to_frame()

        if n_episodes > 1:
            if not result_df.empty:
                result_df["episode"] = episode
            if not station_df.empty:
                station_df["episode"] = episode
            if not agent_load_df.empty:
                agent_load_df["episode"] = episode

        all_results.append(result_df)
        all_station_frames.append(station_df)
        all_agent_load_frames.append(agent_load_df)

    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    station_df = pd.concat(all_station_frames, ignore_index=True) if all_station_frames else pd.DataFrame()
    agent_load_df = pd.concat(all_agent_load_frames, ignore_index=True) if all_agent_load_frames else pd.DataFrame()

    metrics = {
        "total_waiting_time": total_waiting_time,
        "total_boarded": total_boarded,
        "avg_waiting_time": (total_waiting_time / total_boarded) if total_boarded > 0 else 0.0,
        "avg_headway_std": float(np.mean(headway_stds)) if headway_stds else 0.0,
        "avg_rl_ratio": float(np.mean(rl_ratios)) if rl_ratios else 0.0,
        "avg_llm_interventions": float(np.mean(llm_interventions)) if llm_interventions else 0.0,
        "avg_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
    }
    summary = {
        "topology": getattr(env.base_env, "topology", None),
        "n_episodes": n_episodes,
        "n_steps_per_episode": num_steps,
        "avg_waiting_time": metrics["avg_waiting_time"],
        "avg_headway_std": metrics["avg_headway_std"],
        "avg_rl_ratio": metrics["avg_rl_ratio"],
        "avg_llm_interventions": metrics["avg_llm_interventions"],
        "avg_latency_ms": metrics["avg_latency_ms"],
    }
    if physical_episodes:
        shared = {key: float(np.mean([row[key] for row in physical_episodes])) for key in (
            "cost", "waiting_person_seconds", "onboard_person_seconds", "passenger_person_seconds",
            "headway_deviation_seconds", "operation_units")}
        metrics.update(shared)
        summary.update(shared, physical_episodes=physical_episodes,
                       objective_source="DispatchObjective interval costs plus final terminal cost")
    if "coordination_requested" in results_df:
        summary["coordination_diagnostics"] = dict(
            requests=int(results_df["coordination_requested"].sum()),
            strictly_improved_decisions=int(results_df["strictly_improved"].sum()),
            solver_attempts=int(results_df["solver_attempts"].sum()),
            physical_candidate_evaluations=int(results_df["physical_candidate_evaluations"].sum()),
            decisions_with_new_evaluations=int((results_df["physical_candidate_evaluations"] > 0).sum()),
            provider_or_solver_errors=int(results_df["coordination_error_count"].sum()),
            skipped_stages=int(results_df["skipped_stage_count"].sum()),
            safety_overrides=int(results_df["safety_override"].sum()),
            service_review_requests=int(results_df["reference_service_review_required"].sum()),
            budget_exceedances=int(results_df["budget_exceeded"].sum()),
            sources=results_df["coordination_source"].value_counts().to_dict())
    if hasattr(policy, "config"):
        summary["experiment_configuration"] = experiment_configuration(policy.config)
    return results_df, station_df, agent_load_df, metrics, summary


def save_benchmark_outputs(
    *,
    args,
    results_df: pd.DataFrame,
    station_df: pd.DataFrame,
    agent_load_df: pd.DataFrame,
    metrics: Dict[str, float],
    summary: Dict[str, Any],
) -> None:
    diagnostics_dir = os.path.join(args.output_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)

    output_path = args.output or os.path.join(diagnostics_dir, "benchmark_results.csv")
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("分层基准结果已保存至 %s", output_path)

    station_output_path = args.station_output or os.path.join(diagnostics_dir, "legacy_station_events.csv")
    station_df.to_csv(station_output_path, index=False, encoding="utf-8-sig")
    logger.info("站点事件记录已保存至 %s", station_output_path)

    agent_load_output_path = args.agent_load_output or os.path.join(diagnostics_dir, "agent_loads.csv")
    agent_load_df.to_csv(agent_load_output_path, index=False, encoding="utf-8-sig")
    logger.info("Agent载客数记录已保存至 %s", agent_load_output_path)

    metrics_output_path = os.path.join(diagnostics_dir, "legacy_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_output_path, index=False, encoding="utf-8-sig")
    logger.info("全局指标已保存至 %s", metrics_output_path)

    summary_path = os.path.join(diagnostics_dir, "benchmark_summary.json")
    recipe = (getattr(args, "scenario_config_resolved", None) or {}).get("scenario_recipe")
    if recipe is not None:
        summary.update(scenario_recipe=recipe, scenario_sha256=recipe["sha256"],
                       case_id=args.scenario_config_resolved["scenario_name"], episode_seed=args.seed)
    summary["source_sha256"] = getattr(args, 'experiment_source_sha256', {})
    summary["source_changed"] = summary["source_sha256"] != experiment_fingerprints()
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("分层摘要已保存至 %s", summary_path)


def run_demo(
    env,
    policy: HierarchicalLLMPolicy,
    n_steps: int = 100,
):
    """运行演示模式"""
    logger.info("=" * 60)
    logger.info("分层RL-LLM Agent 演示模式")
    logger.info("=" * 60)
    
    td = env.reset()
    
    for step in range(n_steps):
        print(f"\n--- Step {step + 1} ---")
        
        # 获取决策
        td = policy(td)
        
        # 打印决策信息
        if policy.decision_history:
            last_record = policy.decision_history[-1]
            print(f"决策层: {last_record.layer.value}")
            print(f"触发类型: {last_record.trigger_type or 'None'}")
            print(f"置信度: {last_record.confidence:.2f}")
            print(f"延迟: {last_record.latency_ms:.2f}ms")
            
            # 打印部分动作
            actions = last_record.actions
            for i, (agent, action) in enumerate(actions.items()):
                if i >= 3:
                    print(f"  ... ({len(actions) - 3} more agents)")
                    break
                reason = last_record.reasoning.get(agent, "")
                print(f"  {agent}: action={action}, {reason[:50]}...")
        
        # 执行动作
        td = env.step(td)
        policy.notify_step_result(td)
        
        # 简单交互
        try:
            input("\n按Enter继续, Ctrl+C退出...")
        except KeyboardInterrupt:
            print("\n演示结束")
            break
        
        td = env.reset() if td.get("done", False) else td


def run_benchmark(
    env,
    policy: HierarchicalLLMPolicy,
    n_episodes: int = 5,
    n_steps: int = 500,
    output_dir: str = "./results",
) -> Dict[str, Any]:
    """保留兼容接口，实际 benchmark 输出由 run_hierarchical_benchmark 提供。"""
    logger.info(f"开始基准测试: {n_episodes} episodes x {n_steps} steps")
    _, _, _, metrics, summary = run_hierarchical_benchmark(
        env=env,
        policy=policy,
        num_steps=n_steps,
        n_episodes=n_episodes,
        policy_label="Hierarchical_RAG_SmallGuidance",
    )
    logger.info("Benchmark Metrics:")
    logger.info(f"  Average Passenger Waiting Time: {metrics['avg_waiting_time']:.2f} seconds")
    logger.info(f"  Total Boarded Passengers: {metrics['total_boarded']}")
    logger.info(f"  Average Headway Standard Deviation: {metrics['avg_headway_std']:.2f} seconds")
    logger.info(f"  平均RL决策比例: {summary['avg_rl_ratio']:.2%}")
    logger.info(f"  平均LLM介入次数: {summary['avg_llm_interventions']:.1f}")
    return summary


def experiment_fingerprints():
    files = ("LLM/main_hierarchical.py", "LLM/hierarchical_llm_policy.py", "LLM/dispatch_coordination.py",
             "LLM/rag/enhanced_rag_manager.py", "LLM/rag/native_case.py",
             "LLM/dispatch_scope.py", "LLM/experiment_controls.py", "LLM/coupling_trigger.py",
             "LLM/strategic_api.py", "LLM/small_model_guidance.py", "MILP/guidance_labels.py", "MILP/parallel_evaluation.py",
             "LLM/vllm_pool.py", "LLM/serve_vllm.py",
             "MILP/milp_model.py", "MILP/operation_model.py", "MILP/staged_dispatch.py",
             "MILP/collection_protocol.py", "RL/dispatch_training.py",
             "metro_world.py", "dispatch_operations.py", "dispatch_plan.py", "dispatch_candidates.py", "dispatch_objective.py",
             "metro_environment.py", "metro_rl_env.py", "envwrapper.py", "metro_scenario_v2.py",
             "dispatch_timing.py", "dispatch_records.py", "experiment_results.py", "dispatch_diagnostics.py",
             "topology/metro_lines.py", "scenarios/prepare_dispatch.py")
    result = {}
    for name in files:
        with open(os.path.join(PROJECT_ROOT, name), 'rb') as stream:
            result[name] = hashlib.sha256(stream.read()).hexdigest()
    return result


def main():
    """主函数"""
    args = parse_args()
    args.experiment_source_sha256 = experiment_fingerprints()
    if args.torch_threads < 1:
        raise ValueError('--torch-threads must be positive')
    torch.set_num_threads(args.torch_threads)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    args.scenario_config_resolved = scenario_config_from_args(args)
    if args.dispatch_scenario:
        if args.events_file or args.scenario_config:
            raise ValueError('Use --dispatch-scenario or an external scenario file, not both')
        args.scenario_config_resolved.update(runtime_scenario(args.topology, args.dispatch_scenario,
            split=args.scenario_split, repeat=args.scenario_repeat, seed=args.scenario_seed))
        # Match the episode seed used by training and offline collection.
        args.seed = args.scenario_seed + SPLIT_OFFSETS[args.scenario_split] + args.scenario_repeat
    if args.dispatch_config:
        args.scenario_config_resolved["dispatch_config"] = args.dispatch_config
    if args.events_file:
        args.scenario_config_resolved["events_file"] = args.events_file
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    logger.info("=" * 60)
    logger.info("分层RL-LLM Agent地铁调度系统")
    logger.info("=" * 60)
    logger.info(f"运行模式: {args.mode}")
    logger.info(f"拓扑: {args.topology}")

    resolved_n_agents = args.n_agents or get_default_n_agents(args.topology)
    if resolved_n_agents is None:
        raise ValueError(f"未找到 topology {args.topology} 对应的默认列车数")
    logger.info(f"列车数: {resolved_n_agents}")

    if not args.rl_model_path and args.reference_policy == 'marl':
        args.rl_model_path = resolve_default_rl_model_path(args.topology)
        if args.rl_model_path:
            logger.info(f"未显式提供RL模型，自动使用: {args.rl_model_path}")
        else:
            logger.warning("未找到 topology %s 对应的默认RL模型，将退化为默认停留策略", args.topology)
    
    # 创建环境
    logger.info("创建环境...")
    try:
        env, resolved_n_agents = build_env(
            topology=args.topology,
            n_steps=args.n_steps,
            seed=args.seed,
            n_agents=resolved_n_agents,
            scenario_config=args.scenario_config_resolved,
        )
        logger.info("环境创建成功")
    except Exception as e:
        logger.error(f"环境创建失败: {e}")
        if args.require_components or args.dispatch_config or args.topology in LINE_SPECS:
            raise RuntimeError("Native dispatch evaluation requires a real configured environment") from e
        logger.info("使用模拟环境...")
        env = None
    
    # 构建配置
    config = build_config(args)
    logger.info(f"MILP规划时间域: {config.milp_time_horizon}")
    logger.info(f"MILP求解时限: {config.milp_time_limit}s")
    logger.info(f"缓存计划步数: {config.cached_plan_max_steps}")
    logger.info(f"MILP接受门控: {config.milp_acceptance_gate}")
    if config.always_hierarchical:
        logger.info("消融模式: When2Ask trigger 已关闭，保留 hierarchical 内部 plan cache")
    if not config.use_rag:
        logger.info("消融模式: RAG 模块已关闭")
    
    # 创建策略
    logger.info("创建分层策略...")
    device = resolve_policy_device(args)
    logger.info("MARL device: %s; SLM backend: %s", device, args.small_model_backend)
    policy = HierarchicalLLMPolicy(
        env=env,
        config=config,
        n_agents=resolved_n_agents,
        device=device,
    )
    if args.require_components:
        checks = component_checks(policy, config)
        if not all(checks.values()):
            raise RuntimeError(f"Requested complete method has missing components: {checks}")
    
    # 开启 LLM IO 日志
    if getattr(args, "log_llm_io", None):
        policy.set_io_log_file(args.log_llm_io)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行
    if args.mode == 'eval':
        if env is None:
            logger.error("评估模式需要有效的环境")
            return
        results = run_evaluation(env, policy, args.n_steps, args.verbose, execution_output=args.output_dir)
        
        if args.save_history:
            history_path = os.path.join(args.output_dir, "decision_history.json")
            policy.export_decision_history(history_path)
    
    elif args.mode == 'demo':
        if env is None:
            logger.error("演示模式需要有效的环境")
            return
        run_demo(env, policy, args.n_steps)
    
    elif args.mode == 'benchmark':
        if env is None:
            logger.error("基准测试需要有效的环境")
            return
        results_df, station_df, agent_load_df, metrics, summary = run_hierarchical_benchmark(
            env=env,
            policy=policy,
            num_steps=args.n_steps,
            n_episodes=args.episodes,
            policy_label="Hierarchical_RAG_SmallGuidance",
            execution_output=args.output_dir,
            result_context=dict(case_id=args.scenario_config_resolved.get("scenario_name") or f"topology{args.topology}",
                entry=dict(seed=args.seed), recipe=args.scenario_config_resolved.get("scenario_recipe"),
                source_sha256=args.experiment_source_sha256),
        )
        save_benchmark_outputs(
            args=args,
            results_df=results_df,
            station_df=station_df,
            agent_load_df=agent_load_df,
            metrics=metrics,
            summary=summary,
        )
        if args.save_history:
            history_path = os.path.join(args.output_dir, "diagnostics", "decision_history.json")
            policy.export_decision_history(history_path)
    
    elif args.mode == 'train':
        logger.info("训练模式尚未实现，请使用现有的MARL训练脚本")
        logger.info("然后将训练好的模型路径通过 --rl_model_path 传入")
    
    logger.info("运行结束")


if __name__ == "__main__":
    main()
