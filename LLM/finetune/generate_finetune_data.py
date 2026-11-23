"""
Generate SFT + RL Fine-tuning Data for Small Model Guidance

通过 MILP vs MARL 动作对比，生成小规模语言模型的 SFT 微调数据。
小模型学习输出 **求解引导**（focus_agents, warm_start, variable_fixing, strategy_reasoning），
而非直接输出调度决策。

参考 generate_solver_learning_data.py 的 Ground Truth 方法：
- focus_agents = MILP 动作 ≠ MARL 动作的列车
- warm_start / variable_fixing 来自 MILP 求解状态
- 附带 reward 信号支持后续 RL 训练

产物：
- SFT JSONL（ChatML 格式，适配 LLaMA-Factory / OpenAI fine-tuning）
- RL JSONL（含 reward 字段，供 OnlineLearner 使用）

Usage:
    python generate_finetune_data.py --topology 4 --steps 7200 --episodes 1
    python generate_finetune_data.py --topology 4 --steps 7200 --load-trajectory traj.pkl
"""
import sys
import os
import json
import math
import torch
import random
import argparse
import numpy as np
import pickle
from collections import deque
from datetime import datetime
from typing import Dict, List, Set, Any, Optional, Tuple
from tqdm import tqdm
from torchrl.envs import TransformedEnv, RewardSum
from tensordict import TensorDict

# --- 项目路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

from metro_rl_env import MetroRLEnvV2
from Params import rl_parser, apply_topology_defaults, get_default_tick_seconds, get_default_n_agents
from logger import logger
import linedata
from train_run_config import TrainRunConfig
from MILP.milp_ground_truth_benchmark import FastMILPSolver

# ========================================================================
# 常量：小模型 guidance 的 system prompt（SFT 训练与推理共用）
# ========================================================================
GUIDANCE_SYSTEM_PROMPT = """你是一个地铁调度MILP求解加速专家。根据当前网络状态，输出求解引导。

**任务**:
1. 识别 focus_agents: 哪些列车需要 MILP 优化（其最优动作可能与常规 MARL 策略不同）
2. 识别 focus_nodes: 关键节点范围
3. 提供 warm_start 和 variable_fixing 加速 MILP 求解
4. 给出策略推理说明

**focus_agents 判断依据**:
- 在中断区或邻近中断的列车
- 在高等待站点需要决策的列车
- 位置接近可能冲突的列车

**输出格式 (严格 JSON)**:
{
  "focus_agents": [0, 2],
  "focus_nodes": [13, 14, 15],
  "warm_start": {"stay[0,13,0]": 1},
  "variable_fixing": {"stay[0,13,0]": 1},
  "strategy_reasoning": "列车0邻近中断需等待，列车2在高等待站点"
}

**变量命名规则**:
- stay[agent_idx, node_idx, t]: 列车在节点停留
- x[agent_idx, node_idx, t]: 列车在节点位置
- 只用 stay 和 x 变量，不用 move 变量
- 只有站台节点才能 stay=1"""


# ========================================================================
# 辅助函数
# ========================================================================

def _safe_tensor_item(tensor, index: int = 0, default: float = 0.0) -> float:
    """安全地从 tensor 中取标量值。"""
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


def format_state_for_guidance(world, time_horizon: int = 20) -> str:
    """将环境状态格式化为小模型输入（与 generate_solver_learning_data 对齐）。

    不包含 FOCUS 标记——focus_agents 是模型需要预测的输出。
    """
    agents = world.agents
    platform_nodes = world.platform_nodes
    nodes = world.nodes
    env_index = 0

    node_ids = sorted(nodes.keys())
    node_index = {nid: idx for idx, nid in enumerate(node_ids)}

    lines: List[str] = []

    # 问题规模
    lines.append("### 问题规模")
    lines.append(f"- 列车数量: {len(agents)}")
    lines.append(f"- 节点数量: {len(nodes)}")
    lines.append(f"- 站台数量: {len(platform_nodes)}")
    lines.append(f"- 规划horizon: {time_horizon}")

    # 中断信息
    disrupted_segments = world.get_disrupted_segments()
    disrupted_nodes: Set[int] = set()

    if disrupted_segments:
        blockages, speed_limits = [], []
        for seg_id, info in disrupted_segments.items():
            dtype = info.get('type', 'UNKNOWN')
            try:
                parts = seg_id.split("-")
                if len(parts) >= 2:
                    sn, en = int(parts[0]), int(parts[1])
                    if dtype in ('BLOCK', 'BLOCKAGE'):
                        disrupted_nodes.update([sn, en])
                        blockages.append((seg_id, sn, en))
                    elif dtype == 'SPEED_LIMIT':
                        speed_limits.append((seg_id, sn, en, info.get('speed_factor', info.get('factor', 2.0))))
            except (ValueError, IndexError):
                pass

        if blockages:
            lines.append("\n### 区间中断 (BLOCK)")
            for seg_id, sn, en in blockages:
                lines.append(f"- {seg_id}: 节点{sn}(idx={node_index.get(sn,-1)}) <-> {en}(idx={node_index.get(en,-1)})")
        if speed_limits:
            lines.append("\n### 限速区段")
            for seg_id, sn, en, factor in speed_limits:
                lines.append(f"- {seg_id}: 减速{factor}倍")

    # 列车状态
    lines.append("\n### 列车状态")
    for idx, agent in enumerate(agents):
        pos = int(agent.state.position[env_index].item())
        pos_idx = node_index.get(pos, -1)
        passengers = int(_safe_tensor_item(agent.passenger_count, env_index, 0))
        dwell = _safe_tensor_item(getattr(agent.state, "dwell_time", None), env_index, 0)
        expected_dwell = _safe_tensor_item(
            getattr(agent.state, "current_expected_dwell_time", None), env_index, 0
        )

        markers: List[str] = []
        if pos in platform_nodes:
            node_type = "站台"
            remaining = max(0, int(expected_dwell - dwell))
            markers.append(f"停留中{int(dwell)}/{int(expected_dwell)}" if remaining > 0 else "可发车")
        else:
            node_type = "区间"

        if pos in disrupted_nodes:
            markers.append("在中断区")

        next_nodes = world.get_next_nodes(pos)
        for n in next_nodes:
            for seg_key in [f"{pos}-{n.id}", f"{n.id}-{pos}"]:
                info = disrupted_segments.get(seg_key) if disrupted_segments else None
                if info and info.get('type', '') in ('BLOCK', 'BLOCKAGE'):
                    markers.append("邻近中断")
                    break
            else:
                continue
            break

        marker_str = f" [{', '.join(markers)}]" if markers else ""
        lines.append(f"- 列车{idx}: {node_type} 位置{pos}(idx={pos_idx}), 乘客={passengers}{marker_str}")

    # 站台等待
    lines.append("\n### 站台等待")
    high_wait = []
    for nid, node in sorted(platform_nodes.items()):
        w = _safe_tensor_item(node.waiting_passengers, env_index, 0)
        if w > 100:
            high_wait.append((nid, node_index.get(nid, -1), int(w)))
    if high_wait:
        high_wait.sort(key=lambda x: x[2], reverse=True)
        for nid, nidx, cnt in high_wait[:10]:
            lvl = "高" if cnt > 500 else "中"
            lines.append(f"- 节点{nid}(idx={nidx}): {cnt}人 [{lvl}]")
    else:
        lines.append("- 无高等待站点")

    return "\n".join(lines)


def identify_focus_agents(
    world,
    milp_actions_dict: Dict[str, int],
    marl_actions_list: List[int],
    high_wait_threshold: float = 500,
) -> Tuple[List[int], List[int], Dict[int, str]]:
    """通过 MILP vs MARL 动作对比识别 focus_agents（与 generate_solver_learning_data 逻辑一致）。

    Returns:
        focus_agents: 需要 MILP 优化的列车索引
        focus_nodes: 关键节点索引列表
        agent_reasons: 每个 focus_agent 的原因说明
    """
    agents = world.agents
    platform_nodes = world.platform_nodes
    nodes = world.nodes
    env_index = 0

    node_ids = sorted(nodes.keys())
    node_index = {nid: idx for idx, nid in enumerate(node_ids)}

    disrupted_segments = world.get_disrupted_segments() or {}
    disrupted_nodes: Set[int] = set()
    for seg_id, info in disrupted_segments.items():
        if info.get('type', '') in ('BLOCK', 'BLOCKAGE'):
            try:
                parts = seg_id.split("-")
                if len(parts) >= 2:
                    disrupted_nodes.update([int(parts[0]), int(parts[1])])
            except (ValueError, IndexError):
                pass

    focus_agents: List[int] = []
    focus_nodes: Set[int] = set()
    agent_reasons: Dict[int, str] = {}

    # 中断节点 → focus_nodes
    for nid in disrupted_nodes:
        if nid in node_index:
            focus_nodes.add(node_index[nid])

    # 高等待站点 → focus_nodes
    for nid, node in platform_nodes.items():
        w = _safe_tensor_item(node.waiting_passengers, env_index, 0)
        if w > high_wait_threshold and nid in node_index:
            focus_nodes.add(node_index[nid])

    action_names = {0: "停留", 1: "方向1", 2: "方向2", 3: "方向3"}

    for idx, agent in enumerate(agents):
        milp_action = milp_actions_dict.get(agent.name, 0)
        marl_action = marl_actions_list[idx] if idx < len(marl_actions_list) else 0
        if milp_action == marl_action:
            continue

        pos = int(agent.state.position[env_index].item())
        pos_idx = node_index.get(pos, -1)
        reasons = [
            f"动作差异(MILP={action_names.get(milp_action, str(milp_action))}, "
            f"MARL={action_names.get(marl_action, str(marl_action))})"
        ]
        focus_nodes.add(pos_idx)

        if pos in disrupted_nodes:
            reasons.append("在中断区")
        next_nodes = world.get_next_nodes(pos)
        for n in next_nodes:
            for seg_key in [f"{pos}-{n.id}", f"{n.id}-{pos}"]:
                info = disrupted_segments.get(seg_key)
                if info and info.get('type', '') in ('BLOCK', 'BLOCKAGE'):
                    reasons.append("邻近中断")
                    if n.id in node_index:
                        focus_nodes.add(node_index[n.id])
                    break
            else:
                continue
            break
        if pos in platform_nodes:
            w = _safe_tensor_item(platform_nodes[pos].waiting_passengers, env_index, 0)
            if w > high_wait_threshold:
                reasons.append(f"高等待({int(w)}人)")

        focus_agents.append(idx)
        agent_reasons[idx] = ", ".join(reasons)

    return focus_agents, sorted(focus_nodes), agent_reasons


def build_warm_start_from_state(
    world, focus_agents: List[int]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """根据当前环境状态为 focus_agents 构建 warm_start 和 variable_fixing 提示。"""
    platform_nodes = world.platform_nodes
    nodes = world.nodes
    node_ids = sorted(nodes.keys())
    node_index = {nid: idx for idx, nid in enumerate(node_ids)}
    env_index = 0
    focus_set = set(focus_agents)

    warm_start: Dict[str, int] = {}
    variable_fixing: Dict[str, int] = {}

    for idx, agent in enumerate(world.agents):
        if idx not in focus_set:
            continue
        pos = int(agent.state.position[env_index].item())
        pos_idx = node_index.get(pos, -1)
        if pos_idx < 0:
            continue

        # t=0 时的 stay / x 变量
        x_var = f"x[{idx},{pos_idx},0]"
        warm_start[x_var] = 1

        if pos in platform_nodes:
            dwell = _safe_tensor_item(
                getattr(agent.state, "dwell_time", None), env_index, 0
            )
            expected = _safe_tensor_item(
                getattr(agent.state, "current_expected_dwell_time", None), env_index, 0
            )
            stay_var = f"stay[{idx},{pos_idx},0]"
            if dwell < expected:
                # 停站未满 → 一定停留
                warm_start[stay_var] = 1
                variable_fixing[stay_var] = 1
            else:
                warm_start[stay_var] = 0

    return warm_start, variable_fixing


def create_sft_record(
    world,
    focus_agents: List[int],
    focus_nodes: List[int],
    agent_reasons: Dict[int, str],
    warm_start: Dict[str, int],
    variable_fixing: Dict[str, int],
    reward: float,
    episode: int,
    step: int,
    has_disruption: bool,
    time_horizon: int = 20,
) -> dict:
    """创建一条 ChatML 格式的 SFT 训练记录。

    格式与 generate_solver_learning_data.create_ground_truth_record 对齐
    """
    user_content = format_state_for_guidance(world, time_horizon)

    reasoning_parts = [
        f"列车{idx}: {agent_reasons.get(idx, '需要优化')}"
        for idx in focus_agents
    ]

    guidance = {
        "focus_agents": focus_agents,
        "focus_nodes": focus_nodes,
        "warm_start": warm_start,
        "variable_fixing": variable_fixing,
        "strategy_reasoning": (
            "; ".join(reasoning_parts) if reasoning_parts else "无需特殊优化"
        ),
    }

    assistant_content = json.dumps(guidance, ensure_ascii=False, indent=2)

    return {
        "messages": [
            {"role": "system", "content": GUIDANCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "episode": episode,
            "step": step,
            "has_disruption": has_disruption,
            "num_focus_agents": len(focus_agents),
            "focus_agents": focus_agents,
            "reward": reward,
            "data_quality": "verified",
        },
    }


# ========================================================================
# 加载 MARL 策略（复用 generate_solver_learning_data 的逻辑）
# ========================================================================

def load_marl_policy(topology: int, env, device: torch.device):
    """加载 MARL (MAPPO) 策略用于对比。"""
    try:
        from main_marl import load_rl_policy

        model_files = {
            1: "metro_shenzhenline1_mappo_best.pt",
            **{t: f"metro_topology_{t}_mappo_best.pt" for t in range(4, 17)},
        }
        filename = model_files.get(topology, "metro_topology_mappo_best.pt")
        model_path = os.path.join(project_root, "MARL_model", filename)
        if not os.path.exists(model_path):
            logger.warning(f"MARL模型不存在: {model_path}，将使用默认停留策略")
            return None
        logger.info(f"加载MARL模型: {model_path}")
        return load_rl_policy(model_path, env, device)
    except Exception as e:
        logger.warning(f"加载MARL模型失败: {e}，将使用默认停留策略")
        return None


# ========================================================================
# BFS 路径转动作（复用 generate_solver_learning_data 的逻辑）
# ========================================================================

def _build_bfs_action_resolver(world):
    """构建 BFS 最短路径动作解析器，将 MILP 目标位置转为环境动作。"""
    adjacency = world._next_nodes_id_map

    def bfs_next_hop(start: int, target: int, max_depth: int = 200) -> int:
        if start == target:
            return 0
        neighbors = adjacency.get(start, [])
        if not neighbors:
            return 1
        best_idx, best_dist = 0, float('inf')
        for idx, first_hop in enumerate(neighbors):
            if first_hop == target:
                return idx + 1
            visited = {start, first_hop}
            queue = deque([(first_hop, 1)])
            found_dist = float('inf')
            while queue and len(visited) < max_depth:
                node, dist = queue.popleft()
                if dist >= best_dist:
                    break
                for nn in adjacency.get(node, []):
                    if nn == target:
                        found_dist = dist + 1
                        break
                    if nn not in visited:
                        visited.add(nn)
                        queue.append((nn, dist + 1))
                if found_dist < float('inf'):
                    break
            if found_dist < best_dist:
                best_dist = found_dist
                best_idx = idx
        return best_idx + 1

    return bfs_next_hop


# ========================================================================
# 主数据收集循环
# ========================================================================

def run_guidance_data_collection(
    rl_env,
    milp_solver: Optional[FastMILPSolver],
    marl_policy,
    num_episodes: int,
    max_steps: int,
    device: torch.device,
    decision_interval: int = 1,
    milp_result: Optional[Dict[str, Any]] = None,
    high_wait_threshold: float = 500,
    time_horizon: int = 20,
) -> Tuple[List[dict], List[dict]]:
    """收集 SFT + RL 训练数据。

    核心逻辑：
    1. 使用 MILP 轨迹重放动作
    2. 同步获取 MARL 策略动作
    3. 对比差异 → 识别 focus_agents
    4. 生成 ChatML SFT 记录 + reward 标注 RL 记录

    Returns:
        sft_records: SFT ChatML 记录（含 guidance 输出）
        rl_records: 含 reward 的轨迹记录（供 OnlineLearner 使用）
    """
    sft_records: List[dict] = []
    rl_records: List[dict] = []

    world = rl_env.base_env.scenario.world
    n_agents = len(world.agents)
    agent_names = [a.name for a in world.agents]
    platform_ids = set(world.platform_nodes.keys())

    # --- MILP 轨迹 ---
    if milp_result is None:
        if milp_solver is None:
            logger.error("未提供 MILP 求解器或轨迹")
            return [], []
        logger.info("开始 MILP Ground Truth 求解...")
        milp_result = milp_solver.solve()
        if milp_result.get("status") == "failed":
            logger.error("MILP 求解失败")
            return [], []

    trajectory = milp_result.get("trajectory", {})
    milp_positions = trajectory.get("positions", [])
    milp_actions_traj = trajectory.get("actions", [])
    milp_horizon = trajectory.get("horizon", 0)
    milp_di = trajectory.get("decision_interval", decision_interval)

    logger.info(
        f"MILP 求解完成: horizon={milp_horizon}, "
        f"solve_time={milp_result.get('solve_time_sec', 0):.2f}s"
    )

    # 目标位置表：step → agent_idx → target_node_id
    target_positions: Dict[int, Dict[int, int]] = {}
    for t_step in range(len(milp_positions) - 1):
        target_positions[t_step] = {}
        for a_idx in range(n_agents):
            target_positions[t_step][a_idx] = (
                milp_positions[t_step + 1][a_idx]
                if a_idx < len(milp_positions[t_step + 1])
                else -1
            )

    bfs_next_hop = _build_bfs_action_resolver(world)
    agent_target_platform = [-1] * n_agents

    def _resolve_replay_action(agent_idx, agent, raw_milp_action, milp_target):
        """将 MILP 轨迹的位置目标转为当前环境可执行的动作。"""
        current_pos = int(agent.state.position[0].item())
        is_on_platform = current_pos in platform_ids

        if raw_milp_action > 0 and milp_target > 0:
            agent_target_platform[agent_idx] = milp_target

        target = agent_target_platform[agent_idx]

        if is_on_platform:
            dwell_min = float(agent.state.dwell_min_limit[0].item())
            dwell_max = float(agent.state.dwell_max_limit[0].item())
            current_dwell = float(agent.state.time_since_arrival[0].item())
            if dwell_min > 0 and current_dwell < dwell_min:
                return 0
            if dwell_max > 0 and current_dwell >= dwell_max:
                if target > 0:
                    return bfs_next_hop(current_pos, target)
                return min(raw_milp_action, 3) if raw_milp_action > 0 else 1
            if raw_milp_action == 0:
                return 0
            if target > 0:
                return bfs_next_hop(current_pos, target)
            return min(raw_milp_action, 3)

        # 区间节点：直接 BFS 到目标
        if target > 0:
            return bfs_next_hop(current_pos, target)
        return 1

    # --- Episode 循环 ---
    for ep in range(num_episodes):
        logger.info(f"=== Episode {ep+1}/{num_episodes} ===")
        observations = rl_env.reset()
        ep_sft_count = 0

        pbar = tqdm(range(max_steps), desc=f"GuidanceGen Ep{ep+1}")
        for step in pbar:
            world = rl_env.base_env.scenario.world
            disrupted_segments = world.get_disrupted_segments() or {}
            has_disruption = len(disrupted_segments) > 0
            decision_step = min(step // max(1, milp_di), milp_horizon - 2)

            # 1) MARL 动作
            if marl_policy is not None:
                with torch.no_grad():
                    marl_out = marl_policy(observations.to(device))
                    marl_tensor = marl_out.get(("agents", "action"))
                    if marl_tensor is None:
                        marl_tensor = marl_out.get(rl_env.action_key)
                    if marl_tensor is None:
                        raise RuntimeError(
                            "MARL policy output缺少动作键: ('agents','action') 和 rl_env.action_key 均不存在"
                        )
                    marl_actions_list = marl_tensor.squeeze().tolist()
                    if isinstance(marl_actions_list, (int, float)):
                        marl_actions_list = [int(marl_actions_list)]
            else:
                marl_actions_list = [0] * n_agents

            # 2) MILP 动作（重放轨迹）
            milp_step_actions = (
                milp_actions_traj[decision_step]
                if decision_step < len(milp_actions_traj)
                else [1] * n_agents
            )
            milp_targets = target_positions.get(decision_step, {})

            milp_actions_dict: Dict[str, int] = {}
            for a_idx, a_name in enumerate(agent_names):
                agent = world.agents[a_idx]
                raw_a = (
                    milp_step_actions[a_idx]
                    if a_idx < len(milp_step_actions)
                    else 1
                )
                mt = milp_targets.get(a_idx, -1)
                milp_actions_dict[a_name] = _resolve_replay_action(
                    a_idx, agent, raw_a, mt
                )

            # 3) 识别 focus_agents
            focus_agents, focus_nodes, agent_reasons = identify_focus_agents(
                world, milp_actions_dict, marl_actions_list, high_wait_threshold
            )

            # 4) 执行 MILP 动作
            actions_tensor = torch.stack(
                [
                    torch.tensor(
                        [[milp_actions_dict[n]]], dtype=torch.long, device=device
                    ).squeeze(-1)
                    for n in agent_names
                ],
                dim=1,
            )
            policy_output = TensorDict(
                {("agents", "action"): actions_tensor}, batch_size=[1]
            )
            tensordict = rl_env.step(policy_output)

            reward_tensor = tensordict.get(("next", "agents", "reward"))
            reward = (
                reward_tensor.mean().item() if reward_tensor is not None else 0.0
            )

            # 5) 当存在动作差异时，生成训练记录
            if len(focus_agents) > 0:
                warm_start, variable_fixing = build_warm_start_from_state(
                    world, focus_agents
                )

                record = create_sft_record(
                    world=world,
                    focus_agents=focus_agents,
                    focus_nodes=focus_nodes,
                    agent_reasons=agent_reasons,
                    warm_start=warm_start,
                    variable_fixing=variable_fixing,
                    reward=reward,
                    episode=ep,
                    step=step,
                    has_disruption=has_disruption,
                    time_horizon=time_horizon,
                )
                sft_records.append(record)
                ep_sft_count += 1

                # RL 训练记录（含结构化 state 和 reward）
                rl_records.append({
                    "state_description": record["messages"][1]["content"],
                    "guidance": json.loads(record["messages"][2]["content"]),
                    "reward": reward,
                    "episode": ep,
                    "step": step,
                    "has_disruption": has_disruption,
                })

            pbar.set_postfix({
                "reward": f"{reward:.4f}",
                "focus": len(focus_agents),
                "sft": ep_sft_count,
            })

            observations = tensordict.get("next")
            done = tensordict.get(
                ("next", "done"), torch.tensor(False)
            ).any().item()
            if done:
                logger.info(f"Episode {ep+1} 结束于 step {step}")
                break

        try:
            pbar.close()
        except Exception:
            pass
        logger.info(f"Episode {ep+1}: SFT记录数={ep_sft_count}")

    return sft_records, rl_records


# ========================================================================
# main
# ========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT + RL fine-tuning data for small model guidance."
    )
    parser.add_argument('--steps', type=int, default=7200, help="每轮最大步数")
    parser.add_argument('--episodes', type=int, default=1, help="轮数")
    parser.add_argument('--topology', type=int, default=4, help="拓扑ID")
    parser.add_argument('--output', type=str, default=None, help="SFT JSONL输出路径")
    parser.add_argument('--rl-output', type=str, default=None,
                        help="RL JSONL输出路径")
    parser.add_argument('--decision-interval', type=int, default=1,
                        help="决策间隔(秒)")
    parser.add_argument('--seed', type=int, default=111, help="随机种子")
    parser.add_argument('--time-limit', type=int, default=360000,
                        help="MILP求解时限(秒)")
    parser.add_argument('--mip-gap', type=float, default=0.005,
                        help="MILP MIP gap")
    parser.add_argument('--high-wait-threshold', type=float, default=500,
                        help="高等待阈值")
    parser.add_argument('--load-trajectory', type=str, default=None,
                        help="加载已有MILP轨迹(pickle)")
    parser.add_argument('--save-trajectory', type=str, default=None,
                        help="保存MILP轨迹(pickle)")
    parser.add_argument('--skip-milp', action='store_true',
                        help="跳过MILP求解(必须配合 --load-trajectory)")
    args = parser.parse_args()

    # 随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 环境参数
    rl_args = rl_parser.parse_args([])
    apply_topology_defaults(rl_args, args.topology)
    tick_seconds = get_default_tick_seconds(args.topology) or 1
    os.environ["METRO_TOPOLOGY"] = str(args.topology)
    linedata.load_topology(args.topology)

    logger.info("=" * 60)
    logger.info("小模型 Guidance SFT + RL 数据生成器")
    logger.info("=" * 60)
    logger.info(f"拓扑: {args.topology}, Steps: {args.steps}, Episodes: {args.episodes}")

    device = torch.device("cpu")

    headway = getattr(rl_args, "train_headway", 240)
    loops_per_run = getattr(rl_args, "train_loops_per_run", 7)
    first_start_time = getattr(rl_args, "train_first_start_time", 0)
    assignments = getattr(rl_args, "train_assignments", None)
    train_run_config = TrainRunConfig(
        route_id=None,
        headway=headway,
        loops_per_run=loops_per_run,
        first_start_time=first_start_time,
        assignments=assignments,
    )

    start_steps = rl_args.start_steps
    episode_length = args.steps
    max_steps = start_steps + episode_length
    n_agents = get_default_n_agents(args.topology)

    # --- 创建环境 ---
    try:
        base_env = MetroRLEnvV2(
            num_envs=1,
            max_steps=max_steps,
            device=device,
            n_agents=n_agents,
            train_run_config=train_run_config,
            topology=args.topology,
            start_step=start_steps,
            episode_length=episode_length,
            enable_random_blockages=False,
            enable_random_speed_limits=False,
            enable_simulation_blockages=True,
            enable_simulation_speed_limits=True,
            default_interval=tick_seconds,
            seed=args.seed,
            use_float_boarding=True,
        )
        FastMILPSolver.apply_fixed_disruptions(base_env._env.world, args.topology)
        rl_env = TransformedEnv(
            base_env,
            RewardSum(
                in_keys=[base_env.reward_key],
                out_keys=[("agents", "episode_reward")],
            ),
        )
    except Exception as e:
        logger.error(f"创建环境失败: {e}", exc_info=True)
        sys.exit(1)

    # --- MARL 策略 ---
    marl_policy = load_marl_policy(args.topology, rl_env, device)

    # --- MILP 轨迹 ---
    milp_result = None
    if args.load_trajectory and os.path.exists(args.load_trajectory):
        logger.info(f"加载轨迹: {args.load_trajectory}")
        with open(args.load_trajectory, 'rb') as f:
            saved = pickle.load(f)
        milp_result = saved.get("metrics", {})
        traj = saved.get("trajectory")
        if traj:
            milp_result["trajectory"] = traj
    elif args.skip_milp:
        logger.error("需要 --load-trajectory 或取消 --skip-milp")
        return

    milp_solver = None
    if milp_result is None:
        milp_solver = FastMILPSolver(
            topology_id=args.topology,
            n_agents=n_agents,
            total_steps=args.steps,
            decision_interval=args.decision_interval,
            tick_seconds=tick_seconds,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
        )

    # --- 收集数据 ---
    sft_records, rl_records = run_guidance_data_collection(
        rl_env=rl_env,
        milp_solver=milp_solver,
        marl_policy=marl_policy,
        num_episodes=args.episodes,
        max_steps=args.steps,
        device=device,
        decision_interval=args.decision_interval,
        milp_result=milp_result,
        high_wait_threshold=args.high_wait_threshold,
    )

    # --- 保存 MILP 轨迹 ---
    if args.save_trajectory and milp_result and milp_result.get("trajectory"):
        with open(args.save_trajectory, 'wb') as f:
            pickle.dump(
                {
                    "metrics": {
                        k: v for k, v in milp_result.items() if k != "trajectory"
                    },
                    "trajectory": milp_result["trajectory"],
                },
                f,
            )
        logger.info(f"轨迹已保存: {args.save_trajectory}")

    # --- 写入文件 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"guidance_sft_topology{args.topology}_{timestamp}"

    # 1) SFT JSONL
    sft_path = args.output or os.path.join(current_dir, f"{base_name}.jsonl")
    if sft_records:
        with open(sft_path, "w", encoding="utf-8") as f:
            for rec in sft_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"SFT数据已保存: {sft_path} ({len(sft_records)} 条)")
    else:
        logger.warning("未收集到SFT记录")

    # 2) RL JSONL
    rl_path = args.rl_output or os.path.join(current_dir, f"{base_name}_rl.jsonl")
    if rl_records:
        with open(rl_path, "w", encoding="utf-8") as f:
            for rec in rl_records:
                f.write(
                    json.dumps(rec, ensure_ascii=False, default=str) + "\n"
                )
        logger.info(f"RL数据已保存: {rl_path} ({len(rl_records)} 条)")

    # --- 统计 ---
    if sft_records:
        focus_counts = [r["metadata"]["num_focus_agents"] for r in sft_records]
        rewards = [r["metadata"]["reward"] for r in sft_records]
        logger.info("\n" + "=" * 60)
        logger.info("数据统计:")
        logger.info(f"  SFT记录数: {len(sft_records)}")
        logger.info(f"  RL记录数: {len(rl_records)}")
        logger.info(f"  平均focus_agents: {np.mean(focus_counts):.1f}")
        logger.info(f"  平均reward: {np.mean(rewards):.4f}")
        logger.info(f"  reward范围: [{min(rewards):.4f}, {max(rewards):.4f}]")
        logger.info("=" * 60)

    logger.info("数据生成完成！")


if __name__ == "__main__":
    main()
