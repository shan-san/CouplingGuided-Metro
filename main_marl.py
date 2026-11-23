"""Metro System Simulation with RL Policy."""

import sys
import os
import time
import torch
from torch import nn
from tensordict import TensorDictBase, set_get_defaults_to_none
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, MultiAgentMLP
from torchrl.envs import TransformedEnv, RewardSum

from config import Config
from logger import logger, setup_logger
from Params import simu_parser, rl_parser, apply_topology_defaults, get_default_tick_seconds, get_default_n_agents
from linedata import load_topology

MetroWindow = None
QApplication = None
_GUI_IMPORT_ERROR = None


def _load_gui_components():
    global MetroWindow, QApplication, _GUI_IMPORT_ERROR

    if MetroWindow is not None and QApplication is not None:
        return MetroWindow, QApplication
    if _GUI_IMPORT_ERROR is not None:
        raise RuntimeError("GUI 依赖不可用，当前环境无法启用可视化运行") from _GUI_IMPORT_ERROR

    try:
        from gui_lean import MetroWindow as loaded_metro_window, QApplication as loaded_q_application
    except ImportError as exc:
        _GUI_IMPORT_ERROR = exc
        raise RuntimeError("GUI 依赖不可用，当前环境无法启用可视化运行") from exc

    MetroWindow = loaded_metro_window
    QApplication = loaded_q_application
    return MetroWindow, QApplication

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(project_root)
from metro_rl_env import MetroRLEnvV2
from train_run_config import TrainRunConfig
from model_config import PolicyNetConfig


def _normalize_policy_state_dict_keys(state_dict):
    """规范化策略网络权重键名。

    兼容训练时可能出现的包装层差异，例如：
    - `module.0.module._orig_mod.0...` -> `module.0.module.0...`
    - 缺少顶层 `module.` 前缀
    """
    try:
        from tensordict import TensorDictBase as _TDB
        if isinstance(state_dict, _TDB):
            flat = state_dict.flatten_keys(separator=".")
            state = {str(k): v for k, v in flat.items()}
        else:
            state = dict(state_dict)
    except Exception:
        state = dict(state_dict)

    normalized = {}
    for key, value in state.items():
        k = str(key)
        k = k.replace(".module._orig_mod.", ".module.")
        k = k.replace("._orig_mod.", ".")
        normalized[k] = value

    if normalized and not any(str(k).startswith("module.") for k in normalized.keys()):
        normalized = {f"module.{k}": v for k, v in normalized.items()}

    return normalized

# --- 加载RL模型函数 ---
def load_rl_policy(model_path, env, device, rl_args=None):
    """加载训练好的RL策略模型"""
    print(f"正在加载模型: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    
    # 使用安全的权重加载；旧版PyTorch不支持weights_only，做兼容处理
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    # 使用训练保存的超参数（或默认参数）构建配置，不做模型结构推断
    saved_hparams = checkpoint.get('hparams', {})
    base_rl_args = rl_parser.parse_args([])
    effective_args = rl_args if rl_args is not None else base_rl_args
    cfg = PolicyNetConfig.from_args(effective_args)
    if isinstance(saved_hparams, dict) and saved_hparams:
        # 允许checkpoint中的配置覆盖默认/命令行配置
        cfg = PolicyNetConfig.from_hparams({**cfg.to_hparams(), **saved_hparams})

    policy_state_dict = checkpoint.get('policy_state_dict')
    if policy_state_dict is None:
        raise KeyError("checkpoint 缺少 'policy_state_dict'，无法构建策略网络")

    # 从环境spec读取输入/输出维度
    env_input_dim = env.observation_spec["agents", "observation", "train_state"].shape[-1]
    # env_output_dim = env.unbatched_action_spec["agents", "action", "action"].shape[-1]

    n_agents = int(cfg.n_agents)
    share_parameters_policy = bool(cfg.share_parameters_policy)
    n_agent_inputs = int(env_input_dim)
    n_agent_outputs = int(cfg.n_action_dim)
    policy_depth = int(cfg.policy_depth)
    policy_hidden_dim = int(cfg.policy_hidden_dim)

    feature_mode = str(saved_hparams.get("policy_feature_mode", "train_state_only"))
    print(
        "模型配置:",
        f"n_agents={n_agents}",
        f"policy_depth={policy_depth}",
        f"policy_hidden_dim={policy_hidden_dim}",
        f"feature_mode={feature_mode}",
        f"in_dim={n_agent_inputs}",
        f"out_dim={n_agent_outputs}",
    )

    # 重新创建与训练时完全一致的策略网络结构

    # 抑制即将更改的TensorDict get 行为的弃用告警
    try:
        set_get_defaults_to_none(True)
    except Exception:
        pass

    # 构建与训练脚本一致的轻量策略网络
    if feature_mode != "train_state_only":
        feature_mode = "train_state_only"

    # 从环境spec读取与训练一致的观测维度
    train_state_dim = int(env.observation_spec["agents", "observation", "train_state"].shape[-1])

    policy_net = nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=train_state_dim,
            n_agent_outputs=n_agent_outputs,
            n_agents=env.n_agents,
            centralised=False,
            share_params=share_parameters_policy,
            device=device,
            depth=policy_depth,
            num_cells=policy_hidden_dim,
            activation_class=torch.nn.Tanh,
        ),
        torch.nn.LogSoftmax(dim=-1),
    )

    policy_module = TensorDictModule(
        module=policy_net,
        in_keys=[("agents", "observation", "train_state")],
        out_keys=[("agents", "logits")],
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "logits")],
        out_keys=[env.action_key],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=False,
    )

    # 加载权重
    normalized_policy_state_dict = _normalize_policy_state_dict_keys(policy_state_dict)
    try:
        # 直接尝试加载规范化后的权重
        policy.load_state_dict(normalized_policy_state_dict, strict=False)
    except Exception:
        try:
            # 再尝试原始权重，保留旧兼容路径
            policy.load_state_dict(policy_state_dict, strict=False)
        except Exception as exc:
            expected_out = n_agent_outputs
            sample_keys = list(normalized_policy_state_dict.keys())[:5]
            raise RuntimeError(
                "策略网络权重加载失败：请确认训练与仿真环境配置一致，特别是动作维度。"
                f" 期望输出维度: {expected_out}. 示例权重键: {sample_keys}"
            ) from exc
    policy.eval()  # 设置为评估模式
    print("模型加载成功并设置为评估模式。")
    return policy

# --- 修改后的运行函数 ---
def run_with_gui(rl_env, policy, device):
    """使用RL策略运行带GUI的模拟，并显示客流信息"""
    metro_window_cls, q_application_cls = _load_gui_components()
    app = q_application_cls(sys.argv)
    window = metro_window_cls(rl_env.base_env, env_index=0)  # GUI只显示一个环境实例
    window.show()

    # 初始观察
    observations = rl_env.reset()
    done = False
    step_counter = 0
    total_reward = 0.0
    
    # 创建定时器进行定期更新
    def update_simulation():
        nonlocal observations, done, step_counter, total_reward
        
        if done:
            print(f"回合结束，重置环境...")
            observations = rl_env.reset()
            done = False
            step_counter = 0
            total_reward = 0.0
            window.statusBar().showMessage(f"环境已重置")
        else:
            # 从策略获取动作
            with torch.no_grad():
                current_obs = observations.to(device).to(torch.float32)
                policy_output = policy(current_obs)
                
                actions = policy_output[rl_env.action_key]
                policy_actions = actions.cpu().numpy()[0]
                
                # 检查中断影响并更新GUI
                disruption_info = "当前无中断"
                disrupted_segments = rl_env.base_env.scenario.world.get_disrupted_segments()
                if disrupted_segments:
                    disruption_info_parts = []
                    for seg_id, details in disrupted_segments.items():
                        # seg_id 格式为 "start_platform_name-end_platform_name"
                        # 直接使用 seg_id，因为它已经是正确的格式
                        info_str = f"{seg_id}"
                        if details['type'] == 'SPEED_LIMIT':
                            info_str += f" (限速 {details['factor']}x)"
                        else:
                            info_str += " (中断)"
                        disruption_info_parts.append(info_str)
                            
                    disruption_info = "中断/限速: " + ", ".join(disruption_info_parts)
                
                print(f"当前中断/限速区段: {disruption_info}")
                
                policy_action_str = ", ".join([f"列车{i+1}:{('移动' if act == 1 else '停留')}" 
                                      for i, act in enumerate(policy_actions)])
                print(f"步骤 {step_counter}: 策略动作 = {policy_action_str}")
                
                tensordict = rl_env.step(policy_output)
                observations = tensordict.get("next")
                step_counter += 1
                
                # 打印中断影响的反馈
                print("列车实际状态:")
                for i, agent in enumerate(rl_env.base_env.scenario.world.agents):
                    actual_action = agent.action[0].item() if hasattr(agent, 'action') else None
                    action_result = agent.action_result[0].item() if hasattr(agent, 'action_result') else None
                    position = agent.state.position[0].item() if hasattr(agent, 'state') else "未知"
                    
                    status_str = ""
                    if action_result == 3:
                        status_str = " [被中断阻止]"
                    elif action_result == 0:
                        status_str = " [被冲突阻止]"
                    elif action_result == 2:
                        status_str = " [成功移动]"
                    elif action_result == 1:
                        status_str = " [停留]"
                    elif action_result == 4:
                        status_str = " [因限速停留]"
                    elif action_result == 5:
                        status_str = " [未达最小停站]"
                    
                    print(f"  列车{i+1}: 位置 {position}{status_str}")
                
                # 从TensorDict提取完成状态
                done = tensordict.get(("next", "done"), torch.tensor(False)).any().item()
                
                # 获取奖励
                reward = 0.0
                rewards = tensordict.get(("next", "agents", "reward"))
                reward = rewards.mean().item()
                total_reward += reward

                # print the train_state and log the action
                
                # 客流统计
                total_passengers = 0
                total_waiting = 0
                
                for agent in rl_env.base_env.scenario.world.agents:
                    if hasattr(agent, 'passenger_count'):
                        total_passengers += agent.passenger_count[0].item()
                
                for node_id, node in rl_env.base_env.scenario.world.platform_nodes.items():
                    if hasattr(node, 'waiting_passengers') and node.waiting_passengers is not None:
                        total_waiting += node.waiting_passengers[0].item()
                
                # 更新状态栏
                window.statusBar().showMessage(
                    f"步骤: {step_counter} | 总乘客: {total_passengers:.0f}人 | 站点等待: {total_waiting:.0f}人 | "
                    f"奖励: {reward:.4f} | 累计: {total_reward:.4f}"
                )

        # 更新GUI
        window.refresh_trains()
        
    # 设置定时器
    from PyQt6.QtCore import QTimer
    timer = QTimer()
    timer.timeout.connect(update_simulation)
    timer.start(100)  # 100ms更新一次

    return app.exec()

def run_without_gui(rl_env, policy, device, start_steps, max_steps):
    """使用RL策略运行无GUI的模拟，并输出客流信息"""
    observations = rl_env.reset()
    total_steps = start_steps

    try:
        while total_steps < max_steps:
            # 获取策略动作
            with torch.no_grad():
                current_obs = observations.to(device).to(torch.float32)
                train_state = current_obs.get(("agents", "observation", "train_state"))
                if train_state is not None:
                    printable_state = train_state.detach().to("cpu")
                    if printable_state.shape[0] == 1:
                        printable_state = printable_state[0]
                    print(f"[Step {total_steps}] train_state: {printable_state.tolist()}")
                policy_output = policy(current_obs)

                # 修正：正确获取各智能体的动作
                actions = policy_output[rl_env.action_key]
                policy_actions = actions.cpu().numpy()[0]
                printable_actions = actions.detach().to("cpu")
                if printable_actions.shape[-1] == 1:
                    printable_actions = printable_actions.squeeze(-1)
                if printable_actions.shape[0] == 1:
                    action_list = printable_actions[0].tolist()
                else:
                    action_list = printable_actions.tolist()
                logger.info("[Step %d] policy actions: %s", total_steps, action_list)

                # 输出策略想要执行的动作
                policy_action_str = " ".join([f"列车{i+1}:{('移动' if act == 1 else '停留')}" 
                                      for i, act in enumerate(policy_actions)])
                print(f"\n===== 步骤 {total_steps} =====")
                print(f"策略输出动作: {policy_action_str}")
                
                # 执行动作
                tensordict = rl_env.step(policy_output)
                observations = tensordict
                total_steps += 1

                # 获取列车的实际状态 (执行后)
                print("列车实际状态:")
                for i, agent in enumerate(rl_env.base_env.scenario.world.agents):
                    # 获取列车的实际动作状态
                    actual_action = agent.action[0].item() if hasattr(agent, 'action') else None
                    
                    # 获取列车的位置和停留信息
                    position = agent.state.position[0].item() if hasattr(agent, 'state') else "未知"
                    is_at_station = position in rl_env.base_env.scenario.world.platform_nodes
                    
                    # 确定列车的实际状态
                    if actual_action == 0:  # 停留
                        state_str = "停留"
                    elif actual_action == 1:  # 移动
                        state_str = "移动"
                    else:
                        # 从状态信息推断
                        dwell_time = agent.state.dwell_time[0].item() if hasattr(agent.state, 'dwell_time') else 0
                        expected_dwell = agent.state.current_expected_dwell_time[0].item() if hasattr(agent.state, 'current_expected_dwell_time') else 0
                        
                        if dwell_time < expected_dwell and is_at_station:
                            state_str = "停站中"
                        elif dwell_time >= expected_dwell and is_at_station:
                            state_str = "已完成停站"
                        else:
                            state_str = "区间行驶"
                    
                    print(f"  列车{i+1}: {state_str} (位置: {position})")
            
            # 获取奖励
            reward_val = 0.0
            reward = 0.0
            rewards = tensordict.get(("next", "agents", "reward"))
            reward = rewards.mean().item()
            reward_val += reward
            
            print(f"奖励: {reward_val:.4f}")
            
            # 打印客流信息
            print("\n----- 列车客流 -----")
            for i, agent in enumerate(rl_env.base_env.scenario.world.agents):
                position = agent.state.position[0].item() if hasattr(agent, 'state') else "未知"
                
                if hasattr(agent, 'passenger_count'):
                    passenger_count = agent.passenger_count[0].item()
                    print(f"列车 {i+1} (位置: {position}): {passenger_count:.0f}人")
                    
                    # 打印主要目的地
                    if hasattr(agent, 'passengers_by_destination') and agent.passengers_by_destination:
                        top_dests = sorted([(dest, count[0].item()) 
                                          for dest, count in agent.passengers_by_destination.items()
                                          if count[0].item() > 0],
                                         key=lambda x: x[1], reverse=True)[:3]
                        
                        if top_dests:
                            dest_str = ", ".join([f"站{d}({c:.0f}人)" for d, c in top_dests])
                            print(f"  主要目的地: {dest_str}")
                else:
                    print(f"列车 {i+1} (位置: {position}): 无客流数据")
            
            print("\n----- 站点等待客流 -----")
            total_waiting = 0
            for node_id, node in rl_env.base_env.scenario.world.platform_nodes.items():
                if hasattr(node, 'waiting_passengers') and node.waiting_passengers is not None:
                    waiting = node.waiting_passengers[0].item()
                    total_waiting += waiting
                    print(f"站点 {node_id}: {waiting:.0f}人等待")
                else:
                    print(f"站点 {node_id}: 无客流数据")
            
            print(f"\n总计: {total_waiting:.0f}人在站台等待")
            print("=======================")
            
            # 从TensorDict中提取完成状态
            done = tensordict.get(("next", "done"), torch.tensor(False)).any().item()
            if done:
                print(f"回合结束于步骤 {total_steps}，重置环境")
                observations = rl_env.reset()
            
            # 短暂暂停
            # time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("模拟被用户停止")
    finally:
        print("模拟结束")

def main():
    """主入口点"""
    # 解析命令行参数
    sim_args = simu_parser.parse_args()
    rl_args = rl_parser.parse_args([])

    os.environ["METRO_TOPOLOGY"] = str(sim_args.topology)
    apply_topology_defaults(rl_args, sim_args.topology)
    tick_seconds = get_default_tick_seconds(sim_args.topology) or 1

    load_topology(sim_args.topology)
    logger.info(f"当前拓扑: {sim_args.topology}")

    logger.info("启动地铁系统模拟 (使用RL策略)")
    logger.info(f"GUI: {'禁用' if sim_args.nogui else '启用'}")

    if sim_args.topology == 1:
        model_filename = "metro_mappo_best.pt"
    elif sim_args.topology == 2:
        model_filename = "metro_topology_mappo_best.pt"
    elif sim_args.topology == 3:
        model_filename = "metro_shenzhenline1_mappo_best.pt"
    elif sim_args.topology == 4:
        model_filename = "metro_topology_4_mappo_best.pt"
    elif sim_args.topology == 5:
        model_filename = "metro_topology_5_mappo_best.pt"
    elif sim_args.topology == 6:
        model_filename = "metro_topology_6_mappo_best.pt"
    elif sim_args.topology == 7:
        model_filename = "metro_topology_7_mappo_best.pt"
    elif sim_args.topology == 8:
        model_filename = "metro_topology_8_mappo_best.pt"
    elif sim_args.topology == 9:
        model_filename = "metro_topology_9_mappo_best.pt"
    elif sim_args.topology == 10:
        model_filename = "metro_topology_10_mappo_best.pt"
    elif sim_args.topology == 11:
        model_filename = "metro_topology_11_mappo_best.pt"
    elif sim_args.topology == 12:
        model_filename = "metro_topology_12_mappo_best.pt"
    elif sim_args.topology == 13:
        model_filename = "metro_topology_13_mappo_best.pt"
    elif sim_args.topology == 14:
        model_filename = "metro_topology_14_mappo_best.pt"
    elif sim_args.topology == 15:
        model_filename = "metro_topology_15_mappo_best.pt"
    elif sim_args.topology == 16:
        model_filename = "metro_topology_16_mappo_best.pt"
    else:
        model_filename = "metro_mappo_best.pt"
        logger.warning("未知拓扑参数，已回退到默认模型。")

    model_path = os.path.join(project_root, "MARL_model", model_filename)
    if not os.path.exists(model_path):
        logger.error("模型文件不存在: %s", model_path)
        sys.exit(1)

    # 如未显式指定，优先使用模型中记录的智能体数量
    saved_n_agents = None
    checkpoint_meta = None
    try:
        checkpoint_meta = torch.load(model_path, map_location="cpu", weights_only=False)
        saved_hparams = checkpoint_meta.get("hparams", {}) or {}
        if "n_agents" in saved_hparams:
            saved_n_agents = int(saved_hparams["n_agents"])
        elif "n_agents" in checkpoint_meta:
            saved_n_agents = int(checkpoint_meta["n_agents"])
    except Exception as exc:
        logger.warning("无法从模型读取智能体数量，使用参数配置。(%s)", exc)
    finally:
        checkpoint_meta = None

    if saved_n_agents and saved_n_agents != rl_args.n_agents:
        logger.info("根据模型超参数将智能体数量从 %s 调整为 %s", rl_args.n_agents, saved_n_agents)
        rl_args.n_agents = saved_n_agents

    # 设置设备
    # device = torch.device("cuda")  # 也可以使用 "cuda" 如果可用
    device = torch.device(rl_args.device)
    
    # 创建RL环境
    num_envs = 1  # 仿真时只用1个环境
    start_steps = rl_args.start_steps
    max_steps = start_steps + rl_args.max_steps
    n_agents = rl_args.n_agents

    headway = getattr(rl_args, "train_headway", 240)
    loops_per_run = getattr(rl_args, "train_loops_per_run", 7)
    first_start_time = getattr(rl_args, "train_first_start_time", 0)
    train_run_config = TrainRunConfig(
        route_id=None,
        headway=headway,
        loops_per_run=loops_per_run,
        first_start_time=first_start_time,
        assignments=None,
    )

    rl_env = MetroRLEnvV2(
        num_envs=num_envs,
        max_steps=max_steps,
        device=device,
        n_agents=n_agents,
        train_run_config=train_run_config,
        topology=sim_args.topology,
        start_step=start_steps,
        episode_length=rl_args.max_steps,
        # 仿真/评估时，禁用随机事件
        enable_random_blockages=False,
        enable_random_speed_limits=False,
        # 仿真/评估时，启用固定的场景
        enable_simulation_blockages=True,
        enable_simulation_speed_limits=True,
        default_interval=tick_seconds,
    )

    # 应用与训练相同的环境变换
    rl_env = TransformedEnv(
        rl_env,
        RewardSum(in_keys=[rl_env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    logger.info(f"RL环境创建成功，智能体数量: {n_agents}")
        

    
    # 加载训练好的模型
    try:
        logger.info(f"将加载的MARL模型路径: {model_path}")
        policy = load_rl_policy(model_path, rl_env, device, rl_args)
        logger.info("RL策略模型加载成功")
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        sys.exit(1)
    
    # 添加初始列车 (RL环境在重置时会自动处理)
    logger.info(f"初始化 {n_agents} 个列车")
    
    # 运行模拟
    if sim_args.nogui:
        logger.info("运行无GUI模拟 (RL控制)")
        run_without_gui(rl_env, policy, device, start_steps, max_steps)
    else:
        logger.info("运行带GUI模拟 (RL控制)")
        return run_with_gui(rl_env, policy, device)

if __name__ == "__main__":
    sys.exit(main())
