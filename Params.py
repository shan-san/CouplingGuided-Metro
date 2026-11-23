import argparse

TOPOLOGY_DEFAULTS = {
    1: {"n_agents": 12, "tick_seconds": 1},
    2: {"n_agents": 12, "tick_seconds": 1},
    3: {"n_agents": 76, "tick_seconds": 5},
    4: {"n_agents": 10, "tick_seconds": 1},
    5: {"n_agents": 10, "tick_seconds": 1},
    6: {"n_agents": 10, "tick_seconds": 1},
    7: {"n_agents": 10, "tick_seconds": 1},
    8: {"n_agents": 12, "tick_seconds": 1},
    9: {"n_agents": 14, "tick_seconds": 1},
    10: {"n_agents": 14, "tick_seconds": 1},
    11: {"n_agents": 14, "tick_seconds": 1},
    12: {"n_agents": 14, "tick_seconds": 1},
    13: {"n_agents": 12, "tick_seconds": 1},
    14: {"n_agents": 20, "tick_seconds": 1},
    15: {"n_agents": 30, "tick_seconds": 1},
    16: {"n_agents": 60, "tick_seconds": 1},
}

from topology.metro_lines import LINE_SPECS
TOPOLOGY_DEFAULTS.update({tid: {"n_agents": spec["fleet"], "tick_seconds": 30}
                          for tid, spec in LINE_SPECS.items()})

def get_default_n_agents(topology: int) -> int | None:
    defaults = TOPOLOGY_DEFAULTS.get(topology, {})
    return defaults.get("n_agents")

def get_default_tick_seconds(topology: int) -> int | None:
    defaults = TOPOLOGY_DEFAULTS.get(topology, {})
    return defaults.get("tick_seconds")

def apply_topology_defaults(rl_args, topology: int):
    if getattr(rl_args, "n_agents", None) is None:
        default_agents = get_default_n_agents(topology)
        if default_agents is not None:
            rl_args.n_agents = default_agents

simu_parser = argparse.ArgumentParser(description='Metro System Simulation Params')

simu_parser.add_argument('--nogui', action='store_true', help='Run without GUI')
simu_parser.add_argument('--mode', choices=['self_rolling', 'delegated'], default='self_rolling', help='Operation mode')
simu_parser.add_argument('--topology', type=int, choices=sorted(TOPOLOGY_DEFAULTS), default=1, help='仿真拓扑；101-105为具名线路')

rl_parser = argparse.ArgumentParser(description='Reinforcement Learning Params')

# 拓扑配置
rl_parser.add_argument('--topology', type=int, choices=sorted(TOPOLOGY_DEFAULTS), default=1,
                       help='训练环境使用的拓扑 (1-16)')

# 设备配置
rl_parser.add_argument('--device', type=str, default='cpu', help='使用的设备 (cpu/cuda)')

# 采样相关参数
rl_parser.add_argument('--frames_per_batch', type=int, default=60000, help='每次训练迭代收集的帧数')
rl_parser.add_argument('--n_iters', type=int, default=100, help='采样和训练迭代的次数')

# 训练相关参数
rl_parser.add_argument('--num_epochs', type=int, default=3, help='每次训练迭代的优化步骤数')
rl_parser.add_argument('--minibatch_size', type=int, default=8192, help='每个优化步骤的小批量大小')
rl_parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
rl_parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪的最大范数')
rl_parser.add_argument('--max_epochs_per_batch', type=int, default=3, help='每个采样批次允许的最大epoch数，用于限制过度优化')
rl_parser.add_argument('--kl_stop_threshold', type=float, default=0.015, help='KL散度阈值，超过后提前终止当前批次的优化')

# PPO算法相关参数
rl_parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO损失的裁剪值')
rl_parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
rl_parser.add_argument('--lmbda', type=float, default=0.9, help='广义优势估计的lambda值')
rl_parser.add_argument('--entropy_eps', type=float, default=1e-2, help='PPO损失中熵项的系数')

# 环境相关参数
rl_parser.add_argument('--max_steps', type=int, default=7200, help='每个episode的仿真时长，以秒为单位')
rl_parser.add_argument('--start_steps', type=int, default=21600, help='从上午六点钟开始仿真')
rl_parser.add_argument('--n_agents', type=int, default=None, help='智能体数量（留空时按topology自动设置）')
rl_parser.add_argument('--warmup_steps', type=int, default=0, help='每个epoch跳过的优化步数（预热），非全局预热')
rl_parser.add_argument('--enable_random_blockages', action=argparse.BooleanOptionalAction, default=True, help='启用随机线路阻塞')
rl_parser.add_argument('--enable_random_speed_limits', action=argparse.BooleanOptionalAction, default=True, help='启用随机线路限速')


# 运行计划配置参数
rl_parser.add_argument('--train_route_id', type=str, default=None, help='训练环境使用的线路ID，留空则使用默认线路')
rl_parser.add_argument('--train_headway', type=int, default=240, help='列车发车间隔(秒)')
rl_parser.add_argument('--train_loops_per_run', type=int, default=7, help='每次运行的循环次数')
rl_parser.add_argument('--train_first_start_time', type=int, default=0, help='首班车出发时间(相对秒)')

# Actor-Critic网络相关参数
rl_parser.add_argument('--share_parameters_policy', action='store_true', default=True, help='是否共享策略参数')
rl_parser.add_argument('--share_parameters_critic', action='store_true', default=True, help='是否共享价值网络参数')
rl_parser.add_argument('--mappo', action='store_true', default=True, help='使用MAPPO，否则使用IPPO')
rl_parser.add_argument('--policy_depth', type=int, default=3, help='策略网络深度')
rl_parser.add_argument('--policy_hidden_dim', type=int, default=32, help='策略网络隐藏层维度')
rl_parser.add_argument('--critic_depth', type=int, default=3, help='价值网络深度')
rl_parser.add_argument('--critic_hidden_dim', type=int, default=32, help='价值网络隐藏层维度')

# 模型保存相关参数
rl_parser.add_argument('--save_frequency', type=int, default=5, help='每多少次迭代保存一次模型')
rl_parser.add_argument('--save_best', action='store_true', default=True, help='是否保存最佳模型')
