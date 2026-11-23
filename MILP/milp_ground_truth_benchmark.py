"""
MILP Ground Truth Solver (Fast Version)

简化策略:
1. 移除微观乘客OD追踪 (load_by_dest)，大幅减少变量和Big-M约束。
2. 使用 "Headway Adherence" (间隔均衡性) 和 "Effective Dwell" (有效停站) 作为乘客满意度的代理指标。
3. 保持物理约束（阻塞、限速、单一闭塞分区）的严格性。

目标函数设计:
- 移动奖励: 鼓励列车运行
- 停站奖励: 鼓励在站台停靠服务乘客
- 服务奖励: 完成一次有效离站代表完成乘客上下车服务
- 区间停车惩罚: 不应在非站台区域停留
- Headway均衡: 鼓励均匀的发车间隔

与MARL reward的对应关系:
- move_bonus ≈ base_move_bonus (0.05)
- dwell_bonus ≈ dwell_step_scale (0.05) 
- service_reward ≈ 累积的 passenger_board + passenger_alight
- non_platform_penalty ≈ non_platform_hold_penalty (0.05)

新功能: 将MILP轨迹输入MARL环境获取准确reward
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import random
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from MILP import cplex_backend as mip
from Params import TOPOLOGY_DEFAULTS
from linedata import load_topology, get_platform_by_id
from experiment_scenarios import (
    add_scenario_args,
    custom_disruptions_for_milp,
    scenario_config_from_args,
    scenario_kwargs,
    scenario_suffix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ================= Configuration =================

# 核心奖励配置 - 与MARL环境的reward scale对齐
# 参考 metro_agent_v2.py: base_move_bonus=0.05, dwell_step_scale=0.05
REWARD_CONFIG = {
    "move_bonus": 0.05,            # 移动奖励：对齐MARL的base_move_bonus
    "dwell_bonus": 0.05,           # 停站奖励/决策步：对齐MARL的dwell_step_scale
    "service_reward": 0.05,        # 完成服务(离站)奖励：对齐MARL的dwell_depart_bonus
    "headway_penalty": 0.01,       # 间隔不均匀惩罚：保持均衡发车
    "non_platform_penalty": 0.05,  # 区间停车惩罚：对齐MARL的non_platform_hold_penalty
}

# 固定中断配置
FIXED_DISRUPTIONS = {
    "default": {
        "blockages": [
            {"segment": (1, 2), "start_time": 1000, "end_time": 2000},
        ],
        "speed_limits": [
            {"segment": (3, 4), "start_time": 4000, "end_time": 4500, "factor": 2.0},
        ],
    },
    4: {
        "blockages": [
            {"segment": (1, 2), "start_time": 1000, "end_time": 2000},
        ],
        "speed_limits": [
            {"segment": (3, 4), "start_time": 4000, "end_time": 4500, "factor": 2.0},
        ],
    },
}


class FastMILPSolver:
    """快速MILP求解器 - 简化的Ground Truth模型"""
    
    def __init__(
        self,
        topology_id: int = 4,
        n_agents: int = 10,
        total_steps: int = 7200,
        decision_interval: int = 120,
        tick_seconds: int = 1,
        time_limit: Optional[int] = 3600,
        mip_gap: float = 0.005,
        custom_disruptions: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ):
        self.topology_id = topology_id
        self.n_agents = n_agents
        self.total_steps = total_steps
        self.decision_interval = decision_interval
        self.tick_seconds = tick_seconds
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        
        self.horizon = total_steps // decision_interval
        
        # 构建拓扑
        self.nodes, self.edges, self.adj = self._build_topology(topology_id)
        self.platforms = [n['id'] for n in self.nodes if n['is_platform']]
        self.node_map = {n['id']: i for i, n in enumerate(self.nodes)}
        self.plat_map = {pid: i for i, pid in enumerate(self.platforms)}
        
        # 加载中断配置
        self.disruptions = custom_disruptions or FIXED_DISRUPTIONS.get(topology_id, FIXED_DISRUPTIONS["default"])
        
    def _build_topology(self, topo_id: int) -> Tuple[List[Dict], List[Tuple], Dict]:
        """构建拓扑图 - 根据linedata.py的实际定义"""
        nodes = []
        edges = []
        
        if topo_id == 4:
            # Topology 4: 5个车站，10列车，2个中间连接
            # 节点: d1(1), s11-s15(2-6), d2(7), s21-s25(8-12)
            specs = [
                (1, "d1", False), (2, "s11", True), (3, "s12", True),
                (4, "s13", True), (5, "s14", True), (6, "s15", True),
                (7, "d2", False), (8, "s21", True), (9, "s22", True),
                (10, "s23", True), (11, "s24", True), (12, "s25", True),
            ]
            for nid, name, is_p in specs:
                nodes.append({"id": nid, "name": name, "is_platform": is_p})
            
            # 外环: d1->s11->s12->s13->s14->s15->d2->s25->s24->s23->s22->s21->d1
            loop = [1, 2, 3, 4, 5, 6, 7, 12, 11, 10, 9, 8]
            for i in range(len(loop)):
                u, v = loop[i], loop[(i+1) % len(loop)]
                edges.append((u, v, 60))
                edges.append((v, u, 60))
            
            # 中间连接: s12<->s22, s14<->s24
            edges.extend([(3, 9, 60), (9, 3, 60), (5, 11, 60), (11, 5, 60)])
        
        elif topo_id == 5:
            # Topology 5: 7个车站，10列车，2个中间连接
            # 节点: d1(1), s11-s17(2-8), d2(9), s21-s27(10-16)
            specs = [
                (1, "d1", False), (2, "s11", True), (3, "s12", True),
                (4, "s13", True), (5, "s14", True), (6, "s15", True),
                (7, "s16", True), (8, "s17", True), (9, "d2", False),
                (10, "s21", True), (11, "s22", True), (12, "s23", True),
                (13, "s24", True), (14, "s25", True), (15, "s26", True), (16, "s27", True),
            ]
            for nid, name, is_p in specs:
                nodes.append({"id": nid, "name": name, "is_platform": is_p})
            
            # 外环
            loop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 15, 14, 13, 12, 11, 10]
            for i in range(len(loop)):
                u, v = loop[i], loop[(i+1) % len(loop)]
                edges.append((u, v, 60))
                edges.append((v, u, 60))
            
            # 中间连接: s12<->s22, s16<->s26
            edges.extend([(3, 11, 60), (11, 3, 60), (7, 15, 60), (15, 7, 60)])
        
        elif topo_id == 6:
            # Topology 6: 9个车站，10列车，2个中间连接
            # 节点: d1(1), s11-s19(2-10), d2(11), s21-s29(12-20)
            specs = [
                (1, "d1", False), (2, "s11", True), (3, "s12", True),
                (4, "s13", True), (5, "s14", True), (6, "s15", True),
                (7, "s16", True), (8, "s17", True), (9, "s18", True), (10, "s19", True),
                (11, "d2", False), (12, "s21", True), (13, "s22", True), (14, "s23", True),
                (15, "s24", True), (16, "s25", True), (17, "s26", True),
                (18, "s27", True), (19, "s28", True), (20, "s29", True),
            ]
            for nid, name, is_p in specs:
                nodes.append({"id": nid, "name": name, "is_platform": is_p})
            
            # 外环
            loop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 19, 18, 17, 16, 15, 14, 13, 12]
            for i in range(len(loop)):
                u, v = loop[i], loop[(i+1) % len(loop)]
                edges.append((u, v, 60))
                edges.append((v, u, 60))
            
            # 中间连接: s12<->s22, s18<->s28
            edges.extend([(3, 13, 60), (13, 3, 60), (9, 19, 60), (19, 9, 60)])
        
        elif topo_id == 7:
            # Topology 7: 8个车站，10列车，2个中间连接
            # 节点: d1(1), s11-s18(2-9), d2(10), s21-s28(11-18)
            specs = [
                (1, "d1", False), (2, "s11", True), (3, "s12", True),
                (4, "s13", True), (5, "s14", True), (6, "s15", True),
                (7, "s16", True), (8, "s17", True), (9, "s18", True),
                (10, "d2", False), (11, "s21", True), (12, "s22", True),
                (13, "s23", True), (14, "s24", True), (15, "s25", True),
                (16, "s26", True), (17, "s27", True), (18, "s28", True),
            ]
            for nid, name, is_p in specs:
                nodes.append({"id": nid, "name": name, "is_platform": is_p})
            
            # 外环
            loop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 17, 16, 15, 14, 13, 12, 11]
            for i in range(len(loop)):
                u, v = loop[i], loop[(i+1) % len(loop)]
                edges.append((u, v, 60))
                edges.append((v, u, 60))
            
            # 中间连接: s12<->s22, s17<->s27
            edges.extend([(3, 12, 60), (12, 3, 60), (8, 17, 60), (17, 8, 60)])
        
        elif topo_id in [8, 9]:
            # Topology 8: 8个车站，12列车，2个中间连接
            # Topology 9: 8个车站，14列车，2个中间连接
            # 结构相同，只是列车数量不同
            specs = [
                (1, "d1", False), (2, "s11", True), (3, "s12", True),
                (4, "s13", True), (5, "s14", True), (6, "s15", True),
                (7, "s16", True), (8, "s17", True), (9, "s18", True),
                (10, "d2", False), (11, "s21", True), (12, "s22", True),
                (13, "s23", True), (14, "s24", True), (15, "s25", True),
                (16, "s26", True), (17, "s27", True), (18, "s28", True),
            ]
            for nid, name, is_p in specs:
                nodes.append({"id": nid, "name": name, "is_platform": is_p})
            
            # 外环
            loop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 17, 16, 15, 14, 13, 12, 11]
            for i in range(len(loop)):
                u, v = loop[i], loop[(i+1) % len(loop)]
                edges.append((u, v, 60))
                edges.append((v, u, 60))
            
            # 中间连接: s12<->s22, s17<->s27
            edges.extend([(3, 12, 60), (12, 3, 60), (8, 17, 60), (17, 8, 60)])
        
        elif topo_id == 10:
            # Topology 10: 9个车站，14列车，2个中间连接
            specs = [
                (1, "d1", False), (2, "s11", True), (3, "s12", True),
                (4, "s13", True), (5, "s14", True), (6, "s15", True),
                (7, "s16", True), (8, "s17", True), (9, "s18", True), (10, "s19", True),
                (11, "d2", False), (12, "s21", True), (13, "s22", True), (14, "s23", True),
                (15, "s24", True), (16, "s25", True), (17, "s26", True),
                (18, "s27", True), (19, "s28", True), (20, "s29", True),
            ]
            for nid, name, is_p in specs:
                nodes.append({"id": nid, "name": name, "is_platform": is_p})
            
            # 外环
            loop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 19, 18, 17, 16, 15, 14, 13, 12]
            for i in range(len(loop)):
                u, v = loop[i], loop[(i+1) % len(loop)]
                edges.append((u, v, 60))
                edges.append((v, u, 60))
            
            # 中间连接: s12<->s22, s18<->s28
            edges.extend([(3, 13, 60), (13, 3, 60), (9, 19, 60), (19, 9, 60)])
        
        elif topo_id == 11:
            # Topology 11: 9个车站，14列车，4个中间连接
            specs = [
                (1, "d1", False), (2, "s11", True), (3, "s12", True),
                (4, "s13", True), (5, "s14", True), (6, "s15", True),
                (7, "s16", True), (8, "s17", True), (9, "s18", True), (10, "s19", True),
                (11, "d2", False), (12, "s21", True), (13, "s22", True), (14, "s23", True),
                (15, "s24", True), (16, "s25", True), (17, "s26", True),
                (18, "s27", True), (19, "s28", True), (20, "s29", True),
            ]
            for nid, name, is_p in specs:
                nodes.append({"id": nid, "name": name, "is_platform": is_p})
            
            # 外环
            loop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 19, 18, 17, 16, 15, 14, 13, 12]
            for i in range(len(loop)):
                u, v = loop[i], loop[(i+1) % len(loop)]
                edges.append((u, v, 60))
                edges.append((v, u, 60))
            
            # 中间连接: s12<->s22, s14<->s24, s16<->s26, s18<->s28
            edges.extend([
                (3, 13, 60), (13, 3, 60),   # s12<->s22
                (5, 15, 60), (15, 5, 60),   # s14<->s24
                (7, 17, 60), (17, 7, 60),   # s16<->s26
                (9, 19, 60), (19, 9, 60),   # s18<->s28
            ])
        
        elif topo_id == 12:
            # Topology 12: 9个车站，14列车，6个中间连接
            specs = [
                (1, "d1", False), (2, "s11", True), (3, "s12", True),
                (4, "s13", True), (5, "s14", True), (6, "s15", True),
                (7, "s16", True), (8, "s17", True), (9, "s18", True), (10, "s19", True),
                (11, "d2", False), (12, "s21", True), (13, "s22", True), (14, "s23", True),
                (15, "s24", True), (16, "s25", True), (17, "s26", True),
                (18, "s27", True), (19, "s28", True), (20, "s29", True),
            ]
            for nid, name, is_p in specs:
                nodes.append({"id": nid, "name": name, "is_platform": is_p})
            
            # 外环
            loop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 19, 18, 17, 16, 15, 14, 13, 12]
            for i in range(len(loop)):
                u, v = loop[i], loop[(i+1) % len(loop)]
                edges.append((u, v, 60))
                edges.append((v, u, 60))
            
            # 中间连接: s12<->s22, s13<->s23, s14<->s24, s16<->s26, s17<->s27, s18<->s28
            edges.extend([
                (3, 13, 60), (13, 3, 60),   # s12<->s22
                (4, 14, 60), (14, 4, 60),   # s13<->s23
                (5, 15, 60), (15, 5, 60),   # s14<->s24
                (7, 17, 60), (17, 7, 60),   # s16<->s26
                (8, 18, 60), (18, 8, 60),   # s17<->s27
                (9, 19, 60), (19, 9, 60),   # s18<->s28
            ])
        
        else:
            # 通用拓扑: 简单环形
            n_nodes = TOPOLOGY_DEFAULTS.get(topo_id, {}).get("n_agents", 10) + 2
            for i in range(1, n_nodes + 1):
                is_platform = i not in [1, n_nodes // 2 + 1]
                nodes.append({"id": i, "name": f"s{i}" if is_platform else f"d{i}", "is_platform": is_platform})
            
            for i in range(1, n_nodes + 1):
                u, v = i, (i % n_nodes) + 1
                edges.append((u, v, 60))
                edges.append((v, u, 60))
        
        adj = defaultdict(list)
        for u, v, t in edges:
            if v not in adj[u]:
                adj[u].append(v)
        
        return nodes, edges, dict(adj)
    
    def _is_blocked(self, u: int, v: int, t_step: int) -> bool:
        """检查边在给定时间步是否被阻塞"""
        sim_time = t_step * self.decision_interval * self.tick_seconds
        
        for b in self.disruptions.get("blockages", []):
            seg = b["segment"]
            if (u == seg[0] and v == seg[1]) or (u == seg[1] and v == seg[0]):
                if b["start_time"] <= sim_time < b["end_time"]:
                    return True
        return False

    @staticmethod
    def apply_fixed_disruptions(world, topology_id: int) -> None:
        """将FIXED_DISRUPTIONS注入仿真世界的disruption_manager"""
        disruptions = FIXED_DISRUPTIONS.get(topology_id, FIXED_DISRUPTIONS["default"])
        dm = world.disruption_manager
        dm.disruptions.clear()

        for b in disruptions.get("blockages", []):
            u, v = b["segment"]
            segment_id = f"{u}-{v}"
            target_segment = None
            if hasattr(world, "_get_segment_by_id"):
                target_segment = world._get_segment_by_id(segment_id)

            if target_segment:
                dm.add_disruption(target_segment.id, b["start_time"], b["end_time"], "BLOCK")
                logger.info(f"已注入固定中断: {target_segment.id}, 时间: {b['start_time']}-{b['end_time']}")
            else:
                dm.add_disruption(segment_id, b["start_time"], b["end_time"], "BLOCK")
                logger.info(f"已尝试按ID注入固定中断: {segment_id}, 时间: {b['start_time']}-{b['end_time']}")

        for s in disruptions.get("speed_limits", []):
            u, v = s["segment"]
            segment_id = f"{u}-{v}"
            target_segment = None
            if hasattr(world, "_get_segment_by_id"):
                target_segment = world._get_segment_by_id(segment_id)

            if target_segment:
                dm.add_disruption(
                    target_segment.id,
                    s["start_time"],
                    s["end_time"],
                    "SPEED_LIMIT",
                    factor=s.get("factor", 2.0),
                )
                logger.info(f"已注入固定限速: {target_segment.id}, 时间: {s['start_time']}-{s['end_time']}")
            else:
                dm.add_disruption(
                    segment_id,
                    s["start_time"],
                    s["end_time"],
                    "SPEED_LIMIT",
                    factor=s.get("factor", 2.0),
                )
                logger.info(f"已尝试按ID注入固定限速: {segment_id}, 时间: {s['start_time']}-{s['end_time']}")
    
    def solve(self) -> Dict[str, Any]:
        """求解MILP模型"""
        logger.info(f"构建快速MILP模型 | 拓扑: {self.topology_id} | 智能体: {self.n_agents} | 决策步数: {self.horizon}")
        
        build_start = time.perf_counter()
        
        m = mip.Model("Metro_Fast_Schedule")
        m.setParam("OutputFlag", 1)
        if self.time_limit is not None:
            m.setParam("TimeLimit", self.time_limit)
        m.setParam("MIPGap", self.mip_gap)
        m.setParam("MIPFocus", 1)  # 侧重于寻找可行解
        m.setParam("Threads", 0)   # 使用所有核心
        m.setParam("Heuristics", 0.5)
        
        N = len(self.nodes)
        T = self.horizon
        A = self.n_agents
        platform_indices = [self.node_map[pid] for pid in self.platforms]
        
        # ================= 变量定义 =================
        logger.info("创建决策变量...")
        
        # x[a, n, t]: 智能体a在时刻t是否在节点n
        x = m.addVars(A, N, T, vtype=mip.BINARY, name="x")
        
        # move[a, u, v, t]: 智能体a在时刻t从节点u移动到节点v
        move_arcs = []
        for u_id, neighbors in self.adj.items():
            u_idx = self.node_map[u_id]
            for v_id in neighbors:
                v_idx = self.node_map[v_id]
                for t in range(T - 1):
                    if not self._is_blocked(u_id, v_id, t):
                        for a in range(A):
                            move_arcs.append((a, u_idx, v_idx, t))
        
        move = m.addVars(move_arcs, vtype=mip.BINARY, name="move")
        
        # stay[a, n, t]: 智能体a在时刻t停留在节点n
        stay = m.addVars(A, N, T - 1, vtype=mip.BINARY, name="stay")
        
        # is_dwelling[a, t]: 智能体a在时刻t是否在站台
        is_dwelling = m.addVars(A, T, vtype=mip.BINARY, name="is_dwelling")
        
        # depart_event[a, p, t]: 智能体a在时刻t从站台p离开
        depart_event = m.addVars(A, len(self.platforms), T - 1, vtype=mip.BINARY, name="depart")
        
        logger.info(f"变量数: x={A*N*T}, move={len(move_arcs)}, stay={A*N*(T-1)}")
        
        # ================= 约束条件 =================
        logger.info("添加约束条件...")
        
        # 1. 初始位置: 均匀分布在各站台
        for a in range(A):
            start_node_id = self.platforms[a % len(self.platforms)]
            start_node_idx = self.node_map[start_node_id]
            m.addConstr(x[a, start_node_idx, 0] == 1, f"init_pos_{a}")
            for n_idx in range(N):
                if n_idx != start_node_idx:
                    m.addConstr(x[a, n_idx, 0] == 0, f"init_zero_{a}_{n_idx}")
        
        # 2. 位置唯一性: 每个智能体每时刻只能在一个节点
        for a in range(A):
            for t in range(T):
                m.addConstr(m.sum(x[a, n, t] for n in range(N)) == 1, f"one_loc_{a}_{t}")
        
        # 3. 容量约束: 放宽闭塞分区，允许站台容纳多辆车
        # 站台节点：允许最多3辆车同时停靠（模拟多个停车位）
        # 非站台节点（depot）：允许更多车辆（模拟停车场）
        for n in range(N):
            node_id = self.nodes[n]['id']
            is_platform = self.nodes[n]['is_platform']
            capacity = 3 if is_platform else A  # 站台容量3，depot容量无限
            for t in range(T):
                m.addConstr(m.sum(x[a, n, t] for a in range(A)) <= capacity, f"capacity_{n}_{t}")
        
        # 4. 决策约束: stay + move_out = x
        for a in range(A):
            for t in range(T - 1):
                for n_idx in range(N):
                    node_id = self.nodes[n_idx]['id']
                    neighbors = self.adj.get(node_id, [])
                    
                    out_moves = m.sum(
                        move[a, n_idx, self.node_map[v], t]
                        for v in neighbors
                        if (a, n_idx, self.node_map[v], t) in move
                    )
                    m.addConstr(stay[a, n_idx, t] + out_moves == x[a, n_idx, t], f"decision_{a}_{n_idx}_{t}")
        
        # 5. 流守恒: x[t+1] = stay + move_in
        for a in range(A):
            for t in range(T - 1):
                for n_idx in range(N):
                    node_id = self.nodes[n_idx]['id']
                    
                    # 流入: 从邻居移动过来
                    in_moves = m.linear_expr()
                    for u_id, neighbors in self.adj.items():
                        if node_id in neighbors:
                            u_idx = self.node_map[u_id]
                            if (a, u_idx, n_idx, t) in move:
                                in_moves += move[a, u_idx, n_idx, t]
                    
                    m.addConstr(x[a, n_idx, t + 1] == stay[a, n_idx, t] + in_moves, f"flow_{a}_{n_idx}_{t}")
        
        # 6. 停站状态定义: is_dwelling = sum(x[a, platform, t])
        for a in range(A):
            for t in range(T):
                m.addConstr(
                    is_dwelling[a, t] == m.sum(x[a, p_idx, t] for p_idx in platform_indices),
                    f"dwelling_{a}_{t}"
                )
        
        # 7. 离站事件定义: depart = x[platform, t] - x[platform, t+1] >= 0
        for a in range(A):
            for p_out_idx, p_id in enumerate(self.platforms):
                p_idx = self.node_map[p_id]
                for t in range(T - 1):
                    m.addConstr(
                        depart_event[a, p_out_idx, t] >= x[a, p_idx, t] - x[a, p_idx, t + 1],
                        f"depart_{a}_{p_out_idx}_{t}"
                    )
        
        # ================= 目标函数 =================
        logger.info("设置目标函数...")
        
        obj = m.linear_expr()
        
        # MILP目标函数设计原则：
        # 1. 不使用dt_scale缩放 - MILP是高层决策，不是逐步模拟
        # 2. service_reward为主导 - 鼓励完成停站服务周期
        # 3. dwell_bonus为辅助 - 鼓励在站台合理停留
        # 4. move_bonus极小 - 只为打破对称性，避免过度移动
        
        # 1. 移动奖励 (鼓励列车运行) - 每次高层move决策
        obj += REWARD_CONFIG["move_bonus"] * m.sum(move[key] for key in move_arcs)
        
        # 2. 停站奖励 (在站台停靠服务乘客) - 每个决策时间步
        obj += REWARD_CONFIG["dwell_bonus"] * m.sum(
            is_dwelling[a, t] for a in range(A) for t in range(T)
        )
        
        # 3. 服务奖励 (每次离站代表完成一次乘客服务) - 主要优化目标
        obj += REWARD_CONFIG["service_reward"] * m.sum(
            depart_event[a, p, t] 
            for a in range(A) for p in range(len(self.platforms)) for t in range(T - 1)
        )
        
        # 4. 区间停车惩罚 (非站台停留) - 每个决策时间步
        non_platform_indices = [i for i in range(N) if i not in platform_indices]
        obj -= REWARD_CONFIG["non_platform_penalty"] * m.sum(
            stay[a, n_idx, t]
            for a in range(A) for n_idx in non_platform_indices for t in range(T - 1)
        )
        
        m.setObjective(obj, mip.MAXIMIZE)
        
        build_time = time.perf_counter() - build_start
        logger.info(f"模型构建完成，耗时: {build_time:.2f}秒")
        
        # ================= 求解 =================
        logger.info("开始求解...")
        solve_start = time.perf_counter()
        m.optimize()
        solve_time = time.perf_counter() - solve_start
        
        # ================= 结果提取 =================
        if m.status not in (mip.OPTIMAL, mip.TIME_LIMIT):
            logger.error(f"MILP求解失败，状态码: {m.status}")
            if m.status == mip.INFEASIBLE:
                logger.info("Running CPLEX conflict refiner to diagnose infeasibility...")
                m.computeIIS()
                m.write("ground_truth_infeasible.ilp")
            return {"status": "failed", "status_code": m.status}
        
        if m.status == mip.TIME_LIMIT and m.SolCount == 0:
            logger.error("时间限制内未找到可行解")
            return {"status": "timeout", "status_code": m.status}
        
        logger.info(f"求解完成 | 状态: {'最优' if m.status == mip.OPTIMAL else '次优'} | 耗时: {solve_time:.2f}秒")
        
        # 提取指标
        raw_reward = m.objVal
        
        # 归一化为MARL可比较的形式
        # MARL: 每步reward = mean(agents), 总reward = sum(每步mean)
        # MILP: 目标值 = sum(所有agent × 所有决策步)
        # 归一化: MARL_equivalent = (raw / n_agents) * (total_steps / horizon)
        marl_equivalent_reward = (raw_reward / A) * (self.total_steps / self.horizon)
        
        # 计算服务次数
        total_services = sum(
            depart_event[a, p, t].solution_value
            for a in range(A) for p in range(len(self.platforms)) for t in range(T - 1)
            if depart_event[a, p, t].solution_value > 0.5
        )
        
        # 计算headway标准差
        avg_headway_std = self._compute_headway_std(x, depart_event, T)
        
        # ================= 提取轨迹 =================
        trajectory = self._extract_trajectory(x, move, stay, move_arcs, N, A, T)
        
        metrics = {
            "status": "optimal" if m.status == mip.OPTIMAL else "suboptimal",
            "solver_backend": "cplex",
            "cplex_status": m.cplex_status,
            "solver_parameter_mapping": m.parameter_mapping,
            "solver_version": m.native_model.get_cplex().get_version(),
            "topology": self.topology_id,
            "n_agents": self.n_agents,
            "total_steps": self.total_steps,
            "decision_horizon": self.horizon,
            "decision_interval": self.decision_interval,
            "raw_reward": raw_reward,
            "total_reward": marl_equivalent_reward,
            "total_reward_raw": raw_reward,  # 兼容旧接口
            "total_services": total_services,
            # 注意: 该指标来自MILP代理模型，不是MARL环境真实回放值
            "avg_headway_std": avg_headway_std,
            "milp_proxy_avg_headway_std": avg_headway_std,
            "build_time_sec": build_time,
            "solve_time_sec": solve_time,
            "mip_gap": m.MIPGap if hasattr(m, 'MIPGap') else None,
            "trajectory": trajectory,  # 新增: 轨迹数据
        }
        
        return metrics
    
    def _extract_trajectory(self, x, move, stay, move_arcs, N: int, A: int, T: int) -> Dict[str, Any]:
        """
        从MILP解中提取完整轨迹
        
        返回:
            trajectory: {
                "positions": List[List[int]] - 每个决策步每个agent的位置 (T x A)
                "actions": List[List[int]] - 每个决策步每个agent的动作 (T-1 x A)
                    动作: 0=STOP, 1+=MOVE (根据目标节点推断方向)
                "node_id_to_idx": Dict[int, int] - 节点ID到索引的映射
                "idx_to_node_id": Dict[int, int] - 索引到节点ID的映射
            }
        """
        logger.info("提取MILP轨迹...")
        
        # 索引到节点ID的映射
        idx_to_node_id = {i: n['id'] for i, n in enumerate(self.nodes)}
        
        # 提取每个时间步每个agent的位置
        positions = []  # T x A
        for t in range(T):
            step_positions = []
            for a in range(A):
                pos = -1
                for n_idx in range(N):
                    try:
                        if x[a, n_idx, t].solution_value > 0.5:
                            pos = idx_to_node_id[n_idx]
                            break
                    except:
                        pass
                step_positions.append(pos)
            positions.append(step_positions)
        
        # 直接从MILP变量提取动作（而不是从位置推断）
        # 优先级：如果有move决策则移动，否则检查stay决策
        actions = []  # (T-1) x A
        for t in range(T - 1):
            step_actions = []
            for a in range(A):
                curr_pos = positions[t][a]
                next_pos = positions[t + 1][a]
                curr_idx = self.node_map.get(curr_pos, -1)
                
                # 检查是否有move决策
                moved = False
                direction = 0
                if curr_idx >= 0:
                    # 查找所有从当前位置出发的move
                    for (agent_idx, from_idx, to_idx, time_idx) in move_arcs:
                        if agent_idx == a and from_idx == curr_idx and time_idx == t:
                            try:
                                if move[(agent_idx, from_idx, to_idx, time_idx)].solution_value > 0.5:
                                    # 找到move决策
                                    target_node_id = idx_to_node_id[to_idx]
                                    direction = self._get_direction_for_move(curr_pos, target_node_id)
                                    moved = True
                                    break
                            except:
                                pass
                
                if moved:
                    step_actions.append(direction)
                else:
                    # 没有move，表示stay (action=0)
                    step_actions.append(0)
            
            actions.append(step_actions)
        
        return {
            "positions": positions,
            "actions": actions,
            "node_id_to_idx": self.node_map,
            "idx_to_node_id": idx_to_node_id,
            "decision_interval": self.decision_interval,
            "n_agents": A,
            "horizon": T,
        }
    
    def _get_direction_for_move(self, from_node: int, to_node: int) -> int:
        """
        根据移动确定动作方向
        
        在MARL中:
        - 动作 0 = STOP
        - 动作 1 = 默认方向移动 (第一个邻居)
        - 动作 2 = 第二方向移动
        - 动作 3 = 第三方向移动
        
        返回: 动作值 (1, 2, 或 3)
        """
        neighbors = self.adj.get(from_node, [])
        if not neighbors:
            return 1  # 默认方向
        
        for i, neighbor in enumerate(neighbors):
            if neighbor == to_node:
                return i + 1  # 动作从1开始
        
        # 未找到匹配的邻居，返回默认方向
        return 1
    
    def _compute_headway_std(self, x, depart_event, horizon) -> float:
        """计算各站台的平均发车间隔标准差"""
        headway_stds = []
        dt = self.decision_interval * self.tick_seconds
        
        for p_out_idx, p_id in enumerate(self.platforms):
            # 收集该站台的所有离站时间
            departures = []
            for t in range(horizon - 1):
                for a in range(self.n_agents):
                    try:
                        if depart_event[a, p_out_idx, t].solution_value > 0.5:
                            departures.append(t * dt)
                    except:
                        pass
            
            if len(departures) < 2:
                continue
            
            departures.sort()
            headways = np.diff(departures)
            
            if len(headways) > 0:
                headway_stds.append(np.std(headways))
        
        return float(np.mean(headway_stds)) if headway_stds else 0.0


# ================= 兼容旧接口的包装类 =================
class MILPGroundTruthSolver(FastMILPSolver):
    """兼容旧接口的别名"""
    pass


# ================= Metric Recorders =================

class StationEventRecorder:
    """Tracks per-agent station arrival and departure events during simulation."""

    VALID_EVENT_TYPES = {"arrival", "departure"}

    def __init__(self, base_env, policy_label: str) -> None:
        self.base_env = base_env
        self.policy_label = policy_label
        self.metro_env = getattr(base_env, "_env", base_env)
        self.world = getattr(self.metro_env, "world", None)
        self.tick_seconds = float(getattr(self.world, "tick_seconds", 1) or 1) if self.world is not None else 1.0
        self.topology = getattr(base_env, "topology", None)
        self.prev_platforms: Dict[Tuple[str, int], int] = {}
        self.events: List[Dict[str, object]] = []
        self._bootstrap()

    def _bootstrap(self) -> None:
        if self.world is None:
            return
        num_envs = getattr(self.world, "num_envs", 0) or 0
        for env_index in range(num_envs):
            for agent in self.world.agents:
                key = (agent.name, env_index)
                self.prev_platforms[key] = self._get_platform(agent, env_index)

    @staticmethod
    def _tensor_at(value, index: int) -> Optional[float]:
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        if value.ndim == 0:
            return float(value.item())
        if index < 0 or index >= value.shape[0]:
            return None
        return float(value[index].item())

    def _get_platform(self, agent, env_index: int) -> int:
        platform_tensor = getattr(agent.state, "current_platform", None)
        value = self._tensor_at(platform_tensor, env_index)
        if value is None:
            return -1
        return int(value)

    def _get_last_departure_time(self, agent, env_index: int) -> Optional[float]:
        last_departure_tensor = getattr(agent.state, "last_departure_time", None)
        value = self._tensor_at(last_departure_tensor, env_index)
        return value

    def _get_sim_time(self, env_index: int) -> Optional[float]:
        if self.world is None:
            return None
        sim_time_tensor = getattr(self.world, "sim_time", None)
        return self._tensor_at(sim_time_tensor, env_index)

    def _resolve_station_label(self, station_id: int) -> str:
        platform = get_platform_by_id(station_id)
        name = getattr(platform, "name", None)
        if name:
            return str(name)
        return f"s{station_id}"

    def _build_event(
        self,
        *,
        agent_name: str,
        env_index: int,
        station_id: int,
        event_type: str,
        sim_time: float,
        step_index: int,
    ) -> Dict[str, object]:
        safe_event = event_type if event_type in self.VALID_EVENT_TYPES else "unknown"
        return {
            "policy": self.policy_label,
            "topology": self.topology,
            "env_index": env_index,
            "agent": agent_name,
            "station": self._resolve_station_label(station_id),
            "event_type": safe_event,
            "step": step_index,
            "sim_time_tick": sim_time,
            "tick_seconds": self.tick_seconds,
            "sim_time_seconds": sim_time * self.tick_seconds,
        }

    def record(self, step_index: int) -> List[Dict[str, object]]:
        if self.world is None:
            return []

        recorded: List[Dict[str, object]] = []
        num_envs = getattr(self.world, "num_envs", 0) or 0

        for env_index in range(num_envs):
            sim_time = self._get_sim_time(env_index)
            if sim_time is None:
                continue

            for agent in self.world.agents:
                key = (agent.name, env_index)
                current_platform = self._get_platform(agent, env_index)
                previous_platform = self.prev_platforms.get(key)

                if previous_platform is None:
                    self.prev_platforms[key] = current_platform
                    continue

                if current_platform == previous_platform:
                    continue

                if previous_platform >= 0 and current_platform != previous_platform:
                    departure_time = self._get_last_departure_time(agent, env_index)
                    if departure_time is None or not math.isfinite(departure_time):
                        departure_time = sim_time
                    recorded.append(
                        self._build_event(
                            agent_name=agent.name,
                            env_index=env_index,
                            station_id=previous_platform,
                            event_type="departure",
                            sim_time=departure_time,
                            step_index=step_index,
                        )
                    )

                if current_platform >= 0:
                    recorded.append(
                        self._build_event(
                            agent_name=agent.name,
                            env_index=env_index,
                            station_id=current_platform,
                            event_type="arrival",
                            sim_time=sim_time,
                            step_index=step_index,
                        )
                    )

                self.prev_platforms[key] = current_platform

        if recorded:
            self.events.extend(recorded)
        return recorded

    def to_frame(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(
                columns=
                [
                    "policy",
                    "topology",
                    "env_index",
                    "agent",
                    "station",
                    "event_type",
                    "step",
                    "sim_time_tick",
                    "tick_seconds",
                    "sim_time_seconds",
                ]
            )
        return pd.DataFrame(self.events)

    def get_headway_std(self) -> float:
        if not self.events:
            return 0.0
        
        df = pd.DataFrame(self.events)
        # Filter for arrival events
        arrivals = df[df["event_type"] == "arrival"].copy()
        
        if arrivals.empty:
            return 0.0
            
        # Sort by station and time
        arrivals.sort_values(by=["station", "sim_time_seconds"], inplace=True)
        
        # Calculate headways
        headway_stds = []
        for station, group in arrivals.groupby("station"):
            if len(group) < 2:
                continue
            
            # Calculate differences between consecutive arrival times
            times = group["sim_time_seconds"].values
            headways = np.diff(times)
            
            if len(headways) > 0:
                headway_stds.append(np.std(headways))
                
        if not headway_stds:
            return 0.0
            
        return float(np.mean(headway_stds))


class GlobalMetricRecorder:
    """Tracks global metrics like average waiting time."""
    def __init__(self, env):
        self.env = env
        self.total_waiting_time = 0.0
        self.tick_seconds = env.base_env.default_interval
        
        # Initialize previous boarded count
        self.prev_total_boarded = 0.0
        if hasattr(env.base_env, "agents"):
             for agent in env.base_env.agents:
                self.prev_total_boarded += agent.cumulative_boarded.sum().item()

    def update(self) -> Dict[str, float]:
        step_metrics = {
            "waiting_count": 0.0,
            "boarded_count": 0.0,
        }

        # 1. Waiting Time
        current_waiting = 0
        for node in self.env.base_env.world.platform_nodes.values():
            current_waiting += node.waiting_passengers.sum().item()
        self.total_waiting_time += current_waiting * self.tick_seconds
        step_metrics["waiting_count"] = current_waiting

        # 2. Boarded Passengers (Delta)
        current_total_boarded = 0
        for agent in self.env.base_env.agents:
            current_total_boarded += agent.cumulative_boarded.sum().item()
        
        step_boarded = current_total_boarded - self.prev_total_boarded
        self.prev_total_boarded = current_total_boarded
        step_metrics["boarded_count"] = step_boarded

        return step_metrics

    def get_metrics(self):
        total_boarded = 0
        for agent in self.env.base_env.agents:
            total_boarded += agent.cumulative_boarded.sum().item()
        
        avg_waiting_time = self.total_waiting_time / total_boarded if total_boarded > 0 else 0.0
        
        return {
            "total_waiting_time": self.total_waiting_time,
            "total_boarded": total_boarded,
            "avg_waiting_time": avg_waiting_time,
        }


# ================= MARL环境精确控制评估 =================

def evaluate_with_optimal_control(
    topology_id: int,
    total_steps: int,
    tick_seconds: int = 1,
    seed: int = 111,
    device: str = "cpu",
    target_dwell_time: int = 30,
    scenario_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    在MARL环境中使用精确控制策略，获取Ground Truth上界
    
    策略（模拟理想MILP解）:
    1. 在站台停留target_dwell_time秒后离站
    2. 移动时始终选择最快到达下一站的路径
    3. 遵循MARL的dwell_min/dwell_max约束
    4. 尊重阻塞和限速
    
    这代表了一个"理想调度员"的表现，应该是上界
    """
    from metro_rl_env import MetroRLEnvV2
    from torchrl.envs import TransformedEnv
    from torchrl.envs.transforms import RewardSum
    from Params import TOPOLOGY_DEFAULTS, rl_parser, apply_topology_defaults
    from train_run_config import TrainRunConfig
    
    logger.info("=" * 60)
    logger.info("精确控制策略评估 (Ground Truth)")
    logger.info("=" * 60)
    logger.info(f"策略: 目标停站时间={target_dwell_time}秒")
    
    # 获取拓扑配置
    topology_defaults = TOPOLOGY_DEFAULTS.get(topology_id, {"n_agents": 10, "tick_seconds": 1})
    n_agents = topology_defaults.get("n_agents", 10)
    tick_sec = topology_defaults.get("tick_seconds", tick_seconds)
    
    # 创建MARL环境
    load_topology(topology_id)
    
    rl_args = rl_parser.parse_args([])
    rl_args.topology = topology_id
    rl_args = apply_topology_defaults(rl_args, topology_id)
    
    start_steps = getattr(rl_args, "start_steps", 18000)
    episode_length = getattr(rl_args, "max_steps", total_steps)
    max_steps = start_steps + total_steps
    
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
    
    extra_kwargs = scenario_kwargs(scenario_config)

    base_env = MetroRLEnvV2(
        num_envs=1,
        max_steps=max_steps,
        device=device,
        n_agents=n_agents,
        train_run_config=train_run_config,
        topology=topology_id,
        start_step=start_steps,
        episode_length=episode_length,
        use_float_boarding=True,
        enable_random_blockages=False,
        enable_random_speed_limits=False,
        enable_simulation_blockages=True,
        enable_simulation_speed_limits=True,
        default_interval=tick_sec,
        seed=seed,
        **extra_kwargs,
    )
    
    env = TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    
    tensordict_obs = env.reset()
    
    agent_names = [agent.name for agent in base_env.agents]
    world = base_env._env.world
    marl_platform_ids = set(world.platform_nodes.keys())
    
    logger.info(f"MARL站台节点数: {len(marl_platform_ids)}")
    logger.info("开始精确控制执行...")
    
    # Initialize recorders
    station_recorder = StationEventRecorder(base_env, "Optimal_Control")
    metric_recorder = GlobalMetricRecorder(env)
    
    step_rewards = []
    total_reward = 0.0
    
    from tqdm import tqdm
    pbar = tqdm(range(total_steps), desc="Optimal Control")
    
    try:
        for step in pbar:
            action_dict = {}
            
            for agent_idx, agent_name in enumerate(agent_names):
                agent = base_env.agents[agent_idx]
                current_pos = int(agent.state.position[0].item())
                is_on_platform = current_pos in marl_platform_ids
                
                # 获取停站约束
                dwell_min = float(agent.state.dwell_min_limit[0].item())
                dwell_max = float(agent.state.dwell_max_limit[0].item())
                current_dwell = float(agent.state.time_since_arrival[0].item()) if is_on_platform else 0.0
                
                if is_on_platform:
                    # 在站台上，遵循精确停站策略
                    if dwell_min > 0 and current_dwell < dwell_min:
                        # 还没到最小停站时间，必须停留
                        action_val = 0
                    elif dwell_max > 0 and current_dwell >= dwell_max:
                        # 超过最大停站时间，必须离开
                        action_val = 1
                    elif current_dwell >= target_dwell_time:
                        # 达到目标停站时间，离开
                        action_val = 1
                    else:
                        # 继续停留
                        action_val = 0
                else:
                    # 不在站台，移动
                    action_val = 1
                
                action_tensor = torch.tensor([[action_val]], dtype=torch.long, device=device)
                action_dict[agent_name] = action_tensor
            
            from tensordict import TensorDict
            action_td = TensorDict(
                {
                    "agents": TensorDict(
                        {"action": torch.stack([action_dict[n].squeeze(-1) for n in agent_names], dim=1)},
                        batch_size=[1],
                    )
                },
                batch_size=[1],
            )
            
            tensordict_result = env.step(action_td)
            
            reward_tensor = tensordict_result.get(("next", "agents", "reward"))
            step_reward = reward_tensor.mean().item() if reward_tensor is not None else 0.0
            
            step_rewards.append(step_reward)
            total_reward += step_reward
            
            # Record metrics
            station_recorder.record(step)
            metric_recorder.update()
            
            done = tensordict_result.get(("next", "done"), torch.tensor(False)).any().item()
            if done:
                logger.info(f"环境在步骤 {step} 结束")
                break
            
            tensordict_obs = tensordict_result.get("next").clone()
            
            if step % 1000 == 0:
                pbar.set_postfix({"reward": f"{total_reward:.2f}", "avg": f"{total_reward/(step+1):.4f}"})
    
    finally:
        env.close()
    
    avg_step_reward = total_reward / len(step_rewards) if step_rewards else 0.0
    
    # Get metrics from recorders
    metrics = metric_recorder.get_metrics()
    avg_headway_std = station_recorder.get_headway_std()
    
    logger.info("=" * 60)
    logger.info("精确控制策略结果 (Ground Truth)")
    logger.info("=" * 60)
    logger.info(f"总步数: {len(step_rewards)}")
    logger.info(f"总Reward: {total_reward:.4f}")
    logger.info(f"平均每步Reward: {avg_step_reward:.6f}")
    logger.info(f"平均乘客等待时间: {metrics['avg_waiting_time']:.2f} 秒")
    logger.info(f"总上车乘客数: {metrics['total_boarded']:.0f}")
    logger.info(f"平均Headway标准差: {avg_headway_std:.2f} 秒")
    
    return {
        "total_reward": total_reward,
        "step_rewards": step_rewards,
        "avg_step_reward": avg_step_reward,
        "num_steps": len(step_rewards),
        "target_dwell_time": target_dwell_time,
        "avg_waiting_time": metrics["avg_waiting_time"],
        "total_boarded": metrics["total_boarded"],
        "avg_headway_std": avg_headway_std,
    }


def evaluate_trajectory_in_marl(
    trajectory: Dict[str, Any],
    topology_id: int,
    total_steps: int,
    tick_seconds: int = 1,
    seed: int = 111,
    device: str = "cpu",
    scenario_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    将MILP求解得到的轨迹在MARL环境中重放，获取准确的reward
    
    改进策略:
    - 使用MILP的目标站台序列(positions)来指导路由决策
    - 在MARL环境中，根据当前位置和MILP目标站台，选择正确的移动方向
    - 使用与MARL训练相同的停站时间逻辑
    
    Args:
        trajectory: MILP求解器返回的轨迹字典
        topology_id: 拓扑ID
        total_steps: 总仿真步数
        tick_seconds: 每个时间步的秒数
        seed: 随机种子
        device: 计算设备
    
    Returns:
        Dict包含:
            - total_reward: 总reward (MARL计算)
            - step_rewards: 每步reward列表
            - avg_step_reward: 平均每步reward
    """
    from metro_rl_env import MetroRLEnvV2
    from torchrl.envs import TransformedEnv
    from torchrl.envs.transforms import RewardSum
    from Params import TOPOLOGY_DEFAULTS, rl_parser, apply_topology_defaults
    from train_run_config import TrainRunConfig
    
    logger.info("=" * 60)
    logger.info("在MARL环境中评估MILP轨迹")
    logger.info("=" * 60)
    
    # 获取拓扑配置
    topology_defaults = TOPOLOGY_DEFAULTS.get(topology_id, {"n_agents": 10, "tick_seconds": 1})
    n_agents = topology_defaults.get("n_agents", 10)
    tick_sec = topology_defaults.get("tick_seconds", tick_seconds)
    
    # 轨迹信息
    positions = trajectory["positions"]  # 每个决策步每个agent在简化拓扑中的位置 (MILP节点ID)
    actions = trajectory["actions"]      # 每个决策步每个agent的动作 (0=停, 1+=移动)
    decision_interval = trajectory["decision_interval"]
    horizon = trajectory["horizon"]
    
    logger.info(f"轨迹信息: agents={n_agents}, horizon={horizon}, decision_interval={decision_interval}")
    
    # 分析MILP决策
    stop_count = sum(1 for step_actions in actions for a in step_actions if a == 0)
    move_count = sum(1 for step_actions in actions for a in step_actions if a > 0)
    logger.info(f"MILP决策统计: 停留={stop_count}, 移动={move_count}")
    
    # 构建 MILP目标位置表: target_positions[decision_step][agent_idx] = MILP目标节点ID
    # 这些ID与linedata.py中Platform.id一致，也就是MARL的platform_nodes的key
    target_positions = {}
    for t_step in range(len(positions) - 1):
        target_positions[t_step] = {}
        for a_idx in range(n_agents):
            curr_pos = positions[t_step][a_idx] if a_idx < len(positions[t_step]) else -1
            next_pos = positions[t_step + 1][a_idx] if a_idx < len(positions[t_step + 1]) else -1
            # MILP目标: 下一个决策时刻应该在的站台
            target_positions[t_step][a_idx] = next_pos
    
    logger.info(f"构建MILP目标位置表: {len(target_positions)} 个决策步")
    
    # 创建MARL环境
    load_topology(topology_id)
    
    # 解析默认参数
    rl_args = rl_parser.parse_args([])
    rl_args.topology = topology_id
    rl_args = apply_topology_defaults(rl_args, topology_id)
    
    start_steps = getattr(rl_args, "start_steps", 18000)
    episode_length = getattr(rl_args, "max_steps", total_steps)
    max_steps = start_steps + total_steps
    
    # 创建 TrainRunConfig
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
    
    # 创建环境
    extra_kwargs = scenario_kwargs(scenario_config)

    base_env = MetroRLEnvV2(
        num_envs=1,
        max_steps=max_steps,
        device=device,
        n_agents=n_agents,
        train_run_config=train_run_config,
        topology=topology_id,
        start_step=start_steps,
        episode_length=episode_length,
        use_float_boarding=True,
        enable_random_blockages=False,
        enable_random_speed_limits=False,
        enable_simulation_blockages=True,
        enable_simulation_speed_limits=True,
        default_interval=tick_sec,
        seed=seed,
        **extra_kwargs,
    )
    
    env = TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    
    # 重置环境
    tensordict_obs = env.reset()
    
    # 获取agent名称列表和world引用
    agent_names = [agent.name for agent in base_env.agents]
    world = base_env._env.world
    
    # 建立MARL站台信息
    marl_platform_ids = set(world.platform_nodes.keys())
    logger.info(f"MARL站台节点数: {len(marl_platform_ids)}")
    
    # 建立MARL邻接关系 (用于智能路由)
    marl_adjacency = world._next_nodes_id_map  # node_id -> list of next node ids
    
    # 使用BFS预计算从每个节点到所有站台的最短路径下一步
    # 为了效率，我们用简化的BFS
    from collections import deque
    
    def bfs_next_hop(start: int, target: int, max_depth: int = 200) -> int:
        """
        使用BFS找到从start到target的最佳下一步动作
        
        Returns:
            动作索引 (1-based: 1=第一个邻居, 2=第二个邻居, ...)
        """
        if start == target:
            return 0  # 已经到达，停留
        
        neighbors = marl_adjacency.get(start, [])
        if not neighbors:
            return 1
        
        # 对每个邻居进行BFS，看哪个能最快到达目标
        best_neighbor_idx = 0
        best_dist = float('inf')
        
        for idx, first_hop in enumerate(neighbors):
            if first_hop == target:
                return idx + 1  # 直接可达
            
            # BFS from first_hop to target
            visited = {start, first_hop}
            queue = deque([(first_hop, 1)])
            found_dist = float('inf')
            
            while queue and len(visited) < max_depth:
                node, dist = queue.popleft()
                if dist >= best_dist:
                    break
                
                for next_node in marl_adjacency.get(node, []):
                    if next_node == target:
                        found_dist = dist + 1
                        break
                    if next_node not in visited:
                        visited.add(next_node)
                        queue.append((next_node, dist + 1))
                
                if found_dist < float('inf'):
                    break
            
            if found_dist < best_dist:
                best_dist = found_dist
                best_neighbor_idx = idx
        
        return best_neighbor_idx + 1
    
    # 辅助函数: 从当前MARL节点寻找到目标MILP站台的最佳下一步
    def find_best_action_to_target(current_marl_pos: int, target_milp_platform: int) -> int:
        """
        根据当前MARL位置和MILP目标站台，选择最佳动作
        
        Args:
            current_marl_pos: 当前在MARL完整图中的节点ID
            target_milp_platform: MILP目标站台ID (也是linedata.py中的Platform.id)
        
        Returns:
            动作值 (0=停, 1=第一个邻居方向, 2=第二个邻居, ...)
        """
        return bfs_next_hop(current_marl_pos, target_milp_platform)
    
    # 每个agent跟踪的目标站台 (从MILP解获取)
    agent_target_platform = [-1] * n_agents
    
    # Initialize recorders for metrics
    station_recorder = StationEventRecorder(base_env, "MILP_Trajectory")
    metric_recorder = GlobalMetricRecorder(env)
    
    # 每个agent的停站计时器（用于控制合理的停站时间）
    agent_dwell_timers = [0] * n_agents  # 当前停站时长
    agent_last_platform = [-1] * n_agents  # 上一个停靠的站台
    
    step_rewards = []
    total_reward = 0.0
    benchmark_rows: List[Dict[str, Any]] = []
    
    # 在MARL环境中重放MILP轨迹
    logger.info("开始重放MILP轨迹...")
    logger.info("策略: 使用MILP目标位置指导路由决策")
    
    # 统计MILP路由使用情况
    milp_route_used = 0
    default_route_used = 0

    from tqdm import tqdm
    pbar = tqdm(range(total_steps), desc="MILP Trajectory Replay")
    
    try:
        for step in pbar:
            decision_start = time.perf_counter()

            # 确定当前决策步
            decision_step = min(step // decision_interval, horizon - 2)
            
            # 从MILP轨迹获取高层意图: 目标位置和动作
            milp_actions = actions[decision_step]
            milp_targets = target_positions.get(decision_step, {})
            
            # 转换为MARL环境格式的动作
            action_dict = {}
            for agent_idx, agent_name in enumerate(agent_names):
                # 获取当前agent的真实位置
                agent = base_env.agents[agent_idx]
                current_pos = int(agent.state.position[0].item())
                is_on_platform = current_pos in marl_platform_ids
                
                # 获取MILP的意图
                milp_action = milp_actions[agent_idx] if agent_idx < len(milp_actions) else 1
                milp_target = milp_targets.get(agent_idx, -1)
                
                # 更新agent目标 (当MILP指示移动时)
                if milp_action > 0 and milp_target > 0:
                    agent_target_platform[agent_idx] = milp_target
                
                # 智能动作选择
                if is_on_platform:
                    # 在站台上
                    if current_pos != agent_last_platform[agent_idx]:
                        # 刚到达新站台，重置计时器
                        agent_dwell_timers[agent_idx] = 0
                        agent_last_platform[agent_idx] = current_pos
                    
                    agent_dwell_timers[agent_idx] += tick_sec
                    
                    # 获取MARL环境的dwell约束
                    dwell_min = float(agent.state.dwell_min_limit[0].item())
                    dwell_max = float(agent.state.dwell_max_limit[0].item())
                    current_dwell = float(agent.state.time_since_arrival[0].item())
                    
                    # 遵循环境约束
                    if dwell_min > 0 and current_dwell < dwell_min:
                        # 还没到最小停站时间，必须停留
                        action_val = 0
                    elif dwell_max > 0 and current_dwell >= dwell_max:
                        # 超过最大停站时间，必须离开
                        # 使用MILP目标来选择方向
                        target = agent_target_platform[agent_idx]
                        if target > 0:
                            action_val = find_best_action_to_target(current_pos, target)
                        else:
                            action_val = min(milp_action, 3) if milp_action > 0 else 1
                    elif milp_action == 0:
                        # MILP说停留，继续停留
                        action_val = 0
                    else:
                        # MILP说移动，且已满足最小停站时间
                        target = agent_target_platform[agent_idx]
                        if target > 0:
                            action_val = find_best_action_to_target(current_pos, target)
                        else:
                            action_val = min(milp_action, 3)
                else:
                    # 不在站台，需要移动
                    agent_dwell_timers[agent_idx] = 0  # 重置计时器
                    # 使用MILP目标来选择方向
                    target = agent_target_platform[agent_idx]
                    if target > 0:
                        action_val = find_best_action_to_target(current_pos, target)
                        milp_route_used += 1
                    else:
                        action_val = 1  # 默认移动方向
                        default_route_used += 1
                
                # 创建动作tensor
                action_tensor = torch.tensor([[action_val]], dtype=torch.long, device=device)
                action_dict[agent_name] = action_tensor
            
            # 构建TensorDict格式的动作
            from tensordict import TensorDict
            action_td = TensorDict(
                {
                    "agents": TensorDict(
                        {"action": torch.stack([action_dict[n].squeeze(-1) for n in agent_names], dim=1)},
                        batch_size=[1],
                    )
                },
                batch_size=[1],
            )

            decision_time = time.perf_counter() - decision_start
            
            # 执行步进
            tensordict_result = env.step(action_td)
            
            # 获取reward
            reward_tensor = tensordict_result.get(("next", "agents", "reward"))
            if reward_tensor is not None:
                step_reward = reward_tensor.mean().item()
            else:
                step_reward = 0.0
            
            step_rewards.append(step_reward)
            total_reward += step_reward
            
            # Record metrics
            station_recorder.record(step)
            step_metrics = metric_recorder.update()

            benchmark_rows.append(
                {
                    "policy": "MILP_Ground_Truth",
                    "step": step,
                    "reward": step_reward,
                    "decision_time_sec": decision_time,
                    "decision_source": "MILP_Trajectory_Replay",
                    "waiting_passengers": step_metrics.get("waiting_count", 0.0),
                    "boarded_passengers": step_metrics.get("boarded_count", 0.0),
                }
            )
            
            # 检查是否结束
            done = tensordict_result.get(("next", "done"), torch.tensor(False)).any().item()
            if done:
                logger.info(f"环境在步骤 {step} 结束")
                break
            
            # 更新观察
            tensordict_obs = tensordict_result.get("next").clone()
            
            # 更新进度条
            if step % 500 == 0:
                pbar.set_postfix({"reward": f"{total_reward:.2f}", "avg": f"{total_reward/(step+1):.4f}"})
    
    finally:
        env.close()
    
    avg_step_reward = total_reward / len(step_rewards) if step_rewards else 0.0
    
    # Get metrics from recorders
    metrics = metric_recorder.get_metrics()
    avg_headway_std = station_recorder.get_headway_std()
    
    logger.info("=" * 60)
    logger.info("MARL环境评估结果 (MILP轨迹)")
    logger.info("=" * 60)
    logger.info(f"总步数: {len(step_rewards)}")
    logger.info(f"总Reward (MARL): {total_reward:.4f}")
    logger.info(f"平均每步Reward: {avg_step_reward:.6f}")
    logger.info(f"平均乘客等待时间: {metrics['avg_waiting_time']:.2f} 秒")
    logger.info(f"总上车乘客数: {metrics['total_boarded']:.0f}")
    logger.info(f"平均Headway标准差: {avg_headway_std:.2f} 秒")
    logger.info(f"MILP路由使用次数: {milp_route_used}, 默认路由使用次数: {default_route_used}")

    benchmark_results_df = pd.DataFrame(benchmark_rows)
    station_events_df = station_recorder.to_frame()
    
    return {
        "total_reward": total_reward,
        "step_rewards": step_rewards,
        "avg_step_reward": avg_step_reward,
        "num_steps": len(step_rewards),
        "milp_stop_count": stop_count,
        "milp_move_count": move_count,
        "total_waiting_time": metrics["total_waiting_time"],
        "avg_waiting_time": metrics["avg_waiting_time"],
        "total_boarded": metrics["total_boarded"],
        "avg_headway_std": avg_headway_std,
        "benchmark_results_df": benchmark_results_df,
        "station_events_df": station_events_df,
    }


# ================= 主函数 =================

def _parse_args():
    parser = argparse.ArgumentParser(description="MILP Ground Truth Benchmark (Fast Version)")
    parser.add_argument("--topology", type=int, default=4, help="拓扑ID")
    parser.add_argument("--steps", type=int, default=7200, help="模拟步数")
    parser.add_argument("--time_limit", type=int, default=3600, help="MILP求解时间限制(秒)")
    parser.add_argument("--mip_gap", type=float, default=0.005, help="MIP Gap容差")
    parser.add_argument("--decision_interval", type=int, default=120, help="决策间隔(时间步)")
    parser.add_argument("--output", type=str, default=None, help="结果输出路径")
    parser.add_argument("--results-dir", type=str, default=None, help="默认结果输出目录")
    parser.add_argument("--seed", type=int, default=111, help="随机种子")
    parser.set_defaults(eval_marl=True)
    parser.add_argument("--eval_marl", dest="eval_marl", action="store_true", help="在MARL环境中评估MILP轨迹以获取准确reward (默认开启)")
    parser.add_argument("--no_eval_marl", dest="eval_marl", action="store_false", help="跳过MARL环境回放评估")
    parser.add_argument("--optimal_control", action="store_true", help="使用精确控制策略作为Ground Truth (跳过MILP求解)")
    parser.add_argument("--target_dwell", type=int, default=30, help="目标停站时间(秒)，用于精确控制策略")
    parser.add_argument("--skip_milp", action="store_true", help="跳过MILP求解，仅评估已有轨迹（需配合--load_trajectory）")
    parser.add_argument("--save_trajectory", type=str, default=None, help="保存轨迹到指定路径")
    parser.add_argument("--load_trajectory", type=str, default=None, help="从指定路径加载轨迹")
    add_scenario_args(parser)
    return parser.parse_args()


def main():
    args = _parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    scenario_config = scenario_config_from_args(args)
    suffix = scenario_suffix(scenario_config)
    
    topology_defaults = TOPOLOGY_DEFAULTS.get(args.topology, {"n_agents": 10, "tick_seconds": 1})
    n_agents = topology_defaults.get("n_agents", 10)
    tick_seconds = topology_defaults.get("tick_seconds", 1)
    
    # ================= 精确控制模式 (Ground Truth) =================
    if args.optimal_control:
        logger.info("=" * 60)
        logger.info("Ground Truth: 精确控制策略")
        logger.info("=" * 60)
        logger.info(f"拓扑: {args.topology}")
        logger.info(f"智能体数量: {n_agents}")
        logger.info(f"总步数: {args.steps}")
        logger.info(f"目标停站时间: {args.target_dwell}秒")
        
        result = evaluate_with_optimal_control(
            topology_id=args.topology,
            total_steps=args.steps,
            tick_seconds=tick_seconds,
            seed=args.seed,
            device="cpu",
            target_dwell_time=args.target_dwell,
            scenario_config=scenario_config,
        )
        
        # 保存结果
        results_dir = args.results_dir or os.path.join(project_root, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        output_path = args.output or os.path.join(
            results_dir, f"ground_truth_optimal_control_topology{args.topology}{suffix}.csv"
        )
        
        metrics = {
            "status": "optimal_control",
            "topology": args.topology,
            "n_agents": n_agents,
            "total_steps": args.steps,
            "target_dwell_time": args.target_dwell,
            "total_reward": result["total_reward"],
            "avg_step_reward": result["avg_step_reward"],
            "num_steps": result["num_steps"],
            "avg_waiting_time": result.get("avg_waiting_time", 0.0),
            "total_boarded": result.get("total_boarded", 0.0),
            "avg_headway_std": result.get("avg_headway_std", 0.0),
        }
        
        df = pd.DataFrame([metrics])
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"结果已保存至: {output_path}")
        
        return metrics
    
    # ================= MILP求解模式 =================
    # 使用命令行指定的decision_interval（容量约束已放宽，无需根据规模调整）
    decision_interval = args.decision_interval
    
    logger.info("=" * 60)
    logger.info("MILP Ground Truth Benchmark (Fast Version)")
    logger.info("=" * 60)
    logger.info(f"拓扑: {args.topology}")
    logger.info(f"智能体数量: {n_agents}")
    logger.info(f"总步数: {args.steps}")
    logger.info(f"决策间隔: {decision_interval}步")
    logger.info(f"时间限制: {args.time_limit}秒")
    logger.info(f"MIP Gap: {args.mip_gap}")
    
    solver = FastMILPSolver(
        topology_id=args.topology,
        n_agents=n_agents,
        total_steps=args.steps,
        decision_interval=decision_interval,
        tick_seconds=tick_seconds,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        custom_disruptions=custom_disruptions_for_milp(scenario_config),
    )
    
    # 加载或求解轨迹
    import pickle
    
    if args.load_trajectory and os.path.exists(args.load_trajectory):
        logger.info(f"从文件加载轨迹: {args.load_trajectory}")
        with open(args.load_trajectory, 'rb') as f:
            saved_data = pickle.load(f)
        metrics = saved_data.get("metrics", {})
        trajectory = saved_data.get("trajectory", None)
        if trajectory:
            metrics["trajectory"] = trajectory
    elif not args.skip_milp:
        metrics = solver.solve()
    else:
        logger.error("需要 --load_trajectory 或取消 --skip_milp")
        return None
    
    # 保存轨迹
    if args.save_trajectory and "trajectory" in metrics:
        logger.info(f"保存轨迹到: {args.save_trajectory}")
        with open(args.save_trajectory, 'wb') as f:
            pickle.dump({
                "metrics": {k: v for k, v in metrics.items() if k != "trajectory"},
                "trajectory": metrics["trajectory"],
            }, f)
    
    logger.info("=" * 60)
    logger.info("Ground Truth 结果 (MILP目标)")
    logger.info("=" * 60)
    logger.info(f"状态: {metrics.get('status', 'unknown')}")
    
    def safe_log(name, key, fmt=".4f"):
        val = metrics.get(key)
        if val is not None and isinstance(val, (int, float)):
            logger.info(f"{name}: {val:{fmt}}")
        else:
            logger.info(f"{name}: N/A")
    
    safe_log("原始目标值 (Raw Reward)", "raw_reward", ".4f")
    safe_log("MILP估计总Reward", "total_reward", ".4f")
    safe_log("总服务次数", "total_services", ".0f")
    safe_log("MILP代理Headway标准差(秒)", "milp_proxy_avg_headway_std", ".2f")
    safe_log("模型构建时间(秒)", "build_time_sec", ".2f")
    safe_log("求解时间(秒)", "solve_time_sec", ".2f")
    
    # ================= MARL环境评估 =================
    marl_eval_result = None
    replay_benchmark_df = None
    replay_station_df = None
    if args.eval_marl and "trajectory" in metrics and metrics["trajectory"]:
        try:
            marl_eval_result = evaluate_trajectory_in_marl(
                trajectory=metrics["trajectory"],
                topology_id=args.topology,
                total_steps=args.steps,
                tick_seconds=tick_seconds,
                seed=args.seed,
                device="cpu",
                scenario_config=scenario_config,
            )
            
            # 更新metrics
            metrics["marl_total_reward"] = marl_eval_result["total_reward"]
            metrics["marl_avg_step_reward"] = marl_eval_result["avg_step_reward"]
            metrics["marl_num_steps"] = marl_eval_result["num_steps"]
            metrics["marl_total_waiting_time"] = marl_eval_result.get("total_waiting_time", 0.0)
            metrics["marl_avg_waiting_time"] = marl_eval_result.get("avg_waiting_time", 0.0)
            metrics["marl_total_boarded"] = marl_eval_result.get("total_boarded", 0.0)
            metrics["marl_avg_headway_std"] = marl_eval_result.get("avg_headway_std", 0.0)
            replay_benchmark_df = marl_eval_result.get("benchmark_results_df")
            replay_station_df = marl_eval_result.get("station_events_df")

            # 统一输出口径: avg_* 列均为MARL环境真实回放指标
            metrics["avg_step_reward"] = metrics["marl_avg_step_reward"]
            metrics["avg_waiting_time"] = metrics["marl_avg_waiting_time"]
            metrics["avg_headway_std"] = metrics["marl_avg_headway_std"]
            
            logger.info("=" * 60)
            logger.info("MILP轨迹在MARL环境中的评估结果")
            logger.info("=" * 60)
            logger.info(f"MILP代理目标值: {metrics.get('raw_reward', 'N/A'):.2f} (仅用于优化)")
            logger.info(f"MARL真实总Reward: {marl_eval_result['total_reward']:.4f}")
            logger.info(f"MARL平均每步Reward: {marl_eval_result['avg_step_reward']:.6f}")
            logger.info(f"MARL平均乘客等待时间: {marl_eval_result.get('avg_waiting_time', 0):.2f} 秒")
            logger.info(f"MARL平均Headway标准差: {marl_eval_result.get('avg_headway_std', 0):.2f} 秒")
            logger.info(f"MILP决策: 停留={marl_eval_result.get('milp_stop_count', 'N/A')}, 移动={marl_eval_result.get('milp_move_count', 'N/A')}")
            
        except Exception as e:
            logger.error(f"MARL环境评估失败: {e}")
            import traceback
            traceback.print_exc()

    # 统一保证CSV包含真实回放所需字段；未回放时置空，避免与MILP代理值混淆
    if "avg_step_reward" not in metrics:
        metrics["avg_step_reward"] = None
    if "avg_waiting_time" not in metrics:
        metrics["avg_waiting_time"] = None
    if "marl_avg_headway_std" not in metrics:
        metrics["avg_headway_std"] = None
    
    # 保存结果
    results_dir = args.results_dir or os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = args.output or os.path.join(
        results_dir, f"milp_ground_truth_topology{args.topology}{suffix}.csv"
    )

    output_dir = os.path.dirname(output_path) or results_dir
    output_stem = os.path.splitext(os.path.basename(output_path))[0]
    benchmark_output_path = os.path.join(output_dir, f"{output_stem}_benchmark_results.csv")
    station_output_path = os.path.join(output_dir, f"{output_stem}_station_events.csv")
    metrics_output_path = os.path.join(output_dir, f"{output_stem}_metrics.csv")
    
    # 不保存trajectory到CSV
    metrics_for_csv = {k: v for k, v in metrics.items() if k != "trajectory"}
    df = pd.DataFrame([metrics_for_csv])
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"结果已保存至: {output_path}")

    # 兼容benchmark脚本产物：补齐benchmark_results.csv和station_events.csv
    if replay_benchmark_df is not None:
        replay_benchmark_df.to_csv(benchmark_output_path, index=False, encoding="utf-8-sig")
        logger.info(f"Benchmark逐步结果已保存至: {benchmark_output_path}")
    else:
        pd.DataFrame(
            columns=[
                "policy",
                "step",
                "reward",
                "decision_time_sec",
                "decision_source",
                "waiting_passengers",
                "boarded_passengers",
            ]
        ).to_csv(benchmark_output_path, index=False, encoding="utf-8-sig")
        logger.info(f"MARL回放未执行，已写入空benchmark结果模板: {benchmark_output_path}")

    if replay_station_df is not None:
        replay_station_df.to_csv(station_output_path, index=False, encoding="utf-8-sig")
        logger.info(f"站点事件记录已保存至: {station_output_path}")
    else:
        pd.DataFrame(
            columns=[
                "policy",
                "topology",
                "env_index",
                "agent",
                "station",
                "event_type",
                "step",
                "sim_time_tick",
                "tick_seconds",
                "sim_time_seconds",
            ]
        ).to_csv(station_output_path, index=False, encoding="utf-8-sig")
        logger.info(f"MARL回放未执行，已写入空站点事件模板: {station_output_path}")

    # 额外输出精简指标文件，便于与benchmark脚本对齐
    pd.DataFrame(
        [
            {
                "total_waiting_time": metrics.get("marl_total_waiting_time"),
                "total_boarded": metrics.get("marl_total_boarded"),
                "avg_waiting_time": metrics.get("avg_waiting_time"),
                "avg_headway_std": metrics.get("avg_headway_std"),
                "avg_step_reward": metrics.get("avg_step_reward"),
            }
        ]
    ).to_csv(metrics_output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Ground Truth指标文件已保存至: {metrics_output_path}")
    
    return metrics


if __name__ == "__main__":
    main()
