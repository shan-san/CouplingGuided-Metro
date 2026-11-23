from __future__ import annotations

from typing import Dict, Optional, Union, TYPE_CHECKING

from topology.node_type import NodeType
from topology.passenger_flow import PassengerFlow

import os
import torch
import numpy as np

if TYPE_CHECKING:
    from topology.route import Route

ALLOWED_PASSENGER_STATIONS_MAP = {
    1: {
        "d1", "s11", "s12", "s13", "s14", "s15", "s16", "d2", "s21", "s22", "s23", "s24", "s25", "s26",
    },
    2: {
        "d1", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "d2", "s21", "s22", "s23", "s24", "s25", "s26", "s27",
    },
    3: {
        "机场东上行","后瑞上行","固戍上行","西乡上行","坪洲上行","宝体上行","宝安中心上行","前海湾上行",
        "鲤鱼门上行","大新上行","桃园上行","深大上行","高新园上行","白石洲上行","世界之窗上行","华侨城上行",
        "侨城东上行","竹子林上行","车公庙上行","香蜜湖上行","购物公园上行","会展中心上行","岗厦上行","华强路上行",
        "科学馆上行","大剧院上行","老街上行","国贸上行","罗湖 2 道",
        "机场东下行","后瑞下行","固戍下行","西乡下行","坪洲下行","宝体下行","宝安中心下行","前海湾下行",
        "鲤鱼门下行","大新下行","桃园下行","深大下行","高新园下行","白石洲下行","世界之窗下行","华侨城下行",
        "侨城东下行","竹子林下行","车公庙下行","香蜜湖下行","购物公园下行","会展中心下行","岗厦下行","华强路下行",
        "科学馆下行","大剧院下行","老街下行","国贸下行","罗湖 3 道",
    },
    4: {
        "d1", "d2",
        "s11", "s12", "s13", "s14", "s15",
        "s21", "s22", "s23", "s24", "s25",
    },
    5: {
        "d1", "d2",
        "s11", "s12", "s13", "s14", "s15", "s16", "s17",
        "s21", "s22", "s23", "s24", "s25", "s26", "s27",
    },
    6: {
        "d1", "d2",
        "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19",
        "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29",
    },
    7: {
        "d1", "d2",
        "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", 
        "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", 
    },
    8: {
        "d1", "d2",
        "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18",
        "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28",
    },
    9: {
        "d1", "d2",
        "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18",
        "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28",
    },
    10: {
        "d1", "d2",
        "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19",
        "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29",
    },
    11: {
        "d1", "d2",
        "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19",
        "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29",
    },
    12: {
        "d1", "d2",
        "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19",
        "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29",
    },
    13: {
        "d1", "d2",
        "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18",
        "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28",
    },
    14: {
        "d1", "d2",
        "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s110", "s111", "s112", "s113",
        "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s210", "s211", "s212", "s213",
    },
    15: {
        "d1", "d2",
        "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s110", "s111", "s112", "s113", "s114", "s115", "s116", "s117", "s118", "s119", "s120", "s121", "s122", "s123", "s124",
        "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s210", "s211", "s212", "s213", "s214", "s215", "s216", "s217", "s218", "s219", "s220", "s221", "s222", "s223", "s224",
    },
    16: {
        "d1", "d2",
        "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s110", "s111", "s112", "s113", "s114", "s115", "s116", "s117", "s118", "s119", "s120", "s121", "s122", "s123", "s124", "s125", "s126", "s127", "s128", "s129", "s130",
        "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s210", "s211", "s212", "s213", "s214", "s215", "s216", "s217", "s218", "s219", "s220", "s221", "s222", "s223", "s224", "s225", "s226", "s227", "s228", "s229", "s230",
    },
}

from topology.metro_lines import LINE_SPECS, platform_names
ALLOWED_PASSENGER_STATIONS_MAP.update({tid: platform_names(tid) for tid in LINE_SPECS})

_active_topology = None
ALLOWED_PASSENGER_STATIONS = frozenset()

def set_passenger_whitelist(topology_id: int):
    global ALLOWED_PASSENGER_STATIONS, _active_topology
    if topology_id not in ALLOWED_PASSENGER_STATIONS_MAP:
        raise ValueError(f"未知拓扑编号: {topology_id}")
    ALLOWED_PASSENGER_STATIONS = frozenset(ALLOWED_PASSENGER_STATIONS_MAP[topology_id])
    _active_topology = topology_id

set_passenger_whitelist(int(os.environ.get("METRO_TOPOLOGY", "1")))

def is_allowed_station_name(name: str) -> bool:
    return (name or "").strip() in ALLOWED_PASSENGER_STATIONS

# 新增：类型常量与工具函数
TYPE_PLATFORM = 0              # 普通平台
TYPE_SEGMENT = 1               # 区间节点
TYPE_PLATFORM_WHITELIST = 2    # 白名单平台

def is_whitelist_node(node) -> bool:
    return getattr(node, "type", None) == TYPE_PLATFORM_WHITELIST

class Node:
    """
    A class used to represent a Node.
    Attributes
    ----------
    id : int
        The unique identifier for the node.
    weight : int
        The weight associated with the node.
    Methods
    -------
    __repr__()
        Returns a string representation of the Node instance.
    """
    def __init__(self, id, weight, node_type: Optional[NodeType] = None):
        self.id = id
        self.weight = weight
        self.weight_upper_bound = weight
        if self.weight > 1:
            self.weight_upper_bound = self.weight + 10
        self.bfs_tree = None
        self.bfs_tree_nodes = None
        # self.type : int = 1   # 0: platform, 1: segment , 2: switchnode
        self.is_upstream_node : bool # 是否是上游节点
        self.switch_group_id = 0 # 道岔组id，默认为0，表示不是道岔节点（两个道岔为一组）
        self.node_type = node_type or NodeType.POSITION

        # Mapping of route id -> next node id
        self._next_nodes: Dict[Union[int, str], int] = {}

        # 客流控制器
        self.passenger_flow = PassengerFlow(node=self)

    def __repr__(self):
        return f"Node(id={self.id}, weight={self.weight})"

    @property
    def waiting_passengers(self):
        return self.passenger_flow.waiting_passengers

    @waiting_passengers.setter
    def waiting_passengers(self, value):
        self.passenger_flow.waiting_passengers = value

    @property
    def passenger_arrival_rate(self):
        return self.passenger_flow.arrival_rate

    @passenger_arrival_rate.setter
    def passenger_arrival_rate(self, value):
        self.passenger_flow.arrival_rate = value

    @property
    def passenger_destinations(self):
        return self.passenger_flow.destinations

    @passenger_destinations.setter
    def passenger_destinations(self, value):
        self.passenger_flow.destinations = value

    @property
    def type(self) -> NodeType:
        return self.node_type

    @type.setter
    def type(self, value):
        if isinstance(value, NodeType):
            self.node_type = value
        elif isinstance(value, int):
            mapping = {
                0: NodeType.PLATFORM,
                1: NodeType.POSITION,
                2: NodeType.SWITCH,
            }
            if value not in mapping:
                raise ValueError(f"Unsupported node type code: {value}")
            self.node_type = mapping[value]
        elif isinstance(value, str):
            try:
                self.node_type = NodeType(value.lower())
            except ValueError as exc:
                raise ValueError(f"Unsupported node type name: {value}") from exc
        else:
            raise TypeError("Node type must be a NodeType, int, or str")

    def init_passenger_tensor(self, num_envs, device):
        """初始化乘客张量，用于并行环境"""
        self.passenger_flow.init_passenger_tensor(num_envs, device)

    def generate_passengers(self, env_index=None, all_platform_nodes=None, current_time=None):
        """按泊松分布生成乘客"""
        self.passenger_flow.generate_passengers(
            env_index=env_index,
            all_platform_nodes=all_platform_nodes,
            current_time=current_time,
        )

    def _init_destination_distribution(self, all_platform_nodes):
        """初始化乘客目的地分布"""
        self.passenger_flow._init_destination_distribution(all_platform_nodes)

    # ------------------------------------------------------------------
    # Connectivity helpers
    # ------------------------------------------------------------------
    def reset_connectivity(self):
        """Remove all stored next-node pointers for this node."""
        self._next_nodes.clear()

    def register_next_node(self, route_id: Union[int, str], next_node: Optional["Node"]):
        """Register the next node for a specific route.

        Args:
            route_id: Identifier of the route.
            next_node: The next node along the route, or None to clear.
        """
        if next_node is None:
            self._next_nodes.pop(route_id, None)
            return

        self._next_nodes[route_id] = next_node.id

    def getnext(self, route: Optional[Union[int, str, "Route"]] = None):
        """Look up next-node connectivity.

        Without arguments returns a mapping ``route_id -> node_id``; with a
        ``route`` (object or identifier) returns the next node id or ``None``.
        """
        if route is None:
            return dict(self._next_nodes)

        route_id = getattr(route, "id", route)
        return self._next_nodes.get(route_id)
