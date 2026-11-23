import random
from tracemalloc import start
import math

import networkx as nx
import numpy as np
from torch import instance_norm
from typing import Dict, List, Optional, Tuple
"""
This module builds a topology of nodes and edges from given platforms and line segments,
and calculates coordinates for the nodes using NetworkX.
Classes:
    Platform: Represents a platform with an ID and dwell time.
    Node: Represents a node in the topology.
    Edge: Represents an edge between two nodes in the topology.
    LineSegment: Represents a line segment between two platforms.
Functions:
    build_topology(platforms, line_segments, default_interval=1):
        Converts platforms to nodes and line segments to edges, splitting line segments
        into multiple nodes based on travel time.
    calc_coordinates_with_networkx(nodes, edges):
        Calculates the coordinates of nodes using NetworkX's spring layout algorithm.
Usage:
    Create Platform and LineSegment objects, then use build_topology to create nodes and edges.
    Use calc_coordinates_with_networkx to calculate the coordinates of the nodes.
"""

from facility.platform import Platform
from collections import deque
from facility.switchnode import SwitchNode
from topology.node import Node
from topology.node_type import NodeType
from topology.edge import Edge
from tr.linesegment import LineSegment
from logger import logger

from topology.node import Node
from topology.edge import Edge
from topology.route import Route, RouteType

import linedata
from linedata import routes

# 依据名称赋予可以上下客的车站白名单类型
from topology.node import TYPE_PLATFORM, TYPE_PLATFORM_WHITELIST, is_allowed_station_name

INTERMEDIATE_NODE_START = 100
EDGE_START = 1
MAX_BFS_NODES = 100

def _seconds_to_ticks(seconds: int | float, tick_seconds: int) -> int:
    """把秒转换为tick数，向上取整，至少为1（若秒>0）"""
    import math
    if seconds is None:
        seconds = 0
    seconds = max(0, float(seconds))
    if seconds == 0:
        return 0
    return max(1, int(math.ceil(seconds / float(tick_seconds))))



def build_topology(
    platforms: List[Platform],
    line_segments: Optional[List[LineSegment]] = None,
    default_interval: int = 1,
) -> Tuple[
    Dict[int, Node],
    Dict[int, Node],
    List[Edge],
    Dict[LineSegment, Tuple[int, int]],
    Dict[int, List[LineSegment]],
    Dict[LineSegment, List[Node]],
    Dict[int, Node],
]:
# def build_topology(platforms, line_segments, default_interval=1):
    nodes: Dict[int, Node] = {}
    platform_nodes: Dict[int, Node] = {}
    edges: List[Edge] = []
    segments: Dict[LineSegment, Tuple[int, int]] = {}
    node2segments: Dict[int, List[LineSegment]] = {}
    segments2nodes: Dict[LineSegment, List[Node]] = {}
    switch_nodes: Dict[int, Node] = {}
    switchnode_startnode_name: List[str] = []
    # intermediate_nodes = {}  #存储可能通往道岔的中间节点
    
    # 1. build 'line_segments' from the routes
    # If no line_segments provided (or empty), derive them from routes
    if not line_segments:
        seg_map: Dict[Tuple[int, int], LineSegment] = {}

        def add_seg(a, b):
            key = (a.id, b.id)
            if key not in seg_map:
                seg_map[key] = LineSegment(a, b)

        for route in routes:
            nl = route.node_list or []
            if len(nl) < 2:
                continue
            # consecutive pairs
            for i in range(len(nl) - 1):
                add_seg(nl[i], nl[i + 1])
            # close the loop if circular
            if route.route_type == RouteType.CIRCULAR:
                add_seg(nl[-1], nl[0])

        line_segments = list(seg_map.values())

    # Convert platforms to nodes
    for platform in platforms:
        dwell_seconds = getattr(platform, "dwell", 0) or 0
        # 站台停留tick数（向上取整，保留余数）
        scaled_dwell_ticks = _seconds_to_ticks(dwell_seconds, default_interval)
        raw_node_type = getattr(platform, "node_type", None)
        if isinstance(raw_node_type, NodeType):
            node_type = raw_node_type
        elif isinstance(raw_node_type, str):
            try:
                node_type = NodeType(raw_node_type.lower())
            except ValueError:
                node_type = None
        else:
            node_type = None

        if node_type is None:
            node_type = NodeType.PLATFORM if is_allowed_station_name(platform.name) else NodeType.POSITION

        node = Node(platform.id, scaled_dwell_ticks, node_type=node_type)
        node.platform_name = getattr(platform, "name", None)
        if hasattr(platform, "station_name"):
            node.station_name = getattr(platform, "station_name")
        if hasattr(platform, "od_profile"):
            node.od_profile = getattr(platform, "od_profile")
        # 记录真实秒和tick粒度，便于其它模块/GUI转换
        node.tick_seconds = default_interval
        node.dwell_seconds = int(dwell_seconds)
        node.weight = scaled_dwell_ticks
        # 若 Node 定义了上界属性，保持一致，避免BFS等使用未缩放值
        if not hasattr(node, "weight_upper_bound"):
            node.weight_upper_bound = node.weight
        # 标记node是否为可上下车的车站节点
        node.node_type = NodeType.PLATFORM if is_allowed_station_name(platform.name) else NodeType.POSITION
        nodes[platform.id] = node

        if node.type is NodeType.PLATFORM:
            platform_nodes[platform.id] = node

    # Log the segments being used and set random travel time for each segments
    if line_segments:
        pass
        # logger.info(f"Using {len(line_segments)} line segments before expansion:")
        # for seg in line_segments:
        #     sp = getattr(seg.start_platform, 'name', seg.start_platform.id)
        #     ep = getattr(seg.end_platform, 'name', seg.end_platform.id)
            # logger.info(f"  {sp} -> {ep} (id={seg.id})")

    def _clean_station_name(name: str | None) -> str:
        if not name:
            return ""
        return str(name).replace("上行", "").replace("下行", "").strip()

    line1_segment_times = {
        ("罗湖", "国贸"): 120,
        ("国贸", "老街"): 60,
        ("老街", "大剧院"): 120,
        ("大剧院", "科学馆"): 120,
        ("科学馆", "华强路"): 120,
        ("华强路", "岗厦"): 120,
        ("岗厦", "会展中心"): 120,
        ("会展中心", "购物公园"): 60,
        ("购物公园", "香蜜湖"): 180,
        ("香蜜湖", "车公庙"): 120,
        ("车公庙", "竹子林"): 120,
        ("竹子林", "侨城东"): 120,
        ("侨城东", "华侨城"): 120,
        ("华侨城", "世界之窗"): 120,
        ("世界之窗", "白石洲"): 120,
        ("白石洲", "高新园"): 120,
        ("高新园", "深大"): 120,
        ("深大", "桃园"): 180,
        ("桃园", "大新"): 120,
        ("大新", "鲤鱼门"): 120,
        ("鲤鱼门", "前海湾"): 120,
        ("前海湾", "新安"): 120,
        ("新安", "宝安中心"): 120,
        ("宝安中心", "宝体"): 120,
        ("宝体", "坪洲"): 120,
        ("坪洲", "西乡"): 120,
        ("西乡", "固戍"): 120,
        ("固戍", "后瑞"): 120,
        ("后瑞", "机场东"): 120,
    }

    active_topology = getattr(linedata, "ACTIVE_TOPOLOGY", None)
    from topology.metro_lines import LINE_SPECS, depot_segment_times
    depot_times = depot_segment_times(active_topology) if active_topology in LINE_SPECS else {}
    for segment in line_segments:
        pair = (segment.start_platform.id, segment.end_platform.id)
        if pair in depot_times:
            segment.set_travel_time(depot_times[pair])
            continue
        if active_topology == 16:
            start_name = _clean_station_name(getattr(segment.start_platform, "station_name", None))
            end_name = _clean_station_name(getattr(segment.end_platform, "station_name", None))
            base_time = line1_segment_times.get((start_name, end_name))
            if base_time is None:
                base_time = line1_segment_times.get((end_name, start_name))
            if base_time is not None:
                delta = random.randint(-15, 15)
                segment.set_travel_time(max(1, int(base_time + delta)))
                continue

        random_travel_time = random.randint(100, 180)
        segment.set_travel_time(random_travel_time)

    # Convert line segments to nodes and edges
    start_node_id = INTERMEDIATE_NODE_START
    start_edge_id = EDGE_START

    for segment in line_segments:
        start_node = nodes[segment.start_platform.id]
        end_node = nodes[segment.end_platform.id]
        travel_time = int(math.ceil(segment.weight)) if default_interval == 1 else int(segment.weight)
        segments[segment] = (start_node.id, end_node.id)

        node2segments[start_node.id] = []
        node2segments[start_node.id].append(segment)

        # node2segments[start_node.id] = segment
        segments2nodes[segment] = []
        segments2nodes[segment].append(start_node)
        
        # Split the line segment into multiple nodes based on travel time
        interval = default_interval if default_interval > 0 else 1
        num_intervals = travel_time // interval if interval else 0
        assigned_weight = 0
        previous_node = start_node
        remaining = max(0, travel_time)

        for i in range(1, int(num_intervals)):
            part_seconds = min(default_interval, remaining)  # 该tick覆盖的秒数
            start_node_id += 1
            intermediate_node_id = start_node_id

            # 中间节点的停留时长=1 tick
            intermediate_node = Node(intermediate_node_id, 1, node_type=NodeType.POSITION)
            intermediate_node.tick_seconds = default_interval
            intermediate_node.weight = 1
            if not hasattr(intermediate_node, "weight_upper_bound"):
                intermediate_node.weight_upper_bound = 1
            nodes[intermediate_node_id] = intermediate_node
            segments2nodes[segment].append(intermediate_node)
            node2segments[intermediate_node_id] = segment
            # 边仅用于连接，并保留该tick对应的秒数（便于可视化/参考）
            e = Edge(start_edge_id, previous_node, intermediate_node, part_seconds)
            edges.append(e)
            start_edge_id += 1


            previous_node = intermediate_node
            remaining -= part_seconds


        e_last = Edge(start_edge_id, previous_node, end_node, remaining if default_interval == 1 else 0)
        edges.append(e_last)
        start_edge_id += 1

        segments2nodes[segment].append(end_node)
        # 将线段的终点节点也添加到 node2segments 映射中
        node2segments[end_node.id] = segment

    for node in nodes.values():
        node.reset_connectivity()

    for route in routes:
        fill_route_nodes_edges(route, segments2nodes)

    build_route_connectivity(routes)

    # Detailed segment list after expansion
    # logger.info("Detailed segment node sequences (after expansion):")
    # for seg, node_list in segments2nodes.items():
    #     sp = getattr(seg.start_platform, 'name', seg.start_platform.id)
    #     ep = getattr(seg.end_platform, 'name', seg.end_platform.id)
    #     node_ids = [n.id for n in node_list]
        # logger.info(f"  {sp} -> {ep} (travel={seg.weight}) nodes={node_ids}")

    logger.info(
        f"Build finished: nodes={len(nodes)}, platform_nodes={len(platform_nodes)}, edges={len(edges)}, segments={len(segments)}"
    )

    return nodes, platform_nodes, edges, segments, node2segments, segments2nodes, switch_nodes

def calc_coordinates_with_networkx(
    nodes: Dict[int, Node],
    edges: List[Edge],
    bounded: bool = False,
) -> Dict[int, Node]:
    G = nx.Graph()

    # Add nodes to the graph
    for node_id, node in nodes.items():
        G.add_node(node_id, weight=node.weight)

    # Add edges to the graph
    for edge in edges:
        G.add_edge(edge.start_node.id, edge.end_node.id, weight=edge.weight)

    # Calculate the coordinates using networkx layout
    pos = ({node_id: (float(index), 0.) for index, node_id in enumerate(nodes)}
           if bounded else nx.spring_layout(G, scale=2))

    # calculate the bfs sub tree
    for node_id, node in nodes.items():
        feature_nodes = 16 if bounded else MAX_BFS_NODES
        if bounded:
            T = nx.DiGraph()
            T.add_node(node_id)
            queue = deque([node_id])
            while queue and len(T) < feature_nodes:
                current = queue.popleft()
                for neighbor in G.neighbors(current):
                    if neighbor not in T:
                        T.add_edge(current, neighbor)
                        queue.append(neighbor)
                        if len(T) >= feature_nodes:
                            break
        else:
            T = nx.bfs_tree(G, source=node_id, depth_limit=200)
        bfs_nodes, adjacency = bfs_tree_to_weighted_matrix(
            G, T, nodes, max_nodes=feature_nodes
        )
        node.bfs_tree_nodes = bfs_nodes
        # Native rollouts copy these fixed features; array copies avoid walking
        # millions of Python float objects without sharing mutable node state.
        node.bfs_tree = np.asarray(adjacency) if bounded else adjacency
        # print (f"Node {node_id} has {len(T)} nodes in its BFS tree")

    # Update node coordinates
    for node_id, (x, y) in pos.items():
        node = nodes.get(node_id)
        if node is None:
            continue
        node.x = float(x)
        node.y = float(y)

    for edge in edges:
        if hasattr(edge.start_node, "id"):
            edge.start_node = nodes.get(edge.start_node.id, edge.start_node)
        if hasattr(edge.end_node, "id"):
            edge.end_node = nodes.get(edge.end_node.id, edge.end_node)

    return nodes

def bfs_tree_to_weighted_matrix(
    graph: nx.Graph,
    tree: nx.DiGraph,
    nodes_lookup: Dict[int, Node],
    max_nodes: int = MAX_BFS_NODES,
) -> tuple[list[int], list[list[float]]]:
    tree_nodes = list(tree.nodes())[:max_nodes]
    adjacency = [[0.0] * max_nodes for _ in range(max_nodes)]
    index_map = {node_id: idx for idx, node_id in enumerate(tree_nodes)}

    # Store node weights on the diagonal to preserve per-node information
    for node_id, idx in index_map.items():
        node = nodes_lookup.get(node_id)
        if node is not None:
            adjacency[idx][idx] = float(getattr(node, "weight", 0.0))

    for src, dst in tree.edges():
        if src not in index_map or dst not in index_map:
            continue
        i = index_map[src]
        j = index_map[dst]
        weight = graph.get_edge_data(src, dst, default={}).get("weight", 1.0)
        weight = float(weight)
        adjacency[i][j] = weight
        adjacency[j][i] = weight

    return tree_nodes, adjacency

def build_adjacency_matrix(nodes: Dict[int, Node], edges: List[Edge], sparse=False):
    G = nx.Graph()

    # Add nodes to the graph
    for node_id, node in nodes.items():
        G.add_node(node_id, weight=node.weight)

    # Add edges to the graph
    for edge in edges:
        G.add_edge(edge.start_node.id, edge.end_node.id, weight=edge.weight)

    matrix = nx.adjacency_matrix(G)
    return matrix if sparse else matrix.todense()

def fill_route_nodes_edges(
    route: Route,
    segments2nodes: Dict[LineSegment, List[Node]],
) -> None:
    route.nodes.clear()
    base_node_list = route.node_list or []
    if len(base_node_list) < 2:
        return

    def append_segment_nodes(node_list: List[Node], skip_terminal: bool = False) -> None:
        """Append segment nodes to the route, avoiding duplicates."""
        if not node_list:
            return

        last_index = len(node_list) - 1
        for idx, node in enumerate(node_list):
            if skip_terminal and idx == last_index:
                continue

            if route.nodes and route.nodes[-1].id == node.id:
                continue

            route.nodes.append(node)

        if not skip_terminal and node_list:
            last_node = node_list[-1]
            if route.nodes and route.nodes[-1].id == last_node.id:
                return
            route.nodes.append(last_node)

    def resolve_segment_nodes(start_node, end_node) -> Optional[List[Node]]:
        for segment, node_list in segments2nodes.items():
            if (
                getattr(segment.start_platform, "id", None) == getattr(start_node, "id", None)
                and getattr(segment.end_platform, "id", None) == getattr(end_node, "id", None)
            ):
                return node_list

        return None

    pairs: List[Tuple] = []
    for idx in range(len(base_node_list) - 1):
        pairs.append((base_node_list[idx], base_node_list[idx + 1]))

    if route.route_type == RouteType.CIRCULAR:
        pairs.append((base_node_list[-1], base_node_list[0]))

    total_pairs = len(pairs)

    for idx, (start_node, end_node) in enumerate(pairs):
        node_list = resolve_segment_nodes(start_node, end_node)
        if not node_list:
            continue

        skip_terminal = route.route_type == RouteType.CIRCULAR and idx == total_pairs - 1
        append_segment_nodes(node_list, skip_terminal=skip_terminal)


def build_route_connectivity(routes: List[Route]) -> None:
    for route in routes:
        node_sequence = getattr(route, "nodes", [])
        if not node_sequence:
            continue

        total = len(node_sequence)
        for idx, node in enumerate(node_sequence):
            if not hasattr(node, "register_next_node"):
                continue

            next_node = None
            if idx + 1 < total:
                next_node = node_sequence[idx + 1]
            elif route.route_type == RouteType.CIRCULAR and total > 1:
                next_node = node_sequence[0]

            node.register_next_node(route.id, next_node)
