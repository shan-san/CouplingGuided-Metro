from dataclasses import dataclass
import torch
from typing import Dict, List, Optional, Tuple

from logger import logger

from route_selection import RouteSelectionAdapter, RandomRouteSelectionPolicy
from timetable import TimetableRun, TimetableStop

try:
    from linedata import routes as _ALL_ROUTES
except ImportError:  # pragma: no cover - fallback when route data unavailable
    _ALL_ROUTES = []
from topology.node import is_whitelist_node

class MovingState:
    """Environment running state"""
    STOPPED = 0
    RUNNING = 1

class Action:
    """Environment action"""
    STOP = 0               # 停留
    MOVE_DEFAULT = 1       # 移动（默认路径）



@dataclass
class Train:
    """Train state"""
    position: torch.Tensor        # Position on the current edge
    current_edge: torch.Tensor    # ID of the current edge
    current_node: torch.Tensor    # ID of the current node
    current_route: torch.Tensor   # Encoded ID of the current route
    target_station: torch.Tensor  # ID of the target station
    arrival_time: torch.Tensor    # Arrival time
    schedule_time: torch.Tensor   # Scheduled arrival time
    passenger_count: torch.Tensor # Current number of passengers
    is_running: torch.Tensor      # Whether the train is running
    dwell_time: torch.Tensor      # Dwell time at the node
    current_expected_dwell_time: torch.Tensor # Expected dwell time
    previous_dwell_time: torch.Tensor # Previous dwell time
    previous_expected_dwell_time: torch.Tensor # Previous expected dwell time
    assigned_run: torch.Tensor    # Tableau run id assigned to the train (-1 if none)
    run_segment_index: torch.Tensor  # Index of current run segment (-1 if none)
    on_assigned_route: torch.Tensor  # Bool tensor marking whether train is on its run
    previous_node: torch.Tensor   # Last node visited (for segment progress tracking)

    speed_limit_delay: torch.Tensor # Delay steps due to speed limit

class MetroAgentV1:
    """Metro train agent"""

    def __init__(
        self,
        name: str,
        num_envs: int,
        device: torch.device,
        **kwargs
    ):
        self.potential_next_pos = None
        self.action = None
        self.world = None
        self.name = name
        self.num_envs = num_envs
        self.device = device
        self.action_result = None
        self.next_node_options = {}  # 存储每个节点的可选下一节点
        self.next_node_choice = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # 选择哪条路径
   

        # Initialize state
        self.state = None
        self.default_route = _ALL_ROUTES[0] if _ALL_ROUTES else None
        self.default_route_key = getattr(self.default_route, "id", None) if self.default_route is not None else None
        self.route_selection_policy = RandomRouteSelectionPolicy()
        self.route_adapter = RouteSelectionAdapter(
            policy=self.route_selection_policy,
            num_envs=self.num_envs,
            default_route_key=self.default_route_key,
        )
        self.Action = Action
        self._default_move_action_value = Action.MOVE_DEFAULT
        self._init_state()

        # Add passenger capacity and count
        self.passenger_capacity = 2520  # Maximum train capacity
        self.passenger_count = torch.zeros(self.num_envs, device=self.device)  # Current passenger count
        self.passengers_by_destination = {}  # Passengers stored by destination
        
        # Cumulative statistics
        self.cumulative_boarded = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_alighted = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_waiting_time = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_delay = torch.zeros(self.num_envs, device=self.device)

        # Run assignment tracking
        self._active_run_id: int = -1
        self._active_run: Optional[TimetableRun] = None
        self._active_run_stops: List[TimetableStop] = []
        self._active_node_to_stop: Dict[int, List[int]] = {}
        self._segment_nodes: Dict[int, List[int]] = {}
        self._run_start_node: Optional[int] = None
        self._last_segment_index = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )

    def _init_state(self):
        """Initialize agent state"""
        self.state = Train(
            position=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            current_edge=torch.ones(self.num_envs, device=self.device, dtype=torch.float32),
            current_node=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            current_route=torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long),
            target_station=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            arrival_time=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            schedule_time=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            passenger_count=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            is_running=torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            dwell_time=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            current_expected_dwell_time=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            previous_dwell_time=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            previous_expected_dwell_time=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            speed_limit_delay=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            assigned_run=torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long),
            run_segment_index=torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long),
            on_assigned_route=torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            previous_node=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
        )

        self.potential_next_pos = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.action_result = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # Current action
        self.action = torch.zeros(
            self.num_envs,
            device=self.device
        )
        self.prev_action = torch.zeros(
            self.num_envs,
            device=self.device
        )

        if hasattr(self, "route_adapter") and self.route_adapter is not None:
            self.route_adapter.reset_all(self)

        self._last_segment_index = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )

        if hasattr(self, "route_adapter") and self.route_adapter is not None:
            self.route_adapter.reset_all(self)

    def reset_state(self, env_index: Optional[int] = None):
        """Reset agent state"""
        if env_index is not None:
            # Reset a single environment
            self.state.position[env_index] = 0
            self.state.current_edge[env_index] = -1
            self.state.target_station[env_index] = 0
            self.state.arrival_time[env_index] = 0
            self.state.schedule_time[env_index] = 0
            self.state.passenger_count[env_index] = 0
            self.cumulative_boarded[env_index] = 0
            self.cumulative_alighted[env_index] = 0
            self.cumulative_waiting_time[env_index] = 0
            self.cumulative_delay[env_index] = 0
            self.action[env_index] = 0
            self.state.dwell_time[env_index] = 0
            self.state.current_expected_dwell_time[env_index] = 0
            self.state.previous_dwell_time[env_index] = 0
            self.state.previous_expected_dwell_time[env_index] = 0
            self.state.speed_limit_delay[env_index] = 0
            self.state.run_segment_index[env_index] = -1
            self.state.on_assigned_route[env_index] = False
            self.state.previous_node[env_index] = 0
            self._last_segment_index[env_index] = -1
            if hasattr(self, "prev_action") and self.prev_action is not None:
                self.prev_action[env_index] = 0
            if hasattr(self, "route_adapter") and self.route_adapter is not None:
                self.route_adapter.reset_env(self, env_index)
            self._reapply_run_assignment(env_index)
        else:
            # Reset all environments
            self._init_state()
            self.cumulative_boarded.zero_()
            self.cumulative_alighted.zero_()
            self.cumulative_waiting_time.zero_()
            self.cumulative_delay.zero_()
            self._reapply_run_assignment()

    def init_position(self, position: torch.Tensor):
        # self.state.position = position
        self.set_position(position)

    def assign_run(self, run: Optional[TimetableRun], world) -> None:
        """Attach a timetable run to this agent for tracking purposes."""
        if run is None or world is None:
            self._active_run_id = -1
            self._active_run = None
            self._active_run_stops = []
            self._active_node_to_stop = {}
            self._segment_nodes = {}
            self._run_start_node = None
            self._reapply_run_assignment()
            return

        (
            stops,
            node_map,
            segment_nodes,
            start_platform,
        ) = self._prepare_run_metadata(run, world)
        self._active_run_id = run.run_id
        self._active_run = run
        self._active_run_stops = stops
        self._active_node_to_stop = node_map
        self._segment_nodes = segment_nodes
        self._run_start_node = start_platform
        self._reapply_run_assignment()

    def get_run_start_node(self) -> Optional[int]:
        return self._run_start_node

    def _reapply_run_assignment(self, env_index: Optional[int] = None) -> None:
        run_id = self._active_run_id if self._active_run_id is not None else -1
        has_schedule = bool(self._active_run_stops)
        segment_value = 0 if has_schedule else -1
        on_route = bool(has_schedule)

        if env_index is None:
            self.state.assigned_run.fill_(run_id)
            self.state.run_segment_index.fill_(segment_value)
            self.state.on_assigned_route.fill_(on_route)
            self._last_segment_index.fill_(segment_value)
        else:
            self.state.assigned_run[env_index] = run_id
            self.state.run_segment_index[env_index] = segment_value
            self.state.on_assigned_route[env_index] = on_route
            self._last_segment_index[env_index] = segment_value

    def _prepare_run_metadata(
        self,
        run: TimetableRun,
        world,
    ) -> Tuple[
        List[TimetableStop],
        Dict[int, List[int]],
        Dict[int, List[int]],
        Optional[int],
    ]:
        stops = list(run.stops)
        if not stops:
            return [], {}, {}, None

        node_map, segment_nodes = self._build_node_to_stop_map(stops, world)
        start_platform = int(stops[0].platform_id) if stops else None
        return stops, node_map, segment_nodes, start_platform

    def _build_node_to_stop_map(
        self,
        stops: List[TimetableStop],
        world,
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        mapping: Dict[int, List[int]] = {}
        segment_nodes: Dict[int, List[int]] = {}
        segment_lookup = getattr(world, "segment2nodes", {}) or {}
        get_sequence = getattr(world, "get_segment_sequence", None)

        for idx, stop in enumerate(stops):
            nodes = self._nodes_for_segment(idx, stops, segment_lookup, get_sequence)
            segment_nodes[idx] = nodes
            for node_id in nodes:
                bucket = mapping.setdefault(int(node_id), [])
                if idx not in bucket:
                    bucket.append(idx)

        return mapping, segment_nodes

    def _nodes_for_segment(
        self,
        index: int,
        stops: List[TimetableStop],
        segment_lookup: Dict[object, List[object]],
        get_sequence,
    ) -> List[int]:
        nodes: List[int] = []
        stop = stops[index]
        next_stop = self._next_stop_for_index(index, stops)
        segment = getattr(stop, "segment", None)
        if segment is not None and segment_lookup:
            sequence = segment_lookup.get(segment)
            if sequence:
                nodes = [int(getattr(node, "id", node)) for node in sequence]

        if not nodes and callable(get_sequence):
            if next_stop is not None:
                sequence = get_sequence(stop.platform_id, next_stop.platform_id)
                if sequence:
                    nodes = [int(node_id) for node_id in sequence]

        if not nodes:
            if next_stop is not None:
                nodes = [int(stop.platform_id), int(next_stop.platform_id)]
            else:
                nodes = [int(stop.platform_id)]

        start_id = int(stop.platform_id)
        if start_id not in nodes:
            nodes.insert(0, start_id)

        # if next_stop is not None:
        #     end_id = int(next_stop.platform_id)
        #     if end_id not in nodes:
        #         nodes.append(end_id)

        return nodes

    @staticmethod
    def _next_stop_for_index(
        index: int,
        stops: List[TimetableStop],
    ) -> Optional[TimetableStop]:
        if not stops:
            return None
        next_index = index + 1
        if next_index < len(stops):
            return stops[next_index]
        return stops[0]

    def _update_run_tracking(
        self,
        env_index: int,
        previous_node: int,
        node_id: int,
        on_platform: bool,
    ) -> None:
        if not self._active_run_stops or not self._active_node_to_stop:
            self.state.run_segment_index[env_index] = -1
            self.state.on_assigned_route[env_index] = False
            return

        total_segments = len(self._active_run_stops)
        current_idx_raw = int(self.state.run_segment_index[env_index].item())
        valid_current_idx = 0 <= current_idx_raw < total_segments
        current_idx = current_idx_raw if valid_current_idx else -1

        if valid_current_idx:
            nodes_current = self._segment_nodes.get(current_idx, [])
            if not nodes_current:
                nodes_current = [int(self._active_run_stops[current_idx].platform_id)]
        else:
            self.state.run_segment_index[env_index] = -1
            nodes_current = []

        node_id_int = int(node_id)
        prev_node_int = int(previous_node)
        cached_idx_source = current_idx if current_idx >= 0 else int(self._last_segment_index[env_index].item())
        cached_idx = cached_idx_source if cached_idx_source >= 0 else -1
        on_current_segment = node_id_int in nodes_current if nodes_current else False

        if not on_current_segment and current_idx >= 0:
            next_idx = (current_idx + 1) % total_segments
            nodes_next = self._segment_nodes.get(next_idx, [])
            if not nodes_next:
                nodes_next = [int(self._active_run_stops[next_idx].platform_id)]
            if node_id_int in nodes_next:
                current_idx = next_idx
                self.state.run_segment_index[env_index] = current_idx
                nodes_current = nodes_next
                on_current_segment = True

        if not on_current_segment:
            candidates = self._active_node_to_stop.get(node_id_int, [])
            prioritized: List[int] = []
            for idx in candidates:
                idx_mod = idx % total_segments
                nodes_for_idx = self._segment_nodes.get(idx_mod, [])
                if not nodes_for_idx:
                    nodes_for_idx = [int(self._active_run_stops[idx_mod].platform_id)]
                if prev_node_int == node_id_int:
                    prioritized.append(idx)
                    continue
                if prev_node_int in nodes_for_idx:
                    try:
                        prev_pos = nodes_for_idx.index(prev_node_int)
                        nodes_for_idx.index(node_id_int, prev_pos)
                    except ValueError:
                        continue
                    prioritized.append(idx)
            search_candidates = prioritized or candidates
            if search_candidates:
                prev_idx = current_idx
                last_idx = int(self._last_segment_index[env_index].item())
                relative_time = self._relative_time(env_index)
                chosen = self._select_stop_index(search_candidates, prev_idx, last_idx, relative_time)
                if chosen is not None:
                    current_idx = chosen % total_segments
                    self.state.run_segment_index[env_index] = current_idx
                    nodes_current = self._segment_nodes.get(current_idx, [])
                    if not nodes_current:
                        nodes_current = [int(self._active_run_stops[current_idx].platform_id)]
                    on_current_segment = node_id_int in nodes_current

        if not on_current_segment:
            self._last_segment_index[env_index] = cached_idx
            self.state.run_segment_index[env_index] = -1
            self.state.on_assigned_route[env_index] = False
            return

        self.state.on_assigned_route[env_index] = True

        next_idx = (current_idx + 1) % total_segments
        next_stop = self._active_run_stops[next_idx]
        if (
            on_platform
            and node_id_int != prev_node_int
        ):
            self.state.run_segment_index[env_index] = next_idx
            self._last_segment_index[env_index] = next_idx
            return

        self._last_segment_index[env_index] = current_idx

    def _select_stop_index(
        self,
        candidates: List[int],
        prev_idx: int,
        last_idx: int,
        relative_time: Optional[float],
    ) -> Optional[int]:
        if not candidates:
            return None

        total_segments = len(self._active_run_stops)
        if total_segments <= 0:
            return None

        best_idx: Optional[int] = None
        best_score: Optional[float] = None

        prev_idx_mod = prev_idx % total_segments if prev_idx >= 0 else prev_idx
        last_idx_mod = last_idx % total_segments if last_idx >= 0 else last_idx

        for idx in candidates:
            idx_mod = idx % total_segments
            start, end = self._stop_schedule_window(idx_mod)
            if relative_time is None:
                score = 0.0
            else:
                score = self._time_distance_window(start, end, relative_time)

            if idx_mod == prev_idx_mod or idx_mod == last_idx_mod:
                score -= 0.1

            if best_score is None or score < best_score:
                best_idx = idx_mod
                best_score = score

        return best_idx

    def _stop_schedule_window(self, index: int) -> Tuple[float, float]:
        if not self._active_run or not self._active_run_stops:
            return 0.0, 0.0

        total = len(self._active_run_stops)
        if total == 0:
            return 0.0, 0.0

        index %= total
        stop = self._active_run_stops[index]
        depart = float(stop.departure_time)
        next_index = (index + 1) % total
        next_stop = self._active_run_stops[next_index]
        arrival = float(next_stop.arrival_time)

        if next_index <= index:
            cycle = float(getattr(self._active_run, "cycle_time", 0.0) or 0.0)
            if cycle > 0.0:
                arrival += cycle

        if arrival <= depart:
            arrival = depart + 1.0

        return depart, arrival

    def _relative_time(self, env_index: int) -> Optional[float]:
        if self._active_run is None:
            return None

        current_time = self._sim_time_value(env_index)
        if current_time is None:
            return None

        return float(current_time)

    @staticmethod
    def _time_distance_window(start: float, end: float, relative_time: float) -> float:
        start = float(start)
        end = float(end)
        relative_time = float(relative_time)

        if start <= relative_time <= end:
            return 0.0

        if relative_time < start:
            return start - relative_time

        return relative_time - end

    def _sim_time_value(self, env_index: int) -> Optional[float]:
        world_time = getattr(self.world, "sim_time", None)
        if world_time is None:
            return None

        try:
            value = world_time[env_index]
        except (IndexError, TypeError):
            return None

        if isinstance(value, torch.Tensor):
            return float(value.item())

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def set_action(self, action: torch.Tensor):
        """设置智能体动作，不再依赖动作选择路径"""
        if not hasattr(self, "prev_action") or self.prev_action is None:
            self.prev_action = torch.zeros_like(action, device=self.device)

        self.prev_action = self.action.clone()
        self.action = action.clone()

        for i in range(self.num_envs):
            current_action = self.action[i].item()
            current_pos = int(self.state.position[i].item())

            if current_action == Action.STOP:
                continue

            if hasattr(self, "route_adapter") and self.route_adapter is not None:
                self.route_adapter.ensure_route(self, i, current_pos)

    def set_position(self, position: torch.Tensor, env_mask: Optional[List[int]] = None):
        if env_mask is None:
            indices = range(self.num_envs)
        else:
            if isinstance(env_mask, int):  # type: ignore[arg-type]
                indices = [env_mask]
            else:
                indices = list(env_mask)

        platform_nodes = getattr(self.world, "platform_nodes", None)
        if platform_nodes is None:
            platform_nodes = getattr(self.world, "platfrom_nodes", set())

        for i in indices:
            previous_node = int(self.state.position[i].item())
            new_node = int(position[i].item())
            moved = new_node != previous_node
            on_platform = new_node in platform_nodes
            self.state.previous_node[i] = float(previous_node)

            if not moved:  # stay in the same position
                self.state.dwell_time[i] = 1 + self.state.dwell_time[i].item()

                # 如果停留在站台，则继续处理乘客上下车
                if on_platform:
                    # 每次停留都继续处理乘客上下车（先下后上）
                    self.alight_passengers()
                    self.board_passengers()
            else:  # move to the next position
                pos = self.world.get_node(new_node)

                self.state.previous_dwell_time[i] = self.state.dwell_time[i].item()
                self.state.previous_expected_dwell_time[i] = self.state.current_expected_dwell_time[i].item()

                self.state.dwell_time[i] = 1
                self.state.current_expected_dwell_time[i] = pos.weight
                # 重置限速延误
                self.state.speed_limit_delay[i] = 0

                # 如果到达新站台，则处理乘客上下车
                if on_platform:
                    # 先下后上
                    self.alight_passengers()
                    self.board_passengers()

            if moved and on_platform:
                sim_time = self._sim_time_value(i)
                if sim_time is not None:
                    self.state.arrival_time[i] = sim_time
            elif not on_platform:
                self.state.arrival_time[i] = 0.0

            self._update_run_tracking(i, previous_node, new_node, on_platform)

            stop_idx = int(self.state.run_segment_index[i].item())
            on_schedule = (
                self._active_run_stops
                and 0 <= stop_idx < len(self._active_run_stops)
                and bool(self.state.on_assigned_route[i].item())
            )

            if on_platform and on_schedule:
                stop = self._active_run_stops[stop_idx]
                if int(new_node) == int(stop.platform_id):
                    self.state.schedule_time[i] = float(stop.departure_time)
                else:
                    self.state.schedule_time[i] = 0.0
            else:
                self.state.schedule_time[i] = 0.0

        """Set agent position"""
        if env_mask is None:
            self.state.position = position
        else:
            for i in indices:
                self.state.position[i] = position[i]

    def board_passengers(self):
        """Passenger boarding logic"""
        total_boarding = 0
        platform_nodes = getattr(self.world, "platform_nodes", None)
        if platform_nodes is None:
            platform_nodes = getattr(self.world, "platfrom_nodes", set())

        for i in range(self.num_envs):
            current_pos = self.state.position[i].item()
            
            # Passengers board only at platform nodes
            if current_pos in platform_nodes:
                node = self.world.get_node(current_pos)
                
                # Calculate the number of passengers that can board
                available_capacity = self.passenger_capacity - self.passenger_count[i].item()
                boarding = min(available_capacity, node.waiting_passengers[i].item())
                
                if boarding > 0:
                    # Update passenger count
                    self.passenger_count[i] += boarding
                    self.cumulative_boarded[i] += boarding
                    node.waiting_passengers[i] -= boarding
                    total_boarding += boarding
                    
                    # Update passengers by destination - 修正：确保总数相等
                    dest_counts = {}
                    total_assigned = 0
                    sorted_dests = sorted(node.passenger_destinations.items(), key=lambda x: x[1], reverse=True)
                    
                    # 除最后一个外都用 int() 截断
                    for dest, prob in sorted_dests[:-1]:
                        count = int(boarding * prob)
                        dest_counts[dest] = count
                        total_assigned += count
                    
                    # 最后一个目的地获得所有剩余乘客
                    if sorted_dests:
                        last_dest = sorted_dests[-1][0]
                        dest_counts[last_dest] = int(boarding) - total_assigned
                    
                    # 更新字典
                    for dest, count in dest_counts.items():
                        if count > 0:
                            if dest not in self.passengers_by_destination:
                                self.passengers_by_destination[dest] = torch.zeros(self.num_envs, device=self.device)
                            self.passengers_by_destination[dest][i] += count
        
        return total_boarding  # 返回所有环境中上车的乘客总数

    def alight_passengers(self):
        """Passenger alighting logic"""
        total_alighting = 0
        platform_nodes = getattr(self.world, "platform_nodes", None)
        if platform_nodes is None:
            platform_nodes = getattr(self.world, "platfrom_nodes", set())

        for i in range(self.num_envs):
            current_pos = self.state.position[i].item()
            
            # Passengers alight only at platform nodes
            if current_pos in platform_nodes:
                # Find all passengers whose destination is the current station
                alighting = self.passengers_by_destination.get(current_pos, 
                           torch.zeros(self.num_envs, device=self.device))[i].item()
                
                # Update passenger count
                if alighting > 0:
                    self.passenger_count[i] -= alighting
                    self.cumulative_alighted[i] += alighting
                    total_alighting += alighting
                    # Clear passengers who have reached their destination
                    if current_pos in self.passengers_by_destination:
                        self.passengers_by_destination[current_pos][i] = 0
        
        return total_alighting  # 返回所有环境中下车的乘客总数

    def calc_alighting_reward(self, env_idx):
        """计算下车奖励但不实际执行下车操作"""
        current_pos = self.state.position[env_idx].item()
        platform_nodes = getattr(self.world, "platform_nodes", None)
        if platform_nodes is None:
            platform_nodes = getattr(self.world, "platfrom_nodes", set())
        if current_pos in platform_nodes:
            alighting = self.passengers_by_destination.get(current_pos, 
                      torch.zeros(self.num_envs, device=self.device))[env_idx].item()
            return alighting * 0.2
        return 0

    def calc_boarding_reward(self, env_idx):
        """计算上车奖励但不实际执行上车操作"""
        current_pos = self.state.position[env_idx].item()
        platform_nodes = getattr(self.world, "platform_nodes", None)
        if platform_nodes is None:
            platform_nodes = getattr(self.world, "platfrom_nodes", set())
        if current_pos in platform_nodes:
            node = self.world.get_node(current_pos)
            available_capacity = self.passenger_capacity - self.passenger_count[env_idx].item()
            boarding = min(available_capacity, node.waiting_passengers[env_idx].item())
            return boarding * 0.1
        return 0

    def calc_reward(self):
        """Shape rewards around on-time departures while keeping segment progress incentives."""
        reward = torch.zeros(self.num_envs, device=self.device)

        base_step_penalty = 0.01
        free_move_bonus = 0

        wait_reward_scale = 0.08
        wait_horizon = 90.0
        depart_window = 5.0
        on_time_depart_bonus = 0.5
        early_depart_penalty_rate = 0.0025
        early_depart_penalty_cap = 0.4
        late_depart_penalty_rate = 0.003
        late_depart_penalty_cap = 0.6
        late_depart_reward_decay = 180.0
        late_depart_block_penalty = 0.05

        on_segment_bonus = 0
        progress_bonus = 0
        blocked_move_penalty = 1

        arrival_bonus = 0.3
        arrival_tolerance = 5.0
        arrival_late_penalty_rate = 0.0025
        arrival_late_penalty_cap = 0.5
        arrival_early_tolerance = 20.0

        has_schedule = bool(self._active_run_stops) and self._active_run is not None
        tick_seconds = float(getattr(self.world, "tick_seconds", 1) or 1)

        for i in range(self.num_envs):
            reward[i] -= base_step_penalty

            action_val = int(self.action[i].item())
            action_outcome = int(self.action_result[i].item())
            current_node = int(self.state.position[i].item())
            previous_node = int(self.state.previous_node[i].item()) if hasattr(self.state, "previous_node") else current_node
            moved = current_node != previous_node

            if not has_schedule:
                if action_outcome == 2:
                    reward[i] += free_move_bonus
                continue

            segment_idx = int(self.state.run_segment_index[i].item())
            if segment_idx < 0 or segment_idx >= len(self._active_run_stops):
                if action_outcome == 2:
                    reward[i] += free_move_bonus
                continue

            if not bool(self.state.on_assigned_route[i].item()):
                if action_outcome == 2:
                    reward[i] += free_move_bonus
                continue

            stop = self._active_run_stops[segment_idx]

            relative_time = self._relative_time(i)
            depart_time, arrival_target = self._stop_schedule_window(segment_idx)
            start_platform = int(stop.platform_id)
            next_stop = self._next_stop_for_index(segment_idx, self._active_run_stops)
            end_platform = int(next_stop.platform_id) if next_stop is not None else start_platform

            if relative_time is not None:
                time_to_depart = depart_time - relative_time

                if current_node == start_platform:
                    if time_to_depart < 0:
                        self.cumulative_delay[i] += tick_seconds
                    if time_to_depart > depart_window:
                        if action_val == Action.STOP:
                            wait_ratio = max(0.0, min(1.0, 1.0 - time_to_depart / wait_horizon))
                            reward[i] += wait_reward_scale * wait_ratio
                            logger.info(
                                f"{self._sim_time_value(i)}: {self.name}, wait_reward applied: env={i}, node={current_node}, ratio={wait_ratio:.3f}, reward={(wait_reward_scale * wait_ratio):.4f}"
                            )
                        elif action_val == Action.MOVE_DEFAULT and action_outcome == 2:
                            penalty = min(early_depart_penalty_cap, early_depart_penalty_rate * time_to_depart)
                            reward[i] -= penalty
                            logger.info(
                                f"{self._sim_time_value(i)}: {self.name}, early_depart_penalty applied: env={i}, node={current_node}, time_to_depart={time_to_depart:.2f}, penalty={penalty:.4f}"
                            )
                    elif -depart_window <= time_to_depart <= depart_window:
                        if action_val == Action.MOVE_DEFAULT and action_outcome == 2:
                            reward[i] += on_time_depart_bonus
                            logger.info(
                                f"{self._sim_time_value(i)}: {self.name}, on_time_depart_bonus applied: env={i}, node={current_node}, bonus={on_time_depart_bonus:.4f}"
                            )
                    else:  # Already late
                        lateness = abs(time_to_depart)
                        if action_val == Action.STOP:
                            penalty = min(late_depart_penalty_cap, late_depart_penalty_rate * lateness)
                            reward[i] -= penalty
                            logger.info(
                                f"{self._sim_time_value(i)}: {self.name}, late_depart_penalty applied: env={i}, node={current_node}, lateness={lateness:.2f}, penalty={penalty:.4f}"
                            )
                        if action_val == Action.MOVE_DEFAULT and action_outcome == 2:
                            depart_gain = on_time_depart_bonus * max(0.0, 1.0 - lateness / late_depart_reward_decay)
                            reward[i] += depart_gain
                            logger.info(
                                f"{self._sim_time_value(i)}: {self.name}, late_depart_gain applied: env={i}, node={current_node}, lateness={lateness:.2f}, gain={depart_gain:.4f}"
                            )
                        if action_val == Action.MOVE_DEFAULT and action_outcome == 0:
                            reward[i] -= late_depart_block_penalty

            if self.world is not None:
                platform_nodes = getattr(self.world, "platform_nodes", None)
                if platform_nodes is None:
                    platform_nodes = getattr(self.world, "platfrom_nodes", set())
                if current_node in platform_nodes:
                    node = self.world.get_node(current_node)
                    waiting_tensor = getattr(node, "waiting_passengers", None)
                    if waiting_tensor is not None and waiting_tensor.shape[0] > i:
                        waiting_count = float(waiting_tensor[i].item())
                        if waiting_count > 0.0:
                            self.cumulative_waiting_time[i] += waiting_count * tick_seconds

            if action_val == Action.MOVE_DEFAULT:
                if action_outcome == 2:
                    reward[i] += progress_bonus
                elif action_outcome == 0:
                    reward[i] -= blocked_move_penalty
                    logger.info(f"{self._sim_time_value(i)}: {self.name}, blocked_move_penalty applied: env={i}, node={current_node}, prev_node={previous_node}, penalty={blocked_move_penalty}")

        return reward

    def calc_terminal_reward(self, done_mask: torch.Tensor) -> torch.Tensor:
        reward = torch.zeros(self.num_envs, device=self.device)
        if done_mask is None or done_mask.numel() == 0:
            return reward

        delay_weight = 0.0005
        waiting_weight = 0.0001
        reward[done_mask] -= delay_weight * self.cumulative_delay[done_mask]
        reward[done_mask] -= waiting_weight * self.cumulative_waiting_time[done_mask]
        return reward

    def to(self, device: torch.device):
        """Move data to the specified device"""
        self.device = device

        # Move state tensors
        for field in self.state.__dataclass_fields__:
            value = getattr(self.state, field)
            if isinstance(value, torch.Tensor):
                setattr(self.state, field, value.to(device))

        # Move action tensor
        self.action = self.action.to(device)
        if hasattr(self, "prev_action") and self.prev_action is not None:
            self.prev_action = self.prev_action.to(device)

    def potential_step(self, next_pos: torch.Tensor):
        """计算下一步潜在位置，考虑路径选择"""
        # 初始化潜在下一位置张量（如果未提供）
        if next_pos is None:
            next_pos = torch.zeros_like(self.state.position)
        
        for i in range(self.num_envs):
            current_pos = int(self.state.position[i].item())

            if self.action[i].item() == Action.STOP:
                next_pos[i] = current_pos
                continue

            if not hasattr(self, "world") or self.world is None:
                next_pos[i] = current_pos
                continue

            next_node_id = None
            if hasattr(self, "route_adapter") and self.route_adapter is not None:
                next_node_id = self.route_adapter.next_node(self, i, current_pos)

            if next_node_id is None:
                next_pos[i] = current_pos
            else:
                next_pos[i] = next_node_id
        
        self.potential_next_pos = next_pos
        return next_pos
    
    def get_available_paths_count(self, env_index=None):
        """获取当前位置可用路径数量，用于状态观察"""
        if env_index is not None:
            current_pos = self.state.position[env_index].item()
            next_nodes = self.world.get_next_nodes(current_pos)
            return torch.tensor([len(next_nodes)], device=self.device)
        else:
            # 返回所有环境的路径数量
            path_counts = torch.zeros((self.num_envs, 1), device=self.device)
            for i in range(self.num_envs):
                current_pos = self.state.position[i].item()
                next_nodes = self.world.get_next_nodes(current_pos)
                path_counts[i, 0] = len(next_nodes)
            return path_counts
