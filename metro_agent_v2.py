"""Metro agent variant focused on headway management without timetables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import math
import torch

from logger import logger
from route_selection import RouteSelectionAdapter, RandomRouteSelectionPolicy

try:
    from linedata import routes as _ALL_ROUTES
except ImportError:  # pragma: no cover - optional data dependency
    _ALL_ROUTES = []


# 最大支持的移动方向数（不包括停留）
# 动作空间: 0=停留, 1=方向1, 2=方向2, 3=方向3
MAX_MOVE_DIRECTIONS = 3


class Action:
    """Discrete action set with multi-directional movement support.
    
    Action space: [0, 1, 2, ..., MAX_MOVE_DIRECTIONS]
    - 0: STOP (dwell/hold position)
    - 1: MOVE to first available direction (default route if available)
    - 2: MOVE to second available direction
    - 3: MOVE to third available direction
    - ...
    
    If the chosen direction is not available at the current node,
    the agent will fall back to the first available direction (action 1 behavior).
    """

    STOP = 0               # dwell/hold position
    MOVE_DEFAULT = 1       # move along current/default route (direction 1)
    MOVE_DIR_2 = 2         # move to direction 2 (if available)
    MOVE_DIR_3 = 3         # move to direction 3 (if available)
    
    @classmethod
    def is_move_action(cls, action: int) -> bool:
        """Check if action is a move action (not STOP)."""
        return action >= 1
    
    @classmethod
    def get_direction_index(cls, action: int) -> int:
        """Get 0-based direction index from action. Returns -1 for STOP."""
        if action <= 0:
            return -1
        return action - 1  # action 1 -> index 0, action 2 -> index 1, etc.


@dataclass
class TrainState:
    """Tensor container describing the per-environment train state."""

    position: torch.Tensor
    current_route: torch.Tensor
    arrival_time: torch.Tensor
    schedule_time: torch.Tensor
    dwell_time: torch.Tensor
    previous_dwell_time: torch.Tensor
    previous_expected_dwell_time: torch.Tensor
    previous_node: torch.Tensor
    action: torch.Tensor
    prev_action: torch.Tensor
    action_result: torch.Tensor
    potential_next_pos: torch.Tensor
    target_departure_time: torch.Tensor
    dwell_target: torch.Tensor
    time_since_arrival: torch.Tensor
    last_departure_time: torch.Tensor
    headway_gap: torch.Tensor
    arrival_headway: torch.Tensor
    headway_error_last: torch.Tensor
    dwell_min_limit: torch.Tensor
    dwell_max_limit: torch.Tensor
    last_dwell_duration: torch.Tensor
    just_departed: torch.Tensor
    current_platform: torch.Tensor
    speed_limit_delay: torch.Tensor
    previous_position: torch.Tensor
    current_expected_dwell_time: torch.Tensor
    passenger_count: torch.Tensor
    passenger_count_prev: torch.Tensor


@dataclass
class _HeadwayBuffers:
    last_departure: torch.Tensor
    next_due: torch.Tensor
    last_arrival: torch.Tensor


class PlatformHeadwayTracker:
    """Shared utility coordinating headway targets per platform."""

    def __init__(self, headway_seconds: float, num_envs: int, device: torch.device):
        self.headway_seconds = float(headway_seconds)
        self.num_envs = int(num_envs)
        self.device = device
        self._buffers: Dict[int, _HeadwayBuffers] = {}
        self.targets = {}

    def set_target_headway(self, platform_id, seconds):
        if not math.isfinite(seconds) or seconds < 30:
            raise ValueError("Target headway must be finite and at least 30 seconds")
        self.targets[platform_id] = float(seconds)

    def last_departure(self, platform_id, env=0):
        return float(self._ensure_platform(platform_id).last_departure[env])

    def to(self, device: torch.device) -> None:
        if self.device == device:
            return
        self.device = device
        for buffers in self._buffers.values():
            buffers.last_departure = buffers.last_departure.to(device)
            buffers.next_due = buffers.next_due.to(device)
            buffers.last_arrival = buffers.last_arrival.to(device)

    def bootstrap(self, platform_ids: Iterable[int]) -> None:
        for platform_id in platform_ids:
            self._ensure_platform(platform_id)

    def reset(self, env_index: Optional[int] = None) -> None:
        if env_index is None:
            self.targets.clear()
        if env_index is None:
            for buffers in self._buffers.values():
                buffers.last_departure.fill_(float("-inf"))
                buffers.next_due.fill_(float("-inf"))
                buffers.last_arrival.fill_(float("-inf"))
        else:
            for buffers in self._buffers.values():
                buffers.last_departure[env_index] = float("-inf")
                buffers.next_due[env_index] = float("-inf")
                buffers.last_arrival[env_index] = float("-inf")

    def register_arrival(
        self,
        platform_id: int,
        env_index: int,
        arrival_time: float,
        baseline_dwell: float,
    ) -> tuple[float, Optional[float]]:
        buffers = self._ensure_platform(platform_id)
        arrival_time = float(arrival_time)
        baseline_dwell = max(0.0, float(baseline_dwell))

        last_departure = float(buffers.last_departure[env_index].item())
        next_due = float(buffers.next_due[env_index].item())
        previous_arrival = float(buffers.last_arrival[env_index].item())

        earliest_departure = arrival_time + baseline_dwell
        if last_departure == float("-inf"):
            constrained_departure = earliest_departure
        else:
            constrained_departure = max(earliest_departure, last_departure + self.targets.get(platform_id, self.headway_seconds))

        if next_due != float("-inf"):
            constrained_departure = max(constrained_departure, next_due)

        if previous_arrival == float("-inf"):
            arrival_gap = None
        else:
            arrival_gap = arrival_time - previous_arrival

        buffers.last_arrival[env_index] = arrival_time
        buffers.next_due[env_index] = constrained_departure + self.targets.get(platform_id, self.headway_seconds)
        return constrained_departure, arrival_gap

    def register_departure(
        self,
        platform_id: int,
        env_index: int,
        departure_time: float,
    ) -> Optional[float]:
        buffers = self._ensure_platform(platform_id)
        departure_time = float(departure_time)
        previous_departure = float(buffers.last_departure[env_index].item())

        if previous_departure == float("-inf"):
            gap = None
        else:
            gap = departure_time - previous_departure

        buffers.last_departure[env_index] = departure_time
        minimum_due = departure_time + self.targets.get(platform_id, self.headway_seconds)
        if float(buffers.next_due[env_index].item()) < minimum_due:
            buffers.next_due[env_index] = minimum_due
        return gap

    def _ensure_platform(self, platform_id: int) -> _HeadwayBuffers:
        buffers = self._buffers.get(platform_id)
        if buffers is not None:
            return buffers

        last_departure = torch.full(
            (self.num_envs,), float("-inf"), device=self.device, dtype=torch.float32
        )
        next_due = torch.full(
            (self.num_envs,), float("-inf"), device=self.device, dtype=torch.float32
        )
        last_arrival = torch.full(
            (self.num_envs,), float("-inf"), device=self.device, dtype=torch.float32
        )
        buffers = _HeadwayBuffers(
            last_departure=last_departure,
            next_due=next_due,
            last_arrival=last_arrival,
        )
        self._buffers[platform_id] = buffers
        return buffers


class MetroAgentV2:
    """Headway-driven train agent using only stop/move actions."""

    VERSION = "v2"

    def __init__(
        self,
        name: str,
        num_envs: int,
        device: torch.device,
        *,
        headway: float = 240.0,
        dwell_tolerance: float = 30.0,
        default_dwell: Optional[float] = None,
        passenger_capacity: int = 2520,
        use_float_boarding: bool = False,
        **_: object,
    ) -> None:
        self.name = name
        self.num_envs = int(num_envs)
        self.device = device
        self.Action = Action
        self._default_move_action_value = Action.MOVE_DEFAULT

        self.world = None
        self.route = _ALL_ROUTES[0] if _ALL_ROUTES else None
        
        self.headway_seconds = float(headway)
        self.dwell_tolerance = max(0.0, float(dwell_tolerance))
        self._default_dwell = default_dwell
        self.use_float_boarding = use_float_boarding

        self.route_selection_policy = RandomRouteSelectionPolicy()
        default_route_key = getattr(self.route, "id", None)
        self.route_adapter = RouteSelectionAdapter(
            policy=self.route_selection_policy,
            num_envs=self.num_envs,
            default_route_key=default_route_key,
        )

        self.state: TrainState = self._init_state()

        self.headway_tracker: Optional[PlatformHeadwayTracker] = None

        self._primary_route_id = self._resolve_primary_route_id()
        self._route_platform_ids: set[int] = set()

        self.agent_version = self.VERSION
        self.passenger_capacity = int(passenger_capacity)
        self.passengers_by_destination: Dict[int, torch.Tensor] = {}
        self.passenger_waiting_mass: Dict[int, torch.Tensor] = {}
        self.completed_waiting_mass = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        
        # Cumulative statistics
        self.cumulative_boarded = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        self.cumulative_alighted = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        self.cumulative_waiting_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        self.cumulative_delay = torch.zeros(self.num_envs, device=self.device)

        self._reset_control_state()
        self._reset_passenger_state()
        self._set_default_route()

    # ------------------------------------------------------------------
    # Lifecycle & configuration
    # ------------------------------------------------------------------
    def _init_state(self) -> TrainState:
        state = TrainState(
            position=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            current_route=torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long),
            arrival_time=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            schedule_time=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            dwell_time=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            previous_dwell_time=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            previous_expected_dwell_time=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            previous_node=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            action=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            prev_action=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            action_result=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            potential_next_pos=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            target_departure_time=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            dwell_target=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            time_since_arrival=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            last_departure_time=torch.full(
                (self.num_envs,), float("-inf"), device=self.device, dtype=torch.float32
            ),
            headway_gap=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            arrival_headway=torch.zeros(
                self.num_envs, device=self.device, dtype=torch.float32
            ),
            headway_error_last=torch.full(
                (self.num_envs,), -1.0, device=self.device, dtype=torch.float32
            ),
            dwell_min_limit=torch.zeros(
                self.num_envs, device=self.device, dtype=torch.float32
            ),
            dwell_max_limit=torch.zeros(
                self.num_envs, device=self.device, dtype=torch.float32
            ),
            last_dwell_duration=torch.zeros(
                self.num_envs, device=self.device, dtype=torch.float32
            ),
            just_departed=torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            current_platform=torch.full(
                (self.num_envs,), -1, device=self.device, dtype=torch.long
            ),
            speed_limit_delay=torch.zeros(self.num_envs, device=self.device, dtype=torch.long),
            previous_position=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            current_expected_dwell_time=torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            passenger_count=torch.zeros(self.num_envs, device=self.device, dtype=torch.float64),
            passenger_count_prev=torch.zeros(self.num_envs, device=self.device, dtype=torch.float64),
        )

        self.route_adapter.reset_all(self)
        return state

    def _clear_state_slot(self, idx: int) -> None:
        idx = int(idx)
        self.state.position[idx] = 0.0
        self.state.current_route[idx] = -1
        self.state.arrival_time[idx] = 0.0
        self.state.schedule_time[idx] = 0.0
        self.state.dwell_time[idx] = 0.0
        self.state.previous_dwell_time[idx] = 0.0
        self.state.previous_expected_dwell_time[idx] = 0.0
        self.state.previous_node[idx] = 0.0
        self.state.action[idx] = 0.0
        self.state.prev_action[idx] = 0.0
        self.state.action_result[idx] = 0.0
        self.state.potential_next_pos[idx] = 0.0

        self._reset_control_state(idx)
        self._reset_passenger_state(idx)
        self._set_default_route(idx)

    def reset_state(self, env_index: Optional[int] = None) -> None:
        if env_index is None:
            self.state = self._init_state()
            self.cumulative_boarded.zero_()
            self.cumulative_alighted.zero_()
            self.cumulative_waiting_time.zero_()
            self.cumulative_delay.zero_()
            self._reset_control_state()
            self._reset_passenger_state()
            self._set_default_route()
            return

        idx = int(env_index)
        self._clear_state_slot(idx)
        self.cumulative_boarded[idx] = 0
        self.cumulative_alighted[idx] = 0
        self.cumulative_waiting_time[idx] = 0
        self.cumulative_delay[idx] = 0
        self.route_adapter.reset_env(self, idx)

    def set_headway_tracker(self, tracker: PlatformHeadwayTracker) -> None:
        self.headway_tracker = tracker
        if self.world is not None:
            self._ensure_route_platform_ids(force=True)
            if self._route_platform_ids:
                tracker.bootstrap(self._route_platform_ids)

    def configure_headway(self, *, headway: Optional[float] = None, dwell_tolerance: Optional[float] = None) -> None:
        if headway is not None:
            self.headway_seconds = float(headway)
        if dwell_tolerance is not None:
            self.dwell_tolerance = max(0.0, float(dwell_tolerance))

    def assign_run(self, *_: object, **__: object) -> None:
        """Timetables are ignored in the headway controller."""
        return None

    # ------------------------------------------------------------------
    # Route helpers
    # ------------------------------------------------------------------
    def _resolve_primary_route_id(self):
        route_key = getattr(self.route, "id", None)
        try:
            return int(route_key) if route_key is not None else None
        except (TypeError, ValueError):
            return route_key

    def _is_primary_route(self, route_value: object) -> bool:
        if self._primary_route_id is None:
            return False
        try:
            return int(route_value) == int(self._primary_route_id)
        except (TypeError, ValueError):
            return route_value == self._primary_route_id

    def _ensure_route_platform_ids(self, *, force: bool = False) -> None:
        if self.world is None:
            return
        if self._route_platform_ids and not force:
            return

        platform_nodes = getattr(self.world, "platform_nodes", {})
        if not platform_nodes:
            self._route_platform_ids = set()
            return

        candidate_ids: set[int] = set()
        route = self.route
        if route is not None:
            route_nodes = getattr(route, "nodes", None)
            if route_nodes:
                for node in route_nodes:
                    node_id = getattr(node, "id", None)
                    if node_id is None:
                        continue
                    try:
                        candidate_ids.add(int(node_id))
                    except (TypeError, ValueError):
                        continue
            route_node_list = getattr(route, "node_list", None)
            if route_node_list:
                for node_ref in route_node_list:
                    node_id = getattr(node_ref, "id", node_ref)
                    if node_id is None:
                        continue
                    try:
                        candidate_ids.add(int(node_id))
                    except (TypeError, ValueError):
                        continue

        if not candidate_ids:
            self._route_platform_ids = set()
            return

        self._route_platform_ids = {node_id for node_id in candidate_ids if node_id in platform_nodes}

    # ------------------------------------------------------------------
    # Action & movement interfaces
    # ------------------------------------------------------------------
    def set_action(self, action: torch.Tensor) -> None:
        if action.shape[0] != self.num_envs:
            raise ValueError("Action batch dimension mismatch")

        self.state.prev_action = self.state.action.clone()
        self.state.action = action.clone().to(self.device)

        for env_index in range(self.num_envs):
            current_action = int(self.state.action[env_index].item())
            if current_action == Action.STOP:
                continue

            current_node = int(self.state.position[env_index].item())
            self.route_adapter.ensure_route(self, env_index, current_node)

    def potential_step(self, next_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        if next_pos is None:
            next_pos = torch.zeros_like(self.state.position)

        for env_index in range(self.num_envs):
            current_pos = int(self.state.position[env_index].item())
            current_action = int(self.state.action[env_index].item())
            
            if current_action == Action.STOP:
                next_pos[env_index] = current_pos
                continue

            if self.world is None:
                next_pos[env_index] = current_pos
                continue

            # 多方向选择：根据动作值选择移动方向
            direction_index = Action.get_direction_index(current_action)
            next_node = self.route_adapter.next_node_by_direction(
                self, env_index, current_pos, direction_index
            )
            if next_node is None:
                next_pos[env_index] = current_pos
            else:
                next_pos[env_index] = float(next_node)

        self.state.potential_next_pos = next_pos
        return next_pos

    def set_position(self, position: torch.Tensor, env_mask: Optional[Sequence[int]] = None) -> None:
        if self.world is None:
            self.state.position = position.to(self.device)
            return

        platform_nodes = getattr(self.world, "platform_nodes", {})
        indices = env_mask if env_mask is not None else range(self.num_envs)
        new_positions = position.to(self.device)

        for idx in indices:
            idx = int(idx)
            prev_node = int(self.state.position[idx].item())
            new_node = int(new_positions[idx].item())
            moved = new_node != prev_node
            dispatch = getattr(self.world, "dispatch", None)
            prev_on_platform = prev_node in platform_nodes and (not dispatch or dispatch.passenger_service(self, prev_node))
            new_on_platform = new_node in platform_nodes and (not dispatch or dispatch.passenger_service(self, new_node))

            if prev_on_platform and moved:
                dwell_duration = float(self.state.time_since_arrival[idx].item())
                if dwell_duration <= 0.0:
                    dwell_duration = float(self.state.dwell_time[idx].item())
                self.state.last_dwell_duration[idx] = dwell_duration
                self.state.just_departed[idx] = True
                depart_time = self._sim_time_value(idx)
                if depart_time is not None:
                    self.state.last_departure_time[idx] = depart_time
                if self.headway_tracker is not None and depart_time is not None:
                    gap = self.headway_tracker.register_departure(prev_node, idx, depart_time)
                    if gap is not None:
                        self.state.headway_gap[idx] = gap - self.headway_seconds
                    else:
                        self.state.headway_gap[idx] = 0.0
                else:
                    self.state.headway_gap[idx] = 0.0
                self.state.target_departure_time[idx] = 0.0

            if moved:
                self.state.previous_position[idx] = prev_node
                self.state.position[idx] = new_node
                self.state.previous_dwell_time[idx] = self.state.dwell_time[idx]
                self.state.previous_expected_dwell_time[idx] = self.state.current_expected_dwell_time[idx]
                self.state.dwell_time[idx] = 0.0 if getattr(self.world, "second_precision", False) else 1.0
                self.state.speed_limit_delay[idx] = 0
            else:
                self.state.dwell_time[idx] += 1.0

            if new_on_platform:
                node = self.world.get_node(new_node)
                if moved:
                    min_limit, max_limit, baseline_seconds = self._compute_platform_dwell_limits(node)
                    self.state.current_expected_dwell_time[idx] = min_limit
                    self.state.dwell_min_limit[idx] = min_limit
                    self.state.dwell_max_limit[idx] = max_limit
                    self.state.dwell_target[idx] = min_limit
                    self.state.time_since_arrival[idx] = 0.0 if getattr(self.world, "second_precision", False) else 1.0
                    self.state.current_platform[idx] = new_node
                    arrival_time = (getattr(self.world, "_transition_arrival_time", None)
                                    if getattr(self.world, "second_precision", False) else self._sim_time_value(idx))
                    if arrival_time is not None:
                        self.state.arrival_time[idx] = arrival_time
                    if self.headway_tracker is not None and arrival_time is not None:
                        target_departure, arrival_gap = self.headway_tracker.register_arrival(
                            new_node, idx, arrival_time, baseline_seconds
                        )
                        self.state.target_departure_time[idx] = target_departure
                        if arrival_gap is not None:
                            self.state.arrival_headway[idx] = arrival_gap
                            self.state.headway_gap[idx] = arrival_gap - self.headway_seconds
                        else:
                            self.state.arrival_headway[idx] = 0.0
                            self.state.headway_gap[idx] = 0.0
                    else:
                        self.state.target_departure_time[idx] = 0.0
                        self.state.arrival_headway[idx] = 0.0
                        self.state.headway_gap[idx] = 0.0
                else:
                    self.state.time_since_arrival[idx] += 1.0

                self.alight_passengers(idx)
                self.board_passengers(idx)
            else:
                self.state.time_since_arrival[idx] = 0.0
                self.state.current_platform[idx] = -1

        self.state.position = new_positions

    def _sync_passenger_count(self, idx: int) -> None:
        # OD loads are authoritative; independently incremented totals drift.
        self.state.passenger_count[idx] = math.fsum(float(q[idx]) for q in self.passengers_by_destination.values())

    def board_passengers(self, env_index: Optional[int] = None) -> None:
        platform_nodes = getattr(self.world, "platform_nodes", getattr(self.world, "platfrom_nodes", set()))
        indices = [env_index] if env_index is not None else range(self.num_envs)
        for idx in indices:
            idx = int(idx)
            current_pos = int(self.state.position[idx].item())
            if current_pos not in platform_nodes:
                continue
            if self.world.dispatch and not self.world.dispatch.passenger_service(self, current_pos):
                continue
            node = self.world.get_node(current_pos)
            events = self.world.event_view() if hasattr(self.world, "event_view") else None
            if events and not events.boarding_allowed(current_pos, self.name):
                continue
            capacity = events.vehicle_capacity(self.name, self.passenger_capacity) if events else self.passenger_capacity
            available_capacity = max(0., capacity - self.state.passenger_count[idx].item())
            waiting_tensor = getattr(node, "waiting_passengers", None)
            if waiting_tensor is None:
                if hasattr(node, "init_passenger_tensor"):
                    node.init_passenger_tensor(self.num_envs, self.device)
                    waiting_tensor = node.waiting_passengers
            if waiting_tensor is None:
                continue
            waiting = waiting_tensor[idx].item()
            excluded = set()
            if self.world.dispatch:
                excluded = self.world.dispatch.excluded_destinations(self)
                if excluded:
                    node.passenger_flow.ensure_od_queues()
                    waiting = sum(float(q[idx]) for d, q in node.passenger_flow.waiting_by_destination.items() if d not in excluded)
            boarding = min(available_capacity, waiting)
            if events:
                boarding = min(boarding, self.world.boarding_rate_persons_per_second
                               * events.boarding_multiplier(current_pos) * float(self.world.tick_seconds or 1))
            if boarding <= 0:
                continue
            flow = getattr(node, "passenger_flow", None)
            if flow is not None:
                assigned, masses = flow.take_boarding(idx, boarding, with_waiting=True, excluded_destinations=excluded)
                for dest, count in assigned.items():
                    if dest not in self.passengers_by_destination:
                        self.passengers_by_destination[dest] = torch.zeros_like(self.state.passenger_count)
                    self.passengers_by_destination[dest][idx] += count
                    mass = self.passenger_waiting_mass.setdefault(dest, torch.zeros_like(self.state.passenger_count))
                    mass[idx] += masses[dest]
                self._sync_passenger_count(idx)
                self.cumulative_boarded[idx] += math.fsum(assigned.values())
                flow.admit(idx, events=events)
                continue
            self.state.passenger_count[idx] += boarding
            self.cumulative_boarded[idx] += boarding
            node.waiting_passengers[idx] -= boarding
            
            if not node.passenger_destinations:
                # Fallback: if no destinations defined, assume they get off at next station (or random)
                # For now, just don't assign, which is the bug source, but we can't invent destinations here easily.
                # Ideally this should not happen if _init_destination_distribution is called.
                # print("Warning: No passenger destinations defined for node", current_pos)
                pass

            # 按目的地分配乘客，确保总数相等
            if self.use_float_boarding:
                # 浮点模式：直接按比例分配
                for dest, prob in node.passenger_destinations.items():
                    boarding_to_dest = boarding * prob
                    if boarding_to_dest <= 1e-9:
                        continue
                    if dest not in self.passengers_by_destination:
                        self.passengers_by_destination[dest] = torch.zeros_like(self.state.passenger_count)
                    self.passengers_by_destination[dest][idx] += boarding_to_dest
            else:
                # 整数模式：先按比例分配，然后修正余数
                dest_counts = {}
                total_assigned = 0
                sorted_dests = sorted(node.passenger_destinations.items(), key=lambda x: x[1], reverse=True)
                
                for dest, prob in sorted_dests[:-1]:  # 除最后一个外都用 int() 截断
                    count = int(boarding * prob)
                    dest_counts[dest] = count
                    total_assigned += count
                
                # 最后一个目的地获得所有剩余乘客，确保总数相等
                if sorted_dests:
                    last_dest = sorted_dests[-1][0]
                    dest_counts[last_dest] = int(boarding) - total_assigned
                
                # 更新字典
                for dest, count in dest_counts.items():
                    if count > 0:
                        if dest not in self.passengers_by_destination:
                            self.passengers_by_destination[dest] = torch.zeros_like(self.state.passenger_count)
                        self.passengers_by_destination[dest][idx] += count


    def alight_passengers(self, env_index: Optional[int] = None) -> None:
        platform_nodes = getattr(self.world, "platform_nodes", getattr(self.world, "platfrom_nodes", set()))
        indices = [env_index] if env_index is not None else range(self.num_envs)
        for idx in indices:
            idx = int(idx)
            current_pos = int(self.state.position[idx].item())
            if current_pos not in platform_nodes:
                continue
            if self.world.dispatch and not self.world.dispatch.passenger_service(self, current_pos):
                continue
            alighting_tensor = self.passengers_by_destination.get(
                current_pos,
                torch.zeros(self.num_envs, device=self.device),
            )
            alighting = alighting_tensor[idx].item()
            if alighting <= 0:
                continue
            self.cumulative_alighted[idx] += alighting
            if current_pos in self.passenger_waiting_mass:
                self.completed_waiting_mass[idx] += self.passenger_waiting_mass[current_pos][idx]
                self.passenger_waiting_mass[current_pos][idx] = 0
            if current_pos in self.passengers_by_destination:
                self.passengers_by_destination[current_pos][idx] = 0
            self._sync_passenger_count(idx)

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------
    def calc_reward(self) -> torch.Tensor:
        if self.world is not None and self.world.dispatch:
            cost = self.world.dispatch.interval_cost["total"] / max(1, len(self.world.agents))
            return torch.full((self.num_envs,), -cost, device=self.device)
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        reward_breakdown = [[] for _ in range(self.num_envs)]

        base_move_bonus = 0.05
        blocked_move_penalty = 0.05
        dwell_step_scale = 0.05
        dwell_threshold_penalty_scale = 0.05
        dwell_depart_bonus_scale = 0.05
        dwell_threshold_tolerance = 1e-3
        non_platform_hold_penalty = 0.05

        passenger_board_reward = 0.01
        passenger_alight_reward = 0.01
        # Local penalty at the currently served platform.
        passenger_wait_penalty = 0.0025
        # Network-wide penalty to align reward with global avg_waiting_time KPI.
        global_waiting_penalty_scale = 0.001
        # Symmetric headway shaping (penalize both early/late departures).
        headway_deviation_cap = 0.1
        headway_on_time_bonus = 0.02
        # Previously applied as terminal lump sum; now spread per-step.
        terminal_delay_weight = 0.001
        terminal_waiting_weight = 0.0002

        tick_seconds = float(getattr(self.world, "tick_seconds", 1) or 1)
        platform_nodes = getattr(self.world, "platform_nodes", {}) if self.world is not None else {}
        platform_count = max(len(platform_nodes), 1)

        for env_index in range(self.num_envs):
            # current_route_val = self.state.current_route[env_index].item()
            # if not self._is_primary_route(current_route_val):
            #     continue

            action_val = int(self.state.action[env_index].item())
            action_outcome = int(self.state.action_result[env_index].item())

            if Action.is_move_action(action_val) and action_outcome == 2:
                reward[env_index] += base_move_bonus
                reward_breakdown[env_index].append(f"move_success=+{base_move_bonus:.3f}")
            elif Action.is_move_action(action_val) and action_outcome == 0:
                reward[env_index] -= blocked_move_penalty
                reward_breakdown[env_index].append(f"move_blocked=-{blocked_move_penalty:.3f}")

            current_node = int(self.state.position[env_index].item())
            on_platform = False
            if self.world is not None:
                on_platform = current_node in getattr(self.world, "platform_nodes", {})

            dwell_min_limit = float(self.state.dwell_min_limit[env_index].item())
            dwell_max_limit = float(self.state.dwell_max_limit[env_index].item())
            current_dwell_time = 0.0
            if on_platform:
                current_dwell_time = float(self.state.time_since_arrival[env_index].item())
            # 站台停车奖励/惩罚
            if on_platform and action_val == Action.STOP:
                if (
                    dwell_min_limit > 0.0
                    and current_dwell_time + dwell_threshold_tolerance < dwell_min_limit
                ):
                    delta = dwell_min_limit - current_dwell_time
                    step_progress = min(delta, 1.0)
                    bonus = dwell_step_scale * step_progress
                    reward[env_index] += bonus
                    reward_breakdown[env_index].append(
                        f"dwell_min_hold=+{bonus:.3f} (t={current_dwell_time:.1f}, min={dwell_min_limit:.1f})"
                    )
                if (
                    dwell_max_limit > 0.0
                    and current_dwell_time - dwell_threshold_tolerance > dwell_max_limit
                ):
                    delta = current_dwell_time - dwell_max_limit
                    step_excess = min(delta, 1.0)
                    penalty = dwell_step_scale * step_excess
                    reward[env_index] -= penalty
                    reward_breakdown[env_index].append(
                        f"dwell_overstay=-{penalty:.3f} (t={current_dwell_time:.1f}, max={dwell_max_limit:.1f})"
                    )
            # 站外停车惩罚
            elif not on_platform and action_val == Action.STOP:
                reward[env_index] -= non_platform_hold_penalty
                reward_breakdown[env_index].append(
                    f"hold_off_platform=-{non_platform_hold_penalty:.3f}"
                )
            # 离站奖励/惩罚
            if self.state.just_departed[env_index]:
                dwell_target = float(self.state.dwell_target[env_index].item())
                dwell_duration = float(self.state.last_dwell_duration[env_index].item())
                if dwell_target > 0.0 and dwell_duration > 0.0:
                    dwell_error = dwell_duration - dwell_target
                    abs_error = abs(dwell_error)
                    if abs_error <= dwell_threshold_tolerance:
                        reward[env_index] += dwell_depart_bonus_scale
                        reward_breakdown[env_index].append(
                            f"dwell_depart=+{dwell_depart_bonus_scale:.3f}"
                        )
                    elif dwell_error < 0.0:
                        penalty = min(dwell_threshold_penalty_scale, dwell_step_scale * abs_error)
                        reward[env_index] -= penalty
                        reward_breakdown[env_index].append(
                            f"dwell_short=-{penalty:.3f} (err={dwell_error:.1f})"
                        )
                    else:
                        penalty = min(dwell_threshold_penalty_scale, dwell_step_scale * abs_error)
                        reward[env_index] -= penalty
                        reward_breakdown[env_index].append(
                            f"dwell_long=-{penalty:.3f} (err={dwell_error:.1f})"
                        )

                headway_gap = float(self.state.headway_gap[env_index].item())
                if headway_gap != 0.0 and self.headway_seconds > 0:
                    gap_score = min(headway_deviation_cap, abs(headway_gap) / self.headway_seconds)
                    reward[env_index] -= gap_score
                    reward_breakdown[env_index].append(f"headway_dev=-{gap_score:.3f}")
                    delay_increment = abs(headway_gap)
                    self.cumulative_delay[env_index] += delay_increment
                    delay_penalty = terminal_delay_weight * delay_increment
                    reward[env_index] -= delay_penalty
                    reward_breakdown[env_index].append(f"delay_step-{delay_penalty:.4f}")
                elif self.headway_seconds > 0:
                    reward[env_index] += headway_on_time_bonus
                    reward_breakdown[env_index].append(f"headway_on_time=+{headway_on_time_bonus:.3f}")

                self.state.just_departed[env_index] = False
                self.state.dwell_min_limit[env_index] = 0.0
                self.state.dwell_max_limit[env_index] = 0.0
                self.state.dwell_target[env_index] = 0.0
                self.state.last_dwell_duration[env_index] = 0.0
                self.state.headway_gap[env_index] = 0.0

            # Passenger-related rewards
            current_passengers = float(self.state.passenger_count[env_index].item())
            previous_passengers = float(self.state.passenger_count_prev[env_index].item())
            passenger_delta = current_passengers - previous_passengers

            boarded = max(passenger_delta, 0.0)
            alighted = max(-passenger_delta, 0.0)

            if boarded > 0.0:
                bonus = boarded * passenger_board_reward
                reward[env_index] += bonus
                reward_breakdown[env_index].append(f"board+{bonus:.4f}")

            if alighted > 0.0:
                bonus = alighted * passenger_alight_reward
                reward[env_index] += bonus
                reward_breakdown[env_index].append(f"alight+{bonus:.4f}")

            # Global waiting penalty: all platforms contribute every step.
            total_waiting_count = 0.0
            if platform_nodes:
                for node in platform_nodes.values():
                    waiting_tensor = getattr(node, "waiting_passengers", None)
                    if waiting_tensor is None or waiting_tensor.shape[0] <= env_index:
                        continue
                    total_waiting_count += (self.world.waiting_at(node.id, env_index) if hasattr(self.world, "waiting_at")
                                            else float(waiting_tensor[env_index].item()))

            if total_waiting_count > 0.0:
                global_wait_penalty = global_waiting_penalty_scale * (total_waiting_count / platform_count)
                reward[env_index] -= global_wait_penalty
                reward_breakdown[env_index].append(f"wait_global-{global_wait_penalty:.4f}")
                waiting_increment = total_waiting_count * tick_seconds
                self.cumulative_waiting_time[env_index] += waiting_increment
                waiting_step_penalty = terminal_waiting_weight * waiting_increment
                reward[env_index] -= waiting_step_penalty
                reward_breakdown[env_index].append(f"wait_step-{waiting_step_penalty:.4f}")

            waiting_penalty = 0.0
            if self.world is not None and on_platform:
                node = self.world.get_node(current_node)
                waiting_tensor = getattr(node, "waiting_passengers", None)
                if waiting_tensor is not None and waiting_tensor.shape[0] > env_index:
                    waiting_count = float(waiting_tensor[env_index].item())
                    waiting_penalty = waiting_count * passenger_wait_penalty
                    reward[env_index] -= waiting_penalty
                    if waiting_penalty > 0.0:
                        reward_breakdown[env_index].append(f"wait-{waiting_penalty:.4f}")

            # remember passenger count for next step
            self.state.passenger_count_prev[env_index] = current_passengers

        for env_index, components in enumerate(reward_breakdown):
            if not components:
                continue
            total = float(reward[env_index].item())
            # logger.info(f"[{self.name}] reward env={env_index} total={total:.4f} :: " + " | ".join(components))

        return reward

    def calc_terminal_reward(self, done_mask: torch.Tensor) -> torch.Tensor:
        if self.world is not None and self.world.dispatch:
            cost = self.world.dispatch.objective.terminal(self.world)["total"] / max(1, len(self.world.agents))
            return -done_mask.to(self.device, dtype=torch.float32) * cost
        # Terminal penalties are already spread into per-step reward in calc_reward.
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _reset_control_state(self, env_index: Optional[int] = None) -> None:
        if env_index is None:
            self.state.target_departure_time.zero_()
            self.state.dwell_target.zero_()
            self.state.time_since_arrival.zero_()
            self.state.last_departure_time.fill_(float("-inf"))
            self.state.headway_gap.zero_()
            self.state.arrival_headway.zero_()
            self.state.headway_error_last.fill_(-1.0)
            self.state.dwell_min_limit.zero_()
            self.state.dwell_max_limit.zero_()
            self.state.last_dwell_duration.zero_()
            self.state.just_departed.zero_()
            self.state.current_platform.fill_(-1)
        else:
            idx = int(env_index)
            self.state.target_departure_time[idx] = 0.0
            self.state.dwell_target[idx] = 0.0
            self.state.time_since_arrival[idx] = 0.0
            self.state.last_departure_time[idx] = float("-inf")
            self.state.headway_gap[idx] = 0.0
            self.state.arrival_headway[idx] = 0.0
            self.state.headway_error_last[idx] = -1.0
            self.state.dwell_min_limit[idx] = 0.0
            self.state.dwell_max_limit[idx] = 0.0
            self.state.last_dwell_duration[idx] = 0.0
            self.state.just_departed[idx] = False
            self.state.current_platform[idx] = -1

    def _reset_passenger_state(self, env_index: Optional[int] = None) -> None:
        if env_index is None:
            self.state.passenger_count.zero_()
            self.state.passenger_count_prev.zero_()
            for tensor in self.passengers_by_destination.values():
                tensor.zero_()
            self.completed_waiting_mass.zero_()
            for tensor in self.passenger_waiting_mass.values():
                tensor.zero_()
        else:
            idx = int(env_index)
            self.state.passenger_count[idx] = 0.0
            self.state.passenger_count_prev[idx] = 0.0
            for tensor in self.passengers_by_destination.values():
                tensor[idx] = 0.0
            self.completed_waiting_mass[idx] = 0
            for tensor in self.passenger_waiting_mass.values():
                tensor[idx] = 0

    def _set_default_route(self, env_index: Optional[int] = None) -> None:
        if self._primary_route_id is None:
            return
        try:
            default_value = int(self._primary_route_id)
        except (TypeError, ValueError):
            return
        if env_index is None:
            self.state.current_route.fill_(default_value)
        else:
            idx = int(env_index)
            self.state.current_route[idx] = default_value

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

    def _compute_platform_dwell_limits(self, node) -> tuple[float, float, float]:
        tick_seconds = float(getattr(self.world, "tick_seconds", 1) or 1)
        node_tick_seconds = float(getattr(node, "tick_seconds", tick_seconds) or tick_seconds)
        baseline_seconds = getattr(node, "dwell_seconds", None)
        if baseline_seconds is None or baseline_seconds <= 0:
            weight = float(getattr(node, "weight", 0) or 0.0)
            if weight > 0:
                if getattr(node, "dwell_seconds", None) is not None:
                    baseline_seconds = weight
                else:
                    baseline_seconds = weight * node_tick_seconds
            elif self._default_dwell is not None:
                baseline_seconds = float(self._default_dwell)
            else:
                baseline_seconds = tick_seconds
        else:
            baseline_seconds = float(baseline_seconds)

        min_ticks = max(1, int(math.ceil(baseline_seconds / tick_seconds)))
        tolerance_seconds = float(self.dwell_tolerance)
        max_seconds = baseline_seconds + tolerance_seconds
        max_ticks = max(min_ticks, int(math.ceil(max_seconds / tick_seconds)))
        return float(min_ticks), float(max_ticks), baseline_seconds

    # ------------------------------------------------------------------
    # Compatibility accessors for state tensors
    # ------------------------------------------------------------------
    def _assign_state_tensor(self, attr: str, value: torch.Tensor) -> None:
        target = getattr(self.state, attr)
        dtype = target.dtype if isinstance(target, torch.Tensor) else None
        if isinstance(value, torch.Tensor):
            tensor = value.to(self.device)
            if dtype is not None and tensor.dtype != dtype:
                tensor = tensor.to(dtype=dtype)
        else:
            tensor = torch.as_tensor(value, device=self.device, dtype=dtype)
        setattr(self.state, attr, tensor)

    @property
    def action(self) -> torch.Tensor:
        return self.state.action

    @action.setter
    def action(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("action", value)

    @property
    def prev_action(self) -> torch.Tensor:
        return self.state.prev_action

    @prev_action.setter
    def prev_action(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("prev_action", value)

    @property
    def action_result(self) -> torch.Tensor:
        return self.state.action_result

    @action_result.setter
    def action_result(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("action_result", value)

    @property
    def potential_next_pos(self) -> torch.Tensor:
        return self.state.potential_next_pos

    @potential_next_pos.setter
    def potential_next_pos(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("potential_next_pos", value)

    @property
    def target_departure_time(self) -> torch.Tensor:
        return self.state.target_departure_time

    @target_departure_time.setter
    def target_departure_time(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("target_departure_time", value)

    @property
    def dwell_target(self) -> torch.Tensor:
        return self.state.dwell_target

    @dwell_target.setter
    def dwell_target(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("dwell_target", value)

    @property
    def time_since_arrival(self) -> torch.Tensor:
        return self.state.time_since_arrival

    @time_since_arrival.setter
    def time_since_arrival(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("time_since_arrival", value)

    @property
    def last_departure_time(self) -> torch.Tensor:
        return self.state.last_departure_time

    @last_departure_time.setter
    def last_departure_time(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("last_departure_time", value)

    @property
    def headway_gap(self) -> torch.Tensor:
        return self.state.headway_gap

    @headway_gap.setter
    def headway_gap(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("headway_gap", value)

    @property
    def arrival_headway(self) -> torch.Tensor:
        return self.state.arrival_headway

    @arrival_headway.setter
    def arrival_headway(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("arrival_headway", value)

    @property
    def headway_error_last(self) -> torch.Tensor:
        return self.state.headway_error_last

    @headway_error_last.setter
    def headway_error_last(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("headway_error_last", value)

    @property
    def just_departed(self) -> torch.Tensor:
        return self.state.just_departed

    @just_departed.setter
    def just_departed(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("just_departed", value)

    @property
    def current_platform(self) -> torch.Tensor:
        return self.state.current_platform

    @current_platform.setter
    def current_platform(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("current_platform", value)

    @property
    def _current_platform(self) -> torch.Tensor:  # legacy alias
        return self.state.current_platform

    @_current_platform.setter
    def _current_platform(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("current_platform", value)

    @property
    def dwell_min_limit(self) -> torch.Tensor:
        return self.state.dwell_min_limit

    @dwell_min_limit.setter
    def dwell_min_limit(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("dwell_min_limit", value)

    @property
    def dwell_max_limit(self) -> torch.Tensor:
        return self.state.dwell_max_limit

    @dwell_max_limit.setter
    def dwell_max_limit(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("dwell_max_limit", value)

    @property
    def last_dwell_duration(self) -> torch.Tensor:
        return self.state.last_dwell_duration

    @last_dwell_duration.setter
    def last_dwell_duration(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("last_dwell_duration", value)

    @property
    def passenger_count(self) -> torch.Tensor:
        return self.state.passenger_count

    @passenger_count.setter
    def passenger_count(self, value: torch.Tensor) -> None:
        self._assign_state_tensor("passenger_count", value)

    def get_available_paths_count(self, env_index: Optional[int] = None) -> torch.Tensor:
        if self.world is None:
            if env_index is None:
                return torch.zeros((self.num_envs, 1), device=self.device)
            return torch.zeros(1, device=self.device)

        if env_index is not None:
            current_pos = int(self.state.position[env_index].item())
            next_nodes = self.world.get_next_nodes(current_pos)
            return torch.tensor([len(next_nodes)], device=self.device, dtype=torch.float32)

        counts = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        for idx in range(self.num_envs):
            current_pos = int(self.state.position[idx].item())
            next_nodes = self.world.get_next_nodes(current_pos)
            counts[idx, 0] = len(next_nodes)
        return counts

    def init_position(self, position: torch.Tensor) -> None:
        self.set_position(position)

    def to(self, device: torch.device) -> None:
        self.device = device
        for field in self.state.__dataclass_fields__:
            value = getattr(self.state, field)
            if isinstance(value, torch.Tensor):
                setattr(self.state, field, value.to(device))

        if self.headway_tracker is not None:
            self.headway_tracker.to(device)


__all__ = [
    "Action",
    "MetroAgentV2",
    "PlatformHeadwayTracker",
]
