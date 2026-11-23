"""Timed dispatch operations executed by the existing MetroWorldV1.

Vehicle IDs never disappear when service changes. All movement proposals still
pass through the world's joint safety checks; a command is not a teleport.
"""
from collections import deque
import copy
from dataclasses import dataclass, field, asdict
import math
import hashlib
import json

import torch

from dispatch_plan import DispatchPlan
from dispatch_objective import DispatchObjective


@dataclass
class VehicleRecord:
    id: str
    agent: object
    role: str = "service"
    status: str = "IN_SERVICE"
    service_id: str | None = None
    depot: str | None = None
    failed: bool = False
    needs_inspection: bool = False
    has_served: bool = True


@dataclass
class Operation:
    id: str
    command_id: str
    action: str
    vehicle_id: str
    phase: str
    started_at: float
    deadline: float
    path: deque = field(default_factory=deque)
    target: int | None = None
    depot: str | None = None
    partner: str | None = None
    continuation: list = field(default_factory=list)
    wait_reason: str | None = None
    service_id: str | None = None
    turnback_entry: int | None = None
    clear_destinations: list | None = None

    def participants(self):
        return {self.vehicle_id, self.partner} - {None}


class _PatternGeometryCache(dict):
    def __deepcopy__(self, memo):
        # Shared caches contain topology geometry, never live occupancy/events.
        return self


class DispatchOperations:
    """One persistent operation state per physical vehicle, not 15 RL labels."""
    def __init__(self, world, config):
        if world.num_envs != 1:
            raise ValueError("Structured operations currently require one serial environment")
        self.world = world
        self.config = copy.deepcopy(config)
        if self.config.get("reserve_policy", "unrestricted") not in {"unrestricted", "contingency"}:
            raise ValueError("Unknown reserve activation policy")
        self.objective = DispatchObjective(self.config.get("objective"))
        self.interval_cost = self.objective.interval(0., 0., [])
        self.depots = self.config.get("depots", {})
        self.objective.prepare(world, self.depots)
        self.turnback_facilities = {segment: spec for spec in world.event_resources.get("turnbacks", {}).values()
                                    for segment in spec["segments"]}
        self.turnback_ids = {segment: key for key, spec in world.event_resources.get("turnbacks", {}).items()
                            for segment in spec["segments"]}
        self.turnback_receivers = {}
        self._pattern_geometries = _PatternGeometryCache()
        self._receiver_approach_cache = _PatternGeometryCache()
        self._forward_distance_cache = _PatternGeometryCache()
        self._shared_interiors = _PatternGeometryCache({rid: frozenset(
            node.id for sid in spec['segments']
            for node in world.segment2nodes[world._segment_id_map[sid]][1:-1])
            for kind in ('turnbacks', 'shared_routes')
            for rid, spec in world.event_resources.get(kind, {}).items()})
        self.profiles = self.config.get("run_profiles", {"normal": 1., "slow": 1.2, "very_slow": 1.5})
        if self.profiles.get("normal") != 1. or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                or not math.isfinite(v) or v < 1 for v in self.profiles.values()):
            raise ValueError("Run profiles must include normal=1, with finite multipliers >=1")
        self.clearance_seconds = self._positive(self.config.get("clearance_seconds", 30), "clearance_seconds")
        self.turnback_seconds = self._positive(self.config.get("turnback_seconds", 60), "turnback_seconds")
        self.coupling_seconds = self._positive(self.config.get("coupling_seconds", 60), "coupling_seconds")
        self.tow_multiplier = self._positive(self.config.get("tow_multiplier", 2.), "tow_multiplier")
        if self.tow_multiplier < 1:
            raise ValueError("Towing cannot accelerate a normal train")
        nodes = set(world.nodes)
        for key, d in self.depots.items():
            exits = d.get("exit_node_ids", [d["exit_node_id"]])
            if not exits or d["exit_node_id"] not in exits:
                raise ValueError("A depot needs a declared default exit")
            for n in (d["node_id"], *exits):
                if n not in nodes:
                    raise ValueError(f"Unknown depot node in {key}")
            if not isinstance(d["capacity"], int) or isinstance(d["capacity"], bool) or d["capacity"] < 1:
                raise ValueError("A depot needs positive integer storage capacity")
            for name in ("preparation_seconds", "inspection_seconds"):
                self._positive(d[name], name)
            slots = d.get("inspection_capacity", 1)
            if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
                raise ValueError("Inspection capacity must be a positive integer")
            for exit_node in exits:
                if self.path(d["node_id"], exit_node, allow_depot=key) is None:
                    raise ValueError("Depot has no declared directed exit path")
        self.records = {a.name: VehicleRecord(a.name, a) for a in world.agents}
        self.services = {}
        for sid, task in self.config.get("service_tasks", {}).items():
            stops = task.get("stops", [])
            if (not isinstance(sid, str) or not sid or not isinstance(stops, list) or len(stops) < 2
                    or any(not isinstance(n, int) or isinstance(n, bool) or n not in world.platform_nodes for n in stops)
                    or any(n in {d["node_id"] for d in self.depots.values()} for n in stops)):
                raise ValueError("A service task needs an identity and at least two passenger stops")
            self.services[sid] = dict(vehicle_id=None, started_at=None, ended_at=None, stops=list(stops),
                purpose=task.get("purpose", "scheduled"), required=task.get("required", True),
                pattern=task.get("pattern", False), repeat=task.get("repeat", False))
        self.operations = {}
        self.command_results = {}
        self.plans = {}
        self.inventory = {d: set() for d in self.depots}
        self.bookings = {d: set() for d in self.depots}
        self.admission_rates = {}
        self.admission_used = {}
        self.profile_selections = {}
        self.log = []
        self.serial = 0
        self.version = 0
        self.seen_failures = set()
        self.hidden_failure_prior = {}
        self.physical_departures = {}
        self.first_departure_due = {}
        self.pending_transits = {}
        self._logged_phases = {}
        assignments = self.config.get("vehicles", {})
        if set(assignments) - set(self.records):
            raise ValueError("Vehicle configuration references an unregistered physical train")
        depot_nodes = {d["node_id"] for d in self.depots.values()}
        for key, v in self.records.items():
            spec = assignments.get(key, {})
            v.role = spec.get("role", "service")
            if v.role not in ("service", "reserve"):
                raise ValueError("Vehicle role must be service or reserve")
            v.depot = spec.get("depot")
            if v.depot is not None:
                if v.depot not in self.depots:
                    raise ValueError("Unknown initial depot")
                if float(v.agent.passenger_count[0]) > 1e-6:
                    raise ValueError("Initially stored vehicles must be empty")
                self.inventory[v.depot].add(key)
                v.agent.state.position[0] = self.depots[v.depot]["node_id"]
                v.agent.state.previous_position[0] = v.agent.state.position[0]
                v.status = "DEPOT_READY"
                if type(spec.get("has_served", False)) is not bool:
                    raise ValueError("Initial service history must be boolean")
                v.has_served = spec.get("has_served", False)
            else:
                if "initial_node" in spec:
                    if spec["initial_node"] not in world.nodes or spec["initial_node"] in depot_nodes:
                        raise ValueError("Invalid initial service position")
                    v.agent.state.position[0] = spec["initial_node"]
                    v.agent.state.previous_position[0] = spec["initial_node"]
                if self.position(key) in depot_nodes:
                    raise ValueError("An initial service train cannot start inside a storage depot")
                self._start_service(v)
            v.agent.state.current_platform[0] = self.position(key) if v.depot is None and self.position(key) in world.platform_nodes else -1
            if self.service_enabled(v.agent) and self.position(key) in world.platform_nodes:
                remaining = max(0., world.minimum_dwell_steps(v.agent, self.position(key))
                                - float(v.agent.state.dwell_time[0]))
                self.first_departure_due[self.position(key)] = self.now + remaining*self.tick
        if any(len(keys) > self.depots[d]["capacity"] for d, keys in self.inventory.items()):
            raise ValueError("Initial depot inventory exceeds capacity")

    @staticmethod
    def _positive(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
        return float(value)

    @property
    def now(self):
        return float(self.world.sim_time[0])

    @property
    def tick(self):
        return float(self.world.tick_seconds or 1)

    def duration(self, seconds):
        return math.ceil(seconds / self.tick) * self.tick

    def position(self, key):
        return int(self.records[key].agent.state.position[0])

    def event(self, name, **data):
        self.version += 1
        self.log.append(dict(time=self.now, event=name, **data))

    def _start_service(self, v, service_id=None):
        self.serial += 1
        sid = service_id or f"service_{v.id}_{self.serial}"
        if service_id:
            task = self.services.get(sid)
            if (task is None or task["vehicle_id"] is not None or task["stops"][0] != self.position(v.id)
                    or task.get("reserved_vehicle_id", v.id) != v.id):
                raise ValueError("Service task unavailable or origin incompatible with vehicle")
            task.update(vehicle_id=v.id, started_at=self.now)
            task.pop("reserved_vehicle_id", None)
            # Detraining for reassignment is followed by a fresh passenger stop,
            # not an immediate departure using the old service's elapsed dwell.
            v.agent.state.dwell_time[0] = 0
            v.agent.state.time_since_arrival[0] = 0
            v.agent.state.arrival_time[0] = self.now
        else:
            if sid in self.services:
                raise ValueError("Service identity has already been used")
            self.services[sid] = dict(vehicle_id=v.id, started_at=self.now, ended_at=None, stops=None)
        v.service_id, v.status, v.has_served = sid, "IN_SERVICE", True
        self.event("service_started", vehicle_id=v.id, service_id=sid,
                   stops=copy.deepcopy(self.services[sid]["stops"]))

    def _end_service(self, v, *, completed=False):
        if v.service_id:
            self.services[v.service_id]["ended_at"] = self.now
            self.services[v.service_id]["completed"] = completed
            self.event("service_ended", vehicle_id=v.id, service_id=v.service_id)
        v.service_id = None
        self.profile_selections.pop(v.id, None)

    def operation_for(self, key):
        return next((o for o in self.operations.values() if key in o.participants()), None)

    def _pattern_geometry(self, task):
        key = tuple(task["stops"])
        if key not in self._pattern_geometries:
            walk = [key[0]]
            for first, last in zip(key, key[1:]):
                segment = self.world._segment_id_map.get(f"{first}-{last}")
                if segment is None:
                    return None
                walk.extend(n.id for n in self.world.segment2nodes[segment][1:])
            self._pattern_geometries[key] = (tuple(walk), frozenset(walk),
                frozenset(zip(walk, walk[1:])), {p:i for i,p in enumerate(walk[:-1])
                                               if p in self.world.platform_nodes})
        return self._pattern_geometries[key]

    def service_route(self, service_id, origin):
        task = self.services[service_id]
        stops = list(task["stops"])
        if task.get("pattern"):
            cycle = stops[:-1]
            if stops[0] != stops[-1] or origin not in cycle:
                return None
            index = cycle.index(origin)
            stops = cycle[index:] + cycle[:index] + [origin]
            geometry = self._pattern_geometry(task)
            if geometry is None:
                return None
            walk, _, _, indices = geometry
            offset = indices[origin]
            return stops, list(walk[offset+1:-1]) + list(walk[:offset+1])
        elif stops[0] != origin:
            return None
        path = []
        for first, last in zip(stops, stops[1:]):
            segment = self.world._segment_id_map.get(f"{first}-{last}")
            part = ([n.id for n in self.world.segment2nodes[segment][1:]] if segment else self.path(first, last))
            if not part:
                return None
            path.extend(part)
        return stops, path

    def section_need(self, service_id, origin):
        """Current public reasons for a section pattern; not a benefit label."""
        view = self.world.event_view(published=True)
        if not view.records:
            return []
        task = self.services[service_id]
        if task.get("pattern"):
            if origin not in task["stops"][:-1]:
                return []
            geometry = self._pattern_geometry(task)
            if geometry is None:
                return []
            _, nodes, edges, _ = geometry
            stops = task["stops"]
        else:
            route = self.service_route(service_id, origin)
            if route is None:
                return []
            stops, path = route
            nodes, edges = set(path) | {origin}, set(zip([origin] + path, path))
        result = []
        for event in view.active():
            kind, target = event["type"], event["target"]["id"]
            if kind in {"track_blockage", "power_restriction", "speed_restriction", "signal_degradation"}:
                segment = self.world._segment_id_map.get(str(target))
                if segment and not nodes.intersection(n.id for n in self.world.segment2nodes[segment][1:-1]):
                    result.append(event["event_id"])
            elif kind == "platform_closure" and target not in stops:
                result.append(event["event_id"])
            elif kind in {"switch_failure", "turnback_capacity_loss"}:
                table = "switches" if kind == "switch_failure" else "turnbacks"
                spec = view.resources.get(table, {}).get(target, {})
                if kind == "switch_failure":
                    entries = spec.get("route_entry_nodes", {})
                    blocked = {(entry, (self.world.nodes[entry].getnext() or {}).get(rid))
                               for rid in event["parameters"]["disabled_route_ids"]
                               for entry in [entries.get(str(rid), spec.get("node_id"))] if entry in self.world.nodes}
                    if not edges.intersection(blocked):
                        result.append(event["event_id"])
                    continue
                else:
                    affected = set(spec.get("interior_nodes", ())) if event["parameters"]["capacity"] == 0 else nodes
                if not nodes.intersection(affected):
                    result.append(event["event_id"])
            elif kind == "demand_surge" and target in stops and self.world.waiting_at(target) > 0:
                result.append(event["event_id"])
        for event in view.records:
            if event["type"] == "vehicle_breakdown":
                vehicle = self.records[event["target"]["id"]]
                if vehicle.depot is None and vehicle.failed and self.position(vehicle.id) not in nodes:
                    result.append(event["event_id"])
        return sorted(set(result))

    def service_revisable(self, vehicle):
        op = self.operation_for(vehicle)
        if not (op and op.action == "reassign_service" and op.phase == "RUNNING"
                and self.services.get(op.service_id, {}).get("repeat") and self._safe_platform(self.position(vehicle))):
            return False
        return not any(c.command_id == op.command_id and c.locked for p in self.plans.values() for c in p.commands)

    def discretionary_hold_seconds(self, vehicle):
        record = self.records[vehicle]
        pos = self.position(vehicle)
        op = self.operation_for(vehicle)
        if op:
            # Timed phases and resource/signal waits are executed internally.
            # A new primitive pause must not renew an irrevocable operation's
            # delay every decision window.
            return 0.
        if record.depot is not None or record.failed:
            return math.inf
        if float(record.agent.state.speed_limit_delay[0]) > 0:
            return math.inf
        from MILP.operation_model import nominal_action
        action = nominal_action(self.world,record.agent)
        if not action:
            return math.inf
        target = sorted((self.world.nodes[pos].getnext() or {}).items(),key=lambda item:str(item[0]))[action-1][1]
        if target == pos or not self.world.edge_event_allowed(record.agent,pos,target,allow_tow=self.towed(vehicle)):
            return math.inf
        if not self.receiver_allowed(vehicle,target):
            return math.inf
        segment = self.world._step_segment.get((pos,target))
        seconds = self.world.event_view().safety_headway(segment.id if segment else None,30)
        ahead = self.world._get_forward_distances(target,max(1,math.ceil(seconds/self.tick)))
        blocked = any(other.id != vehicle and self.blocks_track(other.agent)
                   and not self.ignore_pair(record.agent,other.agent)
                   and ahead.get(self.position(other.id),math.inf) < seconds/self.tick
                   for other in self.records.values())
        if blocked:
            return math.inf
        if self._safe_platform(pos):
            agent = record.agent
            _, maximum, _ = agent._compute_platform_dwell_limits(self.world.platform_nodes[pos])
            maximum += math.ceil(self.world.event_view().dwell_addition(vehicle,pos)/self.tick)
            tolerance = max(0.,maximum-float(agent.state.dwell_time[0]))*self.tick
            regulation = max(0.,float(agent.state.target_departure_time[0])-self.now)
            return max(tolerance,regulation)
        return 0.

    def discretionary_hold_allowed(self, vehicle):
        from dispatch_timing import decision_seconds
        return self.discretionary_hold_seconds(vehicle) + 1e-6 >= decision_seconds(self.world)

    def blocks_track(self, agent, env=0):
        v = self.records[agent.name]
        return v.depot is None

    def _check_turnback_receiver(self, vehicle, path):
        origin = self.position(vehicle)
        segment = self.world._step_segment.get((origin, path[0])) if path else None
        if not segment or segment.id not in self.turnback_ids or origin != segment.start_platform.id:
            return
        target = segment.end_platform.id
        for op in self.operations.values():
            if vehicle in op.participants() or not op.path or self.position(op.vehicle_id) != target:
                continue
            opposite = self.world._step_segment.get((target, op.path[0]))
            if opposite and opposite.id in self.turnback_ids and opposite.end_platform.id == origin:
                raise ValueError("opposed_turnbacks_block_each_others_receiving_platform")

    def _refresh_turnback_receivers(self):
        """Protect an imminent turnback's receiving platform before it fills.

        Existing occupants may leave. Conflicting approaches yield to an
        occupied crossing, then a train at the gate, then the older commitment.
        """
        requests = {}
        order = {a.name: i for i, a in enumerate(self.world.agents)}
        for op in self.operations.values():
            if not op.path:
                continue
            leader = op.partner if op.action == "rescue_tow" and op.phase in {"TOW", "CLEARING_FAILED"} else op.vehicle_id
            pos = self.position(leader)
            segment = self.world._step_segment.get((pos, op.path[0]))
            priority = 1 if pos in self.world.platform_nodes else 0
            if not segment or segment.id not in self.turnback_ids:
                if len(op.path) < 2 or op.path[0] not in self.world.platform_nodes:
                    continue
                segment = self.world._step_segment.get((op.path[0], op.path[1]))
                priority = 2
            if not segment or segment.id not in self.turnback_ids:
                continue
            rid = self.turnback_ids[segment.id]
            requests.setdefault(rid, []).append((priority, op.started_at, order[op.vehicle_id],
                                                 segment.end_platform.id, op.vehicle_id))
        receivers = {}
        for rid, choices in requests.items():
            spec = self.world.event_resources["turnbacks"][rid]
            capacity = self.world.event_view().turnback_capacity(rid, spec.get("capacity", 1))
            occupied = [choice for choice in choices if choice[0] == 0]
            waiting = sorted(choice for choice in choices if choice[0] != 0)
            selected = occupied + waiting[:max(0, capacity-len(occupied))]
            for _, _, _, platform, vehicle in selected:
                receivers[platform] = vehicle
        self.turnback_receivers = receivers

    def receiver_allowed(self, vehicle, platform):
        op = self.operation_for(vehicle)
        pos = self.position(vehicle)
        for receiver, owner in self.turnback_receivers.items():
            if owner == vehicle or op and owner in op.participants():
                continue
            approach = self._receiving_approach(receiver)
            if platform != receiver and platform not in approach:
                continue
            segment = self.world._step_segment.get((pos, platform))
            if segment and segment.id in self.turnback_ids and pos != segment.start_platform.id:
                continue  # An entered crossing must still be able to clear.
            if pos in approach:
                continue  # Existing approach occupants leave ahead of the turn.
            return False
        return True

    def _receiving_approach(self, platform):
        seconds = self.world.event_view().safety_headway(None,30)
        count = max(1,math.ceil(seconds/self.tick))
        key = (platform,count)
        if key not in self._receiver_approach_cache:
            nodes = set()
            depots = {d['node_id'] for d in self.depots.values()}
            for segment in self.world.base_segments:
                if (segment.end_platform.id == platform and segment.id not in self.turnback_ids
                        and segment.start_platform.id not in depots):
                    nodes.update(n.id for n in self.world.segment2nodes[segment][-(count+1):-1])
            self._receiver_approach_cache[key] = frozenset(nodes)
        return self._receiver_approach_cache[key]

    def _turnback_receiver_clear(self, agent, segment):
        approach = self._receiving_approach(segment.end_platform.id) | {segment.end_platform.id}
        return not any(v.id != agent.name and self.blocks_track(v.agent)
                       and not self.ignore_pair(agent,v.agent) and self.position(v.id) in approach
                       for v in self.records.values())

    def service_enabled(self, agent, env=0):
        v = self.records[agent.name]
        return v.service_id is not None and not v.failed and v.status in ("IN_SERVICE", "HOLDING")

    def service_impairments(self, vehicle, *, view=None):
        """Reported vehicle-specific service loss, not an immobilization flag."""
        view = self.world.event_view(published=True) if view is None else view
        result = []
        for event in view.active(target_kind="vehicle", target=vehicle):
            kind, parameters = event["type"], event["parameters"]
            if (kind == "vehicle_breakdown"
                    or kind in {"dwell_extension", "running_time_variation"} and parameters["additional_seconds"] > 0
                    or kind == "vehicle_capacity_loss" and parameters["passenger_capacity"] < self.records[vehicle].agent.passenger_capacity):
                result.append(event["event_id"])
        record = self.records[vehicle]
        failed, inspection = record.failed, record.needs_inspection
        visible = {event["event_id"] for event in view.records}
        for event_id, prior in self.hidden_failure_prior.items():
            event = self.world.disruption_manager.disruptions.get(event_id, {})
            if event_id not in visible and event.get("target", {}).get("id") == vehicle:
                failed, inspection = prior["failed"], prior["needs_inspection"]
        if (failed or inspection) and any(event["type"] == "vehicle_breakdown"
                and event["target"]["id"] == vehicle for event in view.records):
            result.append("reported_vehicle_still_requires_recovery")
        return result

    def reserve_need(self):
        """Contingency eligibility and a rule recommendation from public inputs.

        One impaired vehicle is one possible replacement, even when it has
        several events. Concurrent/active reserves cover the same allowance;
        ordinary nonzero waiting is not by itself an emergency stock request.
        The recommended count is not a physical capacity bound. Optimization
        may choose additional reserves under the same documented need; the
        trajectory oracle checks paths, capacity and passenger benefit.
        """
        view = self.world.event_view(published=True)
        affected = sorted(key for key, vehicle in self.records.items()
            if self.service_impairments(key, view=view)
            and (vehicle.depot is None or vehicle.role == "service" and vehicle.has_served))
        waiting = sum(self.world.waiting_at(p) for p in self.world.platform_nodes)
        surge = any(event["parameters"]["arrival_intensity_multiplier"] > 1
                    for event in view.active("demand_surge")) and waiting > 0
        overflow = sum(max(0., self.world.waiting_at(p)
                            - view.queue_capacity(p, self.world.safe_platform_queue_capacity))
                       for p in self.world.platform_nodes)
        capacity = max((float(v.agent.passenger_capacity) for v in self.records.values() if v.role == "reserve"), default=1.)
        # Simultaneous demand and vehicle loss may share the same additional train.
        required = max(len(affected), int(surge), math.ceil(overflow / max(1., capacity)))
        deployed = sorted(key for key, v in self.records.items() if v.role == "reserve"
            and key not in affected
            and (v.status in {"PREPARING", "INSERTING"} or self.service_enabled(v.agent)))
        return dict(service_need_detected=required > 0,
                    needed_trains=max(0, required-len(deployed)), affected_vehicles=affected,
                    demand_surge=bool(surge), overflow_persons=overflow,
                    committed_or_active_reserves=deployed,
                    basis="reported_vehicle_loss_or_current_demand_pressure_minus_already_deployed_reserves")

    def passenger_service(self, agent, node):
        op = self.operation_for(agent.name)
        v = self.records[agent.name]
        stops = self.services[v.service_id]["stops"] if v.service_id else None
        return (self.service_enabled(agent) and (stops is None or node in stops)
                and not (op and op.action in {"reassign_service", "reroute", "adjust_departure", "skip_stop"}
                         and (op.phase in {"SERVICE_CLEARING", "SERVICE_TURNING"} or op.turnback_entry == node))
                and not (op and op.action == "skip_stop" and op.target == node))

    def excluded_destinations(self, agent):
        op = self.operation_for(agent.name)
        result = {op.target} if op and op.action == "skip_stop" else set()
        v = self.records[agent.name]
        task = self.services.get(v.service_id, {})
        if task.get("stops"):
            remaining = set(op.path) if op else set()
            result |= set(self.world.platform_nodes) - (set(task["stops"]) & remaining)
        return result

    def ignore_pair(self, first, second):
        op = self.operation_for(first.name)
        return (op is not None and op.action == "rescue_tow" and second.name in op.participants()
                and op.phase in ("APPROACH", "COUPLING", "TOW", "CLEARING_FAILED", "CLEAR_REAR"))

    def towed(self, key):
        op = self.operation_for(key)
        return op is not None and op.action == "rescue_tow" and op.phase in ("TOW", "CLEAR_REAR")

    def phase_allows_dwell(self, agent, node):
        op = self.operation_for(agent.name)
        return self.service_enabled(agent) and not (op and op.action == "skip_stop" and node == op.target)

    def multiplier(self, agent, segment):
        selection = self.profile_selections.get(agent.name)
        factor = self.profiles[selection[1]] if selection and selection[0] == segment.id else 1.
        return max(factor, self.tow_multiplier if self.towed(agent.name) else 1.)

    def path(self, start, target, explicit=None, allow_depot=None, *, view=None, vehicle_id=None):
        depot_nodes = {d["node_id"] for d in self.depots.values()}
        allowed = self.depots[allow_depot]["node_id"] if allow_depot in self.depots else None

        def edge_open(origin, destination):
            if view is None:
                return True
            segment = self.world._step_segment.get((origin, destination))
            routes = [rid for rid, nxt in (self.world.nodes[origin].getnext() or {}).items()
                      if nxt == destination]
            return view.edge_allowed(vehicle_id or "", origin, destination,
                                     segment.id if segment else None, routes)

        if explicit is not None:
            route, seen, current = list(explicit), {start}, start
            if not route or route[-1] != target:
                return None
            for nxt in route:
                if (nxt in seen or nxt not in self.world._has_edge.get(current, ())
                        or nxt in depot_nodes and nxt != allowed or not edge_open(current, nxt)):
                    return None
                seen.add(nxt); current = nxt
            return deque(route)
        queue, parents = deque([start]), {start: None}
        while queue:
            current = queue.popleft()
            if current == target:
                result = []
                while current != start:
                    result.append(current); current = parents[current]
                return deque(reversed(result))
            for nxt in self.world._next_nodes_id_map.get(current, ()):
                if nxt not in parents and (nxt not in depot_nodes or nxt == allowed) and edge_open(current, nxt):
                    parents[nxt] = current; queue.append(nxt)
        return None

    def available_path(self, start, target, *, allow_depot=None, vehicle_id=None, view=None, explicit=None):
        return self.path(start, target, explicit, allow_depot,
                         view=self.world.event_view() if view is None else view, vehicle_id=vehicle_id)

    def _route_open(self, vehicle, path, *, allow_failed=False):
        current = self.position(vehicle)
        agent = self.records[vehicle].agent
        for nxt in path:
            if not self.world.edge_event_allowed(agent, current, nxt, allow_tow=allow_failed, check_load=False):
                return False
            current = nxt
        return True

    def _book(self, depot, keys):
        if depot not in self.depots:
            raise ValueError("unknown_depot")
        if not self.depot_allowed(depot, "entry"):
            raise ValueError("depot_entry_closed")
        if len(self.inventory[depot] | self.bookings[depot] | set(keys)) > self.depots[depot]["capacity"]:
            raise ValueError("depot_capacity_exceeded")
        self.bookings[depot].update(keys)

    def depot_allowed(self, depot, direction):
        return all(e["parameters"][direction + "_allowed"] for e in self.world.event_view().active("depot_access_restriction", "depot", depot))

    def _clear_duration(self, v, destinations=None):
        rate = self.config.get("alighting_rate_persons_per_second", self.world.boarding_rate_persons_per_second)
        self._positive(rate, "alighting_rate_persons_per_second")
        remaining = 0.
        node = self.position(v.id)
        if node in self.world.platform_nodes:
            minimum = max(self.world.minimum_dwell_steps(v.agent, node),
                          self.world.minimum_dwell_steps(v.agent, node, view=self.world.event_view(published=True)))
            remaining = max(0., minimum*self.tick
                            - float(v.agent.state.dwell_time[0])*self.tick)
        load = (float(v.agent.passenger_count[0]) if destinations is None else
                sum(float(q[0]) for d, q in v.agent.passengers_by_destination.items() if d in destinations))
        return self.duration(max(self.clearance_seconds, load / rate, remaining))

    def _safe_platform(self, node):
        return (node in self.world.platform_nodes and node not in {d["node_id"] for d in self.depots.values()}
                and self.world.event_view().boarding_allowed(node))

    def _new_operation(self, c, phase, *, path=(), deadline=None, **fields):
        self.serial += 1
        op = Operation(f"operation_{self.serial}", c.command_id, c.action, c.vehicle_id, phase,
                       self.now, self.now if deadline is None else deadline, path=deque(path), **fields)
        self.operations[op.id] = op
        return op

    def _set_headway(self, platform, seconds):
        tracker = self.world.agents[0].headway_tracker
        tracker.set_target_headway(platform, seconds)
        for agent in self.world.agents:
            if self.position(agent.name) == platform:
                agent.state.target_departure_time[0] = max(self.now, float(agent.state.arrival_time[0])
                    + self.world.minimum_dwell_steps(agent, platform) * self.tick,
                    tracker.last_departure(platform) + seconds)

    def _start_now(self, c):
        p, action = c.parameters, c.action
        if c.expires_at is not None and self.now > c.expires_at:
            raise ValueError("command_expired")
        if c.vehicle_id is None:
            platform = p["platform_id"]
            if platform not in self.world.platform_nodes:
                raise ValueError("unknown_passenger_platform")
            if action == "admission_control":
                self.admission_rates[platform] = p["rate"]
                self.admission_used.pop(platform, None)
            else:
                self._set_headway(platform, p["headway_seconds"])
            return None
        if c.vehicle_id not in self.records:
            raise ValueError("unknown_vehicle")
        v, pos = self.records[c.vehicle_id], self.position(c.vehicle_id)
        if c.expected_node is not None and c.expected_node != pos:
            raise ValueError("expected_position_mismatch")
        if c.expected_service_id is not None and c.expected_service_id != v.service_id:
            raise ValueError("expected_service_mismatch")
        busy = self.operation_for(v.id)
        if action == "run_profile":
            segment = self.world._segment_id_map.get(p["segment_id"])
            seq = self.world.segment2nodes.get(segment, ())
            if (not self.service_enabled(v.agent) or v.failed or segment is None or p["profile_id"] not in self.profiles
                    or pos not in [n.id for n in seq[:-1]] or busy and busy.action not in ("adjust_departure", "reroute", "skip_stop")):
                raise ValueError("run_profile_not_applicable")
            self.profile_selections[v.id] = (segment.id, p["profile_id"])
            return None
        revising = busy and action in {"reassign_service", "short_turn", "withdraw"} and self.service_revisable(v.id)
        if busy and not revising:
            raise ValueError("vehicle_has_in_progress_operation")
        if action == "inspect_repair":
            if (v.depot is None or not v.needs_inspection or self.world.event_view().active("vehicle_breakdown", "vehicle", v.id)):
                raise ValueError("inspection_requires_depot_and_failure_clearance")
            slots = self.depots[v.depot].get("inspection_capacity", 1)
            if sum(o.action == "inspect_repair" and o.depot == v.depot for o in self.operations.values()) >= slots:
                raise ValueError("inspection_resource_busy")
            v.status = "INSPECTING"
            return self._new_operation(c, "INSPECTING", depot=v.depot,
                                       deadline=self.now + self.duration(self.depots[v.depot]["inspection_seconds"]))
        if action in ("reserve_insert", "reenter"):
            if (v.depot is None or v.failed or v.needs_inspection or v.status != "DEPOT_READY"
                    or self.world.event_view().active("vehicle_breakdown", "vehicle", v.id)):
                raise ValueError("vehicle_not_qualified_for_insertion")
            if action == "reserve_insert" and (v.role != "reserve" or v.has_served):
                raise ValueError("not_an_unused_reserve")
            if (v.role == "reserve" and self.config.get("reserve_policy", "unrestricted") == "contingency"
                    and not self.reserve_need()["service_need_detected"]):
                raise ValueError("reserve_requires_observed_service_need")
            if action == "reenter" and not v.has_served and not v.needs_inspection and not any(
                    e.get("vehicle_id") == v.id and e["event"] == "inspection_completed" for e in self.log):
                raise ValueError("reentry_requires_previous_service_or_repair")
            if not self.depot_allowed(v.depot, "exit"):
                raise ValueError("depot_exit_closed")
            depot = self.depots[v.depot]
            exit_node = p.get("exit_node_id", depot["exit_node_id"])
            if exit_node not in depot.get("exit_node_ids", [depot["exit_node_id"]]):
                raise ValueError("undeclared_depot_exit")
            path = self.available_path(pos, exit_node, allow_depot=v.depot, vehicle_id=v.id)
            if path is None or not self._route_open(v.id, path):
                raise ValueError("exit_path_unavailable")
            v.status = "PREPARING"
            return self._new_operation(c, "PREPARING", path=path, depot=v.depot,
                                       deadline=self.now + self.duration(depot["preparation_seconds"]))
        if v.failed and action not in ("evacuate",):
            raise ValueError("failed_vehicle_requires_rescue")
        if action == "rescue_tow":
            if v.needs_inspection:
                raise ValueError("rescuer_not_qualified")
            failed = self.records.get(p["failed"])
            if (failed is None or failed.id == v.id or not failed.failed or failed.depot is not None
                    or self.operation_for(failed.id) is not None):
                raise ValueError("invalid_or_busy_rescue_target")
            depot = p.get("depot")
            if depot is None:
                raise ValueError("rescue_requires_selected_depot")
            approach = self.path(pos, self.position(failed.id), p.get("approach_node_ids"), allow_depot=v.depot)
            tow = self.path(self.position(failed.id), self.depots[depot]["node_id"], p.get("tow_node_ids"), allow_depot=depot) if depot in self.depots else None
            if not approach or tow is None or not tow or not self._route_open(v.id, approach) or not self._route_open(failed.id, tow, allow_failed=True):
                raise ValueError("rescue_paths_unavailable")
            if float(failed.agent.passenger_count[0]) > 1e-6 and not any(
                    self._safe_platform(n) for n in [self.position(failed.id)] + list(tow)[:-1]):
                raise ValueError("rescue_requires_safe_clearance_before_depot")
            corridor = set(approach) - {self.position(failed.id)}
            if any(self.blocks_track(other.agent) and self.position(other.id) in corridor
                   for other in self.records.values() if other.id not in (v.id, failed.id)):
                raise ValueError("rescue_approach_occupied")
            if float(v.agent.passenger_count[0]) > 1e-6 and not self._safe_platform(pos):
                raise ValueError("rescuer_requires_safe_passenger_clearance")
            if v.depot is not None and not self.depot_allowed(v.depot, "exit"):
                raise ValueError("rescuer_depot_exit_closed")
            self._book(depot, {v.id, failed.id})
            self._end_service(v)
            failed.needs_inspection = True
            deadline = self.now + (self.duration(self.depots[v.depot]["preparation_seconds"]) if v.depot else self._clear_duration(v))
            # Stop one node behind the failed train; coupling never overlaps footprints.
            approach.pop()
            v.status, failed.status = "RESCUING", "AWAITING_RESCUE"
            return self._new_operation(c, "CLEARING_RESCUER", path=approach, continuation=list(tow),
                                       depot=depot, partner=failed.id, deadline=deadline)
        if v.depot is not None:
            raise ValueError("stored_vehicle_cannot_run_service_actions")
        if action in ("evacuate", "cancel_service", "reassign_service", "short_turn", "withdraw"):
            if float(v.agent.passenger_count[0]) > 1e-6 and not self._safe_platform(pos):
                raise ValueError("passenger_clearance_requires_safe_platform")
            if action != "withdraw" and not self._safe_platform(pos):
                raise ValueError("action_requires_safe_platform")
            path, depot = (), None
            service_id, clear_destinations = p.get("service_id"), None
            if action == "reassign_service":
                task = self.services.get(service_id)
                if revising and self.services.get(busy.service_id, {}).get("pattern_id") == service_id:
                    raise ValueError("already_following_section_pattern")
                if (task is None or task["vehicle_id"] is not None or task.get("reserved_vehicle_id") is not None
                        or not task.get("pattern") and task["stops"][0] != pos or v.needs_inspection):
                    raise ValueError("service_task_unavailable_or_vehicle_incompatible")
                route = self.service_route(service_id, pos)
                if route is None:
                    raise ValueError("service_task_has_no_directed_route")
                stops, path = route
                if not self._route_open(v.id, path):
                    raise ValueError("service_task_route_unavailable")
                self._check_turnback_receiver(v.id, path)
                if task.get("pattern"):
                    if not self.section_need(service_id, pos):
                        raise ValueError("section_pattern_requires_reported_service_need")
                    reported = {e["target"]["id"] for e in self.world.event_view(published=True).records
                                if e["type"] == "vehicle_breakdown"}
                    if any(other.id in reported and other.failed and other.depot is None
                           and self.position(other.id) in path for other in self.records.values()):
                        raise ValueError("section_service_contains_failed_vehicle")
                    pattern_id = service_id
                    service_id = f"{pattern_id}:{c.command_id}"
                    task = self.services[service_id] = dict(task, stops=stops, pattern=False, pattern_id=pattern_id)
                clear_destinations = [d for d in v.agent.passengers_by_destination if d not in stops or d == pos]
                task["reserved_vehicle_id"] = v.id
            if action == "short_turn":
                segment = self.world._segment_id_map.get(f"{pos}-{p['target_node']}")
                facilities = [spec for spec in self.world.event_resources.get("turnbacks", {}).values()
                              if segment is not None and segment.id in spec["segments"]]
                if not facilities:
                    raise ValueError("no_declared_turnback_facility")
                path = [n.id for n in self.world.segment2nodes[segment][1:]]
                self._check_turnback_receiver(v.id, path)
                if not self._route_open(v.id, path):
                    raise ValueError("turnback_unavailable")
            if action == "withdraw":
                depot = p.get("depot")
                if depot not in self.depots:
                    raise ValueError("withdrawal_requires_selected_depot")
                path = self.available_path(pos, self.depots[depot]["node_id"], explicit=p.get("path_node_ids"),
                                           allow_depot=depot, vehicle_id=v.id)
                if path is None or not path or not self._route_open(v.id, path):
                    raise ValueError("withdrawal_path_unavailable")
                self._check_turnback_receiver(v.id, path)
                self._book(depot, {v.id})
            if revising:
                self._finish(busy, reason="future_service_revised_at_safe_platform")
            v.status = "CLEARING"
            return self._new_operation(c, "CLEARING", path=path, depot=depot,
                                       target=p.get("target_node"), service_id=service_id,
                                       clear_destinations=clear_destinations,
                                       deadline=self.now + self._clear_duration(v, clear_destinations))
        if not self.service_enabled(v.agent):
            raise ValueError("vehicle_not_in_passenger_service")
        if action == "hold":
            if pos not in self.world.platform_nodes or p["seconds"] <= 0:
                raise ValueError("holding_requires_platform_and_positive_duration")
            if p['seconds'] > self.discretionary_hold_seconds(v.id) + 1e-6:
                raise ValueError("hold_exceeds_dwell_tolerance_or_departure_target_without_obstruction")
            v.status = "HOLDING"
            return self._new_operation(c, "HOLDING", deadline=self.now + self.duration(p["seconds"]))
        if action in ("adjust_departure", "reroute", "skip_stop"):
            if pos not in self.world.platform_nodes:
                raise ValueError("route_decision_requires_platform")
            target = p["target_node"]
            if target not in self.world.platform_nodes or target == pos:
                raise ValueError("invalid_service_destination")
            path = self.path(pos, target, p.get("path_node_ids"))
            if not path or not self._route_open(v.id, path):
                raise ValueError("service_route_unavailable")
            self._check_turnback_receiver(v.id, path)
            if action == "adjust_departure":
                if f"{pos}-{target}" not in self.world._segment_id_map:
                    raise ValueError("departure_requires_direct_segment")
                if "headway_seconds" in p:
                    self._set_headway(pos, p["headway_seconds"])
            if action == "skip_stop":
                if float(v.agent.passengers_by_destination.get(target, torch.zeros(1))[0]) > 1e-6:
                    raise ValueError("onboard_passengers_require_skipped_stop")
                segment = next((s for s in self.world.base_segments if s.start_platform.id == target
                                and s.end_platform.id in self.world.platform_nodes), None)
                if segment is None:
                    raise ValueError("no_onward_service_after_skipped_stop")
                path.extend(n.id for n in self.world.segment2nodes[segment][1:])
                if not self._route_open(v.id, path):
                    raise ValueError("onward_service_route_unavailable")
            deadline = p.get("departure_time", self.now)
            if deadline < self.now:
                raise ValueError("departure_time_is_in_the_past")
            return self._new_operation(c, "RUNNING", path=path, target=target, deadline=deadline)
        raise ValueError("unsupported_action")

    def applicability_copy(self):
        """Isolate command preconditions without cloning the physical graph.

        Only for _start_now checks, never for stepping. That method changes
        manager-local records, bookings and operations; its sole physical-state
        setter, _set_headway, is replaced by equivalent value validation.
        """
        probe = copy.copy(self)
        probe.records = {key: copy.copy(value) for key, value in self.records.items()}
        for field in ("services", "operations", "inventory", "bookings", "admission_rates",
                      "admission_used", "profile_selections"):
            setattr(probe, field, copy.deepcopy(getattr(self, field)))
        # _start_now only reads historical entries and appends new log events.
        # Avoid copying every past plan for every candidate of every vehicle.
        probe.command_results = {key:dict(value) for key,value in self.command_results.items()}
        probe.plans = dict(self.plans)
        probe.log = list(self.log)
        def validate_headway(platform, seconds):
            if platform not in self.world.platform_nodes or not math.isfinite(seconds) or seconds < 30:
                raise ValueError("Target headway must be finite and at least 30 seconds")
        probe._set_headway = validate_headway
        return probe

    def submit_plan(self, data):
        try:
            plan = DispatchPlan.from_dict(data, self.tick)
            if plan.plan_id in self.plans:
                raise ValueError("duplicate_plan_id")
            expected_version = self.snapshot(published=plan.snapshot_basis == "published")["version"]
            if plan.snapshot_version != expected_version:
                raise ValueError("stale_snapshot_version")
            if plan.created_at > self.now or any(c.effective_at < self.now for c in plan.commands):
                raise ValueError("plan_time_not_current_or_future")
            if any(c.command_id in self.command_results for c in plan.commands):
                raise ValueError("duplicate_command_id")
            if any(c.participants() - set(self.records) for c in plan.commands):
                raise ValueError("unknown_vehicle")
            replace = []
            for old in self.plans.values():
                for c in old.commands:
                    if self.command_results[c.command_id]["status"] != "pending":
                        continue
                    overlaps = any(c.participants() & n.participants() or c.resources() & n.resources() for n in plan.commands)
                    if overlaps and plan.replace_future:
                        if c.locked:
                            raise ValueError("cannot_replace_locked_command")
                        replace.append(c.command_id)
                    elif any(c.effective_at == n.effective_at and (c.participants() & n.participants() or c.resources() & n.resources()) for n in plan.commands):
                        raise ValueError("pending_command_conflict")
            for cid in replace:
                self.command_results[cid].update(status="cancelled", reason="replaced_before_activation")
            self.plans[plan.plan_id] = plan
            for c in plan.commands:
                self.command_results[c.command_id] = dict(status="pending", action=c.action, plan_id=plan.plan_id)
            self.event("plan_accepted", plan_id=plan.plan_id)
            return dict(accepted=True, reason=None, plan_id=plan.plan_id)
        except (ValueError, KeyError, TypeError) as exc:
            return dict(accepted=False, reason=str(exc), plan_id=None)

    def _activate(self):
        due = [c for p in self.plans.values() for c in p.commands
               if c.effective_at <= self.now and self.command_results[c.command_id]["status"] == "pending"]
        # Validate each simultaneous batch on isolated precondition state. No
        # partial service cancellation or inventory booking escapes rejection.
        for when in sorted({c.effective_at for c in due}):
            batch = [c for c in due if c.effective_at == when]
            try:
                trial = self.applicability_copy()
                for c in batch:
                    trial._start_now(c)
            except (ValueError, KeyError, TypeError) as exc:
                for c in batch:
                    self.command_results[c.command_id].update(status="rejected", reason=str(exc), finished_at=self.now)
                    self.event("command_rejected", command_id=c.command_id, action=c.action,
                               vehicle_id=c.vehicle_id, parameters=copy.deepcopy(c.parameters), reason=str(exc))
                continue
            for c in batch:
                evidence = self.reserve_need() if c.action in {"reserve_insert", "reenter"} else None
                op = self._start_now(c)
                self.command_results[c.command_id].update(status="running" if op else "completed",
                    operation_id=op.id if op else None, started_at=self.now, finished_at=None if op else self.now)
                self.event("command_started", command_id=c.command_id, action=c.action,
                           vehicle_id=c.vehicle_id, parameters=copy.deepcopy(c.parameters),
                           operation_id=op.id if op else None, reserve_need=evidence)

    def _faults(self):
        for e in self.world.disruption_manager.disruptions.values():
            if e["type"] != "vehicle_breakdown" or e["start_seconds"] > self.now or e["event_id"] in self.seen_failures:
                continue
            self.seen_failures.add(e["event_id"])
            v = self.records[e["target"]["id"]]
            self.hidden_failure_prior[e["event_id"]] = dict(status=v.status, failed=v.failed, service_id=v.service_id,
                needs_inspection=v.needs_inspection)
            if e["parameters"]["mode"] == "unavailable_in_depot" and v.depot is None:
                raise ValueError("A depot failure must occur to an actually stored vehicle")
            v.failed = e["parameters"]["mode"] != "can_limp_to_depot"
            v.needs_inspection = True
            self._end_service(v)
            v.status = "FAILED" if v.failed else "LIMPING"
            existing = self.operation_for(v.id)
            if existing and not self.towed(v.id):
                self._finish(existing, "failed", "vehicle_breakdown")
            self.event("vehicle_failed", vehicle_id=v.id, source_event=e["event_id"])

    def _clear(self, key, destinations=None):
        v = self.records[key]
        if float(v.agent.passenger_count[0]) <= 1e-6:
            return
        p = self.position(key)
        if not self._safe_platform(p):
            raise ValueError("clearance_platform_no_longer_safe")
        flow = self.world.platform_nodes[p].passenger_flow
        flow.receive_vehicle(v.agent, 0, destinations=destinations)

    def _clear_turnback(self, op):
        vehicle = self.records[op.vehicle_id]
        origin = self.position(vehicle.id)
        segment = self.world._step_segment.get((origin, op.path[0])) if op.path else None
        target = segment.end_platform.id if segment else None
        # Clearance includes a same-station platform transfer for passengers
        # served on the return leg. Others keep waiting at the original platform.
        if (target in self.world.platform_nodes and self._safe_platform(target)
                and getattr(self.world.nodes[origin], "station_name", None)
                == getattr(self.world.nodes[target], "station_name", None)):
            onward = set(op.path) & set(self.world.platform_nodes)
            self.world.platform_nodes[target].passenger_flow.receive_vehicle(
                vehicle.agent, 0, destinations=onward - {origin})
        self._clear(vehicle.id)

    def _park(self, key, depot):
        v = self.records[key]
        if self.position(key) != self.depots[depot]["node_id"] or float(v.agent.passenger_count[0]) > 1e-6:
            raise ValueError("Cannot park a vehicle before physical arrival and clearance")
        self._end_service(v)
        v.depot, v.status = depot, "DEPOT_READY"
        self.inventory[depot].add(key)
        self.bookings[depot].discard(key)

    def _finish(self, op, status="completed", reason=None):
        self.command_results[op.command_id].update(status=status, finished_at=self.now, reason=reason)
        self.operations.pop(op.id, None)
        for keys in self.bookings.values():
            keys.difference_update(op.participants())
        if op.service_id:
            self.services[op.service_id].pop("reserved_vehicle_id", None)
        self.event("operation_" + status, operation_id=op.id, action=op.action, vehicle_id=op.vehicle_id, reason=reason)

    def _advance_timer(self, op):
        v = self.records[op.vehicle_id]
        if op.phase == "SERVICE_CLEARING" and self.now >= op.deadline:
            try:
                self._clear_turnback(op)
            except ValueError as exc:
                op.wait_reason = str(exc)
                return
            op.phase, op.deadline = "SERVICE_TURNING", self.now + self.duration(self.turnback_seconds)
        if op.phase == "SERVICE_TURNING" and self.now >= op.deadline:
            op.phase = "RUNNING"
        if op.phase in ("CLEARING", "CLEARING_RESCUER") and self.now >= op.deadline:
            try:
                self._clear_turnback(op) if op.action == "short_turn" else self._clear(v.id, op.clear_destinations)
            except ValueError as exc:
                op.wait_reason = str(exc)
                return
            if op.action == "rescue_tow":
                self._end_service(v)
                op.phase = "APPROACH"
            elif op.action == "withdraw":
                self._end_service(v)
                v.status, op.phase = "WITHDRAWING", "RUNNING"
            elif op.action == "short_turn":
                self._end_service(v)
                op.phase, op.deadline = "TURNING", self.now + self.duration(self.turnback_seconds)
            else:
                if op.action in ("cancel_service", "reassign_service"):
                    self._end_service(v)
                if op.action == "reassign_service":
                    self._start_service(v, op.service_id)
                    op.phase = "RUNNING"
                    return
                v.status = "FAILED" if v.failed else ("IN_SERVICE" if v.service_id else "OUT_OF_SERVICE")
                self._finish(op)
                return
        if op.phase == "PREPARING" and self.now >= op.deadline:
            if not self.depot_allowed(op.depot, "exit"):
                op.wait_reason = "depot_exit_closed"
                return
            v.status, op.phase = "INSERTING", "RUNNING"
        if op.phase == "TURNING" and self.now >= op.deadline:
            op.phase = "RUNNING"
        if op.phase == "HOLDING" and self.now >= op.deadline:
            v.status = "IN_SERVICE"
            self._finish(op)
            return
        if op.phase == "INSPECTING" and self.now >= op.deadline:
            if self.world.event_view().active("vehicle_breakdown", "vehicle", v.id):
                op.wait_reason = "failure_not_cleared"
                return
            v.failed = v.needs_inspection = False
            v.status = "DEPOT_READY"
            self.event("inspection_completed", vehicle_id=v.id)
            self._finish(op)
            return
        if op.phase == "APPROACH" and not op.path:
            rear, front = self.position(v.id), self.position(op.partner)
            if front not in self.world._has_edge.get(rear, ()):
                op.wait_reason = "coupling_requires_adjacent_distinct_footprints"
                return
            op.phase, op.deadline = "COUPLING", self.now + self.duration(self.coupling_seconds)
        if op.phase == "COUPLING" and self.now >= op.deadline:
            op.phase, op.path = "TOW", deque(op.continuation)
            failed = self.records[op.partner]
            if float(failed.agent.passenger_count[0]) > 1e-6 and self._safe_platform(self.position(failed.id)):
                op.phase, op.deadline = "CLEARING_FAILED", self.now + self._clear_duration(failed)
        if op.phase == "CLEARING_FAILED" and self.now >= op.deadline:
            try:
                self._clear(op.partner)
            except ValueError as exc:
                op.wait_reason = str(exc)
                return
            op.phase = "TOW"

    def pre_step(self):
        self._faults()
        self._activate()
        for op in list(self.operations.values()):
            self._advance_timer(op)
        for op in self.operations.values():
            v = self.records[op.vehicle_id]
            pos = self.position(v.id)
            if (op.action in {"reassign_service", "reroute", "adjust_departure", "skip_stop", "withdraw"}
                    and op.phase == "RUNNING" and op.path
                    and pos in self.world.platform_nodes and op.turnback_entry != pos):
                segment = self.world._step_segment.get((pos, op.path[0]))
                if segment and any(segment.id in spec["segments"]
                                   for spec in self.world.event_resources.get("turnbacks", {}).values()):
                    op.turnback_entry = pos
                    if op.action == "withdraw":
                        # Withdrawal already cleared passengers before running.
                        op.phase, op.deadline = "SERVICE_TURNING", self.now + self.duration(self.turnback_seconds)
                    else:
                        op.phase, op.deadline = "SERVICE_CLEARING", self.now + self._clear_duration(v)
                    self.event("service_turnback_started", vehicle_id=v.id, operation_id=op.id,
                               service_id=v.service_id, origin_node=pos, target_node=segment.end_platform.id,
                               action="short_turn", source="withdrawal_itinerary" if op.action == "withdraw"
                               else "reassigned_service_itinerary")
            for key in op.participants():
                agent = self.records[key].agent
                agent.potential_next_pos[0] = self.position(key)
            if op.phase in ("RUNNING", "APPROACH") and op.path:
                v.agent.potential_next_pos[0] = op.path[0]
            elif op.phase == "TOW" and op.path:
                self.records[op.partner].agent.potential_next_pos[0] = op.path[0]
                v.agent.potential_next_pos[0] = self.position(op.partner)
            elif op.phase == "CLEAR_REAR":
                v.agent.potential_next_pos[0] = self.depots[op.depot]["node_id"]
        for v in self.records.values():
            if self.operation_for(v.id) is None and (v.failed or not self.service_enabled(v.agent) or v.depot):
                v.agent.potential_next_pos[0] = self.position(v.id)
            pos = self.position(v.id)
            if self.service_enabled(v.agent) and pos in self.world.platform_nodes and pos not in self.first_departure_due:
                remaining = max(0., self.world.minimum_dwell_steps(v.agent,pos)-float(v.agent.state.dwell_time[0]))
                self.first_departure_due[pos] = self.now + remaining*self.tick
        self._refresh_turnback_receivers()
        self._record_phases()

    def _record_phases(self):
        current = {op.id: (op.phase, op.vehicle_id, op.action) for op in self.operations.values()}
        for oid, (phase, vehicle, action) in self._logged_phases.items():
            if current.get(oid) != (phase, vehicle, action):
                self.event("operation_phase_ended", operation_id=oid, phase=phase, vehicle_id=vehicle, action=action)
        for oid, (phase, vehicle, action) in current.items():
            if self._logged_phases.get(oid) != (phase, vehicle, action):
                self.event("operation_phase_started", operation_id=oid, phase=phase, vehicle_id=vehicle, action=action)
        self._logged_phases = current

    def post_step(self):
        for v in self.records.values():
            if v.depot and self.position(v.id) != self.depots[v.depot]["node_id"]:
                self.inventory[v.depot].discard(v.id)
                v.depot = None
            previous = int(v.agent.state.previous_position[0])
            if int(v.agent.action_result[0]) == 2 and previous in self.world.platform_nodes:
                self.physical_departures[previous] = self.now - self.tick if self.world.second_precision else self.now
            if int(v.agent.action_result[0]) == 2:
                self.pending_transits.pop(v.id, None)
        for op in list(self.operations.values()):
            v = self.records[op.vehicle_id]
            pos = self.position(v.id)
            if op.turnback_entry is not None and pos != op.turnback_entry:
                op.turnback_entry = None
            if op.phase in ("RUNNING", "APPROACH") and op.path and pos == op.path[0]:
                op.path.popleft()
            if op.phase == "TOW" and op.path and self.position(op.partner) == op.path[0]:
                op.path.popleft()
                failed = self.records[op.partner]
                if op.path and float(failed.agent.passenger_count[0]) > 1e-6 and self._safe_platform(self.position(failed.id)):
                    op.phase, op.deadline = "CLEARING_FAILED", self.now + self._clear_duration(failed)
                elif not op.path:
                    failed = self.records[op.partner]
                    # Clear the failed vehicle only at a declared safe platform,
                    # never by dropping onboard demand inside a depot.
                    if float(failed.agent.passenger_count[0]) > 1e-6:
                        raise AssertionError("Rescue reached depot with uncleared passengers")
                    self._park(failed.id, op.depot)
                    op.phase = "CLEAR_REAR"
            elif op.phase == "CLEAR_REAR" and pos == self.depots[op.depot]["node_id"]:
                self._park(v.id, op.depot); self._finish(op)
            elif op.phase == "RUNNING" and not op.path:
                if op.action == "withdraw":
                    self._park(v.id, op.depot)
                elif op.action in ("reserve_insert", "reenter", "short_turn"):
                    self._start_service(v)
                elif op.action == "reassign_service":
                    task = self.services[op.service_id]
                    pattern = task.get("pattern_id")
                    if task.get("repeat") and pattern and self.section_need(pattern, pos):
                        route = self.service_route(pattern, pos)
                        if route and self._route_open(v.id, route[1]):
                            op.path = deque(route[1])
                            task["completed_cycles"] = task.get("completed_cycles", 0) + 1
                            self.event("section_cycle_completed", vehicle_id=v.id, service_id=v.service_id,
                                       pattern_id=pattern, completed_cycles=task["completed_cycles"])
                            continue
                    self._end_service(v, completed=True)
                    # A completed passenger task is not a withdrawal command.
                    # Keep the physical train circulating until explicitly reassigned or withdrawn.
                    self._start_service(v)
                self._finish(op)
        for key, selection in list(self.profile_selections.items()):
            segment = self.world._segment_id_map[selection[0]]
            if self.position(key) == segment.end_platform.id:
                self.profile_selections.pop(key, None)
        for op in list(self.operations.values()):
            self._advance_timer(op)
        self._refresh_turnback_receivers()
        self._record_phases()
        self.check_invariants()

    def move_allowed(self, agent, origin, dest):
        op = self.operation_for(agent.name)
        if not self._occupied_resource_allows_entry(agent, origin, dest):
            return False
        if not self.receiver_allowed(agent.name, dest):
            return False
        if self.blocks_reserved_path(agent, dest):
            return False
        segment = self.world._step_segment.get((origin, dest))
        facility = self.turnback_facilities.get(segment.id) if segment else None
        if (facility and origin == segment.start_platform.id
                and not self._turnback_receiver_clear(agent,segment)):
            return False
        if (facility and origin == segment.start_platform.id and facility.get("facility_type") != "terminal" and not (op and (
                op.action == "rescue_tow" or op.action == "short_turn" and op.phase == "RUNNING"
                or op.phase == "RUNNING" and op.turnback_entry == origin))):
            return False
        for depot, spec in self.depots.items():
            exiting = origin == spec["node_id"] or segment and segment.start_platform.id == spec["node_id"]
            entering = dest == spec["node_id"] or segment and segment.end_platform.id == spec["node_id"]
            if exiting and (not op or op.action not in ("reserve_insert", "reenter", "rescue_tow")
                    or not self.depot_allowed(depot, "exit")):
                return False
            if entering and (not op or op.depot != depot or op.action not in ("withdraw", "rescue_tow")
                    or not self.depot_allowed(depot, "entry")):
                return False
        if origin in self.world.platform_nodes and not (op and op.action == "rescue_tow"):
            segment = self.world._step_segment.get((origin, dest))
            minimum = self.world.event_view().safety_headway(segment.id if segment else None, 30)
            if self.now < self.physical_departures.get(origin, -math.inf) + minimum:
                return False
        if op and op.action == "adjust_departure" and self.now < op.deadline:
            return False
        if self.service_enabled(agent) and origin in self.world.platform_nodes:
            target = float(agent.state.target_departure_time[0])
            if origin in agent.headway_tracker.targets and self.now < target:
                return False
        # Charge integer travel extensions before arrival. A blocked proposal
        # keeps its completed travel clock and must still pass physical safety.
        edge = (origin, dest)
        if self.pending_transits.get(agent.name) != edge:
            extra = self.world.edge_delay_steps(agent, origin, dest)
            self.pending_transits[agent.name] = edge
            if extra:
                agent.state.speed_limit_delay[0] = extra - 1
                return False
        return True

    def _occupied_resource_allows_entry(self, agent, origin, dest):
        # Reject an entrant before reservation arbitration, so it cannot
        # prevent the current occupant from clearing the shared route.
        operation = self.operation_for(agent.name)
        owner = operation.id if operation and operation.action == 'rescue_tow' else agent.name
        for rid, interior in self._shared_interiors.items():
            if origin in interior or dest not in interior:
                continue
            owners = set()
            for vehicle in self.records.values():
                if self.position(vehicle.id) not in interior or not self.blocks_track(vehicle.agent):
                    continue
                active = self.operation_for(vehicle.id)
                owners.add(active.id if active and active.action == 'rescue_tow' else vehicle.id)
            turnback = self.world.event_resources.get('turnbacks', {}).get(rid)
            spec = turnback or self.world.event_resources['shared_routes'][rid]
            capacity = (self.world.event_view().turnback_capacity(rid, spec.get('capacity', 1))
                        if turnback is not None else spec.get('capacity', 1))
            if owner not in owners and len(owners) >= capacity:
                return False
        return True

    def admission_budget(self, platform, amount):
        rate = self.admission_rates.get(platform)
        if rate is None:
            return amount
        stamp, used = self.admission_used.get(platform, (self.now, 0.))
        if stamp != self.now:
            used = 0.
        allowed = min(amount, max(0., rate * self.tick - used))
        self.admission_used[platform] = (self.now, used + allowed)
        return allowed

    def blocks_reserved_path(self, agent, dest):
        for op in self.operations.values():
            if op.action == "rescue_tow" and agent.name not in op.participants() and dest in (set(op.path) | set(op.continuation)):
                return True
        return False

    def atomic_tow(self, allowed, positions, desired, masks):
        for op in self.operations.values():
            if op.action != "rescue_tow" or op.phase != "TOW":
                continue
            indices = [self.records[k].agent.agent_index for k in (op.vehicle_id, op.partner)]
            if any(not allowed[i] or any(bool(mask[i]) for mask in masks) or desired[i] == positions[i] for i in indices):
                for i in indices:
                    allowed[i] = False

    def check_invariants(self):
        if set(self.records) != {a.name for a in self.world.agents}:
            raise AssertionError("Physical vehicle identity changed")
        occupied, assigned, participants = {}, set(), set()
        for op in self.operations.values():
            if participants & op.participants():
                raise AssertionError("Vehicle participates in two operations")
            participants |= op.participants()
        for key, v in self.records.items():
            if v.service_id:
                if v.service_id in assigned or self.services[v.service_id]["vehicle_id"] != key or self.services[v.service_id]["ended_at"] is not None:
                    raise AssertionError("Invalid physical vehicle/service assignment")
                assigned.add(v.service_id)
            if v.depot is None:
                pos = self.position(key)
                if pos in occupied:
                    raise AssertionError("Two physical footprints overlap")
                occupied[pos] = key
            elif key not in self.inventory[v.depot] or self.position(key) != self.depots[v.depot]["node_id"]:
                raise AssertionError("Depot registration disagrees with physical position")
            load = float(v.agent.passenger_count[0])
            destination_loads = [float(q[0]) for q in v.agent.passengers_by_destination.values()]
            if not math.isfinite(load) or load < 0 or any(not math.isfinite(q) or q < 0 for q in destination_loads):
                raise AssertionError(f"Invalid onboard passenger load: vehicle={key}, time={self.now}, total={load}")
            destination_total = math.fsum(destination_loads)
            if abs(load - destination_total) > 1e-3:
                raise AssertionError("Onboard passenger destination balance violated: "
                    f"vehicle={key}, time={self.now}, total={load:.12g}, "
                    f"destinations={destination_total:.12g}, difference={load-destination_total:.12g}")
        for depot, keys in self.inventory.items():
            if len(keys | self.bookings[depot]) > self.depots[depot]["capacity"]:
                raise AssertionError("Depot inventory/reservations exceed capacity")

    def snapshot(self, *, published=True, include_version=True, include_command_results=True):
        records = {}
        view = self.world.event_view(published=published)
        visible = {e["event_id"] for e in self.world.get_published_events()}
        for key, v in self.records.items():
            data = {n: getattr(v, n) for n in ("id", "role", "status", "service_id", "depot", "failed", "needs_inspection", "has_served")}
            for eid, prior in self.hidden_failure_prior.items():
                e = self.world.disruption_manager.disruptions[eid]
                if published and eid not in visible and e["target"]["id"] == key:
                    data.update(prior)
            data.update(position=self.position(key), passenger_count=float(v.agent.passenger_count[0]),
                        passengers_by_destination={d: float(q[0]) for d, q in v.agent.passengers_by_destination.items()},
                        passenger_waiting_mass={d: float(q[0]) for d, q in v.agent.passenger_waiting_mass.items()},
                        travel_delay_steps=float(v.agent.state.speed_limit_delay[0]),
                        pending_edge=self.pending_transits.get(key), profile=self.profile_selections.get(key))
            node = self.position(key)
            data["dwell_elapsed_seconds"] = float(v.agent.state.dwell_time[0])*self.tick
            data["dwell_remaining_seconds"] = (max(0., self.world.minimum_dwell_steps(v.agent, node, view=view)*self.tick
                - data["dwell_elapsed_seconds"]) if node in self.world.platform_nodes else 0.)
            records[key] = data
        services = copy.deepcopy(self.services)
        for eid, prior in self.hidden_failure_prior.items():
            if published and eid not in visible and prior["service_id"]:
                services[prior["service_id"]]["ended_at"] = None
        result = dict(time=self.now, vehicles=records, services=services,
                    first_departure_due=dict(self.first_departure_due),
                    operations={k: dict(asdict(o), path=list(o.path), remaining_seconds=max(0., o.deadline-self.now)) for k, o in self.operations.items()},
                    inventory={d: sorted(keys) for d, keys in self.inventory.items()},
                    depot_bookings={d: sorted(keys) for d, keys in self.bookings.items()}, admission_rates=dict(self.admission_rates))
        if include_command_results:
            result["command_results"] = copy.deepcopy(self.command_results)
        if include_version:
            result["version"] = self.version
        if published:
            result.pop("version", None)
            result["events"] = self.world.get_published_events()
            if include_version:
                result["version"] = int(hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:15], 16)
        return result
