"""Shared nonnegative scheduling costs for structured dispatch and its MILP.

Safety is never priced. These configurable weights are defaults for model
validation, not fitted parameters or replacements for historical experiment data.
"""
import math
from collections import deque

from dispatch_plan import PARAMETERS


class PassengerTravelBounds:
    """Ordinary passenger-circulation travel, shared by planning copies."""
    def __init__(self, world, depot_nodes):
        self.tick = float(world.tick_seconds or 1)
        self.depots = frozenset(depot_nodes)
        excluded_edges = set()
        for spec in world.event_resources.get("turnbacks", {}).values():
            if spec.get("facility_type") == "terminal":
                continue
            for segment_id in spec["segments"]:
                segment = world._segment_id_map[segment_id]
                nodes = [n.id for n in world.segment2nodes[segment]]
                excluded_edges.update(zip(nodes, nodes[1:]))
        self.reverse = {node: [] for node in world.nodes if node not in self.depots}
        for origin, successors in world._next_nodes_id_map.items():
            if origin in self.reverse:
                for target in successors:
                    if target in self.reverse and (origin, target) not in excluded_edges:
                        self.reverse[target].append((origin, self.tick))
        self.distances = {}

    def __deepcopy__(self, memo):
        # No event, passenger or vehicle state is cached here.
        return self

    def to(self, destination):
        if destination not in self.distances:
            distances = {destination: 0.}
            queue = deque([destination])
            while queue:
                current = queue.popleft()
                for previous, seconds in self.reverse.get(current, ()):
                    value = distances[current] + seconds
                    if value < distances.get(previous, math.inf):
                        distances[previous] = value
                        (queue.appendleft if seconds == 0 else queue.append)(previous)
            self.distances[destination] = distances
        return self.distances[destination]


class DispatchObjective:
    SCHEMA = "passenger_journey_time_v4_service_continuation"
    DEFAULTS = dict(waiting=1., onboard=1., headway=1., operation=.1,
                    terminal_passenger_seconds=30., terminal_operation=1., terminal_vehicle=1.,
                    terminal_service=1., discount=1.)

    def __init__(self, config=None):
        config = dict(config or {})
        unknown = set(config) - set(self.DEFAULTS) - {"operation_costs"}
        if unknown:
            raise ValueError(f"Unknown objective weights: {sorted(unknown)}")
        self.weights = dict(self.DEFAULTS, **{k: v for k, v in config.items() if k in self.DEFAULTS})
        self.operation_costs = {a: 1. for a in PARAMETERS}
        for action, value in config.get("operation_costs", {}).items():
            if action not in PARAMETERS:
                raise ValueError("Unknown operation cost")
            self.operation_costs[action] = value
        for value in list(self.weights.values()) + list(self.operation_costs.values()):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Objective weights must be nonnegative and finite")
        if not 0 < self.weights["discount"] <= 1:
            raise ValueError("Objective discount must be in (0,1]")

    @staticmethod
    def passenger_state(world):
        inside = outside = stranded = onboard = 0.
        for node in world.platform_nodes.values():
            flow = node.passenger_flow
            if flow.waiting_passengers is not None:
                inside += float(flow.waiting_passengers[0])
                outside += flow.outside_count(0)
        for agent in world.agents:
            load = float(agent.passenger_count[0])
            onboard += load
            if not world.dispatch.service_enabled(agent):
                stranded += load
        return dict(inside=inside, outside=outside, onboard=onboard, stranded=stranded)

    def prepare(self, world, depots):
        nodes = frozenset(spec["node_id"] for spec in depots.values())
        if getattr(world, "_passenger_travel_bounds", None) is None or world._passenger_travel_bounds.depots != nodes:
            world._passenger_travel_bounds = PassengerTravelBounds(world, nodes)

    def remaining_travel(self, world):
        self.prepare(world, world.dispatch.depots)
        bounds = world._passenger_travel_bounds
        total = unreachable = 0.
        def add(origin, destination, count, prefix=None, tail=None, prefix_seconds=0.):
            nonlocal total, unreachable
            if count <= 0:
                return
            seconds = (prefix.get(destination) if prefix is not None else None)
            if seconds is None:
                suffix = bounds.to(destination).get(origin if tail is None else tail)
                seconds = None if suffix is None else prefix_seconds + suffix
            if seconds is None:
                unreachable += count
            else:
                total += seconds * count
        for platform, node in world.platform_nodes.items():
            for queues in (node.passenger_flow.waiting_by_destination, node.passenger_flow.outside_by_destination):
                for destination, count in queues.items():
                    add(platform, destination, float(count[0]))
        for agent in world.agents:
            op = world.dispatch.operation_for(agent.name)
            prefix, tail, duration = {}, None, 0.
            if op and op.path:
                # Passengers follow the committed itinerary, not a free
                # transfer or an unselected intermediate crossover.
                for node in op.path:
                    duration += bounds.tick
                    if not (op.action == "skip_stop" and node == op.target):
                        prefix.setdefault(node, duration)
                    tail = node
            for destination, count in agent.passengers_by_destination.items():
                add(int(agent.state.position[0]), destination, float(count[0]), prefix, tail, duration)
        return total, unreachable

    @staticmethod
    def outstanding_headway(world):
        tracker = world.agents[0].headway_tracker
        now = float(world.sim_time[0])
        total = 0.
        for platform, history in tracker._buffers.items():
            previous = float(history.last_departure[0])
            if math.isfinite(previous):
                # Operating controls may revise a target; they cannot erase
                # deviation from the fixed reference used by departed trains.
                total += max(0., now - previous - tracker.headway_seconds)
            elif platform in world.dispatch.first_departure_due:
                total += max(0., now - world.dispatch.first_departure_due[platform])
        return total

    def interval(self, waiting_person_seconds, headway_deviation_seconds, started_actions, *,
                 onboard_person_seconds=0.):
        raw = dict(waiting_person_seconds=float(waiting_person_seconds),
                   onboard_person_seconds=float(onboard_person_seconds),
                   passenger_person_seconds=float(waiting_person_seconds + onboard_person_seconds),
                   headway_deviation_seconds=float(headway_deviation_seconds),
                   operation_units=sum(self.operation_costs[a] for a in started_actions))
        raw["total"] = (self.weights["waiting"] * raw["waiting_person_seconds"]
                        + self.weights["onboard"] * raw["onboard_person_seconds"]
                        + self.weights["headway"] * raw["headway_deviation_seconds"]
                        + self.weights["operation"] * raw["operation_units"])
        return raw

    def terminal(self, world):
        counts = self.passenger_state(world)
        unfinished = counts["inside"] + counts["outside"] + counts["onboard"]
        # A rescue operation counts once; its two vehicle footprints remain
        # distinct in the physical constraints, not duplicated in this cost.
        work = len(world.dispatch.operations)
        vehicles = sum(v.failed or v.needs_inspection for v in world.dispatch.records.values())
        services = sum(bool(t.get("stops")) and not t.get("completed", False)
                       and not t.get("pattern", False)
                       and (t.get("required", True) or t.get("started_at") is not None)
                       for t in world.dispatch.services.values())
        travel, unreachable = self.remaining_travel(world)
        headway = self.outstanding_headway(world)
        return dict(unfinished_passengers=unfinished, unfinished_operations=work,
                    unrecovered_vehicles=vehicles, unfinished_services=services,
                    remaining_travel_person_seconds=travel, unreachable_passengers=unreachable,
                    outstanding_headway_seconds=headway,
                    total=self.weights["waiting"] * self.weights["terminal_passenger_seconds"] * unfinished
                    + min(self.weights["waiting"], self.weights["onboard"]) * travel
                    + self.weights["headway"] * headway
                    + self.weights["terminal_operation"] * work + self.weights["terminal_vehicle"] * vehicles
                    + self.weights["terminal_service"] * services)

    def manifest(self):
        return dict(schema=self.SCHEMA, weights=dict(self.weights), operation_costs=dict(self.operation_costs),
                    direction="minimize_nonnegative_cost", safety="hard_constraints_only",
                    passenger_time="waiting = inside + outside; onboard = all train loads, including stranded; each passenger once per physical second",
                    terminal_travel="service-continuation estimate: queued passengers use ordinary directed circulation; onboard passengers follow committed paths then ordinary circulation; no free platform transfer or unselected crossover, future arrivals or private clearance times; not an optimal-cost lower bound",
                    terminal_regularity="outstanding positive headway deviation since the last actual departure; same fixed reference as realized departure gaps",
                    priority="feasibility first; weighted service, regularity and operation tradeoffs",
                    marl_reward="negative interval cost divided equally over the existing agents")
