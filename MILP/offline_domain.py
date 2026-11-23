"""Exhaustive stage decisions for the full-horizon discrete offline benchmark.

Unlike operation_model's initial command catalog, this domain is regenerated
at EVERY reachable stage. The domain is finite and explicitly documented:
physical ticks, simple directed command paths, all configured running modes,
and the declared admission/headway grids. There is no learned candidate filter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
import time

from dispatch_plan import DispatchCommand, PARAMETERS
from MILP.operation_model import nominal_action


class OfflineLimit(Exception):
    pass


def check_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise OfflineLimit("time_limit")


def fair_product(iterables, *, deadline=math.inf):
    """Complete lazy product ordered by sum of option indices, not by suffix.

    Each finite-index tuple is visited after finitely many predecessors. Only
    requested prefixes of the component domains are materialized.
    """
    streams = [iter(values) for values in iterables]
    cache, exhausted = [[] for _ in streams], [False for _ in streams]

    def get(axis, index):
        while len(cache[axis]) <= index and not exhausted[axis]:
            check_deadline(deadline)
            try:
                cache[axis].append(next(streams[axis]))
            except StopIteration:
                exhausted[axis] = True
        return index < len(cache[axis])

    def visit(axis, remaining, values):
        check_deadline(deadline)
        if axis == len(streams):
            if remaining == 0:
                yield tuple(values)
            return
        indices = (remaining,) if axis == len(streams)-1 else range(remaining, -1, -1)
        for index in indices:
            if get(axis, index):
                yield from visit(axis+1, remaining-index, [*values, cache[axis][index]])

    if not streams:
        yield ()
        return
    for rank in itertools.count():
        if all(exhausted) and rank > sum(len(values)-1 for values in cache):
            break
        yield from visit(0, rank, [])
        if any(done and not values for done, values in zip(exhausted, cache)):
            break


def command_orders(commands, *, deadline=math.inf):
    """Quotient only adjacent commuting platform controls.

    Platform-only commands write distinct admission/headway keys. They commute
    with each other, but not necessarily with a vehicle command (which can
    change service or departure state). Preserve every order across that fence.
    """
    if all(c["vehicle_id"] is None for c in commands):
        check_deadline(deadline)
        yield list(commands)
        return

    def visit(remaining, prefix, previous_platform):
        check_deadline(deadline)
        if not remaining:
            yield list(prefix)
        for index in remaining:
            platform = commands[index]["vehicle_id"] is None
            if platform and previous_platform is not None and index < previous_platform:
                continue
            yield from visit(tuple(i for i in remaining if i != index),
                             [*prefix, commands[index]], index if platform else None)
    yield from visit(tuple(range(len(commands))), [], None)


@dataclass
class StageControl:
    actions: dict = field(default_factory=dict)
    commands: list = field(default_factory=list)
    pauses: list = field(default_factory=list)

    def to_dict(self):
        return dict(actions=dict(self.actions), commands=list(self.commands), pauses=list(self.pauses))

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


class OfflineDomain:
    def __init__(self, stages, tick_seconds=30., *, admission_rates=(0., 1.), headways=None):
        if type(stages) is not int or stages < 1 or not math.isfinite(tick_seconds) or tick_seconds <= 0:
            raise ValueError("Positive integer stages and finite tick required")
        self.stages, self.tick = stages, float(tick_seconds)
        self.admission_rates = tuple(sorted(set(float(v) for v in admission_rates)))
        if not self.admission_rates or any(not math.isfinite(v) or v < 0 for v in self.admission_rates):
            raise ValueError("Admission grid must be nonempty, finite and nonnegative")
        self.headways = tuple(sorted(set(float(v) for v in headways))) if headways is not None else None
        if self.headways is not None and (not self.headways or any(not math.isfinite(v) or v < 30 for v in self.headways)):
            raise ValueError("Headway grid values must be at least 30 seconds")

    def manifest(self):
        return dict(name="full_stage_discrete_dispatch_v5_compiled_waits", stages=self.stages, tick_seconds=self.tick,
            resource_waits="deterministic per-second resource arbitration compiled and independently replayed",
            action_families=list(PARAMETERS), admission_rates=list(self.admission_rates),
            headways=list(self.headways) if self.headways is not None else "all positive ticks through episode duration, plus initial targets",
            holding_seconds="every integer second through episode end", departure_times="every remaining integer second",
            physical_tick_seconds=1, decision_seconds=self.tick,
            depot_exits="every declared exit for reserve deployment and return to service",
            paths="all simple directed paths permitted by the original command contract",
            primitive_actions="physical successor labels; unobstructed station waiting is bounded by remaining dwell tolerance or the current departure target; physical obstruction waits remain available",
            operation_deferral="committed work advances with its timed phases and checked safety waits; no primitive pause may indefinitely renew its delay",
            turnback_receiving_platform="imminent crossing protects its receiver and approach; approach occupants clear before entry; entered crossings can exit; reciprocal blocking commitments are rejected",
            occupied_resource_entry="an entrant to an occupied shared route is rejected before reservation arbitration; current occupants retain clearance priority",
            withdrawal_turnbacks="return paths retain timed intermediate reversals after passenger clearance, before physical depot registration",
            section_services="all configured reusable section patterns; independent vehicle instances with timed turnbacks",
            simultaneous_commands="all orders modulo adjacent commuting platform controls; no overlapping participants or control resources",
            scope="all vehicles and all platforms at every stage; no fixed MARL continuation",
            exclusions="no arbitrary continuous parameters, repeated-node command paths, or newly invented action families",
            online_comparison="online methods must use this domain or a subset; custom parameters outside it require a new benchmark domain")

    def headway_values(self, world):
        if self.headways is not None:
            return self.headways
        values = {k*self.tick for k in range(1, self.stages+1) if k*self.tick >= 30}
        values |= {float(a.headway_seconds) for a in world.agents}
        values |= {float(v) for v in world.agents[0].headway_tracker.targets.values()}
        return tuple(sorted(v for v in values if v >= 30))

    def paths(self, world, start, *, allowed_depot=None, deadline=math.inf):
        """Every simple nonempty path; no shortest-path or top-k truncation."""
        depots = {spec["node_id"] for spec in world.dispatch.depots.values()}
        stack = [(start, (), frozenset({start}))]
        while stack:
            check_deadline(deadline)
            node, path, seen = stack.pop()
            if path:
                yield path
            if node in depots and path:
                continue
            for nxt in sorted(world._has_edge.get(node, ()), reverse=True):
                if nxt not in seen and (nxt not in depots or nxt == allowed_depot):
                    stack.append((nxt, (*path, nxt), seen | {nxt}))

    def vehicle_commands(self, world, name, stage, *, deadline=math.inf):
        for command in self._vehicle_commands(world, name, stage, deadline=deadline):
            if command is None:
                yield None
            else:
                for offset in range(int(self.tick)):
                    check_deadline(deadline)
                    yield dict(command, offset_seconds=offset)

    def _vehicle_commands(self, world, name, stage, *, deadline=math.inf):
        """Syntactic domains; joint execution checks all state preconditions.

        Do not use one-command applicability as a joint-domain pruning rule.
        Shared service/vehicle state can change when another command starts.
        """
        yield None
        pos = world.dispatch.position(name)
        def command(action, parameters=None):
            return dict(action=action, vehicle_id=name, parameters=parameters or {})
        for action in ("cancel_service", "evacuate", "inspect_repair"):
            yield command(action)
        for action in ("reserve_insert", "reenter"):
            yield command(action)
            exits = {node for depot in world.dispatch.depots.values()
                     for node in depot.get("exit_node_ids", [depot["exit_node_id"]])}
            for node in sorted(exits):
                yield command(action, dict(exit_node_id=node))
        for k in range(1, int(self.stages*self.tick)+1):
            check_deadline(deadline)
            yield command("hold", dict(seconds=k))
        for sid, task in sorted(world.dispatch.services.items()):
            if task.get("stops"):
                yield command("reassign_service", dict(service_id=sid))
        for segment in world.base_segments:
            if pos in [n.id for n in world.segment2nodes[segment][:-1]]:
                for profile in sorted(world.dispatch.profiles):
                    yield command("run_profile", dict(segment_id=segment.id, profile_id=profile))
        if pos in world.platform_nodes:
            for segment in world.base_segments:
                if segment.start_platform.id != pos:
                    continue
                target = segment.end_platform.id
                yield command("short_turn", dict(target_node=target))
            # The native departure contract requires a direct platform segment.
            targets = {segment.end_platform.id for segment in world.base_segments
                       if segment.start_platform.id == pos and segment.end_platform.id in world.platform_nodes}
            for target in sorted(targets):
                for when in range(int(stage*self.tick), int(self.stages*self.tick)+1):
                    check_deadline(deadline)
                    for headway in (None, *self.headway_values(world)):
                        params = dict(target_node=target, departure_time=when)
                        if headway is not None:
                            params["headway_seconds"] = headway
                        yield command("adjust_departure", params)
            for path in self.paths(world, pos, deadline=deadline):
                if path[-1] in world.platform_nodes:
                    params = dict(target_node=path[-1], path_node_ids=list(path))
                    yield command("reroute", params)
                    yield command("skip_stop", params)
        for depot, spec in sorted(world.dispatch.depots.items()):
            for path in self.paths(world, pos, allowed_depot=spec["node_id"], deadline=deadline):
                if path[-1] == spec["node_id"]:
                    yield command("withdraw", dict(depot=depot, path_node_ids=list(path)))
            for failed in sorted(world.dispatch.records):
                if failed == name:
                    continue
                target = world.dispatch.position(failed)
                for approach in self.paths(world, pos,
                        allowed_depot=world.dispatch.depots.get(world.dispatch.records[name].depot, {}).get("node_id"),
                        deadline=deadline):
                    if approach[-1] != target:
                        continue
                    for tow in self.paths(world, target, allowed_depot=spec["node_id"], deadline=deadline):
                        if tow[-1] == spec["node_id"]:
                            yield command("rescue_tow", dict(failed=failed, depot=depot,
                                approach_node_ids=list(approach), tow_node_ids=list(tow)))

    def joint_controls(self, world, stage, *, deadline=math.inf):
        """Fairly stream the complete domain, including noncommuting orders."""
        names = sorted(world.dispatch.records)
        groups = [("vehicle", name) for name in names]
        for platform in sorted(world.platform_nodes):
            groups.extend((("admission", platform), ("headway", platform)))
        def options(kind, key):
            if kind == "vehicle":
                degree = len(world.nodes[world.dispatch.position(key)].getnext() or {})
                preferred = nominal_action(world, world.dispatch.records[key].agent)
                labels = [preferred] + [a for a in range(degree+1) if a != preferred]
                for raw in self.vehicle_commands(world, key, stage, deadline=deadline):
                    for label in labels:
                        for pause in (False, True):
                            yield raw, label, pause
            else:
                yield None, None, False
                values = self.admission_rates if kind == "admission" else self.headway_values(world)
                for value in values:
                    action = "admission_control" if kind == "admission" else "adjust_departure"
                    parameter = "rate" if kind == "admission" else "headway_seconds"
                    yield dict(action=action, vehicle_id=None, parameters={"platform_id": key, parameter: value}), None, False
        for values in fair_product((options(*group) for group in groups), deadline=deadline):
            check_deadline(deadline)
            commands, actions, pauses, participants, resources = [], {}, [], set(), set()
            compatible = True
            for (kind, key), (raw, label, pause) in zip(groups, values):
                claimed, used = set(), set()
                if raw:
                    parsed = dict(raw)
                    offset = parsed.pop("offset_seconds", 0)
                    c = DispatchCommand.from_dict(dict(parsed, command_id="domain", effective_at=stage*self.tick+offset), 1)
                    claimed, used = c.participants(), c.resources()
                    if claimed & participants or used & resources:
                        compatible = False
                        break
                    commands.append(raw)
                participants.update(claimed)
                resources.update(used)
                if kind == "vehicle":
                    actions[key] = label
                    if pause:
                        pauses.append(key)
            if compatible:
                for order in command_orders(commands, deadline=deadline):
                    yield StageControl(dict(actions), order, list(pauses))
