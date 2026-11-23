"""Finite-domain joint dispatch in MILPSolver, checked by the original world.

The master chooses command bundles. A deterministic trajectory subproblem uses
MetroWorldV1's actual operations, boarding and hard safety rules. A rejected
joint assignment receives a no-good cut; a feasible assignment receives its
exact nonnegative cost cut. Certificates concern the declared finite catalog
and current published inputs only, not an unrestricted scheduling optimum.
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import time
from types import MappingProxyType

from MILP import cplex_backend as mip
import torch

from delay_events import DelayEvents, EventView, INFORMATION_EVENTS, EVENT_TARGETS
from dispatch_plan import DispatchCommand, DispatchPlan, PARAMETERS
from dispatch_timing import ContinuationMismatch, compile_resource_waits, decision_seconds, timing_manifest


BACKGROUND_CONFLICT_REASONS = frozenset({
    "shared_resource_conflict", "fixed_continuation_history_changed", "fixed_continuation_not_executable",
    "nonfocus_continuation_changed", "nonfocus_continuation_not_executable"})


def background_blockers(model, value):
    if value.get("feasible") or value.get("reason") not in BACKGROUND_CONFLICT_REASONS:
        return set()
    vehicles = set(value.get("vehicles", []))
    if value.get("vehicle"):
        vehicles.add(value["vehicle"])
    return vehicles & (set(model.base.dispatch.records) - model.focus)


class PublishedPlanningEvents(DelayEvents):
    """No future event record, private clearance time or unreported failure."""
    def __init__(self, records, resources, now):
        super().__init__()
        self.records = copy.deepcopy(records)
        self.resources = copy.deepcopy(resources)
        self.current_time = now
        self._frozen_view = EventView(self.records, self.resources)
        self.disruptions = {}
        for e in records:
            if e.get("active", True) and e["type"] not in INFORMATION_EVENTS:
                self.disruptions[e["event_id"]] = dict(copy.deepcopy(e),
                    start_seconds=min(now, e["reported_at"]), actual_end_seconds=math.inf,
                    first_report_seconds=e["reported_at"], recovery_reports=[])

    def update_disruptions(self, current_time):
        self.current_time = float(current_time)

    def published_records(self):
        return copy.deepcopy(self.records)

    def physical_view(self):
        # Published restrictions are fixed for this conditional rollout. Queries
        # must not repeatedly copy thousands of static resource-node entries.
        return self._frozen_view

    def published_view(self):
        return self.physical_view()

    def __deepcopy__(self, memo):
        result = object.__new__(type(self))
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, value if key == "_frozen_view" else copy.deepcopy(value, memo))
        return result


def planning_copy(world):
    """Project before rollout; never advance the real world's event timeline."""
    visible = world.dispatch_snapshot()
    archive = getattr(world, "execution_records", None)
    result = copy.deepcopy(world, {id(archive): None} if archive is not None else None)
    result.execution_records = None
    result._strict_dispatch_execution = True
    result.disruption_manager = PublishedPlanningEvents(world.get_published_events(), world.event_resources,
                                                       float(world.sim_time[0]))
    result._configured_events = None
    result.pending_events = None
    result.custom_disruptions = None
    result.dispatch.hidden_failure_prior = {}
    for key, state in visible["vehicles"].items():
        record = result.dispatch.records[key]
        for field in ("status", "service_id", "failed", "needs_inspection"):
            setattr(record, field, state[field])
    result.dispatch.services = copy.deepcopy(visible["services"])
    result.dispatch.seen_failures &= set(result.disruption_manager.disruptions)
    return result


def state_guard(world):
    """Operational state, not the initial positions or an opaque plan version."""
    snapshot = world.dispatch.snapshot(published=True, include_version=False, include_command_results=False)
    # Command IDs differ between cached windows. Active operation semantics,
    # phases, participants and remaining work must nevertheless match.
    snapshot["operations"] = sorted([
        {k: v for k, v in op.items() if k not in ("id", "command_id")}
        for op in snapshot["operations"].values()], key=lambda o: o["vehicle_id"])
    snapshot["queues"] = {p: dict(
        inside={d: float(q[0]) for d, q in n.passenger_flow.waiting_by_destination.items()},
        outside={d: float(q[0]) for d, q in n.passenger_flow.outside_by_destination.items()},
        inside_waiting_mass={d: float(q[0]) for d, q in n.passenger_flow.waiting_mass_by_destination.items()},
        outside_waiting_mass={d: float(q[0]) for d, q in n.passenger_flow.outside_mass_by_destination.items()})
        for p, n in world.platform_nodes.items()}
    snapshot["train_clocks"] = {a.name: {field: float(getattr(a.state, field)[0])
        for field in ("dwell_time", "dwell_min_limit", "dwell_max_limit", "current_route",
                      "arrival_time", "target_departure_time", "time_since_arrival")}
        for a in world.agents}
    snapshot["headway_targets"] = dict(world.agents[0].headway_tracker.targets)
    snapshot["headway_history"] = {p: {key: (float(getattr(buf, key)[0]) if math.isfinite(float(getattr(buf, key)[0])) else None)
        for key in ("last_departure", "last_arrival", "next_due")}
        for p, buf in world.agents[0].headway_tracker._buffers.items()}
    snapshot["pending_commands"] = sorted([{k: v for k, v in asdict(c).items() if k != "command_id"}
        for plan in world.dispatch.plans.values() for c in plan.commands
        if world.dispatch.command_results[c.command_id]["status"] == "pending"],
        key=lambda c: (c["effective_at"], c["vehicle_id"] or "", c["action"]))
    snapshot["physical_departures"] = dict(world.dispatch.physical_departures)
    snapshot["objective"] = world.dispatch.objective.manifest()
    snapshot["timing"] = timing_manifest(world)
    snapshot["operating_inputs"] = dict(boarding_rate=world.boarding_rate_persons_per_second,
        queue_capacity=world.safe_platform_queue_capacity, resources=copy.deepcopy(world.event_resources),
        vehicles={a.name: a.passenger_capacity for a in world.agents}, dispatch_config=copy.deepcopy(world.dispatch.config),
        platforms={p: dict(arrival_rate=n.passenger_flow.arrival_rate, destinations=dict(n.passenger_flow.destinations),
                           dwell_seconds=float(n.weight)) for p, n in world.platform_nodes.items()})
    return snapshot


def guard_matches(expected, actual, *, tolerance=1e-4):
    if isinstance(expected, dict) and isinstance(actual, dict):
        return set(expected) == set(actual) and all(guard_matches(expected[k], actual[k], tolerance=tolerance) for k in expected)
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return len(expected) == len(actual) and all(guard_matches(x, y, tolerance=tolerance) for x, y in zip(expected, actual))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool) and isinstance(actual, (int, float)):
        return expected == actual or math.isclose(expected, actual, abs_tol=tolerance, rel_tol=1e-7)
    return expected == actual


def positions(world):
    return {a.name: int(a.state.position[0]) for a in world.agents}


def passenger_total(world):
    state = world.dispatch.objective.passenger_state(world)
    return state["inside"] + state["outside"] + state["onboard"]


def action_for_target(world, agent, target):
    current = int(agent.state.position[0])
    if target == current:
        return 0
    choices = sorted((world.nodes[current].getnext() or {}).items(), key=lambda item: str(item[0]))
    for index, (_, successor) in enumerate(choices):
        if successor == target:
            return index + 1
    raise ValueError("A continuation target is not a directed neighbor")


def nominal_action(world, agent):
    if not world.dispatch.service_enabled(agent) or world.dispatch.operation_for(agent.name):
        return 0
    current = int(agent.state.position[0])
    choices = sorted((world.nodes[current].getnext() or {}).items(), key=lambda item: str(item[0]))
    route = agent.route_adapter._current_route_key[0]
    choices = sorted(enumerate(choices), key=lambda item: item[1][0] != route)
    for index, (_, target) in choices:
        if world.edge_event_allowed(agent, current, target, view=world.event_view()):
            segment = world._step_segment.get((current, target))
            if any(segment and segment.end_platform.id == d["node_id"] for d in world.dispatch.depots.values()):
                continue
            facility = world.dispatch.turnback_facilities.get(segment.id) if segment else None
            if facility and facility.get("facility_type") != "terminal":
                # Choosing an alternate bare edge is not a cleared, timed
                # short-turn operation or an assigned section service.
                continue
            return index + 1
    return 0


def set_actions(world, actions):
    for agent in world.agents:
        agent.set_action(torch.tensor([actions.get(agent.name, 0)], dtype=torch.long, device=agent.device))
        agent.potential_step()


@dataclass
class CommandOption:
    id: str
    commands: list
    actions: dict
    pauses: dict


class EvaluationTimeout(Exception):
    pass


class JointDispatchModel:
    def __init__(self, world, horizon, *, focus=None, candidates=None, passenger_inputs=None,
                 fixed_paths=None, reference_actions=None, start_stages=(0,), deadline=math.inf,
                 reviewed_continuation=True, defer_reference=False, focus_platforms=None, reference_packets=None):
        if world.dispatch is None or world.num_envs != 1 or horizon < 1:
            raise ValueError("Joint dispatch needs one configured V2 world and a positive horizon")
        if type(reviewed_continuation) is not bool:
            raise ValueError("reviewed_continuation must be a bool")
        self.started = time.perf_counter()
        self.reviewed_continuation = reviewed_continuation
        self.continuation_mode = ("reviewed_actual_background_v3" if reviewed_continuation
                                  else "strict_actual_background_v3")
        self.construction_reused = False
        self.timings = dict(snapshot_construction_seconds=0., reference_construction_seconds=0.,
                            catalog_construction_seconds=0., cplex_master_search_seconds=0.,
                            physical_candidate_evaluation_seconds=0.)
        self.deadline = deadline
        self.world = world
        self.base = planning_copy(world)
        self.horizon, self.tick, self.now = int(horizon), decision_seconds(world), float(world.sim_time[0])
        self.token = hashlib.sha256(json.dumps(world.dispatch_snapshot(), sort_keys=True).encode()).hexdigest()[:12]
        ids = {a.name for a in world.agents}
        self.focus = ids - set(fixed_paths or {}) if focus is None else {world.agents[v].name if isinstance(v, int) else v for v in focus}
        if not self.focus <= ids:
            raise ValueError("Unknown focus vehicle")
        self.focus_platforms = frozenset(world.platform_nodes if focus_platforms is None else focus_platforms)
        if not self.focus_platforms <= set(world.platform_nodes):
            raise ValueError("Unknown focus platform")
        self.inputs = self._inputs(passenger_inputs)
        self.reference_packets = copy.deepcopy(reference_packets)
        if self.reference_packets is not None:
            if len(self.reference_packets) != horizon:
                raise ValueError("Reference packets need H checked stages")
            for packet, arrivals in zip(self.reference_packets, self.inputs):
                if (packet["arrival_inputs"] != arrivals or
                        getattr(packet["arrival_inputs"], "profile", None) != getattr(arrivals, "profile", None)):
                    raise ValueError("Reference packets use different passenger inputs")
        self.reference_actions = copy.deepcopy(reference_actions or {})
        for name, sequence in self.reference_actions.items():
            if name not in ids or len(sequence) != horizon or any(type(a) is not int or not 0 <= a <= 3 for a in sequence):
                raise ValueError("Reference actions need H legal labels per known vehicle")
        self.fixed_paths = copy.deepcopy(fixed_paths or {})
        for key, path in self.fixed_paths.items():
            if (key not in ids or len(path) != horizon + 1 or path[0] != positions(world)[key]
                    or any(not isinstance(n, int) or isinstance(n, bool) or n not in world.nodes for n in path)):
                raise ValueError("Fixed trajectories need H+1 known state nodes starting at the current position")
            if not world.second_precision and any(a != b and b not in world._has_edge.get(a, ()) for a, b in zip(path, path[1:])):
                raise ValueError("Non-directed fixed continuation")
        self.timings["snapshot_construction_seconds"] = time.perf_counter() - self.started
        started = time.perf_counter()
        self._nominal_evaluation = None
        self.nominal_requests = self.nominal_paths = self.nominal_history = None
        if not defer_reference:
            self.nominal_requests, self.nominal_paths = self._reference()
        self.timings["reference_construction_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        generated = self._generate(start_stages) if candidates is None else candidates
        self._set_catalog(generated)
        self.timings["catalog_construction_seconds"] = time.perf_counter() - started
        self.construction_timings = MappingProxyType({k: v for k, v in self.timings.items()
                                                     if k.endswith("construction_seconds")})
        self.evaluated = {}
        self.parallel_evaluation_stats = []
        self.cuts = []
        self.best = None
        self.best_selection = None
        self._check_time()

    def _set_catalog(self, generated):
        self.groups = self._catalog(generated)
        self.domain_size = math.prod(len(values) for values in self.groups.values())
        catalog = [(g, [dict(id=o.id, actions=o.actions, pauses=o.pauses,
                            commands=[{k: v for k, v in c.items() if k != "command_id"}
                                      for c in o.commands]) for o in options])
                   for g, options in self.groups.items()]
        self.catalog_fingerprint = hashlib.sha256(json.dumps(dict(
            horizon=self.horizon, tick=self.tick, groups=catalog,
            continuation_mode=self.continuation_mode), sort_keys=True).encode()).hexdigest()

    def refocus(self, focus, platforms, *, candidates=None):
        """Rebuild only free decisions against the same checked full reference."""
        result = self.fork(deadline=self.deadline)
        result.focus = set(focus)
        result.focus_platforms = frozenset(platforms)
        if not result.focus <= set(self.base.dispatch.records) or not result.focus_platforms <= set(self.base.platform_nodes):
            raise ValueError("Unknown refocused decision owner")
        result.fixed_paths = {}
        started = time.perf_counter()
        result._set_catalog(result._generate((0,)) if candidates is None else candidates)
        result.timings["catalog_construction_seconds"] = time.perf_counter() - started
        result._check_time()
        return result

    def prefix(self, horizon):
        """A shorter physical check over the same current-stage command catalog."""
        if type(horizon) is not int or not 1 <= horizon <= self.horizon:
            raise ValueError("Invalid prefix horizon")
        self.ensure_reference()
        if any(c["effective_at"] >= self.now + horizon*self.tick
               for opts in self.groups.values() for op in opts for c in op.commands):
            raise ValueError("Prefix cannot hide future-start commands")
        result = self.fork()
        result.horizon = horizon
        result.inputs = self.inputs[:horizon]
        result.reference_packets = None if self.reference_packets is None else self.reference_packets[:horizon]
        result.nominal_requests = self.nominal_requests[:horizon]
        result.nominal_history = self.nominal_history[:horizon]
        result.nominal_paths = {g: p[:horizon+1] for g, p in self.nominal_paths.items()}
        result.fixed_paths = {g: p[:horizon+1] for g, p in self.fixed_paths.items()}
        result.catalog_fingerprint = hashlib.sha256(
            f"{self.catalog_fingerprint}:prefix:{horizon}".encode()).hexdigest()
        return result

    def fork(self, *, deadline=math.inf, reference_selection=None, reuse_evaluations=False):
        """Start an independent search over this exact published state/catalog.

        The absolute perf_counter deadline is new (unlimited by default). Only
        construction inputs are shared, read-only: all transitions deepcopy the
        published base. No evaluations, cuts, incumbent, or timing/budget from a
        teacher search are inherited by default. Online neighborhood repairs may
        reuse exact point evaluations of this same state/catalog; master
        expressions and neighborhood bounds are never shared. An explicitly supplied, independently
        checked reference may be shared equally by paired collection searches.
        Per-fork construction timings are zero;
        construction_timings retains the shared, immutable original build record.
        Callers must treat base, reference data, inputs and groups as immutable.
        """
        result = object.__new__(type(self))
        for name in ("world", "base", "horizon", "tick", "now", "token", "inputs",
                     "reference_actions", "reference_packets", "fixed_paths", "nominal_requests", "nominal_paths",
                     "nominal_history", "reviewed_continuation", "continuation_mode",
                     "groups", "domain_size", "catalog_fingerprint", "construction_timings", "_nominal_evaluation", "focus_platforms"):
            setattr(result, name, getattr(self, name))
        result.focus = set(self.focus)
        result.started = time.perf_counter()
        result.deadline = deadline
        result.construction_reused = True
        result.timings = dict.fromkeys(self.timings, 0.)
        result.parallel_evaluation_stats = []
        result.evaluated = {}
        result.cuts = []
        result.best = None
        result.best_selection = None
        if reuse_evaluations:
            result.evaluated = dict(self.evaluated)
        if reference_selection is not None:
            key = tuple(reference_selection)
            value = self.evaluated.get(key)
            if (not value or not value.get("feasible")
                    or value.get("evaluation_catalog") != self.catalog_fingerprint
                    or value.get("evaluation_selection") != key):
                raise ValueError("Only an exact checked reference from this catalog may be shared")
            result.evaluated[key] = value
        return result

    def _check_time(self):
        if time.perf_counter() >= self.deadline:
            raise EvaluationTimeout()

    def ensure_reference(self):
        """Catalog-only diagnostics do not need a second nominal rollout."""
        if self.nominal_requests is None:
            started = time.perf_counter()
            try:
                self.nominal_requests, self.nominal_paths = self._reference()
            finally:
                self.timings["reference_construction_seconds"] += time.perf_counter() - started

    def check_reference_source(self, world, inputs):
        """Only capture a rollout executed from this request's published state."""
        if (world is not self.world or not guard_matches(state_guard(self.base), state_guard(world))
                or self.fixed_paths or len(inputs) != self.horizon or inputs != self.inputs
                or any(getattr(a, "profile", None) != getattr(b, "profile", None)
                       for a, b in zip(inputs, self.inputs))):
            raise ValueError("Checked continuation does not match the prepared planning problem")

    def _capture_checked_reference(self, *, requests, history, paths, packets, actions, steps,
                                   terminal, trajectory, terminal_passengers, terminal_guard):
        # Called only after checked_reference has executed every second, checked
        # hard feasibility and passenger conservation. No JSON cost is trusted.
        self._check_time()
        gamma = self.base.dispatch.objective.weights["discount"]
        self.reference_packets = packets
        self.reference_actions = actions
        self.nominal_requests, self.nominal_history, self.nominal_paths = requests, history, paths
        self._nominal_evaluation = dict(feasible=True,
            cost=sum(gamma**k*s["total"] for k, s in enumerate(steps)) + gamma**self.horizon*terminal["total"],
            steps=steps, terminal=terminal, packets=packets, trajectory=trajectory,
            passenger_total=terminal_passengers, terminal_guard=terminal_guard, physical_repair_count=0)

    def fix_background_to_reference(self):
        """Fix nonfocus motion to the just-checked trajectory, including seconds."""
        self.ensure_reference()
        if self._nominal_evaluation is None:
            raise ValueError("Cannot fix background without a feasible checked continuation")
        self.fixed_paths = {name: path for name, path in self.nominal_paths.items() if name not in self.focus}
        reference = tuple(next((i for i, op in enumerate(options)
            if not op.commands and not op.actions and not op.pauses), None) for options in self.groups.values())
        # Changing fixed continuations invalidates other point checks. The
        # nominal trajectory satisfies these particular paths by construction.
        self.evaluated = {} if None in reference else {reference: self.evaluate(reference)}

    def _inputs(self, supplied):
        if supplied is None:
            current = {p: n.passenger_flow.interval_arrivals(self.base.event_view(), self.tick)
                       for p, n in self.base.platform_nodes.items()}
            return [copy.deepcopy(current) for _ in range(self.horizon)]
        if not isinstance(supplied, list) or len(supplied) != self.horizon:
            raise ValueError("Passenger input must specify each of H intervals")
        result = copy.deepcopy(supplied)
        for interval in result:
            if not isinstance(interval, dict) or not set(interval) <= set(self.base.platform_nodes):
                raise ValueError("Unknown passenger origin")
            for origin, values in interval.items():
                if not isinstance(values, dict):
                    raise ValueError("Passenger input needs destination counts")
                for dest, count in values.items():
                    if (dest == origin or dest not in self.base.platform_nodes or isinstance(count, bool)
                            or not isinstance(count, (int, float)) or not math.isfinite(count) or count < 0):
                        raise ValueError("Invalid declared passenger input")
        return result

    def _reference(self):
        world = copy.deepcopy(self.base)
        world._planning_deadline = self.deadline
        requests = []
        self.nominal_history = []
        paths = {key: [value] for key, value in positions(world).items()}
        packets, steps, trajectory = [], [], [world.dispatch.snapshot(published=True)]
        cost, checked = 0., True
        gamma = world.dispatch.objective.weights["discount"]
        initial_passengers = passenger_total(world) + sum(float(a.cumulative_alighted[0]) for a in world.agents)
        arrivals_total = 0.
        for k in range(self.horizon):
            self._check_time()
            guard = packets[-1]["expected_guard"] if packets else state_guard(world)
            actions = {a.name: nominal_action(world, a) for a in world.agents}
            actions.update({name: sequence[k] for name, sequence in self.reference_actions.items()})
            overrides = ({name: path[k + 1] for name, path in self.fixed_paths.items()}
                         if not world.second_precision else {})
            reference_commands = []
            if self.reference_packets is not None:
                packet = self.reference_packets[k]
                applied = apply_dispatch_step(world, dict(checked_dispatch=True, dispatch_packets=self.reference_packets), k)
                if not applied["accepted"]:
                    raise ValueError("Reference packet is not executable: "+str(applied["reason"]))
                actions = dict(packet["actions"])
                overrides = None
                reference_commands = copy.deepcopy(packet["commands"])
            else:
                set_actions(world, actions)
            world.step(arrival_inputs=self.inputs[k], dispatch_targets=overrides)
            if self.reference_packets is not None and not audit_dispatch_step(
                    world, dict(dispatch_packets=self.reference_packets), k)["matched"]:
                raise ValueError("Reference packet replay differs from its checked trajectory")
            stage = ([{name: (a["origin"] if a["conflict"] else a["requested"])
                       for name, a in second.items()} for second in world.last_physical_audits]
                     if world.second_precision else
                     {name: (a["origin"] if a["conflict"] else a["requested"])
                      for name, a in world.last_transition_audit.items()})
            requests.append(stage)
            audits = world.last_physical_audits if world.second_precision else [world.last_transition_audit]
            self.nominal_history.append([{name: row["actual"] for name, row in second.items()}
                                         for second in audits])
            for name, pos in positions(world).items():
                paths[name].append(pos)
            arrivals_total += sum(sum(values.values()) for values in self.inputs[k].values())
            completed = sum(float(a.cumulative_alighted[0]) for a in world.agents)
            if not math.isclose(passenger_total(world)+completed, initial_passengers+arrivals_total,
                                rel_tol=1e-5, abs_tol=1e-3):
                raise AssertionError("Passenger conservation failed in the shared reference")
            checked &= not any(row["conflict"] for row in world.last_transition_audit.values())
            checked &= not self.fixed_paths or all(positions(world)[g] == p[k+1] for g, p in self.fixed_paths.items())
            if self.reviewed_continuation:
                checked &= not self._command_failures(world) and not getattr(world, "last_execution_deviations", [])
            interval = dict(world.dispatch.interval_cost)
            steps.append(interval)
            cost += gamma**k * interval["total"]
            packets.append(dict(guard=guard, commands=reference_commands, actions=actions,
                targets=copy.deepcopy(world.last_physical_requests) if world.second_precision else overrides,
                physical_audits=copy.deepcopy(world.last_physical_audits) if world.second_precision else None,
                expected_positions=positions(world), expected_guard=state_guard(world),
                arrival_inputs=copy.deepcopy(self.inputs[k])))
            trajectory.append(world.dispatch.snapshot(published=True))
        if checked:
            terminal = world.dispatch.objective.terminal(world)
            self._nominal_evaluation = dict(feasible=True, cost=cost+gamma**self.horizon*terminal["total"],
                steps=steps, terminal=terminal, packets=packets, trajectory=trajectory,
                passenger_total=passenger_total(world), terminal_guard=state_guard(world), physical_repair_count=0)
        return requests, paths

    def _generate(self, starts):
        groups = {}
        if not isinstance(starts, (list, tuple)) or not starts or any(not isinstance(k, int) or isinstance(k, bool) or k < 0 or k >= self.horizon for k in starts):
            raise ValueError("Candidate start stages must be inside the planning window")
        for name in sorted(self.focus):
            record = self.base.dispatch.records[name]
            pos = int(record.agent.state.position[0])
            options = [dict(id="continue", commands=[])]
            def add(action, parameters=None):
                for k in starts:
                    options.append(dict(id=f"{action}_{len(options)}", commands=[dict(action=action,
                        vehicle_id=name, stage=k, parameters=parameters or {})]))
            busy = self.base.dispatch.operation_for(name)
            if busy and self.base.dispatch.discretionary_hold_allowed(name):
                options.append(dict(id="defer_movement", commands=[], pauses={name: list(range(self.horizon))}))
            if not busy or self.base.dispatch.service_revisable(name):
                durations = {int(self.tick), 60, 120}
                if pos in self.base.platform_nodes:
                    view = self.base.event_view()
                    rate = self.base.boarding_rate_persons_per_second * view.boarding_multiplier(pos)
                    free = max(0., view.vehicle_capacity(name, record.agent.passenger_capacity)
                               - float(record.agent.passenger_count[0]))
                    if rate > 0:
                        durations.add(max(1, math.ceil(min(free, self.base.waiting_at(pos))/rate)))
                if self.base.second_precision and self.tick > 1:
                    durations.update(seconds for seconds in (1, 7, 17) if seconds < self.tick)
                for seconds in sorted({min(v, int(self.horizon*self.tick)) for v in durations if v > 0}):
                    add("hold", {"seconds": seconds})
                if self.base.dispatch.discretionary_hold_allowed(name):
                    options.append(dict(id="wait_window", commands=[], actions={name: [0] * self.horizon}))
                for family in ("cancel_service", "evacuate", "inspect_repair"):
                    add(family)
                if record.depot is not None:
                    depot = self.base.dispatch.depots[record.depot]
                    for exit_node in depot.get("exit_node_ids", [depot["exit_node_id"]]):
                        for family in ("reserve_insert", "reenter"):
                            add(family, {"exit_node_id": exit_node})
                for depot in self.base.dispatch.depots:
                    add("withdraw", {"depot": depot})
                    for failed in self.base.dispatch.records.values():
                        reported_failure = self.base.event_view().active("vehicle_breakdown", "vehicle", failed.id)
                        if (failed.failed or any(e["parameters"]["mode"] == "immobilized" for e in reported_failure)) and failed.id != name and failed.depot is None:
                            add("rescue_tow", {"failed": failed.id, "depot": depot})
                if (record.depot is None and not record.failed and pos in self.base.platform_nodes
                        and self.base.dispatch.service_impairments(name)):
                    # A matched proposal exposes both stock choices to the same
                    # oracle, without forcing retirement or bypassing qualifications.
                    for depot in self.base.dispatch.depots:
                        for spare in self.base.dispatch.records.values():
                            if (spare.id == name or spare.depot is None or spare.failed or spare.needs_inspection
                                    or spare.status != "DEPOT_READY"
                                    or spare.role != "reserve" and spare.id not in self.focus):
                                continue
                            family = "reserve_insert" if spare.role == "reserve" and not spare.has_served else "reenter"
                            if family == "reenter" and not spare.has_served:
                                continue
                            spec = self.base.dispatch.depots[spare.depot]
                            for exit_node in spec.get("exit_node_ids", [spec["exit_node_id"]]):
                                for stage in starts:
                                    options.append(dict(id=f"replace_{depot}_{spare.id}_{exit_node}_{stage}", commands=[
                                        dict(action="withdraw", vehicle_id=name, stage=stage, parameters=dict(depot=depot)),
                                        dict(action=family, vehicle_id=spare.id, stage=stage,
                                             parameters=dict(exit_node_id=exit_node))]))
                section_patterns = set()
                for sid, task in self.base.dispatch.services.items():
                    if (task["vehicle_id"] is None and not task.get("completed", False)
                            and task.get("reserved_vehicle_id") is None
                            and (pos in task["stops"][:-1] if task.get("pattern") else
                                 pos == task["stops"][0] or any(stage > 0 for stage in starts))):
                        if task.get("pattern") and not self.base.dispatch.section_need(sid, pos):
                            continue
                        if task.get("pattern"):
                            cycle = task["stops"][:-1]
                            start = cycle.index(pos)
                            signature = tuple(cycle[start:] + cycle[:start])
                            if signature in section_patterns:
                                continue
                            section_patterns.add(signature)
                        add("reassign_service", {"service_id": sid})
                for segment in self.base.base_segments:
                    seq = [n.id for n in self.base.segment2nodes[segment]]
                    if pos == segment.start_platform.id:
                        for family in ("adjust_departure", "reroute", "skip_stop", "short_turn"):
                            add(family, {"target_node": segment.end_platform.id})
                    if pos in seq[:-1]:
                        for profile in self.base.dispatch.profiles:
                            add("run_profile", {"segment_id": segment.id, "profile_id": profile})
            groups[name] = options
        # Platform controls are separate decisions. A caller can explicitly
        # narrow or extend their finite numeric grids through candidate bundles.
        for p in sorted(self.focus_platforms):
            if self.base.waiting_at(p) <= 0:
                continue
            # A prior restriction must be releasable, not only tightened to 0/1.
            cap = self.base.event_view().queue_capacity(p, self.base.safe_platform_queue_capacity)
            rates = sorted({0., 1., float(self.base.boarding_rate_persons_per_second), float(cap)})
            groups[f"platform_{p}"] = [dict(id="unchanged", commands=[])] + [
                dict(id=f"admit_{rate}", commands=[dict(action="admission_control", vehicle_id=None,
                    stage=0, parameters=dict(platform_id=p, rate=rate))]) for rate in rates]
            for seconds in (max(30., self.tick), self.base.agents[0].headway_seconds):
                groups[f"platform_{p}"].append(dict(id=f"headway_{seconds}", commands=[dict(
                    action="adjust_departure", vehicle_id=None, stage=0,
                    parameters=dict(platform_id=p, headway_seconds=seconds))]))
        return groups or {"continuation": [dict(id="continue", commands=[])]}

    def _catalog(self, groups):
        if not isinstance(groups, dict) or not groups:
            raise ValueError("A finite nonempty candidate catalog is required")
        result = {}
        for gi, (name, values) in enumerate(groups.items()):
            if not isinstance(name, str) or not isinstance(values, list) or not values:
                raise ValueError("Each named decision group needs candidate options")
            result[name] = []
            seen = set()
            for oi, value in enumerate(values):
                if not isinstance(value, dict) or set(value) - {"id", "commands", "actions", "pauses"}:
                    raise ValueError("Unknown candidate fields")
                oid = value.get("id")
                if not isinstance(oid, str) or not oid or oid in seen:
                    raise ValueError("Candidate IDs must be unique within each group")
                seen.add(oid)
                commands = []
                for ci, raw in enumerate(value.get("commands", [])):
                    if not isinstance(raw, dict):
                        raise ValueError("A candidate command must be a structured object")
                    raw = copy.deepcopy(raw)
                    stage = raw.pop("stage", 0)
                    offset = raw.pop("offset_seconds", 0)
                    if type(offset) is not int or not 0 <= offset < self.tick:
                        raise ValueError("Command offset must be an integer second inside its decision interval")
                    if not isinstance(stage, int) or isinstance(stage, bool) or not 0 <= stage < self.horizon:
                        raise ValueError("Command stage outside the planning window")
                    raw.update(command_id=f"joint_{self.token}_{name}_{oi}_{ci}", effective_at=self.now + stage * self.tick + offset)
                    command = DispatchCommand.from_dict(raw, float(self.base.tick_seconds))
                    if not command.participants() <= set(self.base.dispatch.records):
                        raise ValueError("Unknown candidate vehicle")
                    self.focus |= command.participants()
                    commands.append(asdict(command))
                actions, pauses = copy.deepcopy(value.get("actions", {})), copy.deepcopy(value.get("pauses", {}))
                if not isinstance(actions, dict) or not isinstance(pauses, dict):
                    raise ValueError("Primitive continuations and pauses must be vehicle mappings")
                for key, sequence in actions.items():
                    if key not in self.base.dispatch.records or len(sequence) != self.horizon or any(
                            not isinstance(a, int) or isinstance(a, bool) or not 0 <= a <= 3 for a in sequence):
                        raise ValueError("Primitive continuation must contain H valid action labels")
                for key, stages in pauses.items():
                    if key not in self.base.dispatch.records or any(not isinstance(k, int) or not 0 <= k < self.horizon for k in stages):
                        raise ValueError("Invalid operation movement pauses")
                self.focus |= set(actions) | set(pauses)
                result[name].append(CommandOption(oid, commands, actions, pauses))
        if set(self.fixed_paths) & self.focus:
            raise ValueError("A fixed vehicle continuation cannot simultaneously be a decision variable")
        return result

    def evaluate(self, selection, *, enforce_deadline=True):
        self.ensure_reference()
        started = time.perf_counter()
        try:
            selection = tuple(selection)
            if len(selection) != len(self.groups) or any(type(i) is not int or not 0 <= i < len(opts)
                    for i, opts in zip(selection, self.groups.values())):
                raise ValueError("Evaluation needs one valid option index per decision group")
            if (self._nominal_evaluation is not None and len(self._nominal_evaluation["packets"]) == self.horizon
                    and all(not op.commands and not op.actions and not op.pauses
                            for op in (opts[i] for i, opts in zip(selection, self.groups.values())))):
                if enforce_deadline:
                    self._check_time()
                value = dict(self._nominal_evaluation)
            else:
                value = self._evaluate(selection, enforce_deadline=enforce_deadline)
            value.update(evaluation_catalog=self.catalog_fingerprint, evaluation_selection=tuple(selection))
            value.update(continuation_mode=self.continuation_mode,
                         reviewed_continuation=self.reviewed_continuation)
            value.setdefault("reviewed_continuation_waits", [])
            value.setdefault("corrections", [])
            if enforce_deadline:
                self._check_time()
            return value
        finally:
            self.timings["physical_candidate_evaluation_seconds"] += time.perf_counter() - started

    def evaluate_many(self, selections):
        from MILP.parallel_evaluation import evaluate_batch
        values, stats = evaluate_batch(self, selections)
        self.parallel_evaluation_stats.append(stats)
        if stats["mode"] != "serial":
            self.timings["physical_candidate_evaluation_seconds"] += stats["wall_seconds"]
        return values

    def _run_stage(self, world, due, actions, targets, k, *, route_vehicles=()):
        if due:
            plan = dict(plan_id=f"joint_{self.token}_stage_{k}", snapshot_basis="published",
                        snapshot_version=world.dispatch_snapshot()["version"],
                        created_at=float(world.sim_time[0]), commands=copy.deepcopy(due))
            result = world.submit_dispatch_plan(plan)
            if not result["accepted"]:
                return dict(feasible=False, reason=result["reason"], stage=k)
        # A revised vehicle follows its new service/route, not a stale primitive
        # HOLD copied from the MARL reference before the service was inserted.
        for name in route_vehicles:
            actions[name] = nominal_action(world, world.dispatch.records[name].agent)
        set_actions(world, actions)
        try:
            world.step(arrival_inputs=self.inputs[k], dispatch_targets=targets)
        except ContinuationMismatch as exc:
            protected = exc.vehicle not in self.focus
            return dict(feasible=False, reason=("fixed_continuation_not_executable" if protected else str(exc)),
                vehicle=exc.vehicle, stage=k, second=exc.second, origin=exc.origin,
                requested_target=exc.target, continuation_error=str(exc))
        except ValueError as exc:
            return dict(feasible=False, reason=str(exc), stage=k)
        rejected = self._command_failures(world)
        if rejected:
            return dict(feasible=False, reason="operation_precondition", details=rejected, stage=k)
        return None

    def _command_failures(self, world):
        return {cid: status for cid, status in world.dispatch.command_results.items()
                    if status["status"] in ("failed", "rejected")
                    and (cid not in self.base.dispatch.command_results or
                         (self.reviewed_continuation and
                          self.base.dispatch.command_results[cid]["status"] != status["status"]))}

    @staticmethod
    def _stage_audits(world):
        return world.last_physical_audits if world.second_precision else [world.last_transition_audit]

    def _protected_history_error(self, world, k):
        protected = set(self.base.dispatch.records) - self.focus
        for second, audit in enumerate(self._stage_audits(world)):
            for name in protected:
                if audit[name]["actual"] != self.nominal_history[k][second][name]:
                    kind = "fixed" if name in self.fixed_paths else "nonfocus"
                    return dict(feasible=False, reason=f"{kind}_continuation_not_executable",
                                vehicle=name, stage=k, second=second)
        return None

    def _review_targets(self, world, tow_pairs, k, enforce_deadline):
        return compile_resource_waits(world, tow_pairs, start_seconds=self.now+k*self.tick, stage=k,
                                      check_time=self._check_time if enforce_deadline else None)

    def _evaluate(self, selection, *, enforce_deadline=True):
        if not self.reviewed_continuation:
            return self._evaluate_window(selection, enforce_deadline=enforce_deadline)
        # Two whole-window copies, independent of H. An intercepted proposal is
        # private construction data; only clean execution of its targets is returned.
        proposal = self._evaluate_window(selection, enforce_deadline=enforce_deadline, review=True)
        if not proposal["feasible"]:
            return proposal
        if not proposal["reviewed_continuation_waits"] and not proposal["proposal_intervention_count"]:
            # No intercepted request or speculative travel clock was changed;
            # this already is the clean physical execution of the returned targets.
            return proposal
        replay = self._evaluate_window(selection, enforce_deadline=enforce_deadline,
                                       replay_packets=proposal["packets"])
        if replay["feasible"]:
            replay["reviewed_continuation_waits"] = proposal["reviewed_continuation_waits"]
            replay["corrections"] = copy.deepcopy(proposal["reviewed_continuation_waits"])
            replay["proposal_intervention_count"] = proposal["proposal_intervention_count"]
        return replay

    def _evaluate_window(self, selection, *, enforce_deadline, review=False, replay_packets=None):
        if enforce_deadline:
            self._check_time()
        world = copy.deepcopy(self.base)
        world._planning_deadline = self.deadline if enforce_deadline else math.inf
        chosen = [self.groups[group][selection[index]] for index, group in enumerate(self.groups)]
        commands = [c for option in chosen for c in option.commands]
        control, pauses = {}, {}
        for option in chosen:
            if set(control) & set(option.actions) or set(pauses) & set(option.pauses):
                return dict(feasible=False, reason="duplicate_vehicle_control")
            control.update(option.actions)
            pauses.update(option.pauses)
        revised_at = {name: self.now for name in control}
        for name, stages in pauses.items():
            revised_at[name] = min(revised_at.get(name, math.inf), self.now+min(stages, default=math.inf)*self.tick)
        resource_revised_at = {}
        for command in commands:
            parsed = DispatchCommand.from_dict(command, world.tick_seconds)
            for name in parsed.participants():
                revised_at[name] = min(revised_at.get(name, math.inf), parsed.effective_at)
            for resource in parsed.resources():
                resource_revised_at[resource] = min(resource_revised_at.get(resource, math.inf), parsed.effective_at)
        first_revision = min([*revised_at.values(), *resource_revised_at.values()], default=math.inf)
        if self.reference_packets is not None:
            # Unchanged choices retain complete MARL commands, not just motion
            # labels. A later revision cannot retroactively erase earlier work.
            for packet in self.reference_packets:
                for command in packet["commands"]:
                    parsed = DispatchCommand.from_dict(command, world.tick_seconds)
                    replacement_at = min([revised_at.get(v, math.inf) for v in parsed.participants()] +
                                         [resource_revised_at.get(r, math.inf) for r in parsed.resources()], default=math.inf)
                    if parsed.effective_at < replacement_at:
                        commands.append(copy.deepcopy(command))
        tow_pairs = [(op.participants(), op.command_id, op.started_at)
                     for op in world.dispatch.operations.values() if op.action == "rescue_tow"]
        tow_pairs += [(c.participants(), c.command_id, math.inf)
                      for plan in world.dispatch.plans.values() for c in plan.commands
                      if c.action == "rescue_tow"]
        tow_pairs += [(DispatchCommand.from_dict(c, world.tick_seconds).participants(), c["command_id"], math.inf)
                      for c in commands if c["action"] == "rescue_tow"]
        first_total = passenger_total(world) + sum(float(a.cumulative_alighted[0]) for a in world.agents)
        arrivals_total = 0.
        trajectory, steps, packets = [world.dispatch.snapshot(published=True)], [], []
        reviewed_waits, rebased_stages = [], []
        proposal_interventions = 0
        cost = 0.
        gamma = world.dispatch.objective.weights["discount"]
        for k in range(self.horizon):
            if enforce_deadline:
                self._check_time()
            guard = packets[-1]["expected_guard"] if packets else state_guard(world)
            due = [copy.deepcopy(c) for c in commands
                   if self.now + k * self.tick <= c["effective_at"] < self.now + (k + 1) * self.tick]
            actions = (dict(self.reference_packets[k]["actions"]) if self.reference_packets is not None else
                       {a.name: nominal_action(world, a) for a in world.agents})
            for name, sequence in control.items():
                actions[name] = sequence[k]
            discretionary = {name for name, sequence in control.items()
                             if sequence[k] == 0 and world.dispatch.operation_for(name) is None}
            discretionary |= {name for name, stages in pauses.items() if k in stages}
            for name in discretionary:
                if not world.dispatch.discretionary_hold_allowed(name):
                    return dict(feasible=False, reason="discretionary_hold_requires_safe_platform", vehicle=name, stage=k)
            seconds = len(self.nominal_requests[k]) if world.second_precision else 1
            overrides = [{} for _ in range(seconds)] if world.second_precision else {}
            def constrain(name, value):
                if world.second_precision:
                    for second, mapping in enumerate(overrides):
                        # Replay requests, not occupancy: an intercepted move
                        # starts its travel clock. Replaying only the resulting
                        # stationary node charges that delay again on departure.
                        # _protected_history_error still checks every actual
                        # background position against the reference occupancy.
                        mapping[name] = (self.nominal_requests[k][second][name] if value is None else value)
                else:
                    overrides[name] = self.nominal_requests[k][name] if value is None else value
            for name in self.base.dispatch.records:
                if name in self.fixed_paths:
                    if positions(world)[name] != self.fixed_paths[name][k]:
                        return dict(feasible=False, reason="fixed_continuation_history_changed", vehicle=name, stage=k)
                    constrain(name, None if world.second_precision else self.fixed_paths[name][k + 1])
                elif name not in self.focus:
                    if positions(world)[name] != self.nominal_paths[name][k]:
                        return dict(feasible=False, reason="nonfocus_continuation_changed", vehicle=name, stage=k)
                    constrain(name, None)
                elif self.now+(k+1)*self.tick <= first_revision:
                    constrain(name, None)
                elif world.second_precision:
                    for second, mapping in enumerate(overrides):
                        if self.now+k*self.tick+second*float(world.tick_seconds) < first_revision:
                            mapping[name] = self.nominal_requests[k][second][name]
                # A focus vehicle's continuation is a route policy, not a fixed
                # absolute trajectory. Other joint decisions may change its
                # timing. Only nonfocus/supplied continuations stay fixed.
                if k in pauses.get(name, []):
                    constrain(name, positions(world)[name])
            reviewed_packet = replay_packets[k] if replay_packets is not None else None
            if reviewed_packet is not None:
                # Intercepted requests can leave speculative travel clocks in
                # the proposal. Explicit waits must be executed from the base;
                # only this clean execution supplies final guards and costs.
                if reviewed_packet["commands"] != due:
                    return dict(feasible=False, reason="reviewed_command_mismatch", stage=k)
                overrides = reviewed_packet["targets"]
            error = self._run_stage(world, due, actions, overrides, k,
                route_vehicles={name for name, when in revised_at.items()
                                if when <= self.now+k*self.tick and name not in control}
                               if self.reference_packets is not None else ())
            if error:
                return error
            conflicts = sorted({name for audit in self._stage_audits(world)
                                for name, row in audit.items() if row["conflict"]})
            proposal_interventions += sum(row["conflict"] for audit in self._stage_audits(world)
                                          for row in audit.values())
            if conflicts and not review:
                return dict(feasible=False, reason="shared_resource_conflict", vehicles=conflicts, stage=k)
            if self.reviewed_continuation:
                if getattr(world, "last_execution_deviations", []):
                    return dict(feasible=False, reason="reviewed_replay_deviation", stage=k)
            error = self._protected_history_error(world, k)
            if error:
                return error
            for name, path in self.fixed_paths.items():
                if positions(world)[name] != path[k + 1]:
                    return dict(feasible=False, reason="fixed_continuation_not_executable", vehicle=name, stage=k)
            for name in set(self.base.dispatch.records) - self.focus - set(self.fixed_paths):
                if positions(world)[name] != self.nominal_paths[name][k + 1]:
                    return dict(feasible=False, reason="nonfocus_continuation_not_executable", vehicle=name, stage=k)
            arrivals_total += sum(sum(values.values()) for values in self.inputs[k].values())
            completed = sum(float(a.cumulative_alighted[0]) for a in world.agents)
            if not math.isclose(passenger_total(world) + completed, first_total + arrivals_total, rel_tol=1e-5, abs_tol=1e-3):
                raise AssertionError("Passenger conservation failed in the shared transition")
            expected_guard = state_guard(world)
            if reviewed_packet is not None:
                expected_audits = (reviewed_packet["physical_audits"] if world.second_precision
                                   else [reviewed_packet["transition_audit"]])
                expected = [{name: row["actual"] for name, row in audit.items()} for audit in expected_audits]
                actual = [{name: row["actual"] for name, row in audit.items()} for audit in self._stage_audits(world)]
                if actual != expected:
                    return dict(feasible=False, reason="reviewed_target_trajectory_mismatch", stage=k)
                if (not guard_matches(reviewed_packet["guard"], guard)
                        or not guard_matches(reviewed_packet["expected_guard"], expected_guard)):
                    rebased_stages.append(k)
            if review:
                targets, waits = self._review_targets(world, tow_pairs, k, enforce_deadline)
                reviewed_waits.extend(waits)
            else:
                targets = (copy.deepcopy(world.last_physical_requests)
                           if world.second_precision and (seconds > 1 or self.reviewed_continuation)
                           else ({name: row["requested"] for name, row in world.last_transition_audit.items()}
                                 if self.reviewed_continuation else overrides))
            interval = dict(world.dispatch.interval_cost)
            cost += gamma ** k * interval["total"]
            steps.append(interval)
            packets.append(dict(guard=guard, commands=due, actions=actions,
                                targets=targets,
                                physical_audits=copy.deepcopy(world.last_physical_audits) if world.second_precision else None,
                                transition_audit=copy.deepcopy(world.last_transition_audit) if not world.second_precision else None,
                                expected_positions=positions(world), expected_guard=expected_guard,
                                arrival_inputs=copy.deepcopy(self.inputs[k])))
            trajectory.append(world.dispatch.snapshot(published=True))
        terminal = world.dispatch.objective.terminal(world)
        cost += gamma ** self.horizon * terminal["total"]
        return dict(feasible=True, cost=float(cost), steps=steps, terminal=terminal, packets=packets,
                    trajectory=trajectory, passenger_total=passenger_total(world),
                    terminal_guard=state_guard(world), physical_repair_count=0,
                    reviewed_continuation_waits=reviewed_waits, corrections=copy.deepcopy(reviewed_waits),
                    proposal_intervention_count=proposal_interventions,
                    proposal_state_rebased_stages=rebased_stages)

    def solve(self, *, max_evaluations=None, guidance=None, initial_selection=None,
              solver_parameters=None, threads=1, seed_selections=None, adaptive_seed_completion=False,
              stop_on_background_conflict=False):
        """Return a caller-owned master and checked metadata; dispose on errors.

        Valid sparse warm starts are completed within their one-hot groups and
        hard fixes, then checked first by the exact transition, budget permitting.
        They are only search seeds, never assumed feasible or used as bounds.
        """
        model = mip.Model("metro_joint_command_master")
        try:
            return self._solve(model, max_evaluations=max_evaluations, guidance=guidance,
                               initial_selection=initial_selection, solver_parameters=solver_parameters,
                               threads=threads, seed_selections=seed_selections,
                               adaptive_seed_completion=adaptive_seed_completion,
                               stop_on_background_conflict=stop_on_background_conflict)
        except BaseException:
            model.dispose()
            raise

    def _solve(self, model, *, max_evaluations, guidance, initial_selection,
               solver_parameters, threads, seed_selections, adaptive_seed_completion,
               stop_on_background_conflict):
        model.Params.OutputFlag = 0
        model.Params.Threads = threads
        allowed_parameters = {"MIPFocus", "Heuristics", "RINS", "Seed", "MIPGap"}
        for name, value in (solver_parameters or {}).items():
            if name not in allowed_parameters or type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Unsupported solver parameter: {name}")
            model.setParam(name, value)
        eta = model.addVar(lb=0., name="dispatch_cost")
        variables = {(g, i): model.addVar(vtype=mip.BINARY, name=f"command_{gi}_{i}")
                     for gi, (g, opts) in enumerate(self.groups.items()) for i in range(len(opts))}
        for gi, (g, options) in enumerate(self.groups.items()):
            model.addConstr(model.sum(variables[g, i] for i in range(len(options))) == 1,
                            name=f"one_option_{gi}")
        # Only structural, provably safe exclusions are compiled here. Cross-
        # stage vehicle/resource contention is decided by the exact trajectory.
        claims = {}
        for (g, i), var in variables.items():
            for raw in self.groups[g][i].commands:
                c = DispatchCommand.from_dict(raw, self.base.tick_seconds)
                for resource in [*(('vehicle', key) for key in c.participants()), *c.resources()]:
                    claims.setdefault((c.effective_at, resource), []).append((g, i))
        for index, indices in enumerate(claims.values()):
            if len(indices) > 1:
                model.addConstr(model.sum(variables[g, i] for g, i in indices) <= 1,
                                name=f"simultaneous_claim_{index}")
        model.update()
        diagnostics, warm_values = [], {}
        for kind, attr in (("warm_start", "Start"), ("branching_priority", "BranchPriority"), ("variable_fixing", None)):
            entries = (guidance or {}).get(kind, {})
            if not isinstance(entries, dict):
                diagnostics.append(dict(kind=kind, applied=False, reason="guidance_needs_a_variable_mapping"))
                continue
            for name, value in entries.items():
                var = next((v for v in variables.values() if v.name == name), None) if model.NumVars else None
                valid = (type(value) in (int, float) and math.isfinite(value) and value >= 0
                         and (value == int(value) and value <= 2000000000 if attr == "BranchPriority" else value in (0, 1)))
                if var is None or not valid:
                    diagnostics.append(dict(kind=kind, variable=name, applied=False))
                elif attr:
                    if attr == "Start":
                        model.set_start(var, int(value))
                        warm_values[var.name] = int(value)
                    else:
                        model.set_priority(var, int(value))
                    diagnostics.append(dict(kind=kind, variable=name, applied=True, restricts_domain=False))
                elif value in (0, 1):
                    var.lb = var.ub = value
                    diagnostics.append(dict(kind=kind, variable=name, applied=True, restricts_domain=True))
        model.setObjective(eta, mip.MINIMIZE)
        model.update()
        lower, termination = 0., "time_limit"
        detected_blockers = set()
        reference = tuple(next((i for i, op in enumerate(options) if not op.commands and not op.actions and not op.pauses), 0)
                          for options in self.groups.values())
        if initial_selection is not None:
            reference = tuple(initial_selection)
            if len(reference) != len(self.groups) or any(type(i) is not int or not 0 <= i < len(opts)
                    for i, opts in zip(reference, self.groups.values())):
                raise ValueError("Initial selection must contain one valid index per decision group")
        reference_respects_fixes = all(v.lb <= float(reference[list(self.groups).index(g)] == i) <= v.ub
                                      for (g, i), v in variables.items())
        seeds, seed_records = [], []
        if warm_values:
            # One-hot implications complete sparse starts, never a teacher's
            # incumbent or evaluated-candidate cache. Fixes remain hard bounds.
            trial, specified, completed = [], {}, []
            record = dict(source="guidance_warm_start", warm_start=dict(warm_values),
                          status="pending", completion="reference_then_first_fix_compatible")
            for gi, (group, options) in enumerate(self.groups.items()):
                hints = {i: warm_values[v.name] for (g, i), v in variables.items()
                         if g == group and v.name in warm_values}
                hinted = [i for i in range(len(options))
                          if all(int(i == j) == value for j, value in hints.items())]
                compatible = [i for i in hinted if all(
                    variables[group, j].lb <= int(i == j) <= variables[group, j].ub
                    for j in range(len(options)))]
                if not compatible:
                    record.update(status="rejected", group=group, reason=(
                        "warm_start_conflicts_with_fixes" if hinted else "inconsistent_warm_start"))
                    break
                if hints and len(hinted) == 1:
                    specified[group] = hinted[0]
                else:
                    completed.append(group)
                trial.append(reference[gi] if reference[gi] in compatible else compatible[0])
            else:
                record.update(selection=list(trial), partial=bool(completed),
                              specified_assignment=specified, completed_groups=completed)
                seeds.append((tuple(trial), record))
            seed_records.append(record)
        for trial in seed_selections or []:
            trial = tuple(trial)
            if len(trial) != len(self.groups) or any(type(i) is not int or not 0 <= i < len(opts)
                    for i, opts in zip(trial, self.groups.values())):
                raise ValueError("Seed selection must contain one valid index per decision group")
            if any(sum(trial[list(self.groups).index(g)] == i for g, i in claim) > 1
                   for claim in claims.values()):
                seed_records.append(dict(source="seed_selections", selection=list(trial),
                    status="rejected", reason="master_simultaneous_claim_conflict"))
                continue
            if all(v.lb <= float(trial[list(self.groups).index(g)] == i) <= v.ub
                   for (g, i), v in variables.items()) and all(trial != s for s, _ in seeds):
                record = dict(source="seed_selections", selection=list(trial), status="pending")
                seeds.append((trial, record))
                seed_records.append(record)
        from MILP.parallel_evaluation import worker_count
        imported_assignments = set(self.evaluated)
        parallel_assignments = set()
        if worker_count() != 1 and seeds:
            trials = [trial for trial, _ in seeds if trial not in self.evaluated]
            if max_evaluations is not None:
                trials = trials[:max_evaluations]
            checked = self.evaluate_many(trials)
            self.evaluated.update(checked)
            parallel_assignments = set(checked)
            for trial, record in seeds:
                if trial in checked:
                    value = checked[trial]
                    record.update(status="evaluated_parallel", feasible=value["feasible"])
                    record.update({"cost": value["cost"]} if value["feasible"] else {"reason": value["reason"]})
                    if stop_on_background_conflict:
                        detected_blockers.update(background_blockers(self, value))
        seed_trials = len(parallel_assignments)
        cached_references = cached_feasible = 0
        for cut_index, (assignment, value) in enumerate(list(self.evaluated.items())):
            compatible = all(v.lb <= int(assignment[list(self.groups).index(g)] == i) <= v.ub
                             for (g, i), v in variables.items())
            if not compatible:
                del self.evaluated[assignment]
                continue
            if (value.get("evaluation_catalog") != self.catalog_fingerprint
                    or value.get("evaluation_selection") != assignment):
                raise ValueError("Uncertified reference cache")
            if any(sum(assignment[list(self.groups).index(g)] == i for g, i in claim) > 1
                   for claim in claims.values()):
                del self.evaluated[assignment]
                continue
            selected = model.sum(variables[g, assignment[i]] for i, g in enumerate(self.groups))
            if value["feasible"]:
                model.addConstr(eta >= value["cost"]*(selected-len(self.groups)+1),
                                name=f"shared_reference_cost_{cut_index}")
                if self.best is None or value["cost"] < self.best["cost"]:
                    self.best, self.best_selection = value, assignment
                cached_feasible += int(assignment in imported_assignments)
            else:
                model.addConstr(selected <= len(self.groups)-1,
                                name=f"shared_no_good_{cut_index}")
            cached_references += int(assignment in imported_assignments)
            if assignment in parallel_assignments:
                self.cuts.append(dict(type="exact_cost" if value["feasible"] else "no_good",
                    selection=list(assignment), source="parallel_checked_seed",
                    **({"cost": value["cost"]} if value["feasible"] else {"reason": value["reason"]})))
        cached_assignments = imported_assignments & set(self.evaluated)
        model.update()
        incumbent_history = []
        def record_incumbent():
            if self.best is not None and (not incumbent_history or self.best["cost"] < incumbent_history[-1]["cost"]-1e-6):
                incumbent_history.append(dict(cost=self.best["cost"],
                    physical_evaluations=len(self.evaluated)-cached_references,
                    elapsed_s=time.perf_counter()-self.started))
        record_incumbent()
        first, first_record = reference, None
        first_respects_fixes = reference_respects_fixes
        if seeds and seeds[0][1]["source"] == "guidance_warm_start":
            first, first_record = seeds.pop(0)
            first_respects_fixes = True
            if reference_respects_fixes and first != reference and all(reference != s for s, _ in seeds):
                record = dict(source="reference", selection=list(reference), status="pending")
                seeds.insert(0, (reference, record))
                seed_records.append(record)
        if (time.perf_counter() < self.deadline and max_evaluations != 0
                and first_respects_fixes and first not in self.evaluated):
            if first_record is not None:
                first_record["status"] = "checking"
                seed_trials += 1
            try:
                value = self.evaluate(first)
                if first_record is not None and value["feasible"]:
                    indices = dict(zip(self.groups, first))
                    if any(sum(indices[g] == i for g, i in claim) > 1 for claim in claims.values()):
                        # A physical rollout cannot admit a seed excluded by
                        # the unchanged master's structural safety constraints.
                        value = dict(value, feasible=False, reason="master_simultaneous_claim_conflict")
                if first_record is not None:
                    first_record.update(status="evaluated", feasible=value["feasible"])
                    first_record.update({"cost": value["cost"]} if value["feasible"] else {"reason": value["reason"]})
                self.evaluated[first] = value
                if stop_on_background_conflict:
                    detected_blockers.update(background_blockers(self, value))
                selected = model.sum(variables[g, first[i]] for i, g in enumerate(self.groups))
                prefix = "guidance_seed" if first_record is not None else "reference"
                if value["feasible"]:
                    if self.best is None or value["cost"] < self.best["cost"]:
                        self.best, self.best_selection = value, first
                        record_incumbent()
                    model.addConstr(eta >= value["cost"] * (selected - len(self.groups) + 1), name=f"{prefix}_cost")
                    self.cuts.append(dict(type="exact_cost", selection=list(first), cost=value["cost"]))
                else:
                    model.addConstr(selected <= len(self.groups) - 1, name=f"{prefix}_no_good")
                    self.cuts.append(dict(type="no_good", selection=list(first), reason=value["reason"]))
                model.update()
            except EvaluationTimeout:
                if first_record is not None:
                    first_record["status"] = "evaluation_timeout"
        while not detected_blockers and time.perf_counter() < self.deadline:
            if max_evaluations is not None and len(self.evaluated)-cached_references >= max_evaluations:
                termination = "evaluation_limit"
                break
            model.Params.TimeLimit = max(1e-6, self.deadline - time.perf_counter())
            started = time.perf_counter()
            try:
                model.optimize()
            finally:
                self.timings["cplex_master_search_seconds"] += time.perf_counter() - started
            if model.Status == mip.INFEASIBLE:
                termination = "domain_exhausted"
                break
            if model.Status not in (mip.OPTIMAL, mip.TIME_LIMIT, mip.INTERRUPTED, mip.SUBOPTIMAL):
                termination = "master_error"
                break
            lower = max(0., float(model.ObjBound)) if math.isfinite(model.ObjBound) else 0.
            if not model.SolCount:
                break
            selection = tuple(max(range(len(opts)), key=lambda i: variables[g, i].solution_value)
                              for g, opts in self.groups.items())
            # Bounds always come from the unchanged CPLEX master. Seed trials only
            # choose which exact physical subproblem to evaluate next.
            while seeds and seeds[0][0] in self.evaluated:
                _, record = seeds.pop(0)
                record.update(status="already_evaluated", feasible=self.evaluated[tuple(record["selection"])]["feasible"])
            seed_record = None
            if seeds and not (self.best and lower >= self.best["cost"] - 1e-6):
                selection, seed_record = seeds.pop(0)
                seed_trials += 1
                seed_record["status"] = "checking"
            if selection in self.evaluated:
                if self.best and lower >= self.best["cost"] - 1e-6:
                    termination = "optimality_certificate"
                else:
                    termination = "master_limit"
                break
            try:
                value = self.evaluate(selection)
            except EvaluationTimeout:
                if seed_record is not None:
                    seed_record["status"] = "evaluation_timeout"
                break
            if seed_record is not None:
                seed_record.update(status="evaluated", feasible=value["feasible"])
                if value["feasible"]:
                    seed_record["cost"] = value["cost"]
                else:
                    seed_record["reason"] = value["reason"]
            self.evaluated[selection] = value
            if stop_on_background_conflict:
                detected_blockers.update(background_blockers(self, value))
            selected = model.sum(variables[g, selection[i]] for i, g in enumerate(self.groups))
            if value["feasible"]:
                if self.best is None or value["cost"] < self.best["cost"]:
                    self.best, self.best_selection = value, selection
                    record_incumbent()
                    if adaptive_seed_completion:
                        additions = []
                        for original, _ in seeds:
                            trial = tuple(selection[j] if v == reference[j] else v
                                          for j, v in enumerate(original))
                            if trial in self.evaluated or trial == original or any(trial == s for s, _ in additions):
                                continue
                            if any(sum(trial[list(self.groups).index(g)] == i for g, i in claim) > 1
                                   for claim in claims.values()):
                                continue
                            record = dict(source="incumbent_neighborhood", selection=list(trial), status="pending",
                                          anchor_cost=value["cost"])
                            additions.append((trial, record))
                            seed_records.append(record)
                            if len(additions) == 2:
                                break
                        seeds[:0] = additions
                model.addConstr(eta >= value["cost"] * (selected - len(self.groups) + 1),
                                name=f"exact_cost_{len(self.evaluated)}")
                self.cuts.append(dict(type="exact_cost", selection=list(selection), cost=value["cost"]))
            else:
                model.addConstr(selected <= len(self.groups) - 1, name=f"no_good_{len(self.evaluated)}")
                self.cuts.append(dict(type="no_good", selection=list(selection), reason=value["reason"]))
            model.update()
        if detected_blockers:
            termination = "background_conflict_for_scope_repair"
        for record in seed_records:
            if record["status"] == "pending":
                trial = tuple(record["selection"])
                if trial in self.evaluated:
                    record.update(status="already_evaluated", feasible=self.evaluated[trial]["feasible"])
                else:
                    record.update(status="not_evaluated", reason=termination)
        optimal = self.best is not None and termination in ("optimality_certificate", "domain_exhausted")
        if optimal:
            lower = self.best["cost"]
        status = ("optimal_in_domain" if optimal else "feasible_limit") if self.best else (
            "infeasible_in_domain" if termination == "domain_exhausted" else "no_incumbent_limit")
        upper = self.best["cost"] if self.best else None
        if upper is not None:
            lower = min(lower, upper)
        metadata = dict(status=status, termination=termination, milp_incumbent=self.best is not None,
            objective=-upper if upper is not None else None, cost=upper, cost_lower_bound=lower,
            cost_upper_bound=upper, gap=(upper - lower) / max(1., abs(upper)) if upper is not None else None,
            certificate_scope="declared finite command catalog, applied guidance fixes, current published inputs and fixed continuations",
            domain_size=self.domain_size, catalog_fingerprint=self.catalog_fingerprint,
            reviewed_continuation=self.reviewed_continuation, continuation_mode=self.continuation_mode,
            evaluations=len(self.evaluated), physical_evaluations=len(self.evaluated)-cached_references,
            novel_candidate_evaluations=sum(key != reference for key in self.evaluated if key not in cached_assignments),
            feasible_candidate_evaluations=sum(bool(v.get("feasible")) for v in self.evaluated.values())-cached_feasible,
            infeasible_candidate_reasons=dict(Counter(v.get("reason", "unspecified")
                for v in self.evaluated.values() if not v.get("feasible"))),
            shared_reference_evaluations=cached_references, cuts=self.cuts,
            seed_trials=seed_trials, seed_records=seed_records,
            blocking_background_vehicles=sorted(detected_blockers),
            incumbent_history=incumbent_history,
            selected_options={g: self.groups[g][self.best_selection[i]].id for i, g in enumerate(self.groups)} if self.best else {},
            candidate_catalog={g: [asdict(o) for o in options] for g, options in self.groups.items()},
            focus_vehicles=sorted(self.focus), nonfocus_vehicles=sorted(set(self.base.dispatch.records) - self.focus),
            fixed_continuation_source={name: "supplied" if name in self.fixed_paths else "current_route_with_timed_operations_and_safety_review"
                                      for name in set(self.base.dispatch.records) - self.focus},
            objective_definition=self.base.dispatch.objective.manifest(), guidance_diagnostics=diagnostics,
            solver_parameters=dict(solver_parameters or {}), solver_threads=threads,
            solver_backend="cplex", solver_parameter_mapping=model.parameter_mapping,
            cplex_status=model.cplex_status,
            solver_version=model.native_model.get_cplex().get_version(),
            event_inputs=dict(supported_event_families=sorted(EVENT_TARGETS), published_events=self.base.get_published_events(),
                              restriction_continuation="hold active published restrictions; do not read actual future clearance",
                              demand_input="current declared intensity and OD shares, or explicitly supplied conditional arrivals"),
            passenger_inputs=copy.deepcopy(self.inputs), oracle="MetroWorldV1.step + DispatchOperations",
            timings=dict(self.timings), construction_reused=self.construction_reused,
            parallel_candidate_evaluations=list(self.parallel_evaluation_stats),
            candidate_workers=worker_count(),
            construction_timings=dict(self.construction_timings),
            solve_time=time.perf_counter() - self.started, first_stage_requires_apply=True)
        if self.best:
            metadata.update(dispatch_packets=self.best["packets"], trajectory=self.best["trajectory"],
                step_rewards=[-s["total"] / max(1, len(self.base.agents)) for s in self.best["steps"]],
                network_step_rewards=[-s["total"] for s in self.best["steps"]], cost_components=self.best["steps"],
                terminal_cost=self.best["terminal"], all_step_actions=[p["actions"] for p in self.best["packets"]],
                all_step_targets=[p["expected_positions"] for p in self.best["packets"]],
                terminal_guard=self.best["terminal_guard"], physical_repair_count=0,
                reviewed_continuation_waits=self.best.get("reviewed_continuation_waits", []),
                proposal_state_rebased_stages=self.best.get("proposal_state_rebased_stages", []),
                corrections=self.best.get("corrections", []))
        return model, metadata


def apply_dispatch_step(world, solution, stage=0):
    """Commit one checked stage, retaining actual started operations across solves."""
    packets = solution.get("dispatch_packets", [])
    if not (solution.get("milp_incumbent") or solution.get("checked_dispatch")) or not 0 <= stage < len(packets):
        return dict(accepted=False, reason="no_checked_dispatch_incumbent")
    packet = packets[stage]
    if not guard_matches(packet["guard"], state_guard(world)):
        return dict(accepted=False, reason="operating_state_changed_replan_required")
    if getattr(world, "_dispatch_stage_targets", None) is not None:
        return dict(accepted=False, reason="previous_stage_not_consumed")
    commands = copy.deepcopy(packet["commands"])
    if commands:
        plan_id = "commit_" + hashlib.sha256(json.dumps(commands, sort_keys=True).encode()).hexdigest()[:16]
        result = world.submit_dispatch_plan(dict(plan_id=plan_id, created_at=float(world.sim_time[0]),
            snapshot_basis="published", snapshot_version=world.dispatch_snapshot()["version"], commands=commands))
        if not result["accepted"]:
            return result
    set_actions(world, packet["actions"])
    world._dispatch_stage_targets = copy.deepcopy(packet["targets"])
    world._dispatch_expected_positions = copy.deepcopy(packet["expected_positions"])
    return dict(accepted=True, reason=None, actions=dict(packet["actions"]),
                expected_positions=dict(packet["expected_positions"]))


def audit_dispatch_step(world, solution, stage=0):
    packets = solution.get("dispatch_packets", [])
    if not 0 <= stage < len(packets):
        return dict(matched=False, reason="no_checked_dispatch_stage")
    packet = packets[stage]
    actual = state_guard(world)
    mismatches = [key for key in set(packet["expected_guard"]) | set(actual)
                  if key not in packet["expected_guard"] or key not in actual
                  or not guard_matches(packet["expected_guard"][key], actual[key])]
    trace_match = (packet.get("physical_audits") is None or
                   guard_matches(packet["physical_audits"], world.last_physical_audits))
    if not trace_match:
        mismatches.append("physical_execution")
    return dict(matched=not mismatches, state_mismatches=sorted(mismatches),
                physical_trace_match=trace_match,
                execution_deviations=getattr(world, "last_execution_deviations", []),
                positions_match=positions(world) == packet["expected_positions"],
                constraint_interventions=[name for name, item in world.last_transition_audit.items() if item["conflict"]],
                reason="compare against the declared planning inputs; changed demand or event reports require replanning")
