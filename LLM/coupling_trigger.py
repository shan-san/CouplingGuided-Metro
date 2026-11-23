"""Untrained, evidence-producing checks of shared dispatch decisions.

Only the published planning projection is used. Conditional reference rollouts
are diagnostics, not predictions, optimality certificates or safety filters.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from itertools import combinations

from MILP.operation_model import JointDispatchModel, planning_copy, nominal_action, set_actions, apply_dispatch_step
from dispatch_plan import DispatchCommand
from dispatch_candidates import applicable_options


@dataclass
class CouplingThresholds:
    horizon: int = 3
    service_gap_persons: float = 20.
    persistence_stages: int = 2
    cooldown_stages: int = 10

    def __post_init__(self):
        if (type(self.horizon) is not int or self.horizon < 1
                or type(self.persistence_stages) is not int or self.persistence_stages < 1
                or type(self.cooldown_stages) is not int or self.cooldown_stages < 0
                or not math.isfinite(self.service_gap_persons) or self.service_gap_persons < 0):
            raise ValueError("Invalid coupling trigger thresholds")


def option_footprint(world, option, group=None):
    """Declared paths/participants/OD effects, without assigning numeric IDs meaning."""
    footprint = dict(vehicles=set(option.actions) | set(option.pauses), nodes=set(),
                     depots=set(), tasks=set(), platforms=set(), families=set(), resources=set(), avoids=set())
    if group in world.dispatch.records:
        footprint["vehicles"].add(group)
    for raw in option.commands:
        c = DispatchCommand.from_dict(raw, float(world.tick_seconds))
        footprint["vehicles"] |= c.participants()
        footprint["families"].add(c.action)
        p = c.parameters
        if "depot" in p:
            footprint["depots"].add(p["depot"])
        if "service_id" in p:
            service = world.dispatch.services.get(p["service_id"],{})
            footprint["tasks"].add(f"{p['service_id']}@{c.vehicle_id}" if service.get("pattern") else p["service_id"])
        if "platform_id" in p:
            footprint["platforms"].add(p["platform_id"])
            footprint["nodes"].add(p["platform_id"])
        if c.vehicle_id:
            record = world.dispatch.records[c.vehicle_id]
            pos = world.dispatch.position(c.vehicle_id)
            footprint["nodes"].add(pos)
            if pos in world.platform_nodes:
                footprint["platforms"].add(pos)
            if record.depot:
                footprint["depots"].add(record.depot)
            for key in ("path_node_ids", "approach_node_ids", "tow_node_ids"):
                footprint["nodes"].update(p.get(key, []))
            target = p.get("target_node")
            if "segment_id" in p:
                segment = world._segment_id_map.get(p["segment_id"])
                if segment:
                    target = segment.end_platform.id
            if "depot" in p and target is None:
                spec = world.dispatch.depots[p["depot"]]
                target = spec["node_id"]
            if target is not None:
                path = (world.dispatch.available_path(pos, target, allow_depot=p.get("depot", record.depot),
                            vehicle_id=c.vehicle_id, view=world.event_view(published=True), explicit=p.get("path_node_ids"))
                        if c.action == "withdraw" else
                        world.dispatch.path(pos, target, allow_depot=p.get("depot", record.depot)))
                footprint["nodes"].update(path or [])
            if c.action in ("reserve_insert", "reenter") and record.depot:
                spec = world.dispatch.depots[record.depot]
                target = p.get("exit_node_id", spec["exit_node_id"])
                footprint["nodes"].update(world.dispatch.available_path(pos, target, allow_depot=record.depot,
                    vehicle_id=c.vehicle_id, view=world.event_view(published=True)) or [])
            if c.action == "rescue_tow":
                failed_position = world.dispatch.position(p["failed"])
                footprint["nodes"].update(world.dispatch.path(pos, failed_position, allow_depot=record.depot) or [])
                footprint["nodes"].update(world.dispatch.path(failed_position,
                    world.dispatch.depots[p["depot"]]["node_id"], allow_depot=p["depot"]) or [])
            if c.action == "reassign_service":
                route = world.dispatch.service_route(p["service_id"], pos)
                stops, path = route if route else ([], [])
                footprint["platforms"].update(stops)
                footprint["nodes"].update(path)
                if world.dispatch.services[p["service_id"]].get("pattern"):
                    footprint["avoids"].update(world.dispatch.section_need(p["service_id"], pos))
    for name in list(footprint["vehicles"]):
        record = world.dispatch.records[name]
        footprint["nodes"].add(world.dispatch.position(name))
        if record.depot:
            footprint["depots"].add(record.depot)
        operation = world.dispatch.operation_for(name)
        if operation:
            footprint["vehicles"] |= operation.participants()
            footprint["nodes"].update(operation.path)
            footprint["nodes"].update(operation.continuation)
            if operation.depot:
                footprint["depots"].add(operation.depot)
    for kind in ("turnbacks", "shared_routes"):
        for resource, spec in world.event_resources.get(kind, {}).items():
            nodes = set(spec.get("interior_nodes", ()))
            nodes.update(int(node) for segment in spec["segments"] for node in segment.split("-"))
            if footprint["nodes"] & nodes:
                footprint["resources"].add((kind, resource))
    footprint["platforms"] |= footprint["nodes"] & set(world.platform_nodes)
    return footprint


def event_relevant(event, footprint, world):
    if event.get("event_id") in footprint.get("avoids", ()):
        return True
    kind, ident = event["target"]["kind"], event["target"]["id"]
    if kind == "vehicle":
        return ident in footprint["vehicles"]
    if kind == "depot":
        return ident in footprint["depots"]
    if kind in ("platform", "passenger_origin"):
        return ident in footprint["platforms"]
    if kind == "segment":
        segment = world._segment_id_map.get(str(ident))
        return bool(segment and footprint["nodes"] & {n.id for n in world.segment2nodes[segment]})
    specs = world.event_resources.get("switches" if kind == "switch" else "turnbacks", {})
    resource = specs.get(ident, {})
    if kind == "switch":
        entries = resource.get("route_entry_nodes", {})
        nodes = {entries.get(str(rid), resource.get("node_id"))
                 for rid in event["parameters"]["disabled_route_ids"]}
        return bool(nodes & footprint["nodes"])
    return any(event_relevant(dict(target=dict(kind="segment", id=s)), footprint, world)
               for s in resource.get("segments", []))


def option_footprints(model):
    """Read-only footprints for an immutable published-state command catalog."""
    cached = getattr(model, "_footprint_cache", None)
    if cached is None or cached[0] != model.catalog_fingerprint:
        cached = (model.catalog_fingerprint, {g: [option_footprint(model.base, op, g)
                  for op in options] for g, options in model.groups.items()})
        model._footprint_cache = cached
    return cached[1]


def _union(footprints):
    keys = ("vehicles", "nodes", "depots", "tasks", "platforms", "families", "resources")
    return {key: set().union(*(f[key] for f in footprints)) for key in keys}


class CouplingTrigger:
    def __init__(self, thresholds=None):
        self.thresholds = thresholds or CouplingThresholds()
        self.reset()

    def reset(self):
        self.streaks = {}
        self.last_stage = None
        self.last_result = None

    def inspect(self, world, actions=None, *, candidates=None, cached_packet=None, reference_solution=None):
        cfg = self.thresholds
        from dispatch_timing import decision_seconds
        stage = int(round(float(world.sim_time[0]) / decision_seconds(world)))
        if self.last_stage is not None and stage < self.last_stage:
            self.reset()
        # Repeated queries at the same physical stage never advance persistence.
        previous_streaks = self.streaks if self.last_stage != stage else getattr(self, "stage_start_streaks", {})
        if self.last_stage is not None and stage > self.last_stage + 1:
            previous_streaks = {}
        self.stage_start_streaks = dict(previous_streaks)
        model = JointDispatchModel(world, cfg.horizon, candidates=candidates, defer_reference=True)
        base = model.base
        published = base.get_published_events()
        reference = copy.deepcopy(base)
        conflicts, visited = [], {a.name: {base.dispatch.position(a.name)} for a in base.agents}
        initial = {p: {d: float(q[0]) + float(n.passenger_flow.outside_by_destination.get(d, q.new_zeros(q.shape))[0])
                       for d, q in n.passenger_flow.waiting_by_destination.items()}
                   for p, n in base.platform_nodes.items()}
        for p, n in base.platform_nodes.items():
            for d, q in n.passenger_flow.outside_by_destination.items():
                initial[p].setdefault(d, float(q[0]))
        remaining = copy.deepcopy(initial)
        for k in range(cfg.horizon):
            requests = {a.name: int((actions or {}).get(a.name, nominal_action(reference, a))) if k == 0
                        else nominal_action(reference, a) for a in reference.agents}
            if k == 0 and reference_solution:
                commit = apply_dispatch_step(reference, reference_solution)
                if not commit["accepted"]:
                    raise ValueError("Native reference changed before coupling inspection")
                requests = reference_solution["dispatch_packets"][0]["actions"]
            set_actions(reference, requests)
            # No arrivals in this diagnostic: follow service for passengers already
            # present; no future demand generator or private future data is queried.
            reference.step(arrival_inputs={})
            trace = reference.last_physical_audits if reference.second_precision else [reference.last_transition_audit]
            for second, frame in enumerate(trace):
                for name, audit in frame.items():
                    visited[name].update((audit["origin"], audit["actual"]))
                    if audit["conflict"]:
                        conflicts.append(dict(stage=k, second=second, vehicle=name, origin=audit["origin"], requested=audit["requested"]))
            for p, node in reference.platform_nodes.items():
                flow = node.passenger_flow
                remaining[p] = {}
                for d in set(flow.waiting_by_destination) | set(flow.outside_by_destination):
                    inside = float(flow.waiting_by_destination.get(d, reference.sim_time.new_zeros(1))[0])
                    outside = float(flow.outside_by_destination.get(d, reference.sim_time.new_zeros(1))[0])
                    # No new arrivals were injected. Newly cleared passengers
                    # are current unfinished journeys, not newly forecast demand.
                    remaining[p][d] = inside + outside
        footprints = {}
        families = set()
        masks, rejected = applicable_options(model)
        for group, options in model.groups.items():
            fps = [option_footprint(base, op, group) for op, valid in zip(options, masks[group]) if valid]
            footprints[group] = _union(fps)
            if group in visited:
                footprints[group]["vehicles"].add(group)
            families |= footprints[group]["families"]
            for name in footprints[group]["vehicles"]:
                footprints[group]["nodes"] |= visited.get(name, set())
            footprints[group]["platforms"] |= footprints[group]["nodes"] & set(base.platform_nodes)
        gaps = {p: sum(values.values()) for p, values in remaining.items() if sum(values.values()) > 1e-6}
        stranded = {a.name: float(a.passenger_count[0]) for a in reference.agents
                    if not reference.dispatch.service_enabled(a) and float(a.passenger_count[0]) > 1e-6}
        evidence = []
        for left, right in combinations(footprints, 2):
            a, b = footprints[left], footprints[right]
            resources = sorted(a["nodes"] & b["nodes"])
            shared_resources = sorted(a["resources"] & b["resources"])
            vehicles = sorted(a["vehicles"] & b["vehicles"])
            depots, tasks = sorted(a["depots"] & b["depots"]), sorted(a["tasks"] & b["tasks"])
            affected = sorted((a["platforms"] | b["platforms"]) & set(gaps))
            common_service = sorted(a["platforms"] & b["platforms"] & set(gaps))
            relations = []
            if resources or shared_resources:
                relations.append("C1")
                if len(a["nodes"]) > 1 or len(b["nodes"]) > 1:
                    relations.append("C4")
            if vehicles or depots or tasks:
                relations.append("C2")
            if common_service:
                relations.append("C3")
            if not relations:
                continue
            key = f"{left}|{right}"
            deficit = sum(gaps[p] for p in affected) + sum(stranded.get(v, 0.) for v in a["vehicles"] | b["vehicles"])
            conflicting = any(c["vehicle"] in a["vehicles"] | b["vehicles"] for c in conflicts)
            current_conflicting = any(c["stage"] == 0 and c["vehicle"] in a["vehicles"] | b["vehicles"]
                                      for c in conflicts)
            work_blocked = bool((vehicles or depots or tasks) and any(
                base.dispatch.records[v].failed or base.dispatch.records[v].needs_inspection
                for v in a["vehicles"] | b["vehicles"]))
            local_nodes = set().union(*(visited.get(v, {base.dispatch.position(v)})
                                       for v in a["vehicles"] | b["vehicles"]))
            local_nodes.update(int(g.removeprefix("platform_")) for g in (left, right)
                               if g.startswith("platform_"))
            local_platforms = local_nodes & set(base.platform_nodes)
            local_footprint = dict(nodes=local_nodes, platforms=local_platforms,
                vehicles=a["vehicles"] | b["vehicles"], depots={base.dispatch.records[v].depot
                    for v in a["vehicles"] | b["vehicles"]} - {None})
            local_gap = sum(gaps.get(p, 0.) for p in local_platforms) + sum(
                stranded.get(v, 0.) for v in a["vehicles"] | b["vehicles"])
            pressure = local_gap > cfg.service_gap_persons or conflicting or work_blocked
            streak = previous_streaks.get(key, 0) + 1 if pressure else 0
            self.streaks[key] = streak
            evidence.append(dict(groups=[left, right], relations=sorted(set(relations)),
                vehicles=sorted(a["vehicles"] | b["vehicles"]), nodes=resources,
                shared_resources=shared_resources,
                shared_vehicles=vehicles, depots=depots, tasks=tasks, affected_platforms=affected,
                control_platforms=sorted(((set(resources) & local_nodes) | {int(g.removeprefix("platform_"))
                    for g in (left, right) if g.startswith("platform_")}) & set(base.platform_nodes)),
                reference_window_platforms=sorted(local_platforms),
                local_passenger_gap=local_gap,
                local_event_ids=[e["event_id"] for e in published if e.get("active", True)
                    and event_relevant(e, local_footprint, base)],
                local_information_ids=[e["event_id"] for e in published if event_relevant(e, local_footprint, base)],
                action_families=sorted(a["families"] | b["families"]),
                current_passenger_gap=deficit, reference_conflict=conflicting,
                current_reference_conflict=current_conflicting,
                work_blocked=work_blocked,
                affected_event_ids=[e["event_id"] for e in published if e.get("active", True)
                    and (event_relevant(e, a, base) or event_relevant(e, b, base))],
                persistent_stages=streak, active=pressure and streak >= cfg.persistence_stages))
        keys = {"|".join(e["groups"]) for e in evidence}
        self.streaks = {key: value for key, value in self.streaks.items() if key in keys}
        # A future conflict in a conditional continuation is pressure for
        # reassessment, not proof that the present executable stage is unsafe.
        invalid = any(c["stage"] == 0 for c in conflicts)
        cache_reason = None
        if cached_packet:
            # A changed clock/queue alone is not proof of infeasibility. Check
            # the scheduled command preconditions and requested movement itself.
            trial = planning_copy(world)
            try:
                commands = cached_packet.get("commands", [])
                if commands:
                    accepted = trial.submit_dispatch_plan(dict(plan_id="cache_feasibility_review",
                        snapshot_basis="published", snapshot_version=trial.dispatch_snapshot()["version"],
                        created_at=float(trial.sim_time[0]), commands=commands))
                    if not accepted["accepted"]:
                        raise ValueError(accepted["reason"])
                set_actions(trial, cached_packet["actions"])
                trial.step(arrival_inputs=cached_packet.get("arrival_inputs", {}), dispatch_targets=cached_packet.get("targets", {}))
                if any(row["conflict"] for row in trial.last_transition_audit.values()) or any(
                        result["status"] in ("failed", "rejected") for command in commands
                        for result in [trial.dispatch.command_results[command["command_id"]]]):
                    raise ValueError("cached_continuation_no_longer_feasible")
            except (ValueError, KeyError) as exc:
                invalid, cache_reason = True, str(exc)
        active = [e for e in evidence if e["active"]]
        result = dict(signals=dict(PLAN_INVALID=invalid, COUPLED_SERVICE=bool(active)),
            mandatory_reassessment=invalid, evidence=evidence, reference_conflicts=conflicts,
            cached_invalid_reason=cache_reason, current_od_gap=remaining, stranded_onboard=stranded,
            focus_vehicles=sorted({v for e in active for v in e["vehicles"]} | {c["vehicle"] for c in conflicts}),
            focus_nodes=sorted({n for e in active for n in e["nodes"]}),
            focus_platforms=sorted({p for e in active for p in e["affected_platforms"]}),
            events=published, event_ids=[e["event_id"] for e in published], candidate_rejections=rejected,
            covered_action_families=sorted(families),
            basis="shared candidate resources, vehicle/tasks and current OD service under published inputs",
            limits="conservative spatial candidate overlap; not a time-conflict or improvement certificate",
            thresholds=vars(cfg).copy(), conditional_service_horizon=cfg.horizon)
        self.last_stage, self.last_result = stage, result
        return result
