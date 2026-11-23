"""Focused decisions against fixed, physically checked background operations."""
from __future__ import annotations

from collections import deque
import json
import math

from dispatch_plan import DispatchCommand
from MILP.operation_model import background_blockers, planning_copy


class CouplingRequestQueue:
    """Age active local requests without merging their whole-line OD footprints."""
    def __init__(self):
        self.regions = {}
        self.event_versions = {}

    @staticmethod
    def region_key(row):
        platforms = row.get("control_platforms", row.get("nodes", []))
        if platforms:
            return "platforms:" + ",".join(map(str, sorted(set(platforms))))
        if row.get("depots") or row.get("tasks"):
            return json.dumps([sorted(row.get("depots", [])), sorted(row.get("tasks", []))])
        return "vehicles:" + ",".join(sorted(row.get("vehicles", []) or row.get("groups", [])))

    def update(self, coupling, stage, *, cooldown, service_gap):
        versions = {e["event_id"]: json.dumps({k: e[k] for k in (
            "type", "target", "parameters", "active", "information_version", "predicted_end_seconds") if k in e},
            sort_keys=True) for e in coupling.get("events", [])}
        changed = {k for k in set(versions) | set(self.event_versions)
                   if versions.get(k) != self.event_versions.get(k)}
        self.event_versions = versions
        active = []
        for row in coupling.get("evidence", []):
            key = self.region_key(row)
            entry = self.regions.setdefault(key, dict(last_request=-math.inf, pending_since=None,
                last_seen=stage, pressure=None, urgent=None))
            entry["last_seen"] = stage
            if row.get("active") and entry["pending_since"] is None:
                entry["pending_since"] = stage
            pressure = float(row.get("local_passenger_gap", row.get("current_passenger_gap", 0.)))
            related = set(row.get("local_information_ids", row.get("local_event_ids", row.get("affected_event_ids", []))))
            if related & changed:
                entry["urgent"] = "published_event_update"
            baseline = entry["pressure"]
            if baseline is not None and pressure-baseline >= max(1., service_gap) and pressure >= 1.5*baseline:
                entry["urgent"] = "material_service_deterioration"
            if not row.get("active"):
                continue
            urgent = "current_reference_conflict" if row.get("current_reference_conflict") else entry["urgent"]
            eligible = bool(urgent or stage-entry["last_request"] >= cooldown)
            age = stage-entry["pending_since"]
            row.update(request_region=key, pending_age_stages=age, request_eligible=eligible,
                       request_urgent=urgent)
            rank = (bool(row.get("current_reference_conflict")), age//max(1, cooldown), bool(urgent),
                    bool(row.get("work_blocked")), bool(row.get("reference_conflict")), bool(related), pressure,
                    int(row.get("persistent_stages", 0)))
            active.append((rank, tuple(row.get("groups", [])), row))
        self.regions = {k: v for k, v in self.regions.items()
                        if stage-v["last_seen"] <= max(10, 4*cooldown)}
        active_keys = {item[2]["request_region"] for item in active}
        for key, entry in self.regions.items():
            if key not in active_keys:
                entry["pending_since"] = None
        eligible = [item for item in active if item[2]["request_eligible"]]
        chosen = max(eligible, key=lambda item: (item[0], item[1]))[2] if eligible else None
        coupling["scheduled_relation_groups"] = chosen["groups"] if chosen else []
        return dict(region=chosen.get("request_region") if chosen else None,
                    urgent=chosen.get("request_urgent") if chosen else None,
                    pending_age_stages=chosen.get("pending_age_stages", 0) if chosen else 0,
                    active_regions=len({row[2]["request_region"] for row in active}),
                    eligible_regions=len({row[2]["request_region"] for row in eligible}),
                    basis="regional cooldown, published updates, service deterioration and pending age")

    def mark_requested(self, coupling, stage):
        groups = coupling.get("scheduled_relation_groups", [])
        chosen = next((r for r in coupling.get("evidence", []) if r.get("groups") == groups), None)
        if chosen is not None:
            entry = self.regions[chosen["request_region"]]
            entry.update(last_request=stage, pending_since=None, urgent=None,
                         pressure=float(chosen.get("local_passenger_gap", chosen.get("current_passenger_gap", 0.))))


def _vehicles(world, values):
    names = [a.name for a in world.agents]
    result = set()
    for value in values:
        if type(value) is int and 0 <= value < len(names):
            value = names[value]
        if value not in names:
            raise ValueError(f"Unknown focus vehicle: {value}")
        result.add(value)
    return result


def select_focus(world, specification, coupling):
    """LLM scope ranks active relations; it does not override mandatory partners.

    Selecting one relation is an untrained scheduling heuristic, not a claim
    that other relations are independent. Their operations remain fixed inputs.
    """
    config = specification.get("solver_config", {})
    proposed = _vehicles(world, specification.get("focus_agents", config.get("focus_agents", [])) or [])
    nodes = set(specification.get("focus_nodes", config.get("focus_nodes", [])) or [])
    if not nodes <= set(world.nodes):
        raise ValueError("Unknown focus facility")
    active = [row for row in coupling.get("evidence", []) if row.get("active")]

    def priority(row):
        vehicles = set(row.get("vehicles", []))
        return (bool(row.get("reference_conflict")), bool(row.get("work_blocked")),
                bool(row.get("local_event_ids", row.get("affected_event_ids"))),
                float(row.get("local_passenger_gap", row.get("current_passenger_gap", 0))),
                len(vehicles & proposed), len(nodes & set(row.get("affected_platforms", []))),
                int(row.get("persistent_stages", 0)))

    active.sort(key=lambda row: tuple(str(g) for g in row.get("groups", [])))
    scheduled = coupling.get("scheduled_relation_groups")
    preferred = [row for row in active if row.get("groups") == scheduled] if scheduled else active
    anchor = max(preferred, key=priority) if preferred else None
    mandatory = _vehicles(world, [row["vehicle"] for row in coupling.get("reference_conflicts", [])])
    if anchor is not None:
        seed = _vehicles(world, anchor.get("vehicles", [])) | mandatory
        # OD/service footprints may span the entire line. They belong in the
        # objective, not automatically in the free platform-control domain.
        platforms = set(anchor.get("control_platforms", anchor.get("nodes", []))) & set(world.platform_nodes)
        for group in anchor.get("groups", []):
            if group in world.dispatch.records:
                seed.add(group)
            elif group.startswith("platform_"):
                platforms.add(int(group.removeprefix("platform_")))
        reason = "highest_priority_active_relation_with_llm_scope_preference"
    else:
        seed = proposed | mandatory
        platforms = nodes & set(world.platform_nodes)
        reason = "explicit_scope_without_active_relation"
    return sorted(seed), sorted(platforms), dict(policy="active_relation_with_fixed_background",
        reason=reason, llm_vehicles=sorted(proposed), llm_nodes=sorted(nodes),
        trigger_vehicles=sorted(set(coupling.get("focus_vehicles", []))),
        mandatory_vehicles=sorted(mandatory), anchor_groups=anchor.get("groups", []) if anchor else [],
        passenger_impact_platforms=sorted(anchor.get("affected_platforms", [])) if anchor else [],
        ranking_basis="reference-window events and queues before broad service footprints",
        seed_vehicles=sorted(seed), seed_platforms=sorted(platforms),
        request_region=anchor.get("request_region") if anchor else None,
        pending_age_stages=anchor.get("pending_age_stages", 0) if anchor else 0,
        deferred_active_relations=max(0, len(active)-int(anchor is not None)),
        global_scope_requested=bool(proposed) and len(proposed)==len(world.agents))


def blocking_background_vehicles(model):
    """Only actual failed candidate checks justify releasing a background train."""
    blockers = set()
    for value in model.evaluated.values():
        blockers.update(background_blockers(model, value))
    return sorted(blockers)


def select_experiment_focus(world, specification, coupling, *, mode="coupling", nearest_vehicles=6):
    """Scope variants change free decisions, never the checked background."""
    if mode == "coupling":
        return select_focus(world, specification, coupling)
    names = sorted(a.name for a in world.agents)
    if mode == "full_fleet":
        return names, sorted(world.platform_nodes), dict(policy="full_fleet", llm_scope_ignored=True,
            seed_vehicles=names, seed_platforms=sorted(world.platform_nodes))
    if mode != "nearest" or type(nearest_vehicles) is not int or nearest_vehicles < 1:
        raise ValueError("Invalid nearest-vehicle scope")
    seed, platforms, anchor = select_focus(world, {}, coupling)
    nodes = set(platforms) | {world.dispatch.position(v) for v in seed}
    if not nodes:
        if world.platform_nodes:
            nodes = {min(world.platform_nodes, key=lambda p: (-world.waiting_at(p), p))}
        elif names:
            nodes = {world.dispatch.position(names[0])}
    # Undirected topology hops define proximity only; permissions and shared
    # capacity remain in the common directed physical feasibility model.
    adjacency = {node: set() for node in world.nodes}
    for node, record in world.nodes.items():
        for target in (record.getnext() or {}).values():
            if target in adjacency:
                adjacency[node].add(target)
                adjacency[target].add(node)
    distance = {node: 0 for node in nodes}
    queue = deque(sorted(nodes))
    while queue:
        node = queue.popleft()
        for target in adjacency.get(node, ()):
            if target not in distance:
                distance[target] = distance[node]+1
                queue.append(target)
    selected = sorted(names, key=lambda v: (distance.get(world.dispatch.position(v), float("inf")), v))[:nearest_vehicles]
    return selected, sorted(nodes & set(world.platform_nodes)), dict(policy="nearest_topology_hops",
        nearest_vehicles=nearest_vehicles, anchor_nodes=sorted(nodes), anchor_groups=anchor.get("anchor_groups", []),
        seed_vehicles=selected, llm_scope_ignored=True,
        closure_rule="mandatory partners and executable service alternatives may expand the seed")


def _candidate_groups(world, candidates):
    result = {}
    for group, options in (candidates or {}).items():
        vehicles, platforms, tasks = set(), set(), set()
        if group in world.dispatch.records:
            vehicles.add(group)
        for option in options:
            vehicles |= _vehicles(world, set(option.get("actions", {})) | set(option.get("pauses", {})))
            for raw in option.get("commands", []):
                command = dict(raw)
                stage, offset = command.pop("stage", 0), command.pop("offset_seconds", 0)
                command.setdefault("command_id", "scope_review")
                command.setdefault("effective_at", float(world.sim_time[0]) + stage*world.decision_seconds + offset)
                parsed = DispatchCommand.from_dict(command, world.tick_seconds)
                vehicles |= _vehicles(world, parsed.participants())
                if parsed.vehicle_id is None:
                    platforms.add(parsed.parameters["platform_id"])
                if parsed.action == "reassign_service":
                    sid=parsed.parameters["service_id"]
                    tasks.add(f"{sid}@{parsed.vehicle_id}" if world.dispatch.services[sid].get("pattern") else sid)
        if not platforms <= set(world.platform_nodes):
            raise ValueError("Unknown platform-control facility")
        result[group] = vehicles, platforms, tasks
    return result


def scoped_candidates(world, candidates, focus, platforms):
    if candidates is None:
        return None
    dependencies = _candidate_groups(world, candidates)
    selected = {g: options for g, options in candidates.items()
                if dependencies[g][0] & set(focus) or dependencies[g][1] & set(platforms)}
    return selected or {"continuation": [dict(id="continue", commands=[])]}


def close_reference_platform_users(world, closure, reference_history, *, candidates=None, horizon=1,
                                   reference_commands=(), planning_base=None):
    """A free headway/admission control affects all actual window users.

    Use checked per-second visits, not entire possible service paths. Each new
    participant still undergoes the ordinary vehicle/task/operation closure.
    """
    initial = set(closure["resolved"])
    controlled = set(closure["focus_platforms"])
    dependencies = _candidate_groups(world, candidates)
    while True:
        users = {name for stage in reference_history for frame in stage for name, node in frame.items()
                 if node in controlled}
        users |= {name for name in world.dispatch.records if world.dispatch.position(name) in controlled}
        requested = set(closure["resolved"]) | users
        if requested == set(closure["resolved"]):
            break
        _, expanded = close_focus(world, sorted(requested), closure["focus_platforms"], candidates, horizon,
                                  reference_commands=reference_commands, planning_base=planning_base)
        controlled |= {p for vehicles, platforms, _ in dependencies.values()
                       if vehicles & set(expanded["resolved"]) for p in platforms}
        # Freeing a visiting train does not also free its remote platform
        # controls, which would recursively turn a local request into a fleet solve.
        expanded["focus_platforms"] = sorted(controlled)
        expanded["selection"] = closure.get("selection", {})
        closure = expanded
    closure["reference_window_users_added"] = sorted(set(closure["resolved"]) - initial)
    return closure


def close_reference_motion_users(world, closure, reference_history, *, proposed=(), maximum=12,
                                 candidates=None, horizon=1, reference_commands=(), planning_base=None):
    """Close shared *executed window* paths before constructing SLM variables.

    Depot inventories and OD/service footprints are not motion conflicts. The
    bounded overlap component proposes extra decision owners, not a feasibility
    certificate: the unchanged joint oracle still checks every second.
    """
    if type(maximum) is not int or maximum < 1:
        raise ValueError("Window scope maximum must be a positive integer")
    initial = set(closure["resolved"])
    depots = {d["node_id"] for d in world.dispatch.depots.values()}
    resource_nodes = {}
    for kind in ("turnbacks", "shared_routes"):
        for identifier, resource in getattr(world, "event_resources", {}).get(kind, {}).items():
            for node in resource.get("interior_nodes", []):
                resource_nodes.setdefault(node, set()).add(f"{kind}:{identifier}")
    visits = {v: {} for v in world.dispatch.records}
    for stage, frames in enumerate(reference_history):
        for second, frame in enumerate(frames):
            for vehicle, node in frame.items():
                if node not in depots:
                    when = stage*world.decision_seconds+second
                    visits[vehicle].setdefault(("node", node), when)
                    for resource in resource_nodes.get(node, ()):
                        visits[vehicle].setdefault(("resource", resource), when)
    selected, pending = set(initial), set(initial)
    suggested = set(proposed) & set(visits)
    evidence, omitted = [], set()
    while pending:
        boundary = set().union(*(set(visits[v]) for v in pending))
        neighbors = {v for v in visits if v not in selected and boundary.intersection(visits[v])}
        ordered = sorted(neighbors, key=lambda v: (v not in suggested,
            min(visits[v][n] for n in boundary.intersection(visits[v])), v))
        added = ordered[:max(0, maximum-len(selected))]
        omitted.update(set(ordered)-set(added))
        if not added:
            break
        evidence.extend(dict(vehicle=v,
            shared_nodes=sorted(n for kind, n in boundary.intersection(visits[v]) if kind == "node"),
            shared_resources=sorted(n for kind, n in boundary.intersection(visits[v]) if kind == "resource"),
            llm_suggested=v in suggested) for v in added)
        selected.update(added)
        pending = set(added)
    result = closure
    if selected != initial:
        _, result = close_focus(world, sorted(selected), closure["focus_platforms"], candidates, horizon,
            reference_commands=reference_commands, planning_base=planning_base)
        result["selection"] = closure.get("selection", {})
        # Do not recursively free distant station controls along new services.
        result["focus_platforms"] = list(closure["focus_platforms"])
    result["reference_motion_closure"] = dict(added=sorted(set(result["resolved"])-initial),
        maximum=maximum, omitted=sorted(omitted-set(result["resolved"])), evidence=evidence,
        basis="shared actual reference-window nodes and physical resources, excluding depot storage and OD footprints")
    return result


def rebind_scope_guidance(guidance, before, after):
    """Keep soft hints only for identical choices in the same physical window."""
    if (before.base is not after.base or before.horizon != after.horizon
            or guidance.get("catalog_fingerprint") != before.catalog_fingerprint):
        return {}, dict(reason="physical_window_or_catalog_mismatch", retained=0)

    def catalog(model):
        return {f"command_{gi}_{oi}": json.dumps(dict(group=group, option=option.id,
                    commands=option.commands, actions=option.actions, pauses=option.pauses), sort_keys=True)
                for gi, (group, options) in enumerate(model.groups.items()) for oi, option in enumerate(options)}

    old, new = catalog(before), catalog(after)
    names = {}
    for variable, identity in new.items():
        names.setdefault(identity, []).append(variable)
    result = {}
    for field in ("warm_start", "branching_priority"):
        result[field] = {names[old[v]][0]: value for v, value in guidance.get(field, {}).items()
                         if v in old and len(names.get(old[v], [])) == 1}
    result["catalog_fingerprint"] = after.catalog_fingerprint
    return result, dict(reason="exact_group_measure_and_parameters_match", retained=sum(
        len(result[k]) for k in ("warm_start", "branching_priority")),
        hard_fixes_released=len(guidance.get("variable_fixing", {})))


def close_focus(world, requested, nodes=(), candidates=None, horizon=1, *, reference_commands=(), planning_base=None):
    """Close real decision dependencies, not all possible spatial interactions.

    Background trains are not removed: the caller fixes their checked reference
    trajectories. Shared capacity or overlapping candidate routes alone need
    not make another train a decision variable. Physical feasibility is still
    checked jointly by JointDispatchModel before a plan can be accepted.
    """
    names = [a.name for a in world.agents]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("A positive scope horizon is required")
    focus = _vehicles(world, requested)
    if not set(nodes) <= set(world.nodes):
        raise ValueError("Unknown focus facility")
    base = planning_copy(world) if planning_base is None else planning_base
    platforms = set(nodes) & set(base.platform_nodes)
    reasons, task_choices, additions, candidate_tasks = set(), {}, {}, set()
    dependencies = _candidate_groups(base, candidates)

    def include(vehicles, reason):
        added = set(vehicles)-focus
        if added:
            focus.update(added)
            reasons.add(reason)
            additions.setdefault(reason, set()).update(added)
        return bool(added)

    def applicable(vehicle, action, parameters):
        command = DispatchCommand("scope_eligibility", action, vehicle, float(base.sim_time[0]), parameters)
        try:
            base.dispatch.applicability_copy()._start_now(command)
            return True
        except (ValueError, KeyError):
            return False

    # A task alternative is relevant only at its actual origin, with the same
    # vehicle/route/operation preconditions as execution. Far-away vehicles are
    # not stage-zero reassignment alternatives in the generated command domain.
    changed = True
    while changed:
        before = set(focus), set(platforms), set(candidate_tasks)
        platforms |= {base.dispatch.position(v) for v in focus} & set(base.platform_nodes)
        for vehicle_set, platform_set, task_set in dependencies.values():
            if vehicle_set & focus or platform_set & platforms or task_set & candidate_tasks:
                include(vehicle_set, "custom_bundle_or_service_alternatives")
                platforms |= platform_set
                candidate_tasks |= task_set
        for op in base.dispatch.operations.values():
            if op.participants() & focus:
                include(op.participants(), "committed_operation_participants")
        for raw in reference_commands:
            command = DispatchCommand.from_dict(raw, base.tick_seconds)
            if command.participants() & focus:
                include(command.participants(), "marl_reference_participants")
        for plan in base.dispatch.plans.values():
            for command in plan.commands:
                status = base.dispatch.command_results.get(command.command_id, {}).get("status")
                if status in ("pending", "running") and command.participants() & focus:
                    include(command.participants(), "pending_operation_participants")
        for sid, task in base.dispatch.services.items():
            if (task.get("vehicle_id") is not None or task.get("reserved_vehicle_id") is not None
                    or task.get("completed") or task.get("ended_at") is not None):
                continue
            origins = set(task["stops"][:-1]) if task.get("pattern") else {task["stops"][0]}
            if not origins & platforms:
                continue
            if sid not in task_choices:
                task_choices[sid] = [v for v in names if base.dispatch.position(v) in origins & platforms
                    and applicable(v, "reassign_service", dict(service_id=sid))]
            include(task_choices[sid], "eligible_local_service_alternatives")
        depots = {base.dispatch.records[v].depot for v in focus}-{None}
        for vehicle in names:
            record = base.dispatch.records[vehicle]
            if record.depot not in depots or vehicle in focus:
                continue
            if any(applicable(vehicle, action, dict(exit_node_id=exit_node))
                   for action in ("reserve_insert", "reenter")
                   for exit_node in base.dispatch.depots[record.depot].get("exit_node_ids",
                       [base.dispatch.depots[record.depot]["exit_node_id"]])):
                include({vehicle}, "qualified_same_depot_alternative")
        if candidates is None:
            # Match _generate(): rescuers' candidate bundles also control the
            # failed partner, even if a later applicability check rejects them.
            rescuers = [v for v in focus if base.dispatch.operation_for(v) is None]
            if rescuers:
                failed = {v for v, r in base.dispatch.records.items() if r.failed or any(
                    e["parameters"]["mode"]=="immobilized" for e in base.event_view().active("vehicle_breakdown", "vehicle", v))}
                include({v for v in failed if base.dispatch.records[v].depot is None
                         and any(rescuer!=v for rescuer in rescuers)}, "generated_rescue_participants")
        changed = before != (focus, platforms, candidate_tasks)
    return sorted(names.index(v) for v in focus), dict(requested=list(requested), resolved=sorted(focus),
        nonfocus=sorted(set(names)-focus), focus_platforms=sorted(platforms),
        reasons=sorted(reasons), additions={k:sorted(v) for k,v in additions.items()},
        eligible_service_alternatives=task_choices, policy="dependencies_with_fixed_background",
        safety_basis="fixed nonfocus continuations and joint physical feasibility checks",
        conservative=False)
