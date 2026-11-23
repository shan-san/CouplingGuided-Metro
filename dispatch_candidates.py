"""Current-state command masks; joint resource checks remain separate."""
import copy
import hashlib
import json

from dispatch_plan import DispatchCommand
from dispatch_timing import timing_manifest


SERVICE_REVIEW_MEASURES = frozenset({"reserve_insert", "reenter", "rescue_tow", "inspect_repair",
                                     "reassign_service", "short_turn", "withdraw"})


def public_dispatch_state(world, coupling=None):
    return dict(n_agents=len(world.agents), n_nodes=len(world.nodes), n_platforms=len(world.platform_nodes), timing=timing_manifest(world),
        agent_positions=[dict(node_id=int(a.state.position[0]), position=int(a.state.position[0]),
            at_platform=int(a.state.position[0]) in world.platform_nodes, dwell_time=float(a.state.dwell_time[0]),
            expected_dwell=float(a.state.dwell_min_limit[0]), passengers=float(a.passenger_count[0]),
            capacity=world.event_view(published=True).vehicle_capacity(a.name, a.passenger_capacity)) for a in world.agents],
        waiting_passengers={str(p): world.waiting_at(p) for p in world.platform_nodes},
        disruptions=world.get_published_events(), dispatch_state=world.dispatch_snapshot(),
        queues={str(p): dict(inside={str(d): float(q[0]) for d, q in n.passenger_flow.waiting_by_destination.items()},
                            outside={str(d): float(q[0]) for d, q in n.passenger_flow.outside_by_destination.items()})
                for p, n in world.platform_nodes.items()}, coupling=coupling or {})


def applicable_options(model):
    cached = getattr(model, "_applicability_cache", None)
    if cached is not None and cached[0] == model.catalog_fingerprint:
        return copy.deepcopy(cached[1:])
    mask, reasons = {}, {}
    for group, options in model.groups.items():
        mask[group], reasons[group] = [], []
        for option in options:
            if not option.commands:
                held = {name for name, sequence in option.actions.items() if sequence[0] == 0
                        and model.base.dispatch.operation_for(name) is None}
                held |= {name for name, stages in option.pauses.items() if 0 in stages}
                legal = all(model.base.dispatch.discretionary_hold_allowed(name) for name in held)
                mask[group].append(legal)
                reasons[group].append(None if legal else "discretionary_hold_requires_safe_platform")
                continue
            probe = model.base.dispatch.applicability_copy()
            error = None
            try:
                for raw in option.commands:
                    c = DispatchCommand.from_dict(raw, model.base.tick_seconds)
                    if c.effective_at == model.now:
                        probe._start_now(c)
            except (ValueError, KeyError) as exc:
                error = str(exc)
            mask[group].append(error is None)
            reasons[group].append(error)
    model._applicability_cache = (model.catalog_fingerprint, mask, reasons)
    return copy.deepcopy((mask, reasons))


def stock_start_options(model, masks):
    """Order eligible single-stock trials by the current directed access path.

    This only orders physical trials, not variable bounds or objective terms.
    One exit per stock owner is exposed before its alternate exit choices.
    """
    ranked = []
    for gi, (group, options) in enumerate(model.groups.items()):
        for oi, option in enumerate(options):
            if not masks[group][oi] or len(option.commands) != 1:
                continue
            command = option.commands[0]
            if command["action"] not in ("reserve_insert", "reenter"):
                continue
            vehicle = model.base.dispatch.records[command["vehicle_id"]]
            if vehicle.depot is None:
                continue
            depot = model.base.dispatch.depots[vehicle.depot]
            target = command["parameters"].get("exit_node_id", depot["exit_node_id"])
            path = model.base.dispatch.available_path(model.base.dispatch.position(vehicle.id), target,
                allow_depot=vehicle.depot, vehicle_id=vehicle.id)
            if path is not None:
                ranked.append((len(path), vehicle.id, target, gi, oi))
    ranked.sort()
    first, rest, owners = [], [], set()
    for _, owner, _, gi, oi in ranked:
        (rest if owner in owners else first).append((gi, oi))
        owners.add(owner)
    return first + rest


def checked_stage_fingerprint(solution):
    """Detect edits to an internally checked one-stage trajectory before reuse."""
    payload = {key: solution.get(key) for key in ("cost", "dispatch_packets", "passenger_inputs",
                                               "reference_paths", "reference_actions")}
    payload["arrival_profiles"] = [getattr(row, "profile", None) for row in solution.get("passenger_inputs", [])]
    payload["packet_arrival_profiles"] = [getattr(packet.get("arrival_inputs"), "profile", None)
                                         for packet in solution.get("dispatch_packets", [])]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def solution_from_evaluation(model, value, selection, status="checked_policy"):
    if not value.get("feasible"):
        raise ValueError("Cannot execute an infeasible candidate selection")
    result = dict(checked_dispatch=True, milp_incumbent=False, status=status, cost=value["cost"],
                dispatch_packets=value["packets"], passenger_inputs=model.inputs,
                selected_options={g: model.groups[g][selection[i]].id for i, g in enumerate(model.groups)},
                reference_paths={name: [model.base.dispatch.position(name)] +
                    [p["expected_positions"][name] for p in value["packets"]] for name in model.base.dispatch.records},
                reference_actions={name: [p["actions"][name] for p in value["packets"]] for name in model.base.dispatch.records},
                corrections=copy.deepcopy(value.get("corrections", [])),
                continuation_source="checked native policy commands and committed operations",
                continuation_mode=getattr(model, "continuation_mode", "strict"),
                reviewed_continuation_waits=value.get("reviewed_continuation_waits", []),
                proposal_state_rebased_stages=value.get("proposal_state_rebased_stages", []))
    if model.horizon == 1:
        result["checked_stage_fingerprint"] = checked_stage_fingerprint(result)
    return result


def repair_joint_selection(model, selection, rejected, *, max_rounds=8):
    """Filter a proposal without changing the PPO action or its probability.

    Only the shared transition certifies a repair. Unaffected groups keep their
    proposal; implicated groups return to a checked route/operation continuation.
    No safety constraint is relaxed and no operation is cancelled by the filter.
    """
    if max_rounds < 1:
        raise ValueError("Joint repair needs a positive round limit")
    groups = list(model.groups)
    baseline = tuple(next((i for i, op in enumerate(model.groups[g])
                          if not op.commands and not op.actions and not op.pauses), 0) for g in groups)
    reference = model.evaluate(baseline)
    if not reference["feasible"]:
        baseline = tuple(next((i for i, op in enumerate(model.groups[g])
                               if op.id in ("defer_movement", "wait_window")), baseline[j])
                         for j, g in enumerate(groups))
        reference = model.evaluate(baseline)
    if not reference["feasible"]:
        return None, dict(repaired_groups=groups, repair_attempts=0,
                          repair_reasons=[rejected.get("reason")], used_full_reference=True)
    current, value, reasons = list(selection), rejected, []
    attempts = 0
    for _ in range(max_rounds):
        if value["feasible"]:
            break
        reasons.append(value.get("reason", "unspecified"))
        vehicles = set(value.get("vehicles", [])) | {value.get("vehicle")}
        command_ids = set(value.get("details", {}))
        implicated = []
        for index, g in enumerate(groups):
            op = model.groups[g][current[index]]
            participants = {g} | set(op.actions) | set(op.pauses)
            for command in op.commands:
                participants |= DispatchCommand.from_dict(command, model.base.tick_seconds).participants()
            if current[index] != baseline[index] and (participants & vehicles or any(
                    c["command_id"] in command_ids for c in op.commands)):
                implicated.append(index)
        if not implicated:
            implicated = [i for i in range(len(groups)) if current[i] != baseline[i]]
        if not implicated:
            break
        for index in implicated:
            current[index] = baseline[index]
        attempts += 1
        value = model.evaluate(tuple(current))
    if not value["feasible"]:
        current, value = list(baseline), reference
    changed = [g for i, g in enumerate(groups) if selection[i] != current[i]]
    solution = solution_from_evaluation(model, value, tuple(current), status="checked_repaired_policy")
    solution["corrections"] = [dict(kind="joint_proposal_repair", group=g,
        proposed_option=model.groups[g][selection[i]].id,
        executed_option=model.groups[g][current[i]].id) for i, g in enumerate(groups) if g in changed]
    return solution, dict(repaired_groups=changed, repair_attempts=attempts,
        repair_reasons=copy.deepcopy(reasons), used_full_reference=tuple(current) == baseline)
