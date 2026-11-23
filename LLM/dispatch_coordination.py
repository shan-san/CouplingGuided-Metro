"""Native command coordination shared by the existing LLM policies.

Uses MILPSolver and the original world transition; it is not another simulator
or scheduling model. All timing is synchronous and cooperatively budgeted.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import time
from types import SimpleNamespace

import torch

from MILP.operation_model import (
    JointDispatchModel, apply_dispatch_step, audit_dispatch_step, guard_matches, nominal_action,
    planning_copy, positions, set_actions, state_guard, EvaluationTimeout,
    passenger_total,
)
from dispatch_timing import decision_seconds
from LLM.dispatch_scope import (CouplingRequestQueue, blocking_background_vehicles, close_focus,
    close_reference_platform_users, close_reference_motion_users, rebind_scope_guidance,
    scoped_candidates, select_experiment_focus)
from LLM.experiment_controls import experiment_configuration, validate_experiment_controls

logger = logging.getLogger(__name__)


class CoordinationBudgetExpired(TimeoutError):
    pass


def exception_record(stage, exc):
    return dict(stage=stage, error=str(exc) or repr(exc), error_type=type(exc).__name__)


def policy_entropy(logits, mask=None):
    """Maximum normalized per-agent entropy, independent of fleet size."""
    if logits is None:
        return 0.
    values = torch.as_tensor(logits, dtype=torch.float64).reshape(-1, logits.shape[-1])
    allowed = torch.isfinite(values) if mask is None else torch.as_tensor(mask, dtype=torch.bool).reshape(values.shape)
    result = []
    for row, legal in zip(values, allowed):
        row = row[legal]
        if row.numel() <= 1:
            result.append(0.)
            continue
        if not torch.isfinite(row).all():
            raise ValueError("Nonfinite policy logits on available actions")
        probs = row.softmax(-1)
        result.append(float(-(probs * probs.clamp_min(1e-300).log()).sum() / math.log(len(row))))
    return max(result, default=0.)


def action_mask(world, width):
    mask = torch.zeros(len(world.agents), width, dtype=torch.bool)
    mask[:, 0] = True
    visible = world.dispatch_snapshot()["vehicles"]
    for index, agent in enumerate(world.agents):
        state = visible[agent.name]
        if state["failed"] or state["service_id"] is None or world.dispatch.operation_for(agent.name):
            continue
        origin = int(agent.state.position[0])
        if origin in world.platform_nodes and float(agent.state.dwell_time[0]) < float(agent.state.dwell_min_limit[0]):
            continue
        choices = sorted((world.nodes[origin].getnext() or {}).items(), key=lambda item: str(item[0]))
        for label, (_, target) in enumerate(choices, 1):
            if label < width:
                mask[index, label] = world.edge_event_allowed(agent, origin, target, view=world.event_view(published=True))
    return mask


def event_signature(world):
    return json.dumps(world.get_published_events(), sort_keys=True, allow_nan=False)


def retrieved_specification(context):
    cases = context.get("cases", [])
    if not cases:
        return {}
    case = cases[0]
    native = case.get("action_plan", {}).get("specification")
    if isinstance(native, dict):
        return dict(solver_config=dict(time_horizon=native["planning_horizon"],
            time_limit=native["computational_budget"]), scope_rule="current_coupling_component")
    if isinstance(case.get("specification"), dict):
        return copy.deepcopy(case["specification"])
    plan = case.get("action_plan", {}).get("selected_plan", {})
    if not isinstance(plan, dict):
        return {}
    # Scope IDs still undergo current-world validation; no source-line action
    # or legacy variable is copied into the current dispatch decision.
    return {"solver_config": {k: copy.deepcopy(plan[k]) for k in (
        "focus_agents", "focus_nodes", "time_horizon", "time_limit") if k in plan}}


def coordination_horizon(world, config, focus, base, *, reference_solution=None,
                         remaining_stages=None, passenger_inputs=None):
    from MILP.collection_protocol import choose_horizon
    cap = min(remaining_stages, len(passenger_inputs)) if remaining_stages is not None and passenger_inputs is not None else (
        len(passenger_inputs) if passenger_inputs is not None else remaining_stages)
    if getattr(config, "fixed_planning_horizon", False) or getattr(config, "fixed_comparison_horizon", None) is not None:
        fixed = getattr(config, "fixed_comparison_horizon", config.milp_time_horizon)
        return dict(horizon=min(fixed, cap) if cap is not None else fixed, censored=cap is not None and cap < fixed,
                    reasons=["fixed_experiment_horizon"], requested_stages=fixed)
    if getattr(config, "coordination_horizon_policy", "adaptive") == "fixed":
        return dict(horizon=min(base, cap) if cap is not None else base, censored=cap is not None and cap < base,
                    reasons=["fixed_horizon_policy"], requested_stages=base)
    options = dict(base=base, maximum=getattr(config, "coordination_max_horizon", 24),
        tail_stages=getattr(config, "coordination_service_tail", 3), remaining_stages=cap,
        recovery_work_only=True, focus_vehicles=focus)
    info = choose_horizon(world, **options)
    commands = (reference_solution or {}).get("dispatch_packets", [{}])[0].get("commands", [])
    if commands:
        from dispatch_plan import DispatchCommand
        preview = planning_copy(world)
        try:
            for raw in commands:
                parsed = DispatchCommand.from_dict(raw, preview.tick_seconds)
                if raw["effective_at"] == float(world.sim_time[0]) and parsed.participants() & set(focus):
                    preview.dispatch._start_now(parsed)
            proposed = choose_horizon(preview, **options)
            if proposed["horizon"] > info["horizon"]:
                info = proposed
                info["reasons"].append("current_qualified_proposal_duration_not_a_feasibility_certificate")
        except (ValueError, KeyError) as exc:
            info["reasons"].append("proposal_duration_unavailable:" + type(exc).__name__)
    return info


def checked_reference(world, first_actions, horizon, passenger_inputs=None, first_solution=None, *, deadline=math.inf,
                      reference_model=None):
    """Review a reference continuation, preserving all already-started work.

    Fresh policy actions are used at stage zero; None selects the published
    route continuation at every stage. Later actions follow the current
    route, not a claim of future MARL inference. Conflicting committed movements
    may wait without cancelling their operation. Every correction is logged.
    """
    if first_actions is not None and any(type(a) is not int or not 0 <= a <= 3 for a in first_actions.values()):
        raise ValueError("Reference actions must use legal primitive labels")
    if horizon == 1 and reference_model is None and first_solution and first_solution.get("checked_dispatch"):
        from dispatch_candidates import checked_stage_fingerprint
        packets = first_solution.get("dispatch_packets", [])
        packet = packets[0] if len(packets) == 1 else None
        expected_inputs = passenger_inputs
        if expected_inputs is None:
            view = world.event_view(published=True)
            expected_inputs = [{p: n.passenger_flow.interval_arrivals(view, decision_seconds(world))
                                for p, n in world.platform_nodes.items()}]
        if (packet and packet["actions"] == first_actions and packet["guard"] == state_guard(world)
                and first_solution.get("checked_stage_fingerprint") == checked_stage_fingerprint(first_solution)
                and expected_inputs == [packet["arrival_inputs"]]
                and getattr(expected_inputs[0], "profile", None) == getattr(packet["arrival_inputs"], "profile", None)
                and packet.get("physical_audits") is not None
                and not any(row["conflict"] for frame in packet["physical_audits"] for row in frame.values())):
            return dict(first_solution, reference_reused=True)
    # The prepared base is immutable. Each stage still executes on a private
    # copy, so using it as the initial checkpoint avoids a redundant full clone.
    clone = reference_model.base if reference_model is not None else planning_copy(world)
    current = {p: n.passenger_flow.interval_arrivals(clone.event_view(), decision_seconds(clone))
               for p, n in clone.platform_nodes.items()}
    inputs = copy.deepcopy(passenger_inputs) if passenger_inputs is not None else [copy.deepcopy(current) for _ in range(horizon)]
    if len(inputs) != horizon:
        raise ValueError("Declared arrivals must contain H intervals")
    if reference_model is not None:
        reference_model.check_reference_source(world, inputs)
    for interval in inputs:
        if not isinstance(interval, dict) or not set(interval) <= set(clone.platform_nodes):
            raise ValueError("Unknown declared passenger origin")
        for origin, values in interval.items():
            if not isinstance(values, dict) or any(destination not in clone.platform_nodes or destination == origin
                    or type(count) not in (int, float) or not math.isfinite(count) or count < 0
                    for destination, count in values.items()):
                raise ValueError("Invalid declared passenger destination or count")
    packets, costs, corrections = [], [], []
    requests, history, trajectory = [], [], [clone.dispatch.snapshot(published=True)]
    total = passenger_total(clone) + sum(float(a.cumulative_alighted[0]) for a in clone.agents)
    arrivals_total = 0.
    paths = {name: [node] for name, node in positions(clone).items()}
    sequences = {a.name: [] for a in clone.agents}
    for stage in range(horizon):
        if time.perf_counter() >= deadline:
            raise EvaluationTimeout()
        actions = {a.name: int(first_actions.get(a.name, 0)) if stage == 0 and first_actions is not None else nominal_action(clone, a)
                   for a in clone.agents}
        guard = state_guard(clone)
        native_packet = first_solution["dispatch_packets"][0] if first_solution and stage == 0 else None
        targets = {}
        for _ in range(len(actions) + 1):
            trial = copy.deepcopy(clone)
            trial._planning_deadline = deadline
            if native_packet:
                if not apply_dispatch_step(trial, first_solution)["accepted"]:
                    raise RuntimeError("Native MARL reference changed before review")
                actions = dict(native_packet["actions"])
            set_actions(trial, actions)
            trial.step(arrival_inputs=inputs[stage], dispatch_targets=None if native_packet else targets)
            audits = trial.last_physical_audits if trial.second_precision else [trial.last_transition_audit]
            conflicts = sorted({name for frame in audits for name, item in frame.items() if item["conflict"]})
            if not conflicts:
                break
            if native_packet:
                raise RuntimeError("Native MARL reference conflicts under the declared passenger inputs")
            changed = False
            for name in conflicts:
                if actions[name] != 0:
                    corrections.append(dict(stage=stage, vehicle=name, original_action=actions[name], replacement=0))
                    actions[name] = 0
                    changed = True
                # Active operations override primitive HOLD in pre_step().
                # Defer their physical motion explicitly, including both towing
                # participants, while retaining the operation and remaining path.
                op = trial.dispatch.operation_for(name)
                participants = op.participants() if op else {name}
                for participant in sorted(participants):
                    origin = clone.dispatch.position(participant)
                    if targets.get(participant) != origin:
                        targets[participant] = origin
                        corrections.append(dict(stage=stage, vehicle=participant, kind="defer_movement",
                            original_target=trial.last_transition_audit[participant]["requested"],
                            replacement_target=origin, operation_id=op.id if op else None))
                        changed = True
            if not changed:
                raise RuntimeError("No checked fallback: ongoing operations require joint replanning")
        else:
            raise RuntimeError("Reference review did not converge")
        clone = trial
        arrivals_total += sum(sum(values.values()) for values in inputs[stage].values())
        if not math.isclose(passenger_total(clone) + sum(float(a.cumulative_alighted[0]) for a in clone.agents),
                            total + arrivals_total, rel_tol=1e-5, abs_tol=1e-3):
            raise AssertionError("Passenger conservation failed in the checked continuation")
        packets.append(dict(guard=guard, commands=copy.deepcopy(native_packet["commands"]) if native_packet else [],
            actions=actions, targets=(copy.deepcopy(clone.last_physical_requests) if clone.second_precision and decision_seconds(clone) > clone.tick_seconds else
                copy.deepcopy(native_packet["targets"]) if native_packet else dict(targets)),
            expected_positions=positions(clone), expected_guard=state_guard(clone), arrival_inputs=inputs[stage]))
        packets[-1]["physical_audits"] = copy.deepcopy(clone.last_physical_audits) if clone.second_precision else None
        costs.append(dict(clone.dispatch.interval_cost))
        if reference_model is not None:
            audits = clone.last_physical_audits if clone.second_precision else [clone.last_transition_audit]
            frames = [{name: row["requested"] for name, row in frame.items()} for frame in audits]
            requests.append(frames if clone.second_precision else frames[0])
            history.append([{name: row["actual"] for name, row in frame.items()} for frame in audits])
            trajectory.append(clone.dispatch.snapshot(published=True))
        for name, node in positions(clone).items():
            paths[name].append(node)
            sequences[name].append(actions[name])
    gamma = clone.dispatch.objective.weights["discount"]
    terminal = clone.dispatch.objective.terminal(clone)
    cost = sum(gamma ** k * item["total"] for k, item in enumerate(costs)) + gamma ** horizon * terminal["total"]
    if reference_model is not None:
        reference_model._capture_checked_reference(requests=requests, history=history, paths=paths,
            packets=packets, actions=sequences, steps=costs, terminal=terminal, trajectory=trajectory,
            terminal_passengers=passenger_total(clone), terminal_guard=state_guard(clone))
    return dict(checked_dispatch=True, milp_incumbent=False, status="checked_reference", cost=cost,
        dispatch_packets=packets, reference_paths=paths, reference_actions=sequences,
        passenger_inputs=inputs, corrections=corrections,
        continuation_source=("published route and committed operations" if first_actions is None else
                             "current MARL action then current route and committed operations"))


def search_phase_limit(available, limit, *, guided, repairable):
    """Reserve time for releasing guidance or repairing observed background conflicts."""
    usable = max(0., available - min(1., available * .1))
    fraction = .35 if guided else .6 if repairable else 1.
    return min(limit, usable * fraction)


class DispatchCoordination:
    """Command acceptance, coupling requests, guarded cache and fallback."""
    def __init__(self, env, config, solver):
        self.env, self.config, self.solver = env, config, solver
        validate_experiment_controls(config)
        if type(config.milp_time_horizon) is not int or config.milp_time_horizon < 1:
            raise ValueError("A positive integer coordination horizon is required")
        if not 0 <= config.entropy_threshold <= 1 or config.intervention_cooldown < 0 or config.cached_plan_max_steps < 0:
            raise ValueError("Invalid entropy, cooldown or cache configuration")
        fraction = getattr(config, "solver_reserve_fraction", .5)
        if not math.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError("Solver reserve fraction must be in [0, 1)")
        self.reset()

    @property
    def world(self):
        base = getattr(self.env, "base_env", self.env)
        return base.scenario.world

    def reset(self):
        self.cached = None
        self.index = 0
        self.pending_audit = None
        self.last_request = -math.inf
        self.request_queue = CouplingRequestQueue()
        self.acknowledged_events = "[]"
        self.observed_events = "[]"
        self.pending_event_request = False
        self.history = []
        self.last_record = {}
        from LLM.coupling_trigger import CouplingTrigger, CouplingThresholds
        self.coupling_trigger = CouplingTrigger(CouplingThresholds(
            horizon=getattr(self.config, "coupling_check_horizon", 3),
            service_gap_persons=getattr(self.config, "coupling_service_gap", 20.),
            persistence_stages=getattr(self.config, "coupling_persistence", 2),
            cooldown_stages=self.config.intervention_cooldown))

    def invalidate_cache(self, reason):
        self.cached = None
        self.index = 0
        self.last_record.setdefault("cache_invalidations", []).append(reason)

    def notify_step_result(self, *_):
        if self.pending_audit is None:
            return None
        solution, stage, record = self.pending_audit
        started = time.perf_counter()
        audit = audit_dispatch_step(self.world, solution, stage)
        record["execution_audit"] = audit
        record["post_execution_check_s"] = time.perf_counter() - started
        self.pending_audit = None
        if not audit["matched"] or audit["constraint_interventions"]:
            self.invalidate_cache("execution_state_or_operational_constraints_changed")
        return audit

    def has_active_plan(self):
        return bool(self.cached and self.index < min(len(self.cached["dispatch_packets"]), self.config.cached_plan_max_steps))

    def signals(self, logits=None, *, actions=None, candidates=None, reference_solution=None):
        world = self.world
        mode = getattr(self.config, "trigger_mode", "coupling")
        if mode not in ("legacy", "coupling", "periodic", "event"):
            raise ValueError("Unknown trigger mode")
        if mode in ("coupling", "periodic", "event"):
            packet = self.cached["dispatch_packets"][self.index] if self.has_active_plan() else None
            result = self.coupling_trigger.inspect(world, actions, candidates=candidates, cached_packet=packet,
                reference_solution=reference_solution)
            events = event_signature(world)
            metrics = dict(coupling=result, events=events)
            if mode == "coupling":
                return result["signals"], metrics
            changed = events != self.observed_events
            self.observed_events = events
            if mode == "event":
                self.pending_event_request |= changed
            stage = int(round(float(world.sim_time[0])/decision_seconds(world)))
            # Retain the same feasibility diagnostics and scope evidence. Only
            # the request rule changes; a detected invalid plan is never reused.
            signal = dict(PLAN_INVALID=bool(result["signals"].get("PLAN_INVALID")))
            signal.update(PERIODIC=mode == "periodic" and stage % getattr(self.config, "coordination_period", 10) == 0,
                          EVENT_UPDATE=mode == "event" and self.pending_event_request)
            metrics.update(published_information_changed=changed, pending_event_request=self.pending_event_request,
                           coordination_period=getattr(self.config, "coordination_period", 10))
            return signal, metrics
        queues = [world.waiting_at(p) for p in world.platform_nodes]
        total = getattr(self.config, "waiting_total_threshold", None)
        local = getattr(self.config, "waiting_max_threshold", None)
        total = self.config.waiting_threshold if total is None else total
        local = self.config.waiting_threshold if local is None else local
        if math.isnan(total) or math.isnan(local) or min(total, local) < 0:
            raise ValueError("Queue thresholds must be nonnegative")
        entropy = policy_entropy(logits, action_mask(world, logits.shape[-1])) if logits is not None else 0.
        events = event_signature(world)
        return dict(DISRUPTION=events != self.acknowledged_events,
            HIGH_ENTROPY=entropy > self.config.entropy_threshold,
            HIGH_WAITING=sum(queues) >= total or max(queues, default=0.) >= local), dict(
                entropy=entropy, total_waiting=sum(queues), max_waiting=max(queues, default=0.),
                total_threshold=total if math.isfinite(total) else None,
                local_threshold=local if math.isfinite(local) else None, events=events)

    def decide(self, actions, state, *, logits=None, retrieve=None, specify=None, guide=None, reference_solution=None,
               candidates=None, passenger_inputs=None, max_evaluations=None, started_at=None, budgeted_callbacks=False):
        started = time.perf_counter() if started_at is None else started_at
        if self.history and float(self.world.sim_time[0]) < self.history[-1]["time"]:
            self.reset()
        if self.pending_audit is not None and getattr(self.world, "_dispatch_stage_targets", None) is not None:
            raise RuntimeError("Consume the previously committed dispatch stage before deciding again")
        self.notify_step_result()
        record = dict(time=float(self.world.sim_time[0]), timings_s={key: 0. for key in (
            "trigger_check", "retrieval", "llm", "slm", "scope_check", "reference_check", "reference_guard", "guidance_validation",
            "solve", "fallback_check", "commit_check")}, errors=[], attempts=[], skipped_stages=[],
                      simulation_during_computation=False, timing_mode="synchronous_cooperative")
        self.last_record = record
        def timed(key, callback):
            before = time.perf_counter()
            try:
                return callback()
            finally:
                record["timings_s"][key] = record["timings_s"].get(key, 0.) + time.perf_counter() - before
                logger.info("coordination_stage time=%s stage=%s elapsed_s=%.3f", record["time"], key,
                            time.perf_counter()-before)
        budget = getattr(self.config, "coordination_time_budget_s", None)
        budget = float(self.config.milp_time_limit if budget is None else budget)
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("Coordination budget must be finite and nonnegative")
        allowed = getattr(self.config, "allowed_decision_delay_s", None)
        allowed = budget if allowed is None else float(allowed)
        if not math.isfinite(allowed) or allowed < 0:
            raise ValueError("Allowed decision delay must be finite and nonnegative")
        deadline = started + budget
        signal, metrics = timed("trigger_check", lambda: self.signals(logits, actions=actions,
            candidates=candidates, reference_solution=reference_solution))
        now_stage = int(round(float(self.world.sim_time[0]) / decision_seconds(self.world)))
        proposed_review = bool(state.get("native_marl_proposal", {}).get("quality_guard", {}).get("deferred"))
        review_vehicles = []
        if proposed_review:
            from dispatch_candidates import SERVICE_REVIEW_MEASURES
            from dispatch_plan import DispatchCommand
            review_vehicles = sorted({v for raw in reference_solution["dispatch_packets"][0]["commands"]
                if raw["action"] in SERVICE_REVIEW_MEASURES
                for v in DispatchCommand.from_dict(raw, self.world.tick_seconds).participants()})
        review_groups = []
        regional = None
        if getattr(self.config, "trigger_mode", "coupling") == "coupling":
            if proposed_review:
                participants = review_vehicles
                platforms = sorted({self.world.dispatch.position(v) for v in participants} & set(self.world.platform_nodes))
                depots = sorted({self.world.dispatch.records[v].depot for v in participants}-{None})
                review_groups = ["proposed_service_review:" + ",".join(participants)]
                metrics["coupling"]["evidence"].append(dict(groups=review_groups, relations=[], vehicles=participants,
                    control_platforms=platforms, nodes=platforms, depots=depots, active=True, persistent_stages=1,
                    local_passenger_gap=sum(self.world.waiting_at(p) for p in platforms),
                    action_families=state["native_marl_proposal"]["quality_guard"].get("delayed_measures", []),
                    basis="qualified_delayed_service_proposal_requires_common_window_review"))
            regional = self.request_queue.update(metrics["coupling"], now_stage,
                cooldown=self.config.intervention_cooldown,
                service_gap=getattr(self.config, "coupling_service_gap", 20.))
            record["regional_request"] = regional
            signal["COUPLED_SERVICE"] = bool(signal.get("COUPLED_SERVICE") or review_groups)
        record.update(signals=signal, metrics=metrics, budget_s=budget,
                      experiment=experiment_configuration(self.config))
        if "coupling" in metrics:
            state = dict(state, coupling=metrics["coupling"])
        record["component_availability"] = dict(retrieval=retrieve is not None, llm=specify is not None, slm=guide is not None)
        trigger = next((name for name in ("PLAN_INVALID", "PERIODIC", "EVENT_UPDATE", "COUPLED_SERVICE", "DISRUPTION", "HIGH_WAITING", "HIGH_ENTROPY") if signal.get(name)), None)
        record["trigger"] = trigger
        result, stage = None, 0
        trigger_mode = getattr(self.config, "trigger_mode", "coupling")
        if self.has_active_plan() and (signal.get("PERIODIC") or signal.get("EVENT_UPDATE")):
            self.invalidate_cache("scheduled_reassessment")
        if self.has_active_plan():
            packet = self.cached["dispatch_packets"][self.index]
            if regional and regional["region"] and (regional["urgent"] or
                    regional["pending_age_stages"] >= max(1, self.config.intervention_cooldown)):
                self.invalidate_cache("urgent_or_overdue_regional_request")
            elif signal.get("PLAN_INVALID"):
                self.invalidate_cache("coupling_plan_invalid")
            elif guard_matches(packet["guard"], state_guard(self.world)):
                result, stage = self.cached, self.index
                record["source"] = "cache"
                record["cache_origin"] = self.cached.get("status")
                record["cache_kind"] = "optimization" if self.cached.get("milp_incumbent") else "checked_reference"
            else:
                self.invalidate_cache("current_stage_state_mismatch")
        eligible = bool(regional["region"]) if regional is not None else now_stage-self.last_request >= self.config.intervention_cooldown
        request = bool(self.config.always_hierarchical or signal.get("PLAN_INVALID") or signal.get("PERIODIC")
                       or trigger and eligible)
        review_required = bool(proposed_review and request and result is None and (
            regional is None or metrics["coupling"].get("scheduled_relation_groups") == review_groups))
        record.update(reference_service_review_required=review_required,
                      pending_service_review=proposed_review and not review_required)
        if proposed_review and not review_required:
            # Do not execute an unreviewed long operation while its region waits.
            # Existing committed work is preserved by the checked route fallback.
            reference_solution, actions = None, None
        record["safety_override"] = bool(trigger_mode in ("periodic", "event") and signal.get("PLAN_INVALID"))
        record.update(requested=request and result is None, cooldown_eligible=eligible)
        reference = None
        if result is None and request:
            self.last_request = now_stage
            if regional is not None:
                self.request_queue.mark_requested(metrics["coupling"], now_stage)
            if trigger_mode == "event":
                self.pending_event_request = False
            mode = getattr(self.config, "joint_guidance_mode", "full")
            if mode not in ("full", "no_guidance", "rules", "retrieval", "no_llm", "no_slm"):
                raise ValueError("Unknown joint guidance mode")
            record["guidance_mode"] = mode
            context, specification, guidance = {}, {}, {}
            fraction = getattr(self.config, "solver_reserve_fraction", .5)
            if not math.isfinite(fraction) or not 0 <= fraction < 1:
                raise ValueError("Solver reserve fraction must be in [0, 1)")
            reserve = max(0., deadline-time.perf_counter()) * fraction
            guidance_deadline = deadline - reserve
            record["solver_reserved_s"] = reserve
            callbacks = [("retrieval", retrieve, lambda: retrieve(state), mode not in ("no_guidance", "rules")),
                         ("llm", specify, lambda: specify(state, context, trigger), mode in ("full", "no_slm"))]
            for key, callback, call, enabled in callbacks:
                available = guidance_deadline - time.perf_counter()
                if enabled and callback and available > 0:
                    try:
                        if key == "llm" and budgeted_callbacks:
                            cap = min(getattr(self.config, "llm_request_timeout_s", 30.),
                                      available * (.5 if guide and mode == "full" else 1.))
                            record["llm_timeout_s"] = cap
                            call = lambda: specify(state, context, trigger, timeout_s=cap)
                        value = timed(key, call)
                        if not isinstance(value, dict) or value.get("error"):
                            raise ValueError(value.get("error") if isinstance(value, dict) else "Provider must return a structured object")
                        if key == "retrieval":
                            context = value or {}
                        else:
                            specification = value or {}
                    except Exception as exc:
                        record["errors"].append(exception_record(key, exc))
                elif enabled and callback:
                    record["skipped_stages"].append(dict(stage=key, reason="guidance_budget_exhausted_solver_reserved"))
            if not specification and mode in ("retrieval", "no_llm"):
                specification = retrieved_specification(context)
            config = specification.get("solver_config", {}) if isinstance(specification, dict) else {}
            try:
                if time.perf_counter() >= deadline:
                    record["skipped_stages"].extend(dict(stage=key, reason="coordination_budget_exhausted")
                        for key in ("scope_check", "reference_check", "slm", "solve"))
                    raise CoordinationBudgetExpired()
                horizon = (self.config.milp_time_horizon if getattr(self.config, "fixed_planning_horizon", False)
                           else min(self.config.milp_time_horizon, max(1, int(config.get("time_horizon", self.config.milp_time_horizon)))))
                horizon = getattr(self.config, "fixed_comparison_horizon", horizon)
                limit = min(float(self.config.milp_time_limit), max(0., float(config.get("time_limit", self.config.milp_time_limit))))
                coupling = metrics.get("coupling", {})
                requested_focus, requested_nodes, selection = select_experiment_focus(self.world, specification, coupling,
                    mode=getattr(self.config, "coordination_scope", "coupling"),
                    nearest_vehicles=getattr(self.config, "nearest_vehicles", 6))
                if review_required:
                    requested_focus = sorted(set(requested_focus) | set(review_vehicles))
                scenario = getattr(getattr(self.env, "base_env", self.env), "scenario", None)
                end_stage = getattr(scenario, "_max_steps", None)
                remaining_stages = max(1, int(end_stage)-now_stage) if end_stage is not None else None
                focus, closure = timed("scope_check", lambda: close_focus(self.world,
                    requested_focus, sorted(set(requested_nodes)),
                    candidates, horizon, reference_commands=(reference_solution or {}).get("dispatch_packets", [{}])[0].get("commands", [])))
                horizon_info = timed("scope_check", lambda: coordination_horizon(self.world, self.config,
                    closure["resolved"], horizon, reference_solution=reference_solution,
                    remaining_stages=remaining_stages, passenger_inputs=passenger_inputs))
                horizon = horizon_info["horizon"]
                record["horizon_policy"] = horizon_info
                closure["selection"] = selection
                local_candidates = scoped_candidates(self.world, candidates, closure["resolved"], closure["focus_platforms"])
                platforms = closure["focus_platforms"]
                record.update(specification=copy.deepcopy(specification), focus=closure, horizon=horizon)
                logger.info("coordination_scope %s", json.dumps(dict(time=record["time"], **closure)))
                declared = passenger_inputs[:horizon] if passenger_inputs is not None else None
                # Catalog, reference and guided/released searches belong to one
                # immutable request. Do not rebuild its physical continuation.
                schema = timed("guidance_validation", lambda: JointDispatchModel(self.world, horizon,
                    focus=focus, focus_platforms=platforms, candidates=local_candidates,
                    passenger_inputs=declared, deadline=deadline, defer_reference=True))
                masks = None
                try:
                    reference = timed("reference_check", lambda: checked_reference(self.world, actions, horizon,
                        declared, reference_solution, deadline=deadline, reference_model=schema))
                    equivalent = not review_required and (actions is None or getattr(self.config, "reference_policy", "marl") == "rule" or reference_solution and (
                        reference_solution.get("status") == "checked_route_quality_guard" or
                        reference_solution.get("selected_options") and
                        all(v in ("continue", "unchanged") for v in reference_solution["selected_options"].values())))
                    available = deadline-time.perf_counter()
                    estimate = record["timings_s"]["reference_check"]
                    quality = record["coordination_reference_quality"] = dict(
                        policy_cost=reference["cost"], replaced=False, horizon=horizon,
                        basis="same published state, arrivals, horizon and physical objective")
                    if getattr(self.config, "reference_quality_guard", True) and not equivalent and (
                            review_required or available > 2*estimate+.1):
                        try:
                            route_schema = schema.fork(deadline=time.perf_counter()+available*.5)
                            route = timed("reference_guard", lambda: checked_reference(self.world, None, horizon,
                                reference["passenger_inputs"], deadline=route_schema.deadline, reference_model=route_schema))
                            quality["route_cost"] = route["cost"]
                            quality["service_review_completed"] = True
                            if route["cost"] < reference["cost"]-1e-6:
                                reference = route
                                schema = route_schema
                                schema.deadline = deadline
                                quality["replaced"] = True
                        except (RuntimeError, ValueError, EvaluationTimeout) as exc:
                            quality["unavailable"] = type(exc).__name__
                            if review_required:
                                reference = None
                                raise CoordinationBudgetExpired("Deferred service review did not finish") from exc
                    else:
                        quality["skipped"] = "route_equivalent_or_guard_disabled_or_search_budget_reserved"
                except RuntimeError as exc:
                    record["reference_unavailable"] = str(exc)
                    # A truly infeasible background needs a separate full-scope
                    # repair; never silently describe it as a focused solve.
                    focus = list(range(len(self.world.agents)))
                    platforms = list(self.world.platform_nodes)
                    local_candidates = candidates
                    closure.update(resolved=[a.name for a in self.world.agents], nonfocus=[], focus_platforms=platforms)
                    closure["reasons"].append("reference_infeasible_expand_to_all_vehicles")
                    logger.info("coordination_scope_expanded %s", json.dumps(dict(time=record["time"], **closure)))
                    # Variable indices from the narrower catalog may now refer
                    # to different groups. Never apply them after expansion.
                    guidance = {}
                    schema = None
                    record["guidance_released_reason"] = "reference_infeasible_scope_expanded"
                if schema is not None and reference is not None:
                    refined = timed("scope_check", lambda: close_reference_platform_users(self.world, closure,
                        schema.nominal_history, candidates=candidates, horizon=horizon,
                        reference_commands=reference["dispatch_packets"][0].get("commands", []), planning_base=schema.base))
                    if (getattr(self.config, "coordination_scope", "coupling") == "coupling"
                            and getattr(self.config, "scope_window_closure", False)):
                        refined = timed("scope_check", lambda: close_reference_motion_users(self.world, refined,
                            schema.nominal_history, proposed=refined.get("selection", {}).get("llm_vehicles", []),
                            maximum=getattr(self.config, "scope_window_vehicles", 12), candidates=candidates,
                            horizon=horizon, reference_commands=reference["dispatch_packets"][0].get("commands", []),
                            planning_base=schema.base))
                        record["scope_window_closure"] = refined["reference_motion_closure"]
                    if set(refined["resolved"]) != set(closure["resolved"]):
                        local_candidates = scoped_candidates(self.world, candidates, refined["resolved"], refined["focus_platforms"])
                        schema = timed("scope_check", lambda: schema.refocus(refined["resolved"],
                            refined["focus_platforms"], candidates=local_candidates))
                    closure, platforms = refined, refined["focus_platforms"]
                    focus = [i for i, a in enumerate(self.world.agents) if a.name in closure["resolved"]]
                    record["focus"] = closure
                    logger.info("coordination_scope_refined %s", json.dumps(dict(time=record["time"], **closure)))
                if schema is not None and guide and mode in ("full", "no_llm") and time.perf_counter() < guidance_deadline:
                    try:
                        record["slm_catalog_fingerprint"] = schema.catalog_fingerprint
                        from MILP.guidance_labels import variable_catalog
                        from dispatch_candidates import applicable_options
                        catalog = variable_catalog(schema)
                        masks, _ = timed("guidance_validation", lambda: applicable_options(schema))
                        for gi, (group, options) in enumerate(schema.groups.items()):
                            for oi in range(len(options)):
                                catalog[f"command_{gi}_{oi}"]["applicability_zero"] = not masks[group][oi]
                        guidance_state = dict(state, solver_variables=catalog)
                        available = guidance_deadline-time.perf_counter()
                        if available <= 0:
                            raise CoordinationBudgetExpired("SLM catalog used its guidance budget")
                        # Cap the entire grouped phase, preserving search time.
                        slm_budget = min(available, float(getattr(self.config, "small_model_request_timeout", 30.)))
                        if not math.isfinite(slm_budget) or slm_budget <= 0:
                            raise ValueError("SLM total guidance allowance must be positive")
                        record["slm_timeout_s"] = slm_budget
                        guidance = timed("slm", lambda: guide(guidance_state, context, specification, timeout_s=slm_budget)
                            if budgeted_callbacks else guide(guidance_state, context, specification)) or {}
                        if not isinstance(guidance, dict):
                            guidance = {key: getattr(guidance, key, {}) for key in ("warm_start", "variable_fixing", "branching_priority")}
                    except (EvaluationTimeout, CoordinationBudgetExpired) as exc:
                        record["skipped_stages"].append(dict(stage="slm", reason="guidance_budget_exhausted_during_catalog",
                            error_type=type(exc).__name__))
                    except Exception as exc:
                        record["errors"].append(exception_record("slm", exc))
                elif guide and mode in ("full", "no_llm"):
                    record["skipped_stages"].append(dict(stage="slm", reason="guidance_budget_exhausted_solver_reserved"))
                if mode == "rules":
                    guidance = {"warm_start": {f"command_{g}_0": 1 for g in range(len(schema.groups) if schema else len(candidates or focus))}}
                if mode == "retrieval":
                    guidance = context.get("solver_guidance", {})
                if not isinstance(guidance, dict):
                    record["errors"].append(dict(stage="guidance", error="Expected a variable-guidance mapping"))
                    guidance = {}
                if guidance and record.get("slm_catalog_fingerprint"):
                    guidance = dict(guidance, catalog_fingerprint=record["slm_catalog_fingerprint"])
                fixed = ({name: path for name, path in reference["reference_paths"].items() if name not in closure["resolved"]}
                         if reference else {})
                solve_inputs = reference["passenger_inputs"] if reference else declared
                reference_actions = reference["reference_actions"] if reference else None
                seeds = []
                def propose_seeds(incumbent=None):
                    if schema is None or reference is None:
                        return []
                    from dispatch_candidates import applicable_options
                    from MILP.guidance_labels import neighborhood_seeds, public_option_priority
                    from LLM.coupling_trigger import option_footprints, event_relevant
                    current_masks, _ = applicable_options(schema)
                    if any(not any(not op.commands and not op.actions and not op.pauses for op in options)
                           for options in schema.groups.values()):
                        record["coupled_seed_skip_reason"] = "custom_catalog_without_reference_option"
                        return []
                    footprints = option_footprints(schema)
                    events = [e for e in schema.base.get_published_events() if e.get("active", True)]
                    priorities = {g: {o.id: public_option_priority(schema.base, o, footprints[g][i],
                        [e for e in events if event_relevant(e, footprints[g][i], schema.base)])
                        for i, o in enumerate(options)} for g, options in schema.groups.items()}
                    anchor = None
                    if incumbent and incumbent.get("catalog_fingerprint") == schema.catalog_fingerprint:
                        selected = incumbent.get("selected_options", {})
                        anchor = tuple(next(i for i, op in enumerate(ops) if op.id == selected[g])
                                       for g, ops in schema.groups.items())
                    return [s for s in neighborhood_seeds(schema, current_masks, limit=8, anchor=anchor,
                        joint_first=True, cover_families=False,
                        coupling=dict(evidence=coupling.get("evidence", []), option_priority=priorities))
                        if tuple(s) not in schema.evaluated]
                if schema is not None and reference is not None:
                    schema.fix_background_to_reference()
                    seeds = timed("guidance_validation", propose_seeds)
                    record["coupled_seed_count"] = len(seeds)
                    record["reference_reused_for_search"] = True
                    record["prepared_catalog_fingerprint"] = schema.catalog_fingerprint
                record["guidance"] = copy.deepcopy(guidance)
                record["incumbent_history"] = []
                record["best_known_cost"] = reference.get("cost") if reference is not None else None
                released_background = set()
                expansion_limit = max(0, getattr(self.config, "scope_expansion_agents", 2))
                for attempt in range(4):
                    available = max(0., deadline-time.perf_counter())
                    reserve = min(1., available*.1)
                    record["postsolve_reserved_s"] = reserve
                    repairable = bool(attempt < 3 and len(released_background) < expansion_limit and schema is not None
                        and getattr(self.config, "coordination_scope", "coupling") == "coupling"
                        and set(schema.base.dispatch.records) - schema.focus)
                    constrained = bool(guidance.get("variable_fixing"))
                    remaining = search_phase_limit(available, limit, guided=constrained and attempt < 3,
                                                   repairable=repairable)
                    if constrained:
                        record["guidance_release_reserved_s"] = available-reserve-remaining
                    if remaining <= 0:
                        record["skipped_stages"].append(dict(stage="solve", reason="coordination_budget_exhausted"))
                        break
                    try:
                        if schema is not None and reference is not None:
                            timed("solve", lambda: self.solver.solve_prepared_dispatch(schema, time_limit=remaining,
                                guidance=guidance, max_evaluations=max_evaluations, seed_selections=seeds,
                                optimization_mode=getattr(self.config, "optimization_mode", "joint"),
                                stop_on_background_conflict=repairable))
                        else:
                            timed("solve", lambda: self.solver.solve(getattr(self.env, "base_env", self.env),
                                focus_agent_ids=focus, focus_platform_ids=platforms, time_horizon=horizon, time_limit=remaining,
                                dispatch_candidates=local_candidates, passenger_inputs=solve_inputs,
                                fixed_train_trajectories=fixed, reference_actions=reference_actions,
                                reference_dispatch_packets=reference["dispatch_packets"] if reference else None,
                                llm_guidance=guidance, max_evaluations=max_evaluations,
                                dispatch_optimization_mode=getattr(self.config, "optimization_mode", "joint")))
                        candidate = copy.deepcopy(self.solver.last_solution)
                        reason = ("no_checked_incumbent" if not candidate.get("milp_incumbent") else
                            "worse_than_reference" if reference is not None and candidate["cost"] > reference["cost"]+1e-6 else
                            "no_strict_improvement" if reference is not None and candidate["cost"] >= reference["cost"]-1e-6 else
                            "not_better_than_incumbent" if result is not None and candidate["cost"] >= result["cost"]-1e-6 else
                            "returned_after_coordination_deadline" if time.perf_counter()>deadline else "accepted")
                        record["attempts"].append(dict({k: candidate.get(k) for k in (
                            "status", "termination", "cost", "cost_lower_bound", "cost_upper_bound", "guidance_diagnostics", "certificate_scope", "guidance_catalog_mismatch",
                            "physical_evaluations", "novel_candidate_evaluations", "shared_reference_evaluations",
                            "feasible_candidate_evaluations", "infeasible_candidate_reasons", "blocking_background_vehicles", "seed_records",
                            "construction_reused", "evaluations", "incumbent_history", "timings", "staged_diagnostics")},
                            acceptance=reason, reference_cost=reference.get("cost") if reference else None))
                        if reason == "accepted":
                            result = candidate
                            record["source"] = ("staged_optimization" if
                                getattr(self.config, "optimization_mode", "joint") == "staged" else "joint_optimization")
                            self.cached, self.index = result, 0
                            self.acknowledged_events = metrics["events"]
                            record["best_known_cost"] = candidate["cost"]
                            record["incumbent_scope"] = dict(
                                vehicles=sorted(schema.focus) if schema else list(closure["resolved"]),
                                catalog_fingerprint=candidate.get("catalog_fingerprint"))
                            record["incumbent_history"].append(dict(cost=candidate["cost"],
                                attempt=attempt, catalog=schema.catalog_fingerprint if schema else None,
                                basis="current published state and exact full-window physical evaluation"))
                            if not constrained and not repairable:
                                break
                    except Exception as exc:
                        record["errors"].append(exception_record("solve", exc))
                    available = deadline-time.perf_counter()
                    blockers = blocking_background_vehicles(schema) if schema is not None else []
                    release = blockers[:max(0, expansion_limit-len(released_background))]
                    # Refocusing reuses the checked reference; it does not need
                    # another full reference-build allowance.
                    if (not release or available <= 1.
                            or getattr(self.config, "coordination_scope", "coupling") != "coupling"):
                        if constrained and attempt < 3:
                            guidance = {k: v for k, v in guidance.items() if k != "variable_fixing"}
                            record["guidance_fixes_released"] = True
                            seeds = timed("guidance_validation", lambda: propose_seeds(result))
                            continue
                        if repairable and available > 1.:
                            seeds = timed("guidance_validation", lambda: propose_seeds(result))
                            if seeds:
                                continue
                        break
                    _, enlarged = timed("scope_check", lambda: close_focus(self.world,
                        sorted(set(closure["resolved"]) | set(release)), platforms, candidates, horizon,
                        reference_commands=reference["dispatch_packets"][0].get("commands", []), planning_base=schema.base))
                    enlarged["selection"] = selection
                    local_candidates = scoped_candidates(self.world, candidates, enlarged["resolved"], platforms)
                    prior_schema = schema
                    schema = timed("scope_check", lambda: schema.refocus(enlarged["resolved"], platforms,
                        candidates=local_candidates))
                    schema.fix_background_to_reference()
                    closure = enlarged
                    closure["focus_platforms"] = platforms
                    released_background.update(release)
                    record.update(focus=closure, scope_expansion=dict(released=sorted(released_background),
                        basis="evaluated_background_conflicts", limit=expansion_limit))
                    if getattr(self.config, "scope_window_closure", False) and guidance:
                        guidance, rebound = rebind_scope_guidance(guidance, prior_schema, schema)
                        record.setdefault("guidance_rebindings", []).append(rebound)
                        record["guidance_released_reason"] = "expanded_catalog_soft_hints_rebound_hard_fixes_released"
                    else:
                        guidance = {}
                        record["guidance_released_reason"] = "expanded_catalog_after_background_conflict"
                    seeds = timed("guidance_validation", lambda: propose_seeds(result))
            except CoordinationBudgetExpired:
                pass
            except EvaluationTimeout:
                record["skipped_stages"].append(dict(stage="reference_check_or_solve", reason="coordination_budget_exhausted"))
            except Exception as exc:
                record["errors"].append(exception_record("configuration_or_reference", exc))
        if result is None:
            if reference is None:
                reference = timed("fallback_check", lambda: checked_reference(self.world,
                    None if review_required else actions, 1,
                    passenger_inputs[:1] if passenger_inputs is not None else None,
                    None if review_required else reference_solution))
            result = reference
            record["source"] = "checked_fallback" if request else "checked_execution"
            record["reference_corrections"] = reference["corrections"]
            if len(reference["dispatch_packets"]) > 1:
                # An unchanged incumbent is not an optimization gain, but its
                # checked suffix is still usable under the same state/event guards.
                self.cached, self.index = reference, 0
                self.acknowledged_events = metrics["events"]
                record["cached_reference_stages"] = len(reference["dispatch_packets"])
        commit = timed("commit_check", lambda: apply_dispatch_step(self.world, result, stage))
        if not commit["accepted"]:
            self.invalidate_cache(commit["reason"])
            # Do not send an unchecked integer action after a stale-state rejection.
            raise RuntimeError(f"Native dispatch commit rejected: {commit['reason']}")
        if result is self.cached:
            self.index = stage + 1
        packet = result["dispatch_packets"][stage]
        record.update(commit=commit, source_status=result["status"], accepted_cost=result.get("cost"),
                      reference_cost=reference.get("cost") if reference else None,
                      dispatch_stage=stage, dispatch_commands=copy.deepcopy(packet["commands"]),
                      dispatch_targets=copy.deepcopy(packet["targets"]),
                      expected_positions=copy.deepcopy(packet["expected_positions"]),
                      expected_operations=copy.deepcopy(packet["expected_guard"]["operations"]),
                      total_decision_s=time.perf_counter() - started)
        record["strictly_improved"] = (record["source"] in ("joint_optimization", "staged_optimization") and
            reference is not None and result["cost"] < reference["cost"]-1e-6)
        record["allowed_delay_s"] = allowed
        record["budget_exceeded"] = record["total_decision_s"] > budget
        record["allowed_delay_exceeded"] = record["total_decision_s"] > record["allowed_delay_s"]
        self.history.append(record)
        self.pending_audit = (result, stage, record)
        return commit["actions"], record


def direct_dispatch_action(owner, td, marl_actions, *, solver, horizon, time_limit, use_cache, specify):
    """Legacy direct-LLM entry points share native acceptance and execution."""
    from tensordict import TensorDict
    started = time.perf_counter()
    runtime = getattr(owner, "_dispatch_runtime", None)
    if runtime is None:
        config = SimpleNamespace(milp_time_horizon=horizon, milp_time_limit=time_limit,
            intervention_cooldown=0, waiting_threshold=math.inf, entropy_threshold=1.,
            always_hierarchical=True, cached_plan_max_steps=horizon if use_cache else 0,
            joint_guidance_mode="full" if specify else "no_guidance")
        runtime = owner._dispatch_runtime = DispatchCoordination(owner.env, config, solver)
    first = torch.as_tensor(marl_actions).reshape(-1).tolist() if marl_actions is not None else []
    actions = {a.name: int(first[i]) if i < len(first) else 0 for i, a in enumerate(runtime.world.agents)}
    actions, record = runtime.decide(actions, dict(dispatch_state=runtime.world.dispatch_snapshot()),
                                     specify=specify, started_at=started)
    owner.last_decision_source = record["source"]
    owner.last_coordination_record = record
    tensor = torch.tensor([actions[a.name] for a in runtime.world.agents], device=owner.device).view(1, -1, 1)
    output = TensorDict({owner.action_key: tensor}, batch_size=[1])
    record["total_decision_s"] = time.perf_counter() - started
    record["budget_exceeded"] = record["total_decision_s"] > record["budget_s"]
    record["allowed_delay_exceeded"] = record["total_decision_s"] > record["allowed_delay_s"]
    return output
