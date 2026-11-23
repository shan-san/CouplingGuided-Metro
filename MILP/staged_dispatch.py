"""Restricted two-phase search using the unchanged joint physical model.

Bundles containing any service/recovery command belong to phase one, including
all their movement side effects. Empty bundles are reference continuations;
every other bundle is movement (including hold, profile, departure and primitive
controls). Catalog order breaks ties between empty references. A group without
an empty reference is rejected rather than silently choosing its first measure.

Phase two fixes chosen operation bundles exactly and forbids all other operation
bundles. Nonfocus, fixed and committed work is protected; other groups cannot
override selected operation participants/resources. These sequential restrictions
do not certify the original joint domain.
"""
from __future__ import annotations

import copy
import math
import time

from dispatch_plan import DispatchCommand
from MILP.operation_model import EvaluationTimeout


OPERATION_ACTIONS = frozenset({
    "short_turn", "skip_stop", "cancel_service", "reserve_insert", "withdraw",
    "reenter", "reassign_service", "rescue_tow", "evacuate", "admission_control",
    "inspect_repair",
})
_SCOPE = ("restricted staged procedure on the unchanged finite catalog, published "
          "inputs and protected continuations; not the original joint-domain optimum")


def _kind(option):
    if any(c["action"] in OPERATION_ACTIONS for c in option.commands):
        return "operation"
    return "movement" if option.commands or option.actions or option.pauses else "continuation"


def _footprint(problem, option):
    vehicles, resources = set(option.actions) | set(option.pauses), set()
    for raw in option.commands:
        command = DispatchCommand.from_dict(raw, problem.base.tick_seconds)
        vehicles.update(command.participants())
        resources.update(command.resources())
    return vehicles, resources


def _protected(problem):
    dispatch = problem.base.dispatch
    vehicles = (set(dispatch.records) - set(problem.focus)) | set(problem.fixed_paths)
    resources = set()
    for operation in dispatch.operations.values():
        vehicles.update(operation.participants())
    for plan in dispatch.plans.values():
        for command in plan.commands:
            if dispatch.command_results[command.command_id]["status"] == "pending":
                vehicles.update(command.participants())
                resources.update(command.resources())
    for packet in problem.reference_packets or ():
        for raw in packet["commands"]:
            command = DispatchCommand.from_dict(raw, problem.base.tick_seconds)
            if command.locked:
                vehicles.update(command.participants())
                resources.update(command.resources())
    blocked = set()
    for gi, options in enumerate(problem.groups.values()):
        for oi, option in enumerate(options):
            touched, controls = _footprint(problem, option)
            if (vehicles & touched or resources & controls
                    or any(platform not in problem.focus_platforms for _, platform in controls)):
                blocked.add((gi, oi))
    return blocked, vehicles, resources


def _fixes(problem, allowed, supplied):
    fixes, initial = dict(supplied), []
    for gi, (group, options) in enumerate(problem.groups.items()):
        choices = [oi for oi in allowed[gi] if all(
            supplied.get(f"command_{gi}_{j}", int(oi == j)) == int(oi == j)
            for j in range(len(options)))]
        if not choices:
            raise ValueError(f"Staged restrictions conflict with variable_fixing in {group}")
        for oi in range(len(options)):
            name = f"command_{gi}_{oi}"
            if oi not in choices:
                fixes[name] = 0
            elif len(choices) == 1:
                fixes[name] = 1
        initial.append(choices[0])
    return fixes, tuple(initial)


def solve_staged(problem, *, max_evaluations=None, guidance=None, threads=1):
    """Return ``(caller_owned_master, solution)`` like JointDispatchModel.solve.

    Use the problem's absolute perf_counter deadline, splitting remaining wall
    time equally. Phase one gets ceil(cap/2) attempts; phase two gets all unspent
    attempts, including unused phase-one allocation. None means no evaluation
    cap. Cached points are free; timed-out physical attempts still consume it.
    A zero cap permits cached incumbents only. Caller hard fixes are intersected
    with both phases, never overwritten; incompatible fixes raise ValueError.

    Only exact evaluation points and additive timings are merged into problem.
    No master, cuts or phase lower bounds are carried into the next master.
    Per-phase budgets, attempts, selected options and hard fixes are exposed in
    solution['staged_diagnostics']. Top-level lower bound/gap are deliberately None.
    Conflicting warm-start hints are released and recorded; hard fixes are not.
    """
    if max_evaluations is not None and (type(max_evaluations) is not int or max_evaluations < 0):
        raise ValueError("max_evaluations must be a nonnegative integer or None")
    if type(threads) is not int or threads < 0:
        raise ValueError("threads must be a nonnegative integer")
    started, deadline = time.perf_counter(), problem.deadline
    if not isinstance(deadline, (int, float)) or math.isnan(deadline):
        raise ValueError("problem.deadline must be an absolute perf_counter deadline")
    guidance = copy.deepcopy(guidance or {})
    supplied = guidance.get("variable_fixing", {})
    names = {f"command_{gi}_{oi}" for gi, options in enumerate(problem.groups.values())
             for oi in range(len(options))}
    if not isinstance(supplied, dict) or any(
            name not in names or type(value) not in (int, float) or value not in (0, 1)
            for name, value in supplied.items()):
        raise ValueError("variable_fixing needs known command variables and binary values")
    kinds = [[_kind(option) for option in options] for options in problem.groups.values()]
    reference = []
    for (group, _), row in zip(problem.groups.items(), kinds):
        if "continuation" not in row:
            raise ValueError(f"Staged group {group} needs an explicit empty reference continuation")
        reference.append(row.index("continuation"))
    blocked, protected_vehicles, protected_resources = _protected(problem)
    allowed = [[reference[gi]] + [oi for oi, kind in enumerate(row)
                                if kind == "operation" and (gi, oi) not in blocked]
               for gi, row in enumerate(kinds)]
    fixes1, initial = _fixes(problem, allowed, supplied)
    phase1_deadline = min(deadline, started + max(0., deadline - started) / 2)
    phase1_cap = None if max_evaluations is None else (max_evaluations + 1) // 2
    masters, phases = [], []
    keep = None
    spent = completed = 0
    initial_cached = len(problem.evaluated)

    def run(source, name, phase_deadline, cap, fixes, anchor):
        nonlocal spent, completed
        before = time.perf_counter()
        child = source.fork(deadline=phase_deadline, reuse_evaluations=True)
        # The earlier restricted master may discard incompatible cached points.
        child.evaluated.update(problem.evaluated)
        attempts = finished = 0
        evaluate = child.evaluate

        def counted(selection, **kwargs):
            nonlocal spent, completed, attempts, finished
            key = tuple(selection)
            if key in child.evaluated:
                return child.evaluated[key]
            if (time.perf_counter() >= phase_deadline or cap is not None and attempts >= cap
                    or max_evaluations is not None and spent >= max_evaluations):
                raise EvaluationTimeout()
            attempts += 1
            spent += 1
            value = evaluate(selection, **kwargs)
            finished += 1
            completed += 1
            return value

        child.evaluate = counted
        phase_guidance = dict(guidance, variable_fixing=fixes)
        released = []
        warm = phase_guidance.get("warm_start")
        if isinstance(warm, dict):
            warm = dict(warm)
            for variable, value in list(warm.items()):
                if variable in fixes and value != fixes[variable]:
                    released.append(dict(variable=variable, value=value,
                                         reason="incompatible_with_phase_fixing"))
                    del warm[variable]
            phase_guidance["warm_start"] = warm
        try:
            master, result = child.solve(max_evaluations=cap, guidance=phase_guidance,
                                         initial_selection=anchor, threads=threads)
            masters.append(master)
        finally:
            problem.evaluated.update(child.evaluated)
            for key, seconds in child.timings.items():
                problem.timings[key] += seconds
        phases.append(dict(
            name=name, status=result["status"], termination=result.get("termination"),
            deadline=phase_deadline, wall_budget_s=max(0., phase_deadline - before),
            elapsed_s=time.perf_counter() - before, evaluation_cap=cap,
            physical_evaluations=attempts, completed_physical_evaluations=finished,
            shared_reference_evaluations=result.get("shared_reference_evaluations", 0),
            selected_options=dict(result.get("selected_options", {})), variable_fixing=dict(fixes),
            released_hints=released,
            cost=result.get("cost"), cost_lower_bound=result.get("cost_lower_bound"),
            gap=result.get("gap"), certificate_scope=f"{name} restricted domain only; {_SCOPE}"))
        return child, master, result

    try:
        child1, master1, result1 = run(problem, "operations", phase1_deadline, phase1_cap, fixes1, initial)
        chosen, selected_operations, fixes2 = None, {}, {}
        returned, result = master1, result1
        termination = "phase1_no_incumbent"
        if result1.get("milp_incumbent"):
            chosen = tuple(next(i for i, option in enumerate(options)
                                if option.id == result1["selected_options"][group])
                           for group, options in problem.groups.items())
            vehicles, resources = set(), set()
            for gi, (group, options) in enumerate(problem.groups.items()):
                if kinds[gi][chosen[gi]] == "operation":
                    selected_operations[group] = options[chosen[gi]].id
                    touched, controls = _footprint(problem, options[chosen[gi]])
                    vehicles.update(touched)
                    resources.update(controls)
            allowed2 = []
            for gi, (group, options) in enumerate(problem.groups.items()):
                if group in selected_operations:
                    allowed2.append([chosen[gi]])
                else:
                    choices = []
                    for oi, option in enumerate(options):
                        touched, controls = _footprint(problem, option)
                        if (kinds[gi][oi] != "operation" and (gi, oi) not in blocked
                                and not vehicles & touched and not resources & controls):
                            choices.append(oi)
                    allowed2.append(choices)
            fixes2, _ = _fixes(problem, allowed2, supplied)
            remaining = None if max_evaluations is None else max_evaluations - spent
            if time.perf_counter() >= deadline:
                termination = "wall_time_limit"
            elif remaining == 0:
                termination = "evaluation_limit"
            else:
                _, master2, result2 = run(child1, "movement", deadline, remaining, fixes2, chosen)
                if result2.get("milp_incumbent"):
                    returned, result = master2, result2
                    termination = "phase2_completed"
                else:
                    termination = "phase2_no_incumbent"
        if len(phases) == 1:
            phases.append(dict(name="movement", status="skipped", termination=termination,
                               deadline=deadline, wall_budget_s=0., elapsed_s=0., evaluation_cap=0,
                               physical_evaluations=0, completed_physical_evaluations=0,
                               shared_reference_evaluations=0, selected_options={},
                               variable_fixing=fixes2, released_hints=[], cost=None,
                               cost_lower_bound=None, gap=None,
                               certificate_scope=f"movement phase not run; {_SCOPE}"))
        solution = dict(result)
        solution.update(
            optimization_mode="staged",
            status="staged_feasible" if result.get("milp_incumbent") else "staged_no_incumbent",
            termination=termination, cost_lower_bound=None, gap=None,
            global_optimality_proven=False, certificate_scope=_SCOPE,
            physical_evaluations=spent, completed_physical_evaluations=completed,
            evaluations=len(problem.evaluated), shared_reference_evaluations=initial_cached,
            timings=dict(problem.timings), solve_time=time.perf_counter() - started,
            staged_diagnostics=dict(
                phases=phases, deadline=deadline, wall_budget_s=max(0., deadline - started),
                max_evaluations=max_evaluations, physical_evaluations=spent,
                completed_physical_evaluations=completed,
                reference_selection=dict(zip(problem.groups, reference)),
                phase1_selection=dict(result1.get("selected_options", {})),
                selected_operations=selected_operations,
                bundle_classification={g: {o.id: kind for o, kind in zip(options, row)}
                                       for (g, options), row in zip(problem.groups.items(), kinds)},
                protected_vehicles=sorted(protected_vehicles),
                protected_resources=sorted(protected_resources),
                guidance_fixing_policy="preserve hard fixes or reject incompatible restrictions",
                released_hints=[dict(phase=phase["name"], **hint)
                                for phase in phases for hint in phase["released_hints"]],
                returned_phase=2 if len(masters) == 2 and returned is masters[1] else 1,
                certificate_scope=_SCOPE, global_optimality_proven=False))
        keep = returned
        return returned, solution
    finally:
        for master in masters:
            if master is not keep:
                master.dispose()
