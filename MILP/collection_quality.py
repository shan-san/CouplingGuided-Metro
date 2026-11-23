"""Checked reference construction and label-blind collection deduplication."""
import hashlib
import json
import time

from dispatch_candidates import repair_joint_selection, solution_from_evaluation


def reference_indices(model):
    return tuple(next(i for i, op in enumerate(options)
                      if not op.commands and not op.actions and not op.pauses)
                 for options in model.groups.values())


def checked_policy_reference(model, policy=None):
    """Repair only the current action first, then check its full continuation once.

    All accepted trajectories still use the exact original second-level model.
    No long-window failed policy trials are repeated eight times. The returned
    reference is common information, available equally to subsequent searches.
    """
    default = reference_indices(model)
    selection = default
    short_trials = 0
    timings, repair = {}, None
    if policy is not None:
        started = time.perf_counter()
        selection, _, _ = policy.choose(model, deterministic=True)
        timings["policy_scoring_s"] = time.perf_counter()-started
        proposed = selection
        started = time.perf_counter()
        prefix = model.prefix(1) if model.horizon > 1 else model.fork()
        value = prefix.evaluate(selection)
        short_trials += 1
        if not value["feasible"]:
            repaired, repair = repair_joint_selection(prefix, selection, value, max_rounds=2)
            selection = (tuple(next(i for i, op in enumerate(options)
                                    if op.id == repaired["selected_options"][g])
                               for g, options in model.groups.items()) if repaired else default)
        timings["prefix_check_and_repair_s"] = time.perf_counter()-started
    else:
        proposed = selection
    choices = list(dict.fromkeys((selection, default)))
    waiting = tuple(next((i for i, op in enumerate(options)
                         if op.id in ("wait_window", "defer_movement")), default[j])
                    for j, options in enumerate(model.groups.values()))
    if waiting not in choices:
        choices.append(waiting)
    started = time.perf_counter()
    for index, trial in enumerate(choices):
        value = model.evaluate(trial)
        if value["feasible"]:
            model.evaluated[trial] = value
            solution = solution_from_evaluation(model, value, trial)
            solution.update(reference_selection=trial, reference_full_checks=index+1,
                            reference_prefix_checks=short_trials,
                            reference_policy_fallback=bool(trial != proposed),
                            reference_policy_repair=repair,
                            reference_timings=dict(timings, full_continuation_check_s=time.perf_counter()-started))
            return solution
    raise RuntimeError("No checked reference in the current collection catalog")


def prompt_fingerprint(view):
    return hashlib.sha256(json.dumps(view["messages"][:2], sort_keys=True,
                                     separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def distinct_views(views, seen):
    """Filter before guidance validation, never after certifying a larger bundle."""
    kept, rejected, pending = [], [], set()
    for view in views:
        key = prompt_fingerprint(view)
        if key in seen or key in pending:
            rejected.append(dict(group=view["group"], reason="duplicate_prompt"))
        else:
            kept.append(view)
            pending.add(key)
    if not any(v["changed"] for v in kept):
        return [], rejected + [dict(group=v["group"], reason="no_new_changed_group") for v in kept]
    return kept, rejected


def bundle_coverage(views):
    from LLM.finetune.guidance_view import expand_query
    families, controls, changed, relations = set(), set(), 0, set()
    changed_groups = {v["group"] for v in views if v["changed"]}
    linked = {}
    for view in views:
        query = expand_query(json.loads(view["messages"][1]["content"]))
        label = json.loads(view["messages"][2]["content"])
        changed += bool(view["changed"])
        relations.update(k for k, v in query.get("coupling", {}).get("relations", {}).items() if v)
        if view["changed"]:
            for peer, kinds in query.get("coupling", {}).get("relationships", {}).items():
                if peer in changed_groups:
                    pair = tuple(sorted((view["group"], peer)))
                    linked.setdefault(pair, set()).update(kinds)
        for name, value in label["warm_start"].items():
            if value == 1:
                variable = query["solver_variables"][name]
                families.update(c["action"] for c in variable["commands"])
                controls.update(variable.get("control_measures", []))
    families.update(controls)
    return dict(changed_groups=changed, measure_families=sorted(families),
                control_measure_families=sorted(controls),
                multi_measure=len(families)>1, joint_adjustment=changed>1,
                coupling_relations=sorted(relations), coupled_adjustment=bool(linked),
                coupled_decision_pairs=[dict(groups=list(pair), relations=sorted(kinds))
                                        for pair, kinds in sorted(linked.items())])


def dataset_readiness(counters, split):
    checks = dict(has_samples=counters.get("sft_samples", 0) > 0,
                  active_event_positives=counters.get("active_event_states", 0) > 0,
                  coupled_joint_positives=counters.get("coupled_joint_states", 0) > 0)
    return dict(ready_for_sft=all(checks.values()), readiness_checks=checks,
                readiness_scope="minimum coupled-data sanity checks, not coverage sufficiency or a performance guarantee",
                has_training_samples=checks["has_samples"], split=split)
