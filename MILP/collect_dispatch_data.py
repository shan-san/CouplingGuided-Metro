"""Collect conditional CPLEX teachers, measured search guidance and bounded SFT.

Physical execution remains one-second. Episode duration, adaptive planning
windows and solver wall-clock budgets are separate. No API or private future
event-clearance/passenger realization is used by a plan.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if __name__ == "__main__":
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--output")
    early, _ = bootstrap.parse_known_args()
    if early.output:
        os.environ["METRO_LOG_DIRECTORY"] = str(Path(early.output).resolve()/"runtime_logs")

from dispatch_candidates import applicable_options, public_dispatch_state
from dispatch_plan import PARAMETERS
from dispatch_timing import decision_seconds
from MILP.collection_io import (CollectionJournal, TokenizerWorker, atomic_json, collection_lock,
    directory_fingerprint, repair_failure_tail)
from MILP.collection_protocol import StratifiedSampler, choose_horizon
from MILP.collection_shards import partition_episodes, shard_metadata
from MILP.collection_quality import (checked_policy_reference, distinct_views, prompt_fingerprint, bundle_coverage,
                                     dataset_readiness)
from MILP.operation_model import (JointDispatchModel, apply_dispatch_step, audit_dispatch_step,
    planning_copy, nominal_action, state_guard, guard_matches)
from MILP.guidance_labels import variable_catalog, teacher_guidance, neighborhood_seeds, families
from LLM.dispatch_coordination import checked_reference
from LLM.coupling_trigger import option_footprints, event_relevant
from RL.dispatch_training import NativeDispatchPolicy, balanced_episodes
from scenarios.prepare_dispatch import make_world, events_for, add_curriculum_args, scenarios_from_args
from LLM.small_model_guidance import NATIVE_GUIDANCE_SYSTEM as GUIDANCE_SYSTEM

VERSION = "coupled_teacher_v9"


def public_state(world, coupling=None):
    return public_dispatch_state(world, coupling)


def evaluate_common_window(world, solution, inputs, *, reuse_exact=False):
    packets = solution.get("dispatch_packets", [])
    if (reuse_exact and not solution.get("continuation_mode", "").startswith("reviewed_second_targets_")
            and packets and len(packets) == len(inputs) and solution.get("cost") is not None
            and guard_matches(packets[0]["guard"], state_guard(world))
            and all(guard_matches(p["arrival_inputs"], arrivals) for p, arrivals in zip(packets, inputs))):
        return dict(feasible=True, cost=solution["cost"], validation="shared_exact_transition_already_evaluated")
    if not packets:
        return dict(feasible=False, reason="no_complete_incumbent")
    probe = planning_copy(world)
    gamma = probe.dispatch.objective.weights["discount"]
    cost, extension = 0., None
    for k, arrivals in enumerate(inputs):
        if k < len(packets):
            packet_solution, index = solution, k
        else:
            if extension is None:
                try:
                    extension = checked_reference(probe, {a.name: nominal_action(probe, a) for a in probe.agents},
                                                  len(inputs)-k, inputs[k:])
                except RuntimeError as exc:
                    if str(exc) not in ("No checked fallback: ongoing operations require joint replanning",
                                        "Reference review did not converge"):
                        raise
                    return dict(feasible=False, reason="no_checked_conditional_continuation", details=str(exc))
            packet_solution, index = extension, k-len(packets)
        if not apply_dispatch_step(probe, packet_solution, index)["accepted"]:
            return dict(feasible=False, reason="conditional_plan_guard_mismatch")
        probe.step(arrival_inputs=arrivals)
        audit = audit_dispatch_step(probe, packet_solution, index)
        if not audit["matched"] or audit["constraint_interventions"]:
            return dict(feasible=False, reason="conditional_execution_mismatch", audit=audit)
        cost += gamma**k * probe.dispatch.interval_cost["total"]
    cost += gamma**len(inputs) * probe.dispatch.objective.terminal(probe)["total"]
    return dict(feasible=True, cost=cost, validation="independent_common_window_replay")


def collection_couplings(model, masks):
    """Current dependencies, without a second model or sparse persistent trigger.

    Aggregate applicable-option footprints per group. A shared footprint denotes
    potential dependence, not a predicted conflict or an execution certificate.
    """
    keys = ("vehicles", "nodes", "depots", "tasks", "platforms", "families", "resources")
    merged = {}
    footprints = option_footprints(model)
    events = [e for e in model.base.get_published_events() if e.get("active", True)]
    from MILP.guidance_labels import public_option_priority
    priorities = {}
    for group, options in model.groups.items():
        row = {key: set() for key in keys}
        priorities[group] = {}
        for option, allowed, fp in zip(options, masks[group], footprints[group]):
            if allowed:
                for key in keys:
                    row[key].update(fp[key])
                relevant_events = [e for e in events if event_relevant(e, fp, model.base)]
                priorities[group][option.id] = public_option_priority(model.base, option, fp, relevant_events)
        merged[group] = row
    relevant = {g: {e["event_id"] for e in events if event_relevant(e, fp, model.base)} for g, fp in merged.items()}
    evidence = []
    for (g, left), (h, right) in combinations(merged.items(), 2):
        relations = []
        if left["nodes"] & right["nodes"] or left["resources"] & right["resources"]:
            relations.append("C1")
        if any(left[key] & right[key] for key in ("vehicles", "depots", "tasks")):
            relations.append("C2")
        if any(model.base.waiting_at(p) > 0 for p in left["platforms"] & right["platforms"]):
            relations.append("C3")
        if "C1" in relations and (len(left["nodes"]) > 1 or len(right["nodes"]) > 1):
            relations.append("C4")
        if relations:
            evidence.append(dict(groups=[g, h], relations=relations, active=True,
                affected_event_ids=sorted(relevant[g] | relevant[h]),
                action_families=sorted(left["families"] | right["families"])))
    return dict(evidence=evidence, option_priority=priorities,
        covered_action_families=sorted(set().union(*(v["families"] for v in merged.values()))),
        basis="shared current applicable-option footprints, not persistent-trigger decisions")


def quality_check(result, reference, solution, args):
    if not solution.get("milp_incumbent") or not result.get("feasible"):
        return False, "no_checked_teacher"
    if not reference.get("feasible"):
        return False, "reference_not_feasible"
    if solution.get("gap") is None or solution["gap"] > args.max_gap:
        return False, "finite_domain_gap_limit"
    gain = reference["cost"]-result["cost"]
    threshold = max(args.min_absolute_improvement, args.min_relative_improvement*abs(reference["cost"]))
    if gain < -1e-6:
        return False, "worse_than_reference"
    if args.require_improvement and gain <= max(1e-6, threshold):
        if solution.get("physical_evaluations") == 0 and solution.get("status") != "optimal_in_domain":
            return False, "teacher_search_no_completed_candidate"
        if solution.get("feasible_candidate_evaluations") == 0 and solution.get("status") != "optimal_in_domain":
            return False, "no_feasible_intervention_within_budget"
        return False, "insufficient_reference_improvement"
    return True, "improved" if gain > 1e-6 else "noninferior"


def guidance_effect(plain, guided, *, tolerance=1e-6):
    a, b = plain.get("common_evaluation", {}), guided.get("common_evaluation", {})
    feasible = bool(guided.get("milp_incumbent") and b.get("feasible"))
    plain_ok = bool(plain.get("milp_incumbent") and a.get("feasible"))
    noninferior = feasible and (not plain_ok or b["cost"] <= a["cost"]+tolerance)
    gain = a["cost"]-b["cost"] if feasible and plain_ok else None
    fewer = bool(noninferior and plain_ok and guided.get("status") == plain.get("status") == "optimal_in_domain"
                 and guided.get("physical_evaluations", guided.get("evaluations", 0))
                     < plain.get("physical_evaluations", plain.get("evaluations", 0)))
    attainment = {}
    if noninferior and plain_ok and abs(gain) <= tolerance:
        target = a["cost"]
        for arm, value in (("plain", plain), ("guided", guided)):
            # Local search costs can be compared only when they are also the
            # common-window costs. Never use a shorter-window surrogate here.
            if value.get("cost") is not None and abs(value["cost"]-target) <= tolerance:
                hits = [r["physical_evaluations"] for r in value.get("incumbent_history", [])
                        if r["cost"] <= target+tolerance]
                if hits:
                    attainment[arm] = min(hits)
    earlier = (set(attainment) == {"plain", "guided"} and attainment["guided"] < attainment["plain"])
    useful = noninferior and (not plain_ok or gain > tolerance or fewer or earlier)
    return dict(noninferior=bool(noninferior), useful=bool(useful), objective_gain=gain,
        guided_only_feasible=feasible and not plain_ok, fewer_physical_trials=fewer,
        fewer_trials_to_same_quality=bool(earlier), target_attainment_trials=attainment,
        interpretation="paired finite-budget evidence, not global optimality or learned-model performance")


def solve_copy(template, budget, args, seeds, guidance=None, *, max_evaluations=None):
    reference_selection = getattr(template, "collection_reference_selection", None)
    search = template.fork(deadline=time.perf_counter()+budget, reference_selection=reference_selection)
    native = None
    started = time.perf_counter()
    try:
        native, solution = search.solve(max_evaluations=args.max_evaluations if max_evaluations is None else max_evaluations, guidance=guidance,
            seed_selections=seeds, initial_selection=reference_selection,
            adaptive_seed_completion=True, threads=args.threads, solver_parameters={"Seed": args.solver_seed})
        solution["search_wall_s"] = time.perf_counter()-started
        return solution
    finally:
        if native is not None:
            native.dispose()


def solver_record(solution):
    keys = ("status", "termination", "milp_incumbent", "cost", "gap", "domain_size", "cost_lower_bound",
            "cost_upper_bound", "certificate_scope", "selected_options", "solve_time", "evaluations", "seed_trials",
            "solver_backend", "solver_version", "search_wall_s", "timings", "construction_timings", "seed_records",
            "physical_evaluations", "shared_reference_evaluations", "incumbent_history", "continuation_mode",
            "reviewed_continuation_waits", "proposal_state_rebased_stages", "feasible_candidate_evaluations", "infeasible_candidate_reasons")
    return {key: solution[key] for key in keys if key in solution}


def execution_coverage(world, seen, coverage, label_commands=None):
    for cid, result in world.dispatch.command_results.items():
        action, status = result["action"], result["status"]
        if status in ("running", "completed") and (cid, "started") not in seen:
            coverage["execution_started_families"].update([action])
            if label_commands is not None and cid in label_commands:
                coverage["label_execution_started_families"].update([action])
            seen.add((cid, "started"))
        if status in ("completed", "failed", "rejected", "cancelled") and (cid, status) not in seen:
            coverage["execution_completed_families" if status == "completed" else "execution_failed_families"].update([action])
            if label_commands is not None and cid in label_commands and status == "completed":
                coverage["label_execution_completed_families"].update([action])
            seen.add((cid, status))


def code_identity(args, recipe):
    files = ("MILP/collect_dispatch_data.py", "MILP/collection_io.py", "MILP/collection_protocol.py", "MILP/collection_shards.py",
        "MILP/guidance_labels.py", "MILP/collection_quality.py", "MILP/operation_model.py", "MILP/cplex_backend.py", "dispatch_operations.py",
        "dispatch_candidates.py", "dispatch_objective.py", "metro_world.py", "metro_agent_v2.py",
        "RL/dispatch_training.py", "LLM/dispatch_coordination.py", "LLM/coupling_trigger.py",
        "LLM/finetune/guidance_view.py", "LLM/finetune/prepare_sft.py", "LLM/finetune/chat_format.py",
        "dispatch_plan.py", "dispatch_timing.py", "metro_scenario_v2.py", "topology/passenger_flow.py",
        "buildtopology.py", "timetable.py", "route_selection.py")
    configuration = {k: v for k, v in vars(args).items() if k not in ("resume", "overwrite", "preflight_only", "output", "stop_after_states")}
    return dict(version=VERSION, scenario_sha256=recipe["sha256"], configuration=configuration,
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        reference_checkpoint_sha256=hashlib.sha256(Path(args.marl_checkpoint).read_bytes()).hexdigest() if args.marl_checkpoint else None,
        tokenizer=directory_fingerprint(args.tokenizer))


def preflight_solver():
    from MILP import cplex_backend as mip
    native = mip.Model("teacher_full_license_preflight")
    try:
        native.Params.OutputFlag = 0
        native.addVars(2101, vtype=mip.BINARY)
        native.optimize()
        if not native.SolCount:
            raise RuntimeError("CPLEX preflight returned no solution")
        return dict(solver_backend="cplex", solver_version=native.native_model.get_cplex().get_version(), full_size_check=2101)
    finally:
        native.dispose()


def selected_episodes(args):
    recipe, entries = scenarios_from_args(args, args.split)
    if args.episode_order == "balanced":
        entries = balanced_episodes(entries, recipe["seed"])
    entries = entries[:args.episodes]
    if args.split == "validation" and args.validation_normal_samples == 0:
        entries = [e for e in entries if e["scenario"] != "normal"]
    if not entries:
        raise ValueError("No selected episodes")
    part = shard_metadata(entries, args.num_shards, args.shard_index)
    return recipe, partition_episodes(entries, args.num_shards)[args.shard_index], part


def sampler_for_episode(args, entry, tick_seconds):
    normal_limit = args.validation_normal_samples if args.split == "validation" else args.normal_samples
    # A repeat's pre-event world is shared by every scenario; retain that
    # warm-up once, in the normal episode, rather than once per shard.
    return StratifiedSampler(tick_seconds=tick_seconds, interval=args.decision_interval,
        normal_limit=normal_limit if entry["scenario"] == "normal" else 0,
        phase_limit=args.phase_samples, opportunity_limit=args.opportunity_samples,
        min_gap=args.sample_min_gap, successful_quota=True, attempt_multiplier=args.sample_attempt_multiplier,
        combination_quota=True)


def collect(args):
    if args.split == "test":
        raise ValueError("Test episodes cannot enter SFT/RAG knowledge")
    if not args.tokenizer:
        raise ValueError("--tokenizer is required: raw long snapshots are archives, not training samples")
    if (min(args.steps, args.episodes, args.decision_interval, args.max_horizon, args.checkpoint_every) < 1
            or min(args.horizons) < 1 or min(args.time_limits) <= 0 or args.guidance_validation_budget <= 0
            or min(args.max_evaluations, args.guidance_max_evaluations) < 1 or args.guidance_variable_limit < 1 or args.max_tokens < 513
            or args.torch_threads < 1 or args.sample_min_gap < 0 or args.sample_attempt_multiplier < 1
            or min(args.normal_samples, args.validation_normal_samples, args.phase_samples, args.continuation_groups) < 0
            or min(args.min_absolute_improvement, args.min_relative_improvement) < 0):
        raise ValueError("Invalid sampling, horizon, quality or budget setting")
    if args.resume and args.overwrite:
        raise ValueError("Choose --resume or --overwrite, not both")
    recipe, entries, part = selected_episodes(args)
    identity = code_identity(args, recipe)
    identity["sharding"] = part
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(args.torch_threads)
    policy = NativeDispatchPolicy.load(args.marl_checkpoint) if args.marl_checkpoint else None
    backend = preflight_solver()
    identity["runtime"] = dict(python=sys.version, torch=torch.__version__, **backend)
    worker = TokenizerWorker(args.tokenizer, python=args.tokenizer_python, max_tokens=args.max_tokens,
                             continuation_groups=args.continuation_groups)
    try:
        if args.preflight_only:
            report = dict(event="collection_preflight_complete", **backend, tokenizer=worker.ready,
                scenario_count=len(entries), sharding=part,
                reference_source="native_marl" if policy else "published_route_rule",
                physical_tick_seconds=recipe["physical_tick_seconds"], decision_seconds=recipe["decision_seconds"])
            print(json.dumps(report), flush=True)
            return report
        with collection_lock(output):
            return _collect(args, recipe, entries, output, identity, policy, worker, backend)
    finally:
        worker.close()


def _collect(args, recipe, entries, output, identity, policy, worker, backend):
    journal = CollectionJournal(output, identity, resume=args.resume, overwrite=args.overwrite)
    try:
        return _run_collection(args, recipe, entries, output, identity, policy, worker, backend, journal)
    finally:
        journal.close()


def _run_collection(args, recipe, entries, output, identity, policy, worker, backend, journal):
    from LLM.rag.enhanced_rag_manager import CaseLibrary, StateEncoder
    from LLM.finetune.guidance_view import SCHEMA, FIELDS, merge_guidance
    library, encoder = CaseLibrary(max_cases=100000), StateEncoder(use_embeddings=False)
    counters = Counter(solved_states=0, configurations=0, teacher_qualified=0, qualified=0, sft_samples=0,
        rejected=0, improved=0, tied=0, errors=0, completed_episodes=0, marl_reference_calls=0,
        cached_stages=0, route_reviews=0, rollout_mismatches=0, guidance_pairs=0)
    coverage = {k: Counter() for k in ("strata", "sample_reasons", "active_events", "candidate_families", "applicable_families",
        "scheduled_seed_families", "tested_seed_families", "selected_families", "selected_implicit_controls", "exported_families", "execution_started_families",
        "execution_completed_families", "execution_failed_families", "label_execution_started_families",
        "label_execution_completed_families", "rejection_reasons", "planning_horizons", "token_lengths",
        "accepted_strata", "accepted_event_combinations", "accepted_measure_combinations",
        "joint_event_measure_combinations")}
    timing, seen_execution = Counter(), set()
    cursor = dict(episode=0, stage=0, world=None, sampler=None, cached=None, cached_index=0, label_commands=set(),
                  seen_prompts=set())
    elapsed_prior = 0.
    if journal.restored:
        saved = journal.restored
        counters.update(saved["counters"])
        coverage = saved["coverage"]
        timing, seen_execution, cursor = saved["timing"], saved["seen_execution"], saved["cursor"]
        library.cases, library.stats = saved["library_cases"], saved["library_stats"]
        library.case_index = {case["case_id"]: i for i, case in enumerate(library.cases)}
        elapsed_prior = saved["elapsed_s"]
    failures = output/"failures.jsonl"
    if failures.exists():
        counters["errors"] = repair_failure_tail(failures)
    started = time.perf_counter()
    stop = [False]
    previous_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    source = dict(source_topology=recipe["topology"], source_split=args.split, scenario_recipe=recipe,
        sharding=identity.get("sharding", shard_metadata(entries)),
        sft_export_version=VERSION, solver_backend="cplex", identity=identity,
        reference_source="native_marl_same_current_catalog" if policy else "published_route_rule",
        reference_interpretation="One current policy selection followed by checked conditional continuation, not H future inferences.",
        token_checked=True, tokenizer=worker.ready, max_tokens=args.max_tokens,
        sft_format=SCHEMA, reference_checkpoint=args.marl_checkpoint,
        collection_config=vars(args), **{k: v for k, v in backend.items() if k != "solver_backend"})
    if not args.resume:
        atomic_json(output/"run_manifest.json", source)

    def progress(event, **details):
        report = dict(event=event, completed=False, episode_index=cursor["episode"], total_episodes=len(entries),
            stage=cursor["stage"], elapsed_s=elapsed_prior+time.perf_counter()-started,
            counters=dict(counters), coverage={k: dict(v) for k, v in coverage.items()}, timing_s=dict(timing), **details)
        atomic_json(output/"progress.json", report)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
        return report

    def checkpoint():
        checkpoint_started = time.perf_counter()
        journal.commit(dict(counters=dict(counters), coverage=coverage, timing=timing, seen_execution=seen_execution,
            cursor=cursor, library_cases=library.cases, library_stats=library.stats,
            elapsed_s=elapsed_prior+time.perf_counter()-started))
        timing["checkpoint_s"] += time.perf_counter()-checkpoint_started
        if library.cases and counters["solved_states"] % args.checkpoint_every == 0:
            atomic_json(output/"rag.partial.json", dict(cases=library.cases, stats=library.stats,
                collection_complete=False, committed_sampled_states=counters["solved_states"]))

    reuse_exact = not args.verify_teacher_replay
    try:
        if not args.resume:
            checkpoint()
        while cursor["episode"] < len(entries):
            entry = entries[cursor["episode"]]
            if cursor["world"] is None:
                phase_start = time.perf_counter()
                cursor["world"] = make_world(recipe["topology"], fleet=args.n_agents, seed=entry["seed"],
                    events=events_for(entry, fleet=args.n_agents), tick=recipe["tick_seconds"],
                    initial_passenger_seconds=recipe["tick_seconds"])
                cursor["sampler"] = sampler_for_episode(args, entry, recipe["tick_seconds"])
                timing["world_build_s"] += time.perf_counter()-phase_start
                seen_execution = set()
                progress("episode_started", case_id=entry["case_id"])
                checkpoint()
            world, sampler = cursor["world"], cursor["sampler"]
            while cursor["stage"] < args.steps:
                if stop[0] or (args.stop_after_states and counters["solved_states"] >= args.stop_after_states):
                    checkpoint()
                    return progress("collection_paused", reason="signal_or_requested_smoke_limit")
                stage = cursor["stage"]
                sampling = sampler.observe(world, stage) if args.sampling == "stratified" else (
                    dict(stratum="periodic", reason="periodic", phase="periodic", time_s=float(world.sim_time[0]))
                    if stage % args.decision_interval == 0 else None)
                selected, selected_stage = None, 0
                if sampling:
                    counters["solved_states"] += 1
                    coverage["strata"].update([sampling["stratum"]])
                    coverage["sample_reasons"].update([sampling["reason"]])
                    info = choose_horizon(world, base=max(args.horizons), maximum=args.max_horizon,
                        tail_stages=args.service_tail_stages, remaining_stages=args.steps-stage,
                        recovery_work_only=True,
                        prospective_stock_actions={op["action"] for op in sampling.get("opportunities", [])}
                        ) if args.horizon_policy == "adaptive" else dict(
                        horizon=min(max(args.horizons), args.steps-stage), reasons=["fixed_horizons"], censored=False)
                    horizons = [info["horizon"]] if args.horizon_policy == "adaptive" else sorted({min(h, args.steps-stage) for h in args.horizons})
                    common_horizon = min(args.steps-stage, max(*horizons, args.evaluation_horizon))
                    current = {p: n.passenger_flow.interval_arrivals(world.event_view(published=True), decision_seconds(world))
                               for p, n in world.platform_nodes.items()}
                    inputs = [copy.deepcopy(current) for _ in range(common_horizon)]
                    uid = f"{entry['case_id']}_stage{stage}"
                    progress("state_build_started", case_id=entry["case_id"], sampling=sampling, horizon_policy=info)
                    phase_start = time.perf_counter()
                    templates = {h: JointDispatchModel(world, h, passenger_inputs=inputs[:h], reviewed_continuation=True)
                                 for h in sorted(set(horizons+[common_horizon]))}
                    timing["state_build_s"] += time.perf_counter()-phase_start
                    common_model = templates[common_horizon]
                    phase_start = time.perf_counter()
                    baseline = checked_policy_reference(common_model, policy)
                    common_model.collection_reference_selection = baseline["reference_selection"]
                    if policy:
                        counters["marl_reference_calls"] += 1
                    counters["reference_fallback_states"] += baseline["reference_policy_fallback"]
                    counters["reference_full_checks"] += baseline["reference_full_checks"]
                    timing.update(baseline["reference_timings"])
                    base_result = evaluate_common_window(world, baseline, inputs, reuse_exact=reuse_exact)
                    timing["reference_s"] += time.perf_counter()-phase_start
                    selected = baseline
                    phase_start = time.perf_counter()
                    masks_by_horizon = {h: applicable_options(templates[h])[0] for h in horizons}
                    coupling = collection_couplings(templates[max(horizons)], masks_by_horizon[max(horizons)])
                    state = public_state(world, coupling)
                    timing["applicability_and_coupling_s"] += time.perf_counter()-phase_start
                    coverage["active_events"].update({e["type"] for e in state["disruptions"] if e.get("active", True)})
                    qualified, accepted, joint = [], False, False
                    for horizon in horizons:
                        template, masks = templates[horizon], masks_by_horizon[horizon]
                        if template is not common_model:
                            local_reference = checked_policy_reference(template, policy)
                            template.collection_reference_selection = local_reference["reference_selection"]
                        coverage["planning_horizons"].update([str(horizon)])
                        seeds = neighborhood_seeds(template, masks, coverage["tested_seed_families"],
                            limit=args.neighborhood_trials, coupling=coupling,
                            anchor=template.collection_reference_selection, joint_first=True)
                        for seed in seeds:
                            for i, opts in zip(seed, template.groups.values()):
                                coverage["scheduled_seed_families"].update({c["action"] for c in opts[i].commands})
                        schema = variable_catalog(template)
                        invalid = []
                        for gi, (group, options) in enumerate(template.groups.items()):
                            for oi, option in enumerate(options):
                                name = f"command_{gi}_{oi}"
                                coverage["candidate_families"].update(families(schema[name]))
                                if masks[group][oi]:
                                    coverage["applicable_families"].update(families(schema[name]))
                                else:
                                    invalid.append(name)
                                    schema[name]["applicability_zero"] = True
                        for budget in args.time_limits:
                            counters["configurations"] += 1
                            progress("teacher_search_started", case_id=entry["case_id"], horizon=horizon, budget_s=budget)
                            phase_start = time.perf_counter()
                            solution = solve_copy(template, budget, args, seeds)
                            timing["teacher_search_s"] += time.perf_counter()-phase_start
                            # Proposed but timed-out/unreached seeds are not
                            # evidence that a measure has actually been explored.
                            for trial in solution.get("seed_records", []):
                                if trial.get("status") == "evaluated":
                                    coverage["tested_seed_families"].update({
                                        family for gi, oi in enumerate(trial["selection"])
                                        if oi != template.collection_reference_selection[gi]
                                        for family in families(schema[f"command_{gi}_{oi}"])})
                            phase_start = time.perf_counter()
                            result = evaluate_common_window(world, solution, inputs, reuse_exact=reuse_exact)
                            timing["teacher_validation_s"] += time.perf_counter()-phase_start
                            valid, reason = quality_check(result, base_result, solution, args)
                            record = dict(case_id=entry["case_id"], split=args.split, stage=stage, horizon=horizon,
                                budget_s=budget, evaluation_horizon=common_horizon, source_topology=recipe["topology"],
                                sampling=sampling, horizon_policy=info, reference_evaluation=base_result,
                                reference_policy_fallback=baseline["reference_policy_fallback"],
                                reference_full_checks=baseline["reference_full_checks"],
                                common_evaluation=result, qualified=bool(valid), qualification_reason=reason,
                                shared_construction_timings=dict(template.construction_timings), **solver_record(solution))
                            journal.write("solver_records.jsonl", record)
                            if valid:
                                qualified.append((result["cost"], solution["search_wall_s"], solution, template, budget, schema, invalid, seeds))
                            else:
                                coverage["rejection_reasons"].update([reason])
                    if qualified:
                        _, _, teacher, template, budget, schema, invalid, seeds = min(qualified, key=lambda v: (v[0], v[1]))
                        counters["teacher_qualified"] += 1
                        selected = teacher
                        guidance, label_coverage = teacher_guidance(schema, teacher["selected_options"], invalid=invalid,
                                                                  limit=args.guidance_variable_limit)
                        coverage["selected_families"].update(label_coverage["selected_families"])
                        coverage["selected_implicit_controls"].update(v["option"] for v in schema.values()
                            if teacher["selected_options"].get(v["group"]) == v["option"]
                            and not families(v) and (v.get("actions") or v.get("pauses")))
                        specification = dict(scope_rule="current_coupling_component", planning_horizon=template.horizon,
                            computational_budget=budget, focus_region_rule="shared vehicle, infrastructure and passenger-service relationships")
                        specification["continuation_mode"] = template.continuation_mode
                        payload = dict(state=state, specification=specification, solver_variables=schema)
                        phase_start = time.perf_counter()
                        bounded = worker.views(payload, guidance, teacher["selected_options"])
                        timing["tokenizer_s"] += time.perf_counter()-phase_start
                        views, duplicates = distinct_views(bounded["views"], cursor["seen_prompts"])
                        counters["duplicate_groups"] += sum(r["reason"] == "duplicate_prompt" for r in duplicates)
                        shape = bundle_coverage(views)
                        counters["overlength_groups"] += len(bounded["rejected"])
                        queries = [json.loads(v["messages"][1]["content"]) for v in views]
                        predictions = [json.loads(v["messages"][2]["content"]) for v in views]
                        exported = merge_guidance(queries, predictions) if views else {f: {} for f in FIELDS}
                        progress("guidance_validation_started", case_id=entry["case_id"], bounded_groups=len(views),
                            budget_s=args.guidance_validation_budget)
                        phase_start = time.perf_counter()
                        paired = {}
                        order = ("plain", "guided") if counters["teacher_qualified"] % 2 else ("guided", "plain")
                        if views:
                            shared_fixes = dict(exported["variable_fixing"])
                            for arm in order:
                                value = solve_copy(template, args.guidance_validation_budget, args, seeds,
                                    guidance=exported if arm == "guided" else dict(warm_start={},
                                        variable_fixing=shared_fixes, branching_priority={}),
                                    max_evaluations=args.guidance_max_evaluations)
                                common = evaluate_common_window(world, value, inputs, reuse_exact=reuse_exact)
                                paired[arm] = dict(solver_record(value), common_evaluation=common)
                            counters["guidance_pairs"] += 1
                            effect = guidance_effect(paired["plain"], paired["guided"])
                        else:
                            effect = dict(useful=False, noninferior=False, reason="no_bounded_training_groups")
                        timing["paired_guidance_validation_s"] += time.perf_counter()-phase_start
                        check = dict(effect, arms=paired, arm_order=list(order), equal_budget_s=args.guidance_validation_budget,
                            equal_max_evaluations=args.guidance_max_evaluations, same_solver_seed=args.solver_seed,
                            shared_applicability_fixes=dict(exported["variable_fixing"]),
                            same_candidate_catalog_sha256=hashlib.sha256(json.dumps(variable_catalog(template), sort_keys=True).encode()).hexdigest(),
                            independent_searches=True, teacher_evaluations_reused=False, tested_guidance=exported)
                        check["shared_reference_information"] = True
                        accepted = bool(views and effect["useful" if args.guidance_quality == "useful" else "noninferior"])
                        metadata = dict(case_id=uid, source_case_id=uid, split=args.split, source_topology=recipe["topology"],
                            sft_export_version=VERSION,
                            episode_id=entry["case_id"], episode_seed=entry["seed"], sampling=sampling, horizon_policy=info,
                            layout_revision=recipe["layout_revision"], solver_backend="cplex", teacher_status=teacher["status"],
                            teacher_gap=teacher["gap"], teacher_selection=teacher["selected_options"],
                            teacher_common_cost=min(q[0] for q in qualified), reference_common_cost=base_result["cost"],
                            reference_policy_fallback=baseline["reference_policy_fallback"],
                            certificate_scope=teacher["certificate_scope"], label_coverage=label_coverage,
                            guidance_validation=check, accepted_for_sft=accepted,
                            rejected_training_groups=bounded["rejected"]+duplicates,
                            joint_label_coverage=shape)
                        journal.write("teacher_states.jsonl", dict(messages=[dict(role="system", content=GUIDANCE_SYSTEM),
                            dict(role="user", content=json.dumps(payload, ensure_ascii=False)),
                            dict(role="assistant", content=json.dumps(guidance))], metadata=metadata))
                        journal.write("solver_records.jsonl", dict(event="guidance_validation", case_id=entry["case_id"],
                            stage=stage, horizon=template.horizon, accepted_for_sft=accepted, **check))
                        if accepted:
                            cursor["seen_prompts"].update(prompt_fingerprint(v) for v in views)
                            counters["multi_measure_states"] += shape["multi_measure"]
                            counters["joint_adjustment_states"] += shape["joint_adjustment"]
                            counters["active_event_states"] += bool(sampling.get("active_event_types"))
                            counters["multi_event_states"] += len(sampling.get("active_event_types", [])) > 1
                            joint = (shape["multi_measure"] and shape["coupled_adjustment"]
                                     and len(sampling.get("active_event_types", [])) > 1)
                            counters["coupled_joint_states"] += joint
                            coverage["accepted_strata"].update([sampling["stratum"]])
                            event_key = "+".join(sampling.get("active_event_types", [])) or "no_active_physical_event"
                            measure_key = "+".join(shape["measure_families"]) or "implicit_control"
                            coverage["accepted_event_combinations"].update([event_key])
                            coverage["accepted_measure_combinations"].update([measure_key])
                            if joint:
                                coverage["joint_event_measure_combinations"].update([event_key+" -> "+measure_key])
                            cursor["label_commands"].update(c["command_id"] for p in teacher["dispatch_packets"] for c in p["commands"])
                            counters["qualified"] += 1
                            counters["improved" if min(q[0] for q in qualified) < base_result["cost"]-1e-6 else "tied"] += 1
                            for view in views:
                                meta = {k: v for k, v in metadata.items() if k not in ("teacher_selection", "guidance_validation")}
                                meta.update(case_id=uid+"::"+view["group"], input_encoding=SCHEMA, group=view["group"],
                                    token_count=view["tokens"], tokenizer=bounded.get("tokenizer", args.tokenizer),
                                    max_tokens=args.max_tokens, changed_decision=view["changed"],
                                    guidance_validation={k: v for k, v in check.items() if k not in ("arms", "tested_guidance")})
                                journal.write("sft.jsonl", dict(messages=view["messages"], metadata=meta))
                                counters["sft_samples"] += 1
                                coverage["token_lengths"].update([str(view["tokens"])])
                                coverage["exported_families"].update(families(schema[view["selected_variable"]]))
                                coverage.setdefault("exported_control_families", Counter()).update(
                                    schema[view["selected_variable"]].get("control_measures", []))
                        else:
                            counters["rejected"] += 1
                            coverage["rejection_reasons"].update(["no_bounded_groups" if not views else "guidance_not_"+args.guidance_quality])
                        library.add_case(uid, encoder.encode_state(state), json.dumps(state, ensure_ascii=False),
                            action_plan=dict(specification=specification, selected_plan=dict(time_horizon=template.horizon,
                                time_limit=budget, scope_rule="current_coupling_component")),
                            outcome=dict(reward=-min(q[0] for q in qualified), reference_cost=base_result["cost"], feasible=True),
                            metadata=dict(source_topology=recipe["topology"], source_split=args.split, case_id=uid,
                                native_dispatch=True, teacher_status=teacher["status"], teacher_gap=teacher["gap"],
                                accepted_for_sft=accepted, scope_mapping="recompute current coupling; never copy source variable IDs"))
                    else:
                        counters["rejected"] += 1
                    sampler.feedback(sampling, accepted=bool(qualified and accepted), coupled_joint=bool(joint))
                    progress("state_complete", case_id=entry["case_id"], sampling=sampling)
                    del templates
                else:
                    cached, index = cursor["cached"], cursor["cached_index"]
                    if (args.reference_frequency != "stage" and cached is not None and index < len(cached["dispatch_packets"])
                            and guard_matches(cached["dispatch_packets"][index]["guard"], state_guard(world))):
                        selected, selected_stage = cached, index
                        counters["cached_stages"] += 1
                    else:
                        phase_start = time.perf_counter()
                        if policy and args.reference_frequency == "stage":
                            selected = policy.propose(world, deterministic=True)[0]
                            counters["marl_reference_calls"] += 1
                        else:
                            visible = planning_copy(world)
                            actions = {a.name: nominal_action(visible, a) for a in visible.agents}
                            selected = checked_reference(world, actions, min(max(args.horizons), args.steps-stage))
                            counters["route_reviews"] += 1
                        timing["rollout_reference_s"] += time.perf_counter()-phase_start
                phase_start = time.perf_counter()
                if not apply_dispatch_step(world, selected, selected_stage)["accepted"]:
                    raise RuntimeError("Offline replay commit rejected")
                world.step()
                audit = audit_dispatch_step(world, selected, selected_stage)
                execution_coverage(world, seen_execution, coverage, cursor["label_commands"])
                if not audit["matched"]:
                    counters["rollout_mismatches"] += 1
                    journal.write("solver_records.jsonl", dict(event="rollout_state_change", case_id=entry["case_id"],
                        stage=stage, audit=audit, note="Real inputs/reports changed; conditional continuation invalidated."))
                    cursor.update(cached=None, cached_index=0)
                else:
                    cursor.update(cached=selected, cached_index=selected_stage+1)
                timing["physical_rollout_s"] += time.perf_counter()-phase_start
                cursor["stage"] += 1
                if sampling or cursor["stage"] % args.checkpoint_every == 0:
                    checkpoint()
                if cursor["stage"] % args.checkpoint_every == 0:
                    progress("rollout_progress", case_id=entry["case_id"], simulation_seconds=float(world.sim_time[0]))
            counters["completed_episodes"] += 1
            cursor.update(episode=cursor["episode"]+1, stage=0, world=None, sampler=None, cached=None, cached_index=0, label_commands=set())
            checkpoint()
            progress("episode_complete", case_id=entry["case_id"])
        rag = dict(cases=library.cases, collection_complete=True)
        if library.cases:
            rag["stats"] = library.stats
        atomic_json(output/"rag.json", rag)
        missing = {key: sorted(set(PARAMETERS)-set(coverage[key])) for key in (
            "applicable_families", "selected_families", "exported_families", "execution_completed_families", "label_execution_completed_families")}
        summary = dict(source, **dict(counters), counters=dict(counters), completed=True, elapsed_s=elapsed_prior+time.perf_counter()-started,
            coverage={k: dict(v) for k, v in coverage.items()}, missing_measure_coverage=missing, timing_s=dict(timing),
            **dataset_readiness(counters, args.split), all_measures_supervised=not missing["exported_families"],
            archive="teacher_states.jsonl is not training input; sft.jsonl contains token-checked group examples",
            quality="Conditional feasible incumbents and paired guidance tests, not full-future global optima.")
        atomic_json(output/"summary.json", summary)
        atomic_json(output/"progress.json", dict(summary, event="collection_complete"))
        print(json.dumps(dict(event="collection_complete", output=str(output), **dict(counters),
            ready_for_sft=summary["ready_for_sft"], missing_measure_coverage=missing)), flush=True)
        return summary
    except BaseException as exc:
        counters["errors"] += 1
        repair_failure_tail(output/"failures.jsonl")
        with (output/"failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(event="collection_failed", episode=cursor["episode"], stage=cursor["stage"],
                error=str(exc), error_type=type(exc).__name__, elapsed_s=elapsed_prior+time.perf_counter()-started))+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        progress("collection_failed", error=str(exc), error_type=type(exc).__name__,
            resume="Resume uses the last committed world and truncates streams to its byte offsets.")
        raise
    finally:
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_curriculum_args(parser)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--sampling", choices=["stratified", "periodic"], default="stratified")
    parser.add_argument("--decision-interval", type=int, default=20)
    parser.add_argument("--normal-samples", type=int, default=1)
    parser.add_argument("--validation-normal-samples", type=int, default=0)
    parser.add_argument("--sample-min-gap", type=int, default=5)
    parser.add_argument("--sample-attempt-multiplier", type=int, default=2)
    parser.add_argument("--phase-samples", type=int, default=4)
    parser.add_argument("--opportunity-samples", type=int, default=4,
                        help="Reserved samples for newly available repair/reentry/rescue/service qualifications after ordinary quotas")
    parser.add_argument("--horizon-policy", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--horizons", nargs="+", type=int, default=[6])
    parser.add_argument("--max-horizon", type=int, default=60)
    parser.add_argument("--service-tail-stages", type=int, default=6)
    parser.add_argument("--evaluation-horizon", type=int, default=6)
    parser.add_argument("--time-limits", nargs="+", type=float, default=[60.])
    parser.add_argument("--max-evaluations", type=int, default=24)
    parser.add_argument("--neighborhood-trials", type=int, default=12)
    parser.add_argument("--guidance-validation-budget", type=float, default=30.)
    parser.add_argument("--guidance-max-evaluations", type=int, default=4,
                        help="Identical physical-trial cap for both fresh guidance-validation searches")
    parser.add_argument("--guidance-quality", choices=["useful", "noninferior"], default="useful")
    parser.add_argument("--episode-order", choices=["balanced", "sequential"], default="balanced")
    parser.add_argument("--reference-frequency", choices=["sample", "stage"], default="sample")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="Small tensor/scorer thread pool; no CPU affinity or CPLEX thread limit")
    parser.add_argument("--solver-seed", type=int, default=111)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--verify-teacher-replay", action="store_true")
    parser.add_argument("--max-gap", type=float, default=1.)
    parser.add_argument("--min-absolute-improvement", type=float, default=0.)
    parser.add_argument("--min-relative-improvement", type=float, default=0.)
    parser.add_argument("--marl-checkpoint", default=None)
    parser.add_argument("--n-agents", type=int, default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--require-improvement", action="store_true")
    parser.add_argument("--guidance-variable-limit", type=int, default=64)
    parser.add_argument("--tokenizer", help="Local target Qwen tokenizer, required for collection")
    parser.add_argument("--tokenizer-python", help="Use e.g. the Unsloth environment's Python for the tokenizer only")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--continuation-groups", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--stop-after-states", type=int, default=0, help="Pause after N sampled states for a smoke test")
    return parser


def main():
    collect(build_parser().parse_args())


if __name__ == "__main__":
    main()
