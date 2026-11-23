"""Run/replay native rolling baselines on five seeded metro topologies.

All methods see published inputs only. Complete episodes, not different planning
windows' objectives, are compared. A missing trained artifact is never a rule
fallback masquerading as a learned method.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.methods import BaselineController, BaselineSettings, METHODS, PURE_METHODS
from LLM.dispatch_coordination import checked_reference
from MILP.operation_model import (EvaluationTimeout, apply_dispatch_step, audit_dispatch_step, nominal_action,
                                  set_actions, state_guard, guard_matches, passenger_total)
from MILP.offline_baseline import SOURCE_FILES
from scenarios.prepare_dispatch import read_scenarios, make_world, events_for, current_arrivals, EPISODE_STAGES
from topology.metro_lines import LINE_SPECS, line_id
from dispatch_timing import timing_manifest
from experiment_results import EpisodeResults, RESULT_SCHEMA, replay_episode

BASELINE_SOURCES = (*SOURCE_FILES, "dispatch_candidates.py", "LLM/coupling_trigger.py",
                    "LLM/dispatch_coordination.py", "LLM/dispatch_scope.py",
                    "baseline/methods.py", "baseline/guidance.py", "baseline/prepare.py", "baseline/benchmark.py")
BASELINE_SOURCES = (*BASELINE_SOURCES, "baseline/heuristics.py")
BASELINE_SOURCES = (*BASELINE_SOURCES, "dispatch_diagnostics.py")


def fingerprints():
    return {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in BASELINE_SOURCES}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def state_fingerprint(world):
    return hashlib.sha256(json.dumps(state_guard(world), sort_keys=True).encode()).hexdigest()


def passenger_ledger(world, initial_mass, arrivals):
    completed = sum(float(a.cumulative_alighted[0]) for a in world.agents)
    remaining = passenger_total(world)
    residual = initial_mass + arrivals - completed - remaining
    return dict(initial_passengers=initial_mass, arrivals=arrivals, completed=completed,
                remaining=remaining, residual=residual,
                conserved=math.isclose(initial_mass+arrivals, completed+remaining, rel_tol=1e-7, abs_tol=1e-3))


def classify_execution_audit(audit, *, inputs_changed=False):
    if audit["matched"]:
        return "matched"
    if (set(audit["state_mismatches"]) <= {"events"} and audit["physical_trace_match"]
            and audit["positions_match"] and not audit["constraint_interventions"]):
        return "published_information_revision_only"
    return "changed_conditional_inputs" if inputs_changed else "unexpected_execution_mismatch"


def physical_event_signature(world):
    # Analysis only, after/before execution; never passed to a controller.
    return json.dumps([{k: e[k] for k in ("event_id", "type", "target", "parameters")}
                       for e in world.event_view().records], sort_keys=True)


def controller_for(method, args):
    return BaselineController(method, BaselineSettings(horizon=args.horizon, budget_s=args.budget,
        repair_s=args.repair_seconds, iterations=args.iterations, release_fraction=args.release_fraction,
        confidence=args.confidence, seed=args.seed, threads=args.threads, max_evaluations=args.max_evaluations,
        command_stages=tuple(args.command_stages), adaptive_rule_config=args.adaptive_rule_config,
        horizon_policy=getattr(args,"horizon_policy","operation_aware"),
        max_operation_horizon=getattr(args,"max_operation_horizon",12),
        minimum_repair_s=getattr(args, "minimum_repair_seconds", 0.),
        max_rule_horizon=max(args.max_rule_horizon, args.horizon), population_size=args.population_size),
        source_topology=line_id(args.source_line), retrieval_data=args.retrieval_data,
        mlp_checkpoint=args.mlp_checkpoint, strategy_path=args.llm_strategy, marl_checkpoint=args.marl_checkpoint)


def run_case(recipe, entry, method, args):
    output = Path(args.output)/entry["case_id"]/method
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    sources = fingerprints()
    controller = controller_for(method, args)
    world = make_world(recipe["topology"], fleet=args.n_agents, seed=entry["seed"],
        events=events_for(entry, fleet=args.n_agents), initial_passenger_seconds=recipe["tick_seconds"])
    initial_mass = passenger_total(world)+sum(float(a.cumulative_alighted[0]) for a in world.agents)
    arrivals_total = 0.
    trajectory_digest, inputs_digest = hashlib.sha256(), hashlib.sha256()
    audit_classes, failed_candidates = Counter(), Counter()
    feasible_novel = 0
    manifest = dict(method=method, entry=entry, recipe=recipe, n_agents=args.n_agents, steps=args.steps,
                    solver_backend=controller.provenance["solver_backend"],
                    source_sha256=sources, provider=controller.provenance,
                    objective_definition=world.dispatch.objective.manifest(),
                    initial_state_sha256=state_fingerprint(world),
                    online_information="published events and current declared arrivals; no actual future inputs",
                    synchronous_planning=True, clock_start="06:00:00", timing=timing_manifest(world))
    results = EpisodeResults(output, world, method=method, case_id=entry["case_id"], requested_stages=args.steps,
        entry=entry, recipe=recipe, configuration=manifest, source_sha256=sources)
    cost = waiting = onboard = headway = operation_units = decision_seconds = 0.
    exceedances = interventions = mismatches = fallback_count = completed_stages = 0
    seed_only = search_starved = strict_improvements = novel_evaluations = 0
    status, error, started = "complete", None, time.perf_counter()
    try:
        for stage in range(args.steps):
            request_start = time.perf_counter()
            try:
                solution, metadata = controller.decide(world)
            except EvaluationTimeout:
                # Fallback time is included, even when it exceeds the budget.
                fallback_count += 1
                solution = checked_reference(world, {a.name: nominal_action(world, a) for a in world.agents}, 1)
                metadata = dict(method=method, request_status="no_incumbent_before_budget", fallback=True,
                                global_optimality_proven=False)
            else:
                fallback_count += int(metadata.get("fallback", False))
            seed_only += int(metadata.get("seed_only", metadata.get("fallback", False)))
            search_starved += int(metadata.get("search_starved", metadata.get("fallback", False)))
            strict_improvements += int(metadata.get("strictly_improved", False))
            novel_evaluations += metadata.get("novel_candidate_evaluations", 0)
            feasible_novel += metadata.get("feasible_novel_evaluations", 0)
            failed_candidates.update(metadata.get("infeasible_candidate_reasons", {}))
            commit = apply_dispatch_step(world, solution)
            if not commit["accepted"]:
                raise RuntimeError(f"Dispatch commit rejected: {commit}")
            elapsed = time.perf_counter()-request_start
            decision_seconds += elapsed
            exceeded = elapsed > args.budget
            exceedances += int(exceeded)
            packet = solution["dispatch_packets"][0]
            physical_events_before = physical_event_signature(world)
            world.step()
            arrivals = world.last_arrival_inputs
            audit = audit_dispatch_step(world, solution)
            interventions += len(audit["constraint_interventions"])
            mismatches += int(not audit["matched"])
            inputs_changed = (physical_events_before != physical_event_signature(world)
                or not guard_matches(packet["expected_guard"].get("events"), world.get_published_events())
                or not guard_matches(packet["arrival_inputs"], arrivals)
                or not guard_matches(getattr(packet["arrival_inputs"], "profile", None), arrivals.profile))
            audit_class = classify_execution_audit(audit, inputs_changed=inputs_changed)
            audit_classes[audit_class] += 1
            arrivals_total += sum(sum(values.values()) for values in arrivals.values())
            ledger = passenger_ledger(world, initial_mass, arrivals_total)
            if not ledger["conserved"]:
                raise RuntimeError(f"Realized passenger conservation failed: {ledger}")
            interval = world.dispatch.interval_cost
            cost += world.dispatch.objective.weights["discount"]**stage * interval["total"]
            waiting += interval["waiting_person_seconds"]
            onboard += interval["onboard_person_seconds"]
            headway += interval["headway_deviation_seconds"]
            operation_units += interval["operation_units"]
            actual_execution_hash = hashlib.sha256(json.dumps(world.last_physical_audits, sort_keys=True).encode()).hexdigest()
            trajectory_digest.update(actual_execution_hash.encode())
            inputs_digest.update(json.dumps(dict(arrivals=arrivals, profile=arrivals.profile,
                published_events=world.get_published_events()), sort_keys=True).encode())
            results.annotate(decision_s=elapsed, budget_s=args.budget, metadata=metadata, audit=audit,
                audit_classification=audit_class, inputs_changed=inputs_changed, planned_targets=packet["targets"],
                fallback=metadata.get("fallback", False), strictly_improved=metadata.get("strictly_improved", False))
            completed_stages += 1
            if audit_class == "unexpected_execution_mismatch" or audit["constraint_interventions"] and not inputs_changed:
                raise RuntimeError(f"Checked execution differs under unchanged inputs: {audit}")
            if stage % 10 == 0:
                print(json.dumps(dict(case=entry["case_id"], method=method, stage=stage+1,
                    actual_cost_so_far=cost, decision_s=elapsed, budget_exceeded=exceeded)), flush=True)
    except Exception as exc:
        status, error = "error", repr(exc)
    terminal = world.dispatch.objective.terminal(world)
    total = cost+world.dispatch.objective.weights["discount"]**completed_stages*terminal["total"]
    summary = dict(method=method, case_id=entry["case_id"], status=status, error=error, completed_stages=completed_stages,
        solver_backend=controller.provenance["solver_backend"],
        cost=total, terminal=terminal, waiting_person_seconds=waiting, headway_deviation_seconds=headway,
        onboard_person_seconds=onboard, passenger_person_seconds=waiting+onboard,
        realized_interval_cost=cost, discounted_terminal_cost=total-cost,
        objective_definition=world.dispatch.objective.manifest(),
        passenger_ledger=passenger_ledger(world, initial_mass, arrivals_total),
        physical_trajectory_sha256=trajectory_digest.hexdigest(), realized_inputs_sha256=inputs_digest.hexdigest(),
        operation_units=operation_units, completed_passengers=sum(float(a.cumulative_alighted[0]) for a in world.agents),
        boarded_passengers=sum(float(a.cumulative_boarded[0]) for a in world.agents),
        average_decision_s=decision_seconds/max(1, completed_stages), total_decision_s=decision_seconds,
        budget_exceedances=exceedances, fallbacks=fallback_count, resource_interventions=interventions,
        seed_only_decisions=seed_only, search_starved_decisions=search_starved,
        strictly_improved_decisions=strict_improvements, novel_candidate_evaluations=novel_evaluations,
        feasible_novel_evaluations=feasible_novel, infeasible_candidate_reasons=dict(failed_candidates),
        execution_audit_classes=dict(audit_classes),
        execution_guard_mismatches=mismatches, source_changed=fingerprints() != sources,
        full_two_hour_full_fleet=status == "complete" and completed_stages == EPISODE_STAGES
            and (args.n_agents is None or args.n_agents == recipe["line"]["fleet"]),
        elapsed_episode_s=time.perf_counter()-started, global_optimality_proven=False,
        provider=controller.provenance, selector_final=[] if method in PURE_METHODS else controller.selector.report(),
        metrics_scope="same realized physical episode; planning-window costs are diagnostic only")
    return results.finish(status=status, error=error, diagnostics=summary,
                          computation_time_s=decision_seconds, source_changed=summary["source_changed"])


def replay_case(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("result_schema") == RESULT_SCHEMA:
        return replay_episode(path)
    if manifest["source_sha256"] != fingerprints():
        raise ValueError("Baseline or physical source changed; do not silently replay a different model")
    recipe, entry = manifest["recipe"], manifest["entry"]
    world = make_world(recipe["topology"], fleet=manifest["n_agents"], seed=entry["seed"],
                       events=events_for(entry, fleet=manifest["n_agents"]), initial_passenger_seconds=recipe["tick_seconds"])
    if state_fingerprint(world) != manifest["initial_state_sha256"]:
        raise ValueError("Replay initial physical state differs")
    cost, count = 0., 0
    with (path.parent/"decisions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["stage"] != count:
                raise ValueError("Replay stages are incomplete or out of order")
            control = row["control"]
            if control["commands"]:
                result = world.submit_dispatch_plan(dict(plan_id=f"baseline_replay_{count}", snapshot_basis="published",
                    snapshot_version=world.dispatch_snapshot()["version"], created_at=float(world.sim_time[0]), commands=control["commands"]))
                if not result["accepted"]:
                    raise ValueError("Replay command rejected")
            set_actions(world, control["actions"])
            arrivals = {int(p): {int(d): v for d, v in values.items()} for p, values in row["actual_arrivals"].items()}
            profile = [{int(p): {int(d): v for d, v in values.items()} for p, values in second.items()}
                       for second in row["actual_arrival_profile"]]
            world.step(arrival_inputs=profile, dispatch_targets=control["targets"])
            actual = {v.id: world.dispatch.position(v.id) for v in world.dispatch.records.values()}
            if actual != row["actual_positions"]:
                raise ValueError(f"Replay physical positions differ at stage {count}")
            if state_fingerprint(world) != row["actual_state_sha256"]:
                raise ValueError(f"Replay vehicle, operation, passenger or resource state differs at stage {count}")
            digest = hashlib.sha256(json.dumps(world.last_physical_audits, sort_keys=True).encode()).hexdigest()
            if digest != row["actual_execution_sha256"]:
                raise ValueError(f"Replay second-resolution trajectory differs at stage {count}")
            if abs(world.dispatch.interval_cost["total"]-row["interval_cost"]["total"]) > 1e-5:
                raise ValueError("Replay objective differs")
            cost += world.dispatch.objective.weights["discount"]**count*world.dispatch.interval_cost["total"]
            count += 1
    total = cost+world.dispatch.objective.weights["discount"]**count*world.dispatch.objective.terminal(world)["total"]
    summary = json.loads((path.parent/"summary.json").read_text(encoding="utf-8"))
    if abs(total-summary["cost"]) > 1e-4:
        raise ValueError("Replay total objective differs")
    return dict(verified=True, stages=count, cost=total, complete=count == manifest["steps"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line", default="sz20")
    parser.add_argument("--source-line", default="sz20")
    parser.add_argument("--cases", nargs="+", choices=("B", "D", "F", "G"), default=["B", "D", "F"])
    parser.add_argument("--scenario-seed", type=int, default=111)
    parser.add_argument("--scenario-repeats", type=int, default=1)
    parser.add_argument("--methods", nargs="+", choices=(*METHODS, "all"), default=["rh_milp", "alns_mip", "bandit_alns", "rule_guided"])
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--allow-short-test", action="store_true")
    parser.add_argument("--n-agents", type=int)
    parser.add_argument("--horizon", type=int, default=3, help="Base planning stages")
    parser.add_argument("--horizon-policy", choices=("fixed", "operation_aware"), default="operation_aware")
    parser.add_argument("--max-operation-horizon", type=int, default=12)
    parser.add_argument("--budget", type=float, default=10.)
    parser.add_argument("--repair-seconds", type=float, default=2.)
    parser.add_argument("--minimum-repair-seconds", type=float, default=0.,
                        help="Minimum LNS repair allowance; capped by remaining request budget (0 keeps prior allocation)")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=20, help="GA population / PSO swarm size; no pretrained model")
    parser.add_argument("--release-fraction", type=float, default=.35)
    parser.add_argument("--confidence", type=float, default=.9)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="Physical simulator tensor threads, independent of CPLEX --threads")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--max-evaluations", type=int)
    parser.add_argument("--command-stages", nargs="+", type=int, default=[0],
                        help="Common candidate start-stage grid for every method; e.g. 0 1 2")
    parser.add_argument("--adaptive-rule-config", action="store_true",
                        help="Enable G02 window/budget rules; off for fixed-window paired controls")
    parser.add_argument("--max-rule-horizon", type=int, default=6)
    parser.add_argument("--retrieval-data")
    parser.add_argument("--mlp-checkpoint")
    parser.add_argument("--marl-checkpoint")
    parser.add_argument("--llm-strategy")
    parser.add_argument("--check-ready", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--replay", help="Path to a baseline manifest.json; no model inference or optimization")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    if args.replay:
        result = replay_case(args.replay)
        print(json.dumps(result)); return result
    methods = list(METHODS) if "all" in args.methods else list(dict.fromkeys(args.methods))
    readiness = []
    for method in methods:
        try:
            controller = controller_for(method, args)
            readiness.append(dict(method=method, ready=True, provenance=controller.provenance))
        except (ValueError, TypeError, OSError, ImportError, RuntimeError) as exc:
            readiness.append(dict(method=method, ready=False, reason=str(exc)))
    if args.check_ready:
        print(json.dumps(readiness, ensure_ascii=False, indent=2)); return readiness
    missing = [row for row in readiness if not row["ready"]]
    if missing and not args.preview:
        raise ValueError(f"Requested methods are not ready; no experiments started: {missing}")
    if args.steps < 1 or args.steps != EPISODE_STAGES and not args.allow_short_test:
        parser.error("Use --steps 240 for formal runs; shorter checks require --allow-short-test")
    if not args.output and not args.preview:
        parser.error("--output required")
    if args.output and Path(args.output).exists() and not args.preview:
        raise FileExistsError("Use a fresh baseline output directory")
    lines = [spec["key"] for spec in LINE_SPECS.values()] if args.line == "all" else [args.line]
    rows = []
    for line in lines:
        recipe, entries = read_scenarios(line, "test", ["combination_"+c for c in args.cases],
                                         repeats=args.scenario_repeats, seed=args.scenario_seed)
        for entry in entries:
            for method in methods:
                if args.preview:
                    print(f"{entry['case_id']} | {method} | {args.steps} stages | H={args.horizon} | {args.budget}s/request")
                    continue
                rows.append(run_case(recipe, entry, method, args))
                with (Path(args.output)/"comparison.csv").open("w", newline="", encoding="utf-8") as stream:
                    fields = ("case_id", "method", "status", "completed_stages", "cost", "waiting_person_seconds",
                              "onboard_person_seconds", "passenger_person_seconds",
                              "completed_passengers", "average_decision_s", "budget_exceedances", "resource_interventions")
                    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader(); writer.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
