"""Five named topologies, held-out B/D/F, and full 06:00--08:00 offline search.

No MARL/LLM/SLM model is loaded. Timeout is not an optimality certificate.
The implicit state-network formulation uses the original physical simulator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from MILP.offline_domain import OfflineDomain, StageControl
from MILP.offline_global import NativeOfflineProblem, OfflineGlobalSolver
from scenarios.prepare_dispatch import EPISODE_STAGES, events_for, make_world, read_scenarios
from topology.metro_lines import LINE_SPECS, line_id
from experiment_results import EpisodeResults


SOURCE_FILES = (
    "MILP/cplex_backend.py", "MILP/offline_domain.py", "MILP/offline_global.py", "MILP/offline_baseline.py",
    "MILP/operation_model.py", "MILP/parallel_evaluation.py", "metro_agent_v2.py", "metro_agent_v1.py", "metro_scenario_v2.py",
    "metro_world.py", "dispatch_operations.py", "dispatch_objective.py", "delay_events.py",
    "dispatch_plan.py", "dispatch_timing.py", "dispatch_records.py", "experiment_results.py", "dispatch_diagnostics.py", "buildtopology.py",
    "scenarios/prepare_dispatch.py", "topology/metro_lines.py",
    "topology/passenger_flow.py", "topology/node.py", "linedata.py", "route_selection.py")


def source_fingerprints():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


LOADED_SOURCE_SHA256 = source_fingerprints()


def clock(seconds):
    total = 21600+int(round(seconds))
    return f"{total//3600:02d}:{total//60%60:02d}:{total%60:02d}"


def dump_json(path, value):
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def run_case(line, recipe, entry, args):
    started = time.perf_counter()
    if source_fingerprints() != LOADED_SOURCE_SHA256:
        raise RuntimeError("Source files changed since process startup; restart in a new output directory")
    output = Path(args.output)/entry["case_id"]
    output.mkdir(parents=True, exist_ok=True)
    protected = ("summary.json", "search.jsonl", "controls.jsonl", "timetable.csv", "arrivals.csv",
                 "incumbent.json", "run_manifest.json", "cplex_bounds.log", "gurobi_bounds.log")
    if any((output/name).exists() for name in protected):
        raise FileExistsError(f"Existing baseline results: {output}; choose a new output directory")
    world = make_world(recipe["topology"], fleet=args.n_agents, seed=entry["seed"],
        events=events_for(entry, fleet=args.n_agents), tick=recipe["tick_seconds"],
        initial_passenger_seconds=recipe["tick_seconds"])
    domain = OfflineDomain(args.steps, recipe["tick_seconds"], admission_rates=args.admission_rates,
                           headways=args.headways)
    problem = NativeOfflineProblem(world, domain)
    initial_controls = None
    if args.warm_start:
        saved = json.loads(Path(args.warm_start).read_text(encoding="utf-8"))
        for key, expected in (("case_id", entry["case_id"]), ("scenario", entry),
                              ("domain", domain.manifest()), ("source_sha256", LOADED_SOURCE_SHA256),
                              ("fleet", len(world.agents)), ("objective", problem.objective),
                              ("physical_events", events_for(entry, fleet=args.n_agents))):
            if saved.get(key) != expected:
                raise ValueError(f"Warm-start checkpoint differs in {key}; refusing a different benchmark")
        initial_controls = [StageControl.from_dict(c) for c in saved["controls"]]
        if len(initial_controls) != args.steps:
            raise ValueError("Warm start does not contain the complete episode")
    manifest = dict(case_id=entry["case_id"], scenario=entry, domain=domain.manifest(), solver_backend="cplex",
                    source_sha256=LOADED_SOURCE_SHA256, objective=problem.objective,
                    physical_events=events_for(entry, fleet=args.n_agents), fleet=len(world.agents),
                    warm_start=args.warm_start, search_state_restored=False)
    results = EpisodeResults(output, world, method="offline_milp", case_id=entry["case_id"],
        requested_stages=args.steps, entry=entry, recipe=recipe, configuration=manifest,
        source_sha256=LOADED_SOURCE_SHA256, online=False)
    diagnostics = output/"diagnostics"
    diagnostics.mkdir(exist_ok=True)
    dump_json(diagnostics/"run_manifest.json", manifest)
    with (diagnostics/"arrivals.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("stage", "clock", "origin", "destination", "persons"))
        for stage, interval in enumerate(problem.inputs):
            for origin, destinations in interval.items():
                for destination, count in destinations.items():
                    writer.writerow((stage, clock(stage*domain.tick), origin, destination, count))
    print(f"Start {entry['case_id']}: {len(world.agents)} vehicles, {args.steps} stages", flush=True)
    with (diagnostics/"search.jsonl").open("w", encoding="utf-8") as log:
        def progress(value):
            log.write(json.dumps(value, allow_nan=False)+"\n")
            log.flush()
            results.progress(phase="searching", search=value)
            print(json.dumps(dict(case=entry["case_id"], **value), allow_nan=False), flush=True)
        def checkpoint(cost, controls, info):
            dump_json(diagnostics/"incumbent.json", dict(manifest, cost=cost,
                controls=[control.to_dict() for control in controls], progress=info,
                validation="native transitions checked; independent final replay pending",
                optimality_proven=False))
        solver = OfflineGlobalSolver(problem, time_limit=args.time_limit, max_nodes=args.max_nodes,
            max_transitions=args.max_transitions, threads=args.threads,
            absolute_gap=args.absolute_gap, relative_gap=args.relative_gap, progress=progress,
            on_incumbent=checkpoint, initial_controls=initial_controls, log_path=diagnostics/"cplex_bounds.log")
        result, controls = solver.solve(model_path=diagnostics/"search_relaxation.lp")
    computation_time_s = time.perf_counter()-started
    result.update(case_id=entry["case_id"], line=line, topology=recipe["topology"],
        fleet=len(world.agents), stages=args.steps, start_clock="06:00:00", end_clock=clock(args.steps*domain.tick),
        is_full_two_hour_baseline=args.steps == EPISODE_STAGES, scenario_recipe=recipe,
        is_full_fleet=args.n_agents is None or args.n_agents == recipe["line"]["fleet"],
        is_formal_baseline=args.steps == EPISODE_STAGES and (
            args.n_agents is None or args.n_agents == recipe["line"]["fleet"]),
        scenario=entry, physical_events=events_for(entry, fleet=args.n_agents),
        objective=problem.objective, marl_checkpoint=None, language_models_used=False,
        torch_num_threads=torch.get_num_threads(),
        source_sha256=LOADED_SOURCE_SHA256)
    result["verified"] = False
    if controls is not None:
        verification_started = time.perf_counter()
        with (diagnostics/"controls.jsonl").open("w", encoding="utf-8") as stream:
            for stage, control in enumerate(controls):
                stream.write(json.dumps(dict(stage=stage, **control.to_dict()), ensure_ascii=False, allow_nan=False)+"\n")
        try:
            verification = problem.replay(controls, record_directory=output, result_writer=results)
            if not math.isclose(verification["cost"], result["upper_bound"], rel_tol=1e-7, abs_tol=1e-5):
                raise AssertionError("Independent replay objective differs from the search incumbent")
            result.update(verified=True, verification=verification,
                schedule_path=str(output/"timetable.csv"))
        except Exception as exc:
            result.update(status="verification_failed", optimality_proven=False,
                unverified_search_upper_bound=result["upper_bound"], upper_bound=None,
                absolute_gap=None, relative_gap=None, verification_error=repr(exc))
        result["elapsed_verification_s"] = time.perf_counter()-verification_started
    result["source_changed_during_run"] = source_fingerprints() != LOADED_SOURCE_SHA256
    if result["source_changed_during_run"]:
        result.update(status="source_changed", optimality_proven=False)
    result["elapsed_total_s"] = time.perf_counter()-started
    summary = results.finish(status="complete" if result["verified"] else
        ("error" if result.get("verification_error") else "no_solution"), error=result.get("verification_error"),
        diagnostics=result, computation_time_s=computation_time_s,
        verified=result["verified"], bounds=result, source_changed=result["source_changed_during_run"])
    print(json.dumps({k: result.get(k) for k in (
        "case_id", "status", "termination", "verified", "optimality_proven", "lower_bound", "upper_bound",
        "relative_gap", "elapsed_total_s")}, ensure_ascii=False), flush=True)
    return dict(result, **summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line", required=True, help="sz20, sz1, sz10, sz6b, sh16, or all")
    parser.add_argument("--cases", nargs="+", choices=("B", "D", "F", "G"), default=["B", "D", "F"])
    parser.add_argument("--scenario-seed", type=int, default=111)
    parser.add_argument("--scenario-repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=EPISODE_STAGES)
    parser.add_argument("--allow-short-test", action="store_true", help="Explicitly permit non-two-hour regression instances")
    parser.add_argument("--time-limit", type=float, default=3600., help="Search seconds per case, including the primal seed")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="Optional exact search state limit; omitted means unlimited")
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--threads", type=int, default=0,
                        help="0: automatic CPLEX threads and unchanged Torch defaults; positive: explicit thread count")
    parser.add_argument("--absolute-gap", type=float, default=1e-6)
    parser.add_argument("--relative-gap", type=float, default=0.)
    parser.add_argument("--admission-rates", nargs="+", type=float, default=[0., 1.])
    parser.add_argument("--headways", nargs="+", type=float, default=None)
    parser.add_argument("--n-agents", type=int, default=None, help="Reduced regression fleet; omit for actual baseline")
    parser.add_argument("--warm-start", help="Saved incumbent.json for this exact case/domain/source; replayed before reuse, not search-tree resume")
    parser.add_argument("--preview", action="store_true", help="Print case matrix without solving or writing files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.warm_start and (args.line == "all" or len(args.cases) != 1 or args.scenario_repeats != 1):
        parser.error("--warm-start requires one line, one case and --scenario-repeats 1")
    if args.steps != EPISODE_STAGES and not args.allow_short_test:
        parser.error("A full two-hour baseline requires --steps 240; use --allow-short-test only for regression")
    if args.steps < 1 or not math.isfinite(args.time_limit) or args.time_limit < 0 or args.threads < 0:
        parser.error("Invalid steps, time budget or threads")
    if not args.preview and (Path(args.output)/"comparison.csv").exists():
        raise FileExistsError("Existing baseline comparison; choose a new output directory")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    lines = [v["key"] for v in LINE_SPECS.values()] if args.line == "all" else [LINE_SPECS[line_id(args.line)]["key"]]
    results = []
    for line in lines:
        recipe, entries = read_scenarios(line, "test", ["combination_"+c for c in args.cases],
            repeats=args.scenario_repeats, seed=args.scenario_seed)
        for entry in entries:
            if args.preview:
                print(f"{entry['case_id']} | fleet={args.n_agents or recipe['line']['fleet']} | "
                      f"06:00:00--{clock(args.steps*recipe['tick_seconds'])} | seed={entry['seed']}")
                continue
            result = run_case(line, recipe, entry, args)
            results.append(result)
            root = Path(args.output)
            with (root/"comparison.csv").open("w", newline="", encoding="utf-8") as stream:
                fields = ("case_id", "line", "fleet", "status", "termination", "verified", "optimality_proven",
                          "lower_bound", "upper_bound", "relative_gap", "elapsed_search_s", "elapsed_total_s")
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)
    return results


if __name__ == "__main__":
    main()
