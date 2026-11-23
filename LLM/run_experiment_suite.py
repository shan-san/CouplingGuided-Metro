"""Local, dependency-free experiment matrix launcher.

Preview (no model imports, API requests, or files written)::

    python LLM/run_experiment_suite.py --suite all --preview

Run into a NEW directory; credentials are inherited only from the environment::

    python LLM/run_experiment_suite.py --suite ablation --output-dir results/ablations \
        --small-model-url http://localhost:8000/v1 --small-model-model metro-slm-v9 \
        --small-model-path /models/adapter --rag-db-path /data/cases.json \
        --llm-model strategic-model --llm-base-url https://example.org/v1

Non-matrix runtime tuning can be repeated as argv tokens, for example
``--child-arg=--llm-thinking-mode --child-arg=disabled``. Split, physical-input,
matrix-control, credential, and output overrides are deliberately disallowed.
An existing output directory is never reused, including after interruption.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
SUITES = ("ablation", "coordination", "budget_window")
VARIANTS = ("full", "rule_reference", "periodic_trigger", "event_trigger", "no_rag",
            "full_fleet", "nearest_scope")
LINES = ("sz20", "sz1", "sz10", "sz6b", "sh16")
METRICS = ("cost", "waiting_person_seconds", "onboard_person_seconds", "passenger_person_seconds",
           "headway_deviation_seconds", "operation_units",
           "average_decision_s", "total_decision_s", "completed_passengers", "unfinished_passengers",
           "avg_waiting_time", "avg_headway_std", "avg_latency_ms", "avg_llm_interventions")
# Only non-physical, non-matrix tuning may pass through to the runtime.
EXTRA_VALUES = {"--llm-request-timeout", "--llm-thinking-mode", "--llm-reasoning-effort",
                "--solver-reserve-fraction", "--small-model-timeout", "--small-model-tokenizer",
                "--torch-threads", "--entropy_threshold", "--waiting_threshold",
                "--waiting-total-threshold", "--waiting-max-threshold", "--coupling-check-horizon",
                "--coupling-service-gap", "--coupling-persistence", "--cached-plan-max-steps",
                "--coordination-horizon-policy", "--coordination-max-horizon", "--coordination-service-tail",
                "--scope-expansion-agents", "--scope-window-vehicles"}
EXTRA_FLAGS = {"--use_embeddings", "--save_history", "--verbose", "--scope-window-closure"}
SOURCE_FILES = ("LLM/run_experiment_suite.py", "LLM/main_hierarchical.py",
                "LLM/rag/enhanced_rag_manager.py", "LLM/rag/native_case.py",
                "LLM/vllm_pool.py", "LLM/serve_vllm.py",
                "LLM/experiment_controls.py", "LLM/hierarchical_llm_policy.py",
                "LLM/dispatch_coordination.py", "LLM/coupling_trigger.py", "LLM/dispatch_scope.py",
                "MILP/operation_model.py", "MILP/parallel_evaluation.py", "MILP/milp_model.py", "MILP/staged_dispatch.py", "MILP/guidance_labels.py",
                "MILP/collection_protocol.py", "RL/dispatch_training.py",
                "scenarios/prepare_dispatch.py", "delay_events.py", "dispatch_operations.py", "dispatch_plan.py",
                "dispatch_objective.py", "dispatch_candidates.py", "dispatch_timing.py", "dispatch_records.py",
                "experiment_results.py", "dispatch_diagnostics.py", "metro_world.py",
                "metro_environment.py", "metro_rl_env.py", "envwrapper.py", "metro_scenario_v2.py",
                "topology/metro_lines.py")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    result.add_argument("--suite", "--suites", nargs="+", choices=(*SUITES, "all"), default=["all"])
    result.add_argument("--variants", nargs="+", choices=VARIANTS, help="Ablation subset, in canonical order")
    result.add_argument("--line", "--lines", nargs="+", choices=LINES, default=["sz10"])
    result.add_argument("--cases", nargs="+", choices=tuple("ABCDEFG"), default=["B"])
    result.add_argument("--scenario-seed", type=int, default=111)
    result.add_argument("--repeats", type=positive_int, default=1,
                        help="Repeat indices 0..N-1; NOT independent scenario seeds")
    result.add_argument("--steps", type=positive_int, default=240,
                        help="Required completed physical stages; 240 for formal evaluation")
    result.add_argument("--budgets", nargs="+", type=positive_number, default=[30, 60, 90, 180, 300])
    result.add_argument("--horizons", nargs="+", type=positive_int, default=[3, 6, 12])
    result.add_argument("--budget", type=positive_number, default=300)
    result.add_argument("--milp-time-limit", type=positive_int, default=180)
    result.add_argument("--horizon", type=positive_int, default=6)
    result.add_argument("--allowed-decision-delay", type=positive_number,
                        help="Reported latency threshold; defaults to each task's total budget")
    result.add_argument("--coordination-period", type=positive_int, default=10)
    result.add_argument("--nearest-vehicles", type=positive_int, default=6)
    result.add_argument("--cooldown", type=int, default=10)
    result.add_argument("--marl-dir", type=Path, default=ROOT / "MARL_model")
    result.add_argument("--small-model-path", type=Path)
    result.add_argument("--rag-db-path", type=Path)
    result.add_argument("--small-model-url")
    result.add_argument("--small-model-model", default="metro-slm-v9")
    result.add_argument("--llm-model", default="deepseek-v4-pro")
    result.add_argument("--llm-base-url")
    result.add_argument("--llm-max-input-tokens", type=positive_int, default=131072)
    result.add_argument("--llm-max-output-tokens", type=positive_int, default=4096)
    result.add_argument("--llm-request-timeout", type=positive_number, default=60)
    result.add_argument("--small-model-timeout", type=positive_number, default=60)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--child-arg", "--extra-arg", action="append", default=[],
                        help="Repeat for each allowed runtime argv token; use --child-arg=--FLAG")
    result.add_argument("--max-workers", type=positive_int, default=4)
    result.add_argument("--output-dir", type=Path, default=ROOT / "results" / "experiment_suite")
    result.add_argument("--preview", action="store_true", help="Print matrix JSON; do not write or launch")
    result.add_argument("--status", action="store_true", help="Read existing queue status; do not launch")
    return result


def _absolute(value):
    return str(value.expanduser().resolve()) if value is not None else None


def _safe_text(value):
    if any(ord(char) < 32 for char in value):
        raise ValueError("control characters are not allowed in arguments")
    if "://" in value:
        url = urlsplit(value)
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("URLs must not contain credentials, query strings, or fragments")
    return value


def _extra_arguments(tokens):
    index = 0
    while index < len(tokens):
        token = _safe_text(tokens[index])
        flag, separator, value = token.partition("=")
        if flag in EXTRA_FLAGS and not separator:
            index += 1
            continue
        if flag not in EXTRA_VALUES:
            raise ValueError(f"runtime override is not permitted: {flag}; use dedicated suite options")
        if not separator:
            index += 1
            if index >= len(tokens) or tokens[index].startswith("--"):
                raise ValueError(f"missing value for {flag}")
            value = tokens[index]
        _safe_text(value)
        if not value:
            raise ValueError(f"missing value for {flag}")
        index += 1
    return list(tokens)


def _number(value):
    return format(value, ".15g")


def _source_provenance():
    files = {}
    for name in SOURCE_FILES:
        path = ROOT / name
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    git = {}
    for key, command in (("commit", ["git", "rev-parse", "HEAD"]),
                         ("status", ["git", "status", "--porcelain", "--untracked-files=no"])):
        try:
            value = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=5)
            git[key] = value.stdout.strip() if value.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            git[key] = None
    return {"source_sha256": files, "git": git, "python_executable": sys.executable,
            "python_version": sys.version, "project_root": str(ROOT)}


def _path_identity(value):
    if value is None:
        return None
    path = Path(value)
    try:
        stat = path.stat()
        return dict(path=str(path), exists=True, is_file=path.is_file(), size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns, fingerprint="path/size/mtime; weights not loaded or content-hashed")
    except OSError:
        return dict(path=str(path), exists=False)


def build_matrix(args):
    """Build fully explicit commands without importing the runtime or inference stack."""
    suites = [name for name in SUITES if "all" in args.suite or name in args.suite]
    if args.variants and "ablation" not in suites:
        raise ValueError("--variants applies only to the ablation suite")
    if args.cooldown < 0:
        raise ValueError("--cooldown must be nonnegative")
    if args.scenario_seed < 0:
        raise ValueError("--scenario-seed must be nonnegative")
    for name in ("line", "cases", "budgets", "horizons", "variants"):
        values = getattr(args, name)
        if values is not None and len(values) != len(set(values)):
            raise ValueError(f"duplicate --{name} values would create output collisions")
    extra = _extra_arguments(args.child_arg)
    config = {key: (_absolute(value) if isinstance(value, Path) else value)
              for key, value in vars(args).items() if key not in ("preview", "status")}
    for key in ("small_model_url", "llm_base_url", "small_model_model", "llm_model"):
        if config[key] is not None:
            _safe_text(config[key])
    output = Path(config["output_dir"])
    tasks = []
    for suite in suites:
        if suite == "ablation":
            rows = [(name, args.budget, args.horizon) for name in VARIANTS
                    if args.variants is None or name in args.variants]
        elif suite == "coordination":
            rows = [(name, args.budget, args.horizon) for name in ("joint", "staged")]
        else:
            rows = [(f"budget{_number(budget)}_horizon{horizon}", budget, horizon)
                    for budget in args.budgets for horizon in args.horizons]
        for variant, budget, horizon in rows:
            isolated = suite == "coordination"
            controls = dict(reference_policy="rule" if isolated or variant == "rule_reference" else "marl",
                            trigger_mode="periodic" if isolated or variant == "periodic_trigger" else
                                         "event" if variant == "event_trigger" else "coupling",
                            coordination_period=1 if isolated else args.coordination_period,
                            coordination_scope="full_fleet" if isolated or variant == "full_fleet" else
                                               "nearest" if variant == "nearest_scope" else "coupling",
                            nearest_vehicles=args.nearest_vehicles,
                            optimization_mode=variant if isolated else "joint",
                            fixed_planning_horizon=suite in ("coordination", "budget_window"),
                            joint_guidance_mode="no_guidance" if isolated else "full",
                            use_rag=not (isolated or variant == "no_rag"), use_small_model=not isolated,
                            cooldown=0 if isolated else args.cooldown, budget=budget, horizon=horizon,
                            milp_time_limit=args.milp_time_limit,
                            allowed_decision_delay=budget if args.allowed_decision_delay is None else args.allowed_decision_delay)
            for line in args.line:
                for case in args.cases:
                    for repeat in range(args.repeats):
                        task_id = f"{suite}--{variant}--{line}--{case}--seed{args.scenario_seed}--repeat{repeat:03d}"
                        task_root = output / "tasks" / task_id
                        results, logs = task_root / "results", task_root / "logs"
                        command = [sys.executable, str(ROOT / "LLM" / "main_hierarchical.py"),
                                   "--mode", "benchmark", "--topology", line, "--dispatch-scenario", f"combination_{case}",
                                   "--scenario-split", "test", "--scenario-seed", str(args.scenario_seed),
                                   "--scenario-repeat", str(repeat), "--episodes", "1", "--n_steps", str(args.steps),
                                   "--device", args.device, "--output_dir", str(results),
                                   "--log_llm_io", str(logs / "llm_io.jsonl"), "--require-components", "--save_history"]
                        for field in ("reference_policy", "trigger_mode", "coordination_period", "coordination_scope",
                                      "nearest_vehicles", "optimization_mode", "joint_guidance_mode", "cooldown"):
                            command.extend(["--" + field.replace("_", "-"), str(controls[field])])
                        command.extend(["--coordination-budget", _number(budget),
                                        "--allowed-decision-delay", _number(controls["allowed_decision_delay"]),
                                        "--milp_horizon", str(horizon), "--milp_time_limit", str(args.milp_time_limit)])
                        if controls["fixed_planning_horizon"]:
                            command.append("--fixed-planning-horizon")
                        if not controls["use_rag"]:
                            command.append("--ablate_rag")
                        checkpoint = str(Path(config["marl_dir"]) / f"{line}.pt") if controls["reference_policy"] == "marl" else None
                        if checkpoint:
                            command.extend(["--rl_model_path", checkpoint])
                        if not isolated:
                            command.extend(["--use_small_model", "--small-model-backend", "vllm",
                                            "--small_model_model", args.small_model_model, "--llm_model", args.llm_model,
                                            "--llm-max-input-tokens", str(args.llm_max_input_tokens),
                                            "--llm-max-output-tokens", str(args.llm_max_output_tokens),
                                            "--llm-request-timeout", _number(args.llm_request_timeout),
                                            "--small-model-timeout", _number(args.small_model_timeout)])
                            for key, flag in (("small_model_path", "--small_model_path"), ("small_model_url", "--small_model_url"),
                                              ("llm_base_url", "--llm_base_url")):
                                if config[key] is not None:
                                    command.extend([flag, config[key]])
                            if controls["use_rag"] and config["rag_db_path"] is not None:
                                command.extend(["--rag_db_path", config["rag_db_path"]])
                        # Guidance tuning must not re-enable components in the isolated comparison.
                        command.extend(extra)
                        tasks.append(dict(id=task_id, suite=suite, variant=variant, line=line, case=case,
                                          scenario_seed=args.scenario_seed, scenario_repeat=repeat,
                                          episode_seed=args.scenario_seed + 200000 + repeat,
                                          expected_episodes=1, expected_stages=args.steps,
                                          formal=args.steps == 240, controls=controls, checkpoint=checkpoint,
                                          results_dir=str(results), log_dir=str(logs), cwd=str(logs),
                                          log_path=str(logs / "process.log"), command=command))
    model_provenance = dict(
        marl_checkpoints={path: _path_identity(path) for path in sorted({t["checkpoint"] for t in tasks if t["checkpoint"]})},
        slm=dict(enabled=any(t["controls"]["use_small_model"] for t in tasks), backend="vllm",
                 service_url=config["small_model_url"], served_model=config["small_model_model"],
                 adapter=_path_identity(config["small_model_path"])),
        rag=dict(enabled=any(t["controls"]["use_rag"] for t in tasks), database=_path_identity(config["rag_db_path"]),
                 default_resolution="runtime default if path omitted; require-components forbids missing retained RAG"),
        llm=dict(enabled=any(t["controls"]["joint_guidance_mode"] == "full" for t in tasks),
                 model=config["llm_model"], base_url=config["llm_base_url"],
                 max_input_tokens=config["llm_max_input_tokens"], max_output_tokens=config["llm_max_output_tokens"]))
    manifest = dict(schema_version=1, config=config, provenance=_source_provenance(), models=model_provenance,
                    scenario_recipe=dict(split="test", scenario="combination_<case>",
                                         seed=args.scenario_seed, split_offset=200000,
                                         repeat_indices=list(range(args.repeats)),
                                         episode_seed="scenario_seed + 200000 + scenario_repeat",
                                         pairing="identical line/case/seed/repeat across variants; no split edits",
                                         generation="existing scenarios.prepare_dispatch.runtime_scenario unchanged"),
                    environment=dict(inheritance="parent environment passed unchanged; values never recorded",
                                     credentials="OPENAI_API_KEY via environment only, never command arguments"), tasks=tasks)
    manifest["matrix_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest


def _atomic_json(path, payload):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _now():
    return datetime.now(timezone.utc).isoformat()


def validate_completion(task):
    root = Path(task["results_dir"])
    paths = [root/"summary.json"] if (root/"summary.json").exists() else sorted(root.glob("episode_*/summary.json"))
    if paths:
        from experiment_results import RESULT_SCHEMA
        episodes = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        if any(row.get("result_schema") != RESULT_SCHEMA or not row.get("complete") or row.get("source_changed")
               for row in episodes):
            raise ValueError("common episode result is incomplete or its source changed")
        summary = dict(episodes[0], physical_episodes=episodes, n_episodes=len(episodes),
                       n_steps_per_episode=episodes[0]["requested_stages"])
    else:
        paths = list(root.glob("hierarchical*_benchmark_summary.json"))
        if len(paths) != 1:
            raise ValueError(f"expected a common physical benchmark summary; found {len(paths)}")
        summary = json.loads(paths[0].read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("benchmark summary must be an object")
    if summary.get("source_changed") is not False:
        raise ValueError("runtime source stability must be recorded and verified")
    episodes = summary.get("physical_episodes")
    if not isinstance(episodes, list) or len(episodes) != task["expected_episodes"]:
        raise ValueError("missing or incomplete physical_episodes")
    for row in episodes:
        if not isinstance(row, dict) or type(row.get("completed_stages")) is not int or row["completed_stages"] != task["expected_stages"]:
            raise ValueError(f"physical completed_stages must equal {task['expected_stages']}")
        if not isinstance(row.get("cost"), (int, float)) or isinstance(row["cost"], bool) or not math.isfinite(row["cost"]):
            raise ValueError("physical episode cost must be finite")
    for key, expected in (("n_episodes", task["expected_episodes"]), ("n_steps_per_episode", task["expected_stages"]),
                          ("case_id", f"{task['line']}_test_{task['scenario_repeat']}_combination_{task['case']}"),
                          ("episode_seed", task["episode_seed"])):
        if summary.get(key) != expected:
            raise ValueError(f"benchmark {key} does not match matrix")
    metrics = {key: summary[key] for key in METRICS
               if isinstance(summary.get(key), (int, float)) and not isinstance(summary[key], bool)
               and math.isfinite(summary[key])}
    metrics["cost"] = sum(row["cost"] for row in episodes) / len(episodes)
    result = dict(summary_path=str(paths[0]), summary_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                  completed_stages=[row["completed_stages"] for row in episodes], metrics=metrics)
    if "experiment_configuration" not in summary:
        raise ValueError("runtime must record the actual experiment controls")
    actual = summary["experiment_configuration"]
    aliases = dict(use_small_model="use_small_model_guidance", cooldown="intervention_cooldown",
                   horizon="milp_time_horizon", budget="coordination_time_budget_s",
                   allowed_decision_delay="allowed_decision_delay_s")
    if not isinstance(actual, dict):
        raise ValueError("runtime experiment configuration must be an object")
    for key, expected in task["controls"].items():
        if actual.get(aliases.get(key, key)) != expected:
            raise ValueError("runtime experiment configuration does not match requested controls")
    result["experiment_configuration"] = {aliases.get(key, key): actual[aliases.get(key, key)]
                                           for key in task["controls"]}
    return result


def _snapshot(manifest, states, started_at, interrupted=False):
    counts = {state: 0 for state in ("pending", "running", "completed", "failed")}
    counts.update(Counter(row["status"] for row in states))
    finished = not counts["pending"] and not counts["running"]
    groups = defaultdict(list)
    for task, state in zip(manifest["tasks"], states):
        groups[f"{task['suite']}/{task['variant']}"].append(state)
    aggregates = {}
    for key, rows in groups.items():
        group = dict(counts=dict(Counter(row["status"] for row in rows)))
        for metric in METRICS:
            values = [row["metrics"][metric] for row in rows if row["status"] == "completed" and metric in row["metrics"]]
            if values:
                group[f"mean_{metric}"] = sum(value / len(values) for value in values)
        aggregates[key] = group
    return dict(schema_version=1, matrix_sha256=manifest["matrix_sha256"], started_at=started_at,
                updated_at=_now(), status="interrupted" if interrupted else
                "failed" if finished and counts["failed"] else "completed" if finished else "running",
                counts=counts, total=len(states), aggregates=aggregates, tasks=states)


def _stop(process):
    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            process.wait()


def run_matrix(manifest):
    """Dynamically refill a bounded pool of child processes, continuing after failures."""
    output = Path(manifest["config"]["output_dir"])
    tasks = manifest["tasks"]
    for task in tasks:
        if task["checkpoint"] and not Path(task["checkpoint"]).is_file():
            raise ValueError(f"MARL checkpoint not found: {task['checkpoint']}")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError("output directory already exists; choose a NEW directory (resume is not supported)") from exc
    _atomic_json(output / "matrix.json", manifest)
    states = [dict(id=task["id"], status="pending", log_path=task["log_path"]) for task in tasks]
    started, running, next_index = _now(), {}, 0

    def write_status(interrupted=False):
        value = _snapshot(manifest, states, started, interrupted)
        _atomic_json(output / "queue_status.json", value)
        return value

    write_status()
    old_handler = signal.getsignal(signal.SIGTERM)

    def interrupted_handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted_handler)
    interrupted = False
    try:
        while next_index < len(tasks) or running:
            while next_index < len(tasks) and len(running) < manifest["config"]["max_workers"]:
                index = next_index
                next_index += 1
                task, state = tasks[index], states[index]
                handle = None
                try:
                    Path(task["results_dir"]).mkdir(parents=True, exist_ok=False)
                    Path(task["log_dir"]).mkdir(parents=True, exist_ok=False)
                    handle = open(task["log_path"], "wb")
                    process = subprocess.Popen(task["command"], cwd=task["cwd"], stdout=handle,
                                               stderr=subprocess.STDOUT, env=os.environ.copy(),
                                               start_new_session=os.name == "posix")
                    running[index] = (process, handle)
                    state.update(status="running", pid=process.pid, started_at=_now())
                except OSError as exc:
                    if handle is not None:
                        handle.close()
                    state.update(status="failed", error=f"child launch failed ({type(exc).__name__})", finished_at=_now())
                write_status()
            for index in list(running):
                process, handle = running[index]
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del running[index]
                state = states[index]
                state.update(returncode=code, finished_at=_now())
                try:
                    if code != 0:
                        raise ValueError(f"child exited with code {code}")
                    state.update(validate_completion(tasks[index]), status="completed")
                except (ValueError, OSError, TypeError, OverflowError):
                    # Child output is untrusted; never copy its text (possibly credentials) into status.
                    state.update(status="failed", error=f"exit {code}; missing, invalid, or incomplete physical summary"
                                 if code == 0 else f"child exited with code {code}")
                write_status()
            if running:
                time.sleep(0.05)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        # Ignore a second termination request while reaping every child.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for index, (process, handle) in running.items():
            _stop(process)
            handle.close()
            states[index].update(status="failed", error="launcher interrupted", returncode=process.returncode, finished_at=_now())
        signal.signal(signal.SIGTERM, old_handler)
        final = write_status(interrupted)
        _atomic_json(output / "summary.json", final)
    print(json.dumps({"status": final["status"], "counts": final["counts"], "output_dir": str(output)}, sort_keys=True))
    return 130 if interrupted else 1 if final["counts"]["failed"] else 0


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.status:
            if args.preview:
                raise ValueError("--status and --preview are mutually exclusive")
            print((args.output_dir.expanduser().resolve() / "queue_status.json").read_text(encoding="utf-8"))
            return 0
        matrix = build_matrix(args)
        if args.preview:
            print(json.dumps(matrix, indent=2, sort_keys=True, allow_nan=False))
            return 0
        return run_matrix(matrix)
    except (ValueError, OSError) as exc:
        cli.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
