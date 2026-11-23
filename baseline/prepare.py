"""Prepare source-only G13 supervision and frozen LLM-LNS metro strategies.

The LLM proposes a bounded numerical neighborhood DSL, never executable Python.
Training, validation and test episodes are checked before expensive work starts.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from baseline.guidance import FEATURE_VERSION, GuidanceMLP, file_hash, load_sft_cases
from baseline.methods import BaselineController, BaselineSettings, SCORE_KEYS, validate_strategy
from MILP.operation_model import apply_dispatch_step, audit_dispatch_step, nominal_action, planning_copy, set_actions
from LLM.dispatch_coordination import checked_reference
from scenarios.prepare_dispatch import read_scenarios, make_world, events_for, current_arrivals
from topology.metro_lines import LAYOUT_REVISION, line_id


def fresh_output(path):
    result = Path(path)
    if result.exists():
        raise FileExistsError(f"Refusing to overwrite {result}")
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def train_mlp(args):
    output = fresh_output(args.output)
    if output.with_suffix(".training.json").exists():
        raise FileExistsError("Existing training log; choose a new checkpoint path")
    source = line_id(args.source_line)
    train = load_sft_cases(args.train_data, source, "train")
    validation = load_sft_cases(args.validation_data, source, "validation")
    if {c["episode_id"] for c in train} & {c["episode_id"] for c in validation}:
        raise ValueError("Training and validation episodes overlap")
    if min(args.epochs, args.hidden, args.batch_size) < 1 or args.learning_rate <= 0:
        raise ValueError("Positive supervised training sizes and learning rate required")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = GuidanceMLP(args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    samples = [s for c in train for s in c["samples"]]
    held_out = [s for c in validation for s in c["samples"]]
    best_loss, best_weights, history = float("inf"), None, []
    started = time.perf_counter()
    def loss_on(batch):
        losses = []
        for sample in batch:
            logits = model(torch.tensor(sample["features"], dtype=torch.float32)).unsqueeze(0)
            losses.append(torch.nn.functional.cross_entropy(logits, torch.tensor([sample["target"]])))
        return torch.stack(losses).mean()
    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(samples)
        total = 0.
        for index in range(0, len(samples), args.batch_size):
            batch = samples[index:index+args.batch_size]
            optimizer.zero_grad()
            loss = loss_on(batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite supervised loss")
            loss.backward(); optimizer.step()
            total += float(loss.detach())*len(batch)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_on(held_out))
        row = dict(epoch=epoch+1, training_loss=total/len(samples), validation_loss=val_loss)
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_loss:
            best_loss, best_weights = val_loss, copy.deepcopy(model.state_dict())
    metadata = dict(source_topology=source, training_complete=True, method="G13 ordinary group-choice MLP",
        train_data_sha256=file_hash(args.train_data), validation_data_sha256=file_hash(args.validation_data),
        train_episodes=sorted({c["episode_id"] for c in train}), validation_episodes=sorted({c["episode_id"] for c in validation}),
        train_groups=len(samples), validation_groups=len(held_out), epochs=args.epochs, seed=args.seed,
        training_seconds=time.perf_counter()-started, best_validation_loss=best_loss,
        no_target_finetuning=True, no_post_sft_rl=True,
        feature_basis="action/event indicators and normalized operating features; no topology-specific IDs")
    torch.save(dict(kind="metro_guidance_mlp_v1", feature_version=FEATURE_VERSION, hidden=args.hidden,
                    metadata=metadata, weights=best_weights), output)
    output.with_suffix(".training.json").write_text(json.dumps(dict(metadata=metadata, history=history), indent=2)+"\n")
    return metadata


def collect_snapshots(args):
    output = fresh_output(args.output)
    if min(args.episodes, args.steps, args.snapshot_interval) < 1:
        raise ValueError("Positive episode, step and snapshot sizes required")
    recipe, entries = read_scenarios(args.line, args.split, args.scenarios,
                                     repeats=args.scenario_repeats, seed=args.scenario_seed)
    count = 0
    print(json.dumps(dict(event="snapshot_collection_start", line=args.line, split=args.split,
        episodes=min(args.episodes, len(entries)), steps=args.steps, output=str(output),
        reference_policy="published_route_rule_v2")), flush=True)
    with output.open("w", encoding="utf-8") as stream:
        for entry in entries[:args.episodes]:
            print(json.dumps(dict(event="snapshot_case_start", case_id=entry["case_id"],
                split=args.split, seed=entry["seed"])), flush=True)
            world = make_world(recipe["topology"], fleet=args.n_agents, seed=entry["seed"], events=events_for(entry, fleet=args.n_agents),
                               initial_passenger_seconds=recipe["tick_seconds"])
            history = []
            for stage in range(args.steps):
                if stage % args.snapshot_interval == 0:
                    stream.write(json.dumps(dict(kind="metro_baseline_snapshot_v1", source_topology=recipe["topology"],
                        split=args.split, entry=entry, stage=stage, n_agents=args.n_agents,
                        recipe=recipe, history=history, reference_policy="published_route_rule_v2"), allow_nan=False)+"\n")
                    stream.flush(); count += 1
                # Derive both the route action and its review from published
                # inputs; private clearance must not release a train early.
                visible = planning_copy(world)
                solution = checked_reference(visible, {a.name: nominal_action(visible, a) for a in visible.agents}, 1)
                packet = solution["dispatch_packets"][0]
                if not apply_dispatch_step(world, solution)["accepted"]:
                    raise RuntimeError(f"Snapshot commit rejected: case={entry['case_id']}, stage={stage}")
                arrivals = current_arrivals(world)
                world.step()
                arrivals = world.last_arrival_inputs
                audit = audit_dispatch_step(world, solution)
                if audit["constraint_interventions"]:
                    details = {name: world.last_transition_audit[name] for name in audit["constraint_interventions"]}
                    raise RuntimeError("Snapshot resource correction: " + json.dumps(dict(
                        case_id=entry["case_id"], split=args.split, stage=stage,
                        simulation_seconds=float(world.sim_time[0]), transitions=details,
                        state_mismatches=audit["state_mismatches"])))
                history.append(dict(commands=packet["commands"], actions=packet["actions"], targets=world.last_physical_requests,
                                    planned_targets=packet["targets"],
                                    arrival_profile=arrivals.profile))
                if (stage+1) % args.snapshot_interval == 0 or stage+1 == args.steps:
                    print(json.dumps(dict(event="snapshot_progress", case_id=entry["case_id"],
                        stage=stage+1, simulation_seconds=float(world.sim_time[0]), snapshots=count)), flush=True)
            print(json.dumps(dict(event="snapshot_case_complete", case_id=entry["case_id"], stages=args.steps)), flush=True)
    result = dict(snapshots=count, split=args.split, source_topology=recipe["topology"], source_sha256=file_hash(output),
                  reference_policy="published_route_rule_v2")
    print(json.dumps(dict(event="snapshot_collection_complete", **result)), flush=True)
    return result


def read_snapshots(path, source, split, limit):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("Empty source snapshot corpus")
    for row in rows:
        if (row.get("kind") != "metro_baseline_snapshot_v1" or row.get("source_topology") != source
                or row.get("recipe", {}).get("layout_revision") != LAYOUT_REVISION
                or row.get("recipe", {}).get("physical_tick_seconds") != 1
                or row.get("split") != split or row["entry"]["split"] != split
                or len(row["history"]) != row["stage"]
                or any(f"combination_{k}" in row["entry"]["scenario"] for k in ("B", "D", "F"))):
            raise ValueError("Source snapshot split/topology/history mismatch or test contamination")
    if limit < 1:
        raise ValueError("Positive snapshot limit required")
    episodes = {}
    for row in rows:
        episodes.setdefault(row["entry"]["case_id"], []).append(row)
    groups = [sorted(values, key=lambda r: r["stage"]) for values in episodes.values()]
    selected = []
    # Round-robin episodes, starting near the event period instead of taking
    # only the first normal episode's first few snapshots.
    for offset in range(max(map(len, groups))):
        for group in groups:
            if offset < len(group):
                selected.append(group[(len(group)//3+offset) % len(group)])
                if len(selected) == limit:
                    return selected
    return selected


def restore_snapshot(row):
    if row.get("recipe", {}).get("physical_tick_seconds") != 1:
        raise ValueError("Cannot restore a 30-second physical snapshot into the second-resolution model")
    world = make_world(row["source_topology"], fleet=row["n_agents"], seed=row["entry"]["seed"],
        events=events_for(row["entry"], fleet=row["n_agents"]), initial_passenger_seconds=row["recipe"]["tick_seconds"])
    for stage, packet in enumerate(row["history"]):
        if packet["commands"]:
            commands = [dict(raw, command_id=f"baseline_restore_{stage}_{i}")
                        for i, raw in enumerate(packet["commands"])]
            result = world.submit_dispatch_plan(dict(plan_id=f"baseline_restore_{stage}", snapshot_basis="published",
                snapshot_version=world.dispatch_snapshot()["version"], created_at=float(world.sim_time[0]), commands=commands))
            if not result["accepted"]:
                raise ValueError("Source snapshot command replay rejected")
        set_actions(world, packet["actions"])
        profile = [{int(p): {int(d): v for d, v in values.items()} for p, values in second.items()}
                   for second in packet["arrival_profile"]]
        world.step(arrival_inputs=profile, dispatch_targets=packet["targets"])
        if any(a["conflict"] for a in world.last_transition_audit.values()):
            raise ValueError("Source snapshot replay changed physical feasibility")
    return world


def score_strategy(strategy, snapshots, settings):
    results = []
    for row in snapshots:
        controller = BaselineController("alns_mip", settings, source_topology=row["source_topology"])
        controller.strategy = strategy
        solution, report = controller.decide(restore_snapshot(row))
        results.append(dict(case_id=row["entry"]["case_id"], stage=row["stage"], cost=solution["cost"],
            initial_cost=report["initial_cost"], improvement=(report["initial_cost"]-solution["cost"])/max(1., abs(report["initial_cost"])),
            decision_s=report["decision_s"], budget_exceeded=report["budget_exceeded"]))
    return dict(score=sum(r["improvement"] for r in results)/len(results), rows=results)


def request_strategies(args, prompt):
    import requests
    base = (args.api_base or os.getenv("BASELINE_LLM_BASE_URL", "")).rstrip("/")
    key = os.getenv("BASELINE_LLM_API_KEY", "")
    if not base or not key:
        raise ValueError("Set BASELINE_LLM_BASE_URL and BASELINE_LLM_API_KEY for offline strategy generation")
    response = requests.post(base+"/chat/completions", headers={"Authorization": "Bearer "+key},
        json=dict(model=args.llm_model, temperature=.7, messages=[dict(role="system", content=
            "Design metro MILP neighborhood scoring policies. Return JSON only, with a strategies list. "
            "Do not generate code, actions, safety constraints, IDs or future information."), dict(role="user", content=prompt)]),
        timeout=args.api_timeout)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = "\n".join(content.splitlines()[1:-1])
    return json.loads(content), dict(response_sha256=hashlib.sha256(content.encode()).hexdigest(), model=args.llm_model)


def evolve(args):
    output = fresh_output(args.output)
    if min(args.outer_rounds, args.population, args.snapshot_limit) < 1:
        raise ValueError("Positive evolution and evaluation sizes required")
    source = line_id(args.source_line)
    train = read_snapshots(args.train_snapshots, source, "train", args.snapshot_limit)
    validation = read_snapshots(args.validation_snapshots, source, "validation", args.snapshot_limit)
    if {r["entry"]["case_id"] for r in train} & {r["entry"]["case_id"] for r in validation}:
        raise ValueError("Evolution and validation episodes overlap")
    settings = BaselineSettings(horizon=args.horizon, budget_s=args.budget, repair_s=args.repair_seconds,
                                iterations=args.iterations, seed=args.seed, threads=args.threads)
    started, elite, evaluations, calls = time.perf_counter(), None, [], []
    for outer in range(args.outer_rounds):
        prompt = json.dumps(dict(problem="finite native metro command decisions with shared resources and OD passenger queues",
            feature_keys=SCORE_KEYS, meaning=dict(queue="current platform queue /1000", events="relevant published events",
                coupling="shared-resource neighbor fraction", load="onboard /2520", busy="in-progress operation", platform="platform control group"),
            output_schema=dict(strategies=[dict(weights={k: "number in [-10,10]" for k in SCORE_KEYS},
                release_fraction="0.05..1", repair_fraction="0.05..1", temperature="0..1")]),
            candidates_requested=args.population, outer_round=outer,
            feedback=[dict(strategy=v["strategy"], score=v["training"]["score"]) for v in evaluations[-args.population:]],
            elite=elite["strategy"] if elite else None))
        proposed, call = request_strategies(args, prompt)
        calls.append(dict(call, outer_round=outer))
        values = proposed.get("strategies")
        if not isinstance(values, list) or not values:
            raise ValueError("LLM did not return a strategy population")
        for inner, raw in enumerate(values[:args.population]):
            strategy = validate_strategy(raw)
            result = score_strategy(strategy, train, settings)
            candidate = dict(outer_round=outer, inner_candidate=inner, strategy=strategy, training=result)
            evaluations.append(candidate)
            if elite is None or result["score"] > elite["training"]["score"]:
                elite = candidate
            print(json.dumps(dict(outer_round=outer, inner_candidate=inner, source_score=result["score"])), flush=True)
    # Validation never enters the prompts or the elite-selection loop.
    held_out = score_strategy(elite["strategy"], validation, settings)
    artifact = dict(kind="metro_llm_lns_dsl_v1", origin="llm_api", source_topology=source,
        layout_revision=LAYOUT_REVISION,
        strategy=elite["strategy"], validation_complete=True, validation=held_out,
        training=elite["training"], evolution=evaluations, calls=calls,
        train_snapshot_sha256=file_hash(args.train_snapshots), validation_snapshot_sha256=file_hash(args.validation_snapshots),
        settings=asdict(settings), preparation_seconds=time.perf_counter()-started,
        original_llm_lns_reproduction=False,
        adaptation="outer score-feedback prompts and inner source-MIP policy evaluation; bounded numerical DSL instead of arbitrary generated code")
    output.write_text(json.dumps(artifact, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train-mlp")
    train.add_argument("--source-line", default="sz20")
    train.add_argument("--train-data", required=True)
    train.add_argument("--validation-data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--hidden", type=int, default=64)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=.001)
    train.add_argument("--seed", type=int, default=111)
    snapshot = commands.add_parser("collect-snapshots")
    snapshot.add_argument("--line", default="sz20")
    snapshot.add_argument("--split", choices=("train", "validation"), required=True)
    snapshot.add_argument("--scenarios", nargs="+", default=None)
    snapshot.add_argument("--scenario-seed", type=int, default=111)
    snapshot.add_argument("--scenario-repeats", type=int, default=1)
    snapshot.add_argument("--episodes", type=int, default=4)
    snapshot.add_argument("--steps", type=int, default=240)
    snapshot.add_argument("--snapshot-interval", type=int, default=30)
    snapshot.add_argument("--n-agents", type=int, default=None)
    snapshot.add_argument("--output", required=True)
    llm = commands.add_parser("evolve-llm-lns")
    llm.add_argument("--source-line", default="sz20")
    llm.add_argument("--train-snapshots", required=True)
    llm.add_argument("--validation-snapshots", required=True)
    llm.add_argument("--output", required=True)
    llm.add_argument("--llm-model", required=True)
    llm.add_argument("--api-base")
    llm.add_argument("--api-timeout", type=float, default=120.)
    llm.add_argument("--outer-rounds", type=int, default=2)
    llm.add_argument("--population", type=int, default=3)
    llm.add_argument("--snapshot-limit", type=int, default=4)
    llm.add_argument("--horizon", type=int, default=3)
    llm.add_argument("--budget", type=float, default=10.)
    llm.add_argument("--repair-seconds", type=float, default=2.)
    llm.add_argument("--iterations", type=int, default=10)
    llm.add_argument("--threads", type=int, default=1)
    llm.add_argument("--seed", type=int, default=111)
    args = parser.parse_args(argv)
    function = {"train-mlp": train_mlp, "collect-snapshots": collect_snapshots, "evolve-llm-lns": evolve}[args.command]
    return function(args)


if __name__ == "__main__":
    main()
