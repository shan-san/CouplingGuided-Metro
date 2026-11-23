"""Native-command MAPPO preparation/training in the original MetroRL world.

Shared candidate scorer, separate vehicle/platform decision groups, pooled
centralized critic. TorchRL supplies clipped PPO and GAE. No trigger is learned.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, MaskedCategorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.functional import generalized_advantage_estimate

from delay_events import EVENT_TARGETS
from dispatch_plan import PARAMETERS
from dispatch_candidates import applicable_options, solution_from_evaluation, repair_joint_selection, SERVICE_REVIEW_MEASURES
from MILP.operation_model import JointDispatchModel, apply_dispatch_step, audit_dispatch_step
from LLM.coupling_trigger import option_footprints, event_relevant
from scenarios.prepare_dispatch import (
    make_world, events_for, current_arrivals, add_curriculum_args, scenarios_from_args, EPISODE_STAGES,
)
from topology.metro_lines import LINE_SPECS, LAYOUT_REVISION, line_id
from experiment_results import EpisodeResults, write_json

FAMILIES = tuple(PARAMETERS)
EVENTS = tuple(sorted(EVENT_TARGETS))
FEATURE_VERSION = "native_candidates_v4_seconds"
FEATURE_DIM = len(FAMILIES) + len(EVENTS) + 33


def relevant_event(event, footprint, world):
    return event_relevant(event, footprint, world)


def candidate_observation(model):
    world = model.base
    masks, rejection_reasons = applicable_options(model)
    groups = list(model.groups)
    width = max(len(options) for options in model.groups.values())
    features = torch.zeros(len(groups), width, FEATURE_DIM)
    mask = torch.zeros(len(groups), width, dtype=torch.bool)
    footprints = option_footprints(model)
    members = {key: defaultdict(int) for key in ("nodes", "vehicles", "depots", "tasks", "resources")}
    group_bits, option_count = {}, 0
    for group, items in footprints.items():
        group_bits[group] = 0
        for fp in items:
            bit = 1 << option_count
            group_bits[group] |= bit
            for key in members:
                for value in fp[key]:
                    members[key][value] |= bit
            option_count += 1
    events = world.get_published_events()
    current_view = world.event_view()
    occupied = Counter(world.dispatch.position(v.id) for v in world.dispatch.records.values())
    for gi, group in enumerate(groups):
        for oi, (option, fp) in enumerate(zip(model.groups[group], footprints[group])):
            mask[gi, oi] = masks[group][oi]
            flags = [float(a in fp["families"]) for a in FAMILIES]
            flags += [float(any(e["type"] == kind and e.get("active", True) and relevant_event(e, fp, world)
                                for e in events)) for kind in EVENTS]
            records = [world.dispatch.records[v] for v in fp["vehicles"]]
            platforms = [world.platform_nodes[p] for p in fp["platforms"]]
            queues = [world.waiting_at(p) for p in fp["platforms"]]
            outside = sum(node.passenger_flow.outside_count(0) for node in platforms)
            related = 0
            for key in members:
                for value in fp[key]:
                    related |= members[key][value]
            claims = (related & ~group_bits[group]).bit_count()
            load = sum(float(v.agent.passenger_count[0]) for v in records)
            capacities = sum(world.event_view().vehicle_capacity(v.id, v.agent.passenger_capacity) for v in records)
            operation = [world.dispatch.operation_for(v.id) for v in records]
            params = option.commands[0]["parameters"] if option.commands else {}
            relevant = [e for e in events if e.get("active", True) and relevant_event(e, fp, world)]
            targets = set(fp["platforms"])
            if option.commands and option.commands[0]["action"] == "skip_stop":
                targets.discard(params["target_node"])
            od_match = sum(float(q[0]) for node in platforms for d, q in node.passenger_flow.waiting_by_destination.items()
                           if d in targets)
            reported_rate = sum(sum(node.passenger_flow.interval_arrivals(current_view, model.tick).values()) for node in platforms)
            extra = [float(group.startswith("platform_")), float(not option.commands and not option.actions and not option.pauses),
                float(any(v.failed for v in records)), float(any(v.needs_inspection for v in records)),
                float(any(v.depot is not None for v in records)), float(any(v.service_id for v in records)),
                min(10., load / max(1., capacities)), min(10., sum(queues) / 1000.), min(10., outside / 1000.),
                min(10., len(fp["nodes"]) / max(1., len(world.nodes))),
                min(10., claims / max(1., option_count)),
                float(any(operation)), min(10., max((max(0., op.deadline - model.now) for op in operation if op), default=0.) / 300.),
                min(10., float(params.get("seconds", params.get("headway_seconds", 0.))) / 300.),
                min(10., float(params.get("rate", 0.)) / 10.),
                float(bool(option.actions) or bool(option.pauses)),
                min(10., max([1.] + [e["parameters"].get("travel_time_multiplier", 1.) for e in relevant])/5.),
                min(10., max([0.] + [e["parameters"].get("additional_seconds", 0.) for e in relevant])/300.),
                min(10., max([0.] + [e["parameters"].get("minimum_safety_headway_seconds", 0.) for e in relevant])/300.),
                min((current_view.boarding_multiplier(p) for p in fp["platforms"]), default=1.),
                min(10., max((world.waiting_at(p)/max(1., current_view.queue_capacity(p, world.safe_platform_queue_capacity))
                              for p in fp["platforms"]), default=0.)),
                world.dispatch.profiles.get(params.get("profile_id"), 1.)/5.,
                min(10., sum(float(v.agent.passengers_by_destination.get(params.get("target_node"), torch.zeros(1))[0])
                             for v in records)/max(1., capacities)),
                min(1., od_match/max(1., sum(queues))), min(10., reported_rate/1000.)]
            clocks = [v.agent.state for v in records]
            def mean_clock(name, scale):
                return min(10., sum(float(getattr(s, name)[0]) for s in clocks) / max(1, len(clocks)) / scale)
            extra += [mean_clock("dwell_time", 10.), mean_clock("dwell_min_limit", 10.),
                mean_clock("dwell_max_limit", 10.), mean_clock("time_since_arrival", 300.),
                min(10., max((max(0., float(s.target_departure_time[0])-model.now) for s in clocks), default=0.)/300.),
                min(10., sum(abs(float(s.headway_gap[0])) for s in clocks)/max(1, len(clocks))/300.),
                float(occupied[params.get("target_node")] > sum(world.dispatch.position(v.id) == params.get("target_node")
                                                                for v in records)),
                min(10., world.waiting_at(params["target_node"])/1000.)
                    if params.get("target_node") in world.platform_nodes else 0.]
            features[gi, oi] = torch.tensor(flags + extra)
    if not mask.any(-1).all() or not torch.isfinite(features).all():
        raise ValueError("Invalid native candidate features/mask")
    # Central critic only. Actor consumes each candidate's associated local objects.
    valid = features[mask]
    central = torch.cat((valid.mean(0), valid.max(0).values)).expand(len(groups), -1).clone()
    return TensorDict(dict(features=features, mask=mask, central=central), batch_size=[len(groups)]), rejection_reasons


class CandidateScorer(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, features):
        return self.layers(features).squeeze(-1)


class NativeDispatchPolicy:
    def __init__(self, hidden=64, device="cpu"):
        self.device = torch.device(device)
        self.hidden = hidden
        scorer = TensorDictModule(CandidateScorer(hidden), in_keys=["features"], out_keys=["logits"])
        self.actor = ProbabilisticActor(scorer, in_keys=["logits", "mask"], out_keys=["action"],
            distribution_class=MaskedCategorical, return_log_prob=True).to(self.device)
        self.critic = TensorDictModule(nn.Sequential(nn.Linear(2*FEATURE_DIM, hidden), nn.Tanh(), nn.Linear(hidden, 1)),
            in_keys=["central"], out_keys=["state_value"]).to(self.device)

    def choose(self, model, *, deterministic=False):
        """Score the current catalog without running a trajectory subproblem."""
        from torchrl.envs.utils import set_exploration_type, ExplorationType
        td, reasons = candidate_observation(model)
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM):
            td = self.actor(td.to(self.device))
            self.critic(td)
        selection = tuple(int(v) for v in td["action"].tolist())
        return selection, td.cpu(), reasons

    def propose(self, world, *, deterministic=False, model=None, quality_guard=False, defer_service_review=False):
        model = model or JointDispatchModel(world, 1)
        selection, td, reasons = self.choose(model, deterministic=deterministic)
        result = model.evaluate(selection)
        attempted = copy.deepcopy(result)
        rejected = not result["feasible"]
        repair = dict(repaired_groups=[], repair_attempts=0, repair_reasons=[], used_full_reference=False)
        if rejected:
            solution, repair = repair_joint_selection(model, selection, result)
            from LLM.dispatch_coordination import checked_reference
            if solution is None:
                solution = checked_reference(world, {a.name: 0 for a in world.agents}, 1, model.inputs)
        else:
            solution = solution_from_evaluation(model, result, selection)
        quality = dict(enabled=quality_guard, replaced=False,
                       comparison="same-state one-window cost including the common terminal objective")
        delayed = sorted({command["action"] for command in solution["dispatch_packets"][0]["commands"]
                          if command["action"] in SERVICE_REVIEW_MEASURES})
        if quality_guard and defer_service_review and delayed:
            quality.update(deferred=True, delayed_measures=delayed,
                           comparison="deferred to same-state operation-aware coordination window")
        elif quality_guard:
            baseline = tuple(next((i for i, op in enumerate(options)
                                   if not op.commands and not op.actions and not op.pauses), 0)
                             for options in model.groups.values())
            reference = model.evaluate(baseline)
            quality.update(policy_cost=solution["cost"], route_cost=reference.get("cost"),
                           route_feasible=reference["feasible"])
            if reference["feasible"] and reference["cost"] < solution["cost"]-1e-6:
                solution = solution_from_evaluation(model, reference, baseline, status="checked_route_quality_guard")
                quality["replaced"] = True
        return solution, td.cpu(), dict(rejected=rejected, attempted_selection=list(selection),
            rejection=attempted if rejected else None, mask_rejections=reasons,
            repair=repair, repair_fraction=len(repair["repaired_groups"])/max(1, len(model.groups)),
            fallback_corrections=copy.deepcopy(solution.get("corrections", [])), quality_guard=quality)

    def save(self, path, metadata):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(kind="native_dispatch_mappo_v1", feature_version=FEATURE_VERSION, feature_dim=FEATURE_DIM,
            layout_revision=LAYOUT_REVISION,
            hidden=self.hidden, action_families=list(FAMILIES), event_families=list(EVENTS),
            actor=self.actor.state_dict(), critic=self.critic.state_dict(), metadata=metadata), path)

    @classmethod
    def load(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (payload.get("kind") != "native_dispatch_mappo_v1" or payload.get("feature_version") != FEATURE_VERSION
                or payload.get("action_families") != list(FAMILIES) or payload.get("event_families") != list(EVENTS)):
            raise ValueError("Incompatible native dispatch checkpoint; second-resolution execution requires native_candidates_v4_seconds. "
                             "Keep older checkpoints for auditing and retrain into a new output directory.")
        policy = cls(payload["hidden"], device)
        policy.actor.load_state_dict(payload["actor"])
        policy.critic.load_state_dict(payload["critic"])
        policy.metadata = payload["metadata"]
        if payload.get("layout_revision") != LAYOUT_REVISION:
            raise ValueError(
                "Checkpoint uses an older or unspecified topology layout revision; "
                f"expected {LAYOUT_REVISION!r}. Retrain MARL after the topology update "
                "and keep the old checkpoint only for historical auditing."
            )
        return policy


def balanced_episodes(entries, seed):
    """Interleave event, combination and recovery contexts before repeating."""
    rng = random.Random(seed)
    variants = defaultdict(list)
    for entry in entries:
        variants[entry["scenario"]].append(entry)
    for values in variants.values():
        rng.shuffle(values)
    result = []
    for repeat in range(max(map(len, variants.values()))):
        buckets = defaultdict(list)
        for scenario, values in variants.items():
            if repeat < len(values):
                buckets[scenario.split("_", 1)[0]].append(values[repeat])
        for values in buckets.values():
            rng.shuffle(values)
        queues = {key: deque(values) for key, values in buckets.items()}
        while any(queues.values()):
            for key in ("combination", "single", "operation", "single", "single", "normal"):
                if queues.get(key):
                    result.append(queues[key].popleft())
    return result


def action_counts(model, td, solution, info):
    counts = {name: Counter() for name in FAMILIES}
    for gi, options in enumerate(model.groups.values()):
        for oi, option in enumerate(options):
            for command in option.commands:
                values = counts[command["action"]]
                values["candidates"] += 1
                values["applicable"] += int(td["mask"][gi, oi])
                values["proposed"] += int(info["attempted_selection"][gi] == oi)
    for command in solution["dispatch_packets"][0]["commands"]:
        counts[command["action"]]["committed"] += 1
    return counts


def observe_execution(world, before_positions, log_start, counts):
    for event in world.dispatch.log[log_start:]:
        family = event.get("action")
        if event["event"] == "command_started":
            counts[family]["started"] += 1
            if world.dispatch.command_results[event["command_id"]].get("operation_id") is None:
                counts[family]["completed"] += 1
        elif event["event"] == "operation_completed":
            counts[family]["completed"] += 1
        elif event["event"] in ("operation_failed", "command_rejected"):
            family = family or world.dispatch.command_results[event["command_id"]]["action"]
            counts[family]["failed"] += 1
    return dict(actions={key: dict(value) for key, value in counts.items() if value},
        moved_vehicles=sum(before_positions[a.name] != int(a.state.position[0]) for a in world.agents),
        boarded_total=sum(float(a.cumulative_boarded[0]) for a in world.agents),
        alighted_total=sum(float(a.cumulative_alighted[0]) for a in world.agents),
        passengers=world.dispatch.objective.passenger_state(world))


def summarize_coverage(rows, previous=None):
    previous = previous or {}
    counts = {name: Counter(previous.get("action_counts", {}).get(name, {})) for name in FAMILIES}
    events = Counter(previous.get("observed_event_steps", {}))
    cases = Counter(previous.get("visited_episode_counts", {}))
    for row in rows:
        events.update(row["observed_event_types"])
        if row["stage"] == 0:
            cases[row["case_id"]] += 1
        for family, values in row.get("actions", {}).items():
            counts[family].update(values)
    return dict(action_counts={key: dict(value) for key, value in counts.items()},
        missing_applicable_actions=[key for key, value in counts.items() if not value["applicable"]],
        missing_started_actions=[key for key, value in counts.items() if not value["started"]],
        missing_completed_actions=[key for key, value in counts.items() if not value["completed"]],
        observed_event_steps=dict(events), missing_events=sorted(set(EVENTS)-set(events)),
        visited_episode_counts=dict(cases), validation="coverage diagnostics, not a scheduling-performance certificate")


def train(args):
    if min(args.n_iters, args.steps, args.num_epochs, args.frames_per_batch, args.minibatch_size) < 1:
        raise ValueError("Positive training sizes required")
    if args.frames_per_batch % args.steps:
        raise ValueError("frames-per-batch must be a multiple of steps to collect complete episodes")
    max_hours = getattr(args, "max_training_hours", None)
    if max_hours is not None and (not math.isfinite(max_hours) or max_hours <= 0):
        raise ValueError("max_training_hours must be finite and positive")
    if not math.isfinite(args.reward_scale) or args.reward_scale <= 0 or not math.isfinite(args.rejection_penalty) or args.rejection_penalty < 0:
        raise ValueError("Invalid reward scale or repair penalty")
    threads = getattr(args, "torch_threads", 1)
    if threads < 1:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(threads)
    if args.prepare_only:
        return _train(args)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("TensorBoard logging requires tensorboard. Install it with: python -m pip install tensorboard") from exc
    key = LINE_SPECS[line_id(args.line)]["key"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path(getattr(args, "tensorboard_dir", "runs")).expanduser()
    log_dir = root / f"dispatch_{key}" / f"{timestamp}_{os.getpid()}"
    with SummaryWriter(log_dir=str(log_dir.resolve()), flush_secs=15) as writer:
        print(f"TensorBoard logs: {writer.log_dir}", flush=True)
        return _train(args, writer)


def _train(args, writer=None):
    max_hours = getattr(args, "max_training_hours", None)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    recipe, entries = scenarios_from_args(args, "train")
    policy = NativeDispatchPolicy(args.hidden, args.device)
    loss = ClipPPOLoss(policy.actor, policy.critic, clip_epsilon=.2, entropy_coeff=.01, normalize_advantage=False)
    optimizer = torch.optim.Adam(loss.parameters(), lr=args.lr)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() for name in ("latest.pt", "training.jsonl", "training_summary.json")):
        raise FileExistsError("Training output already contains a run; use a new output directory")
    metadata = dict(source_topology=recipe["topology"], scenario_recipe=recipe, scenario_sha256=recipe["sha256"],
        configured_episode_ids=[entry["case_id"] for entry in entries], source_episode_ids=[], seed=args.seed,
        source_fleet=args.n_agents or LINE_SPECS[recipe["topology"]]["fleet"], tick_seconds=recipe["tick_seconds"],
        physical_tick_seconds=recipe["physical_tick_seconds"], decision_seconds=recipe["decision_seconds"],
        episode_seconds=args.steps*recipe["tick_seconds"], episodes_per_iteration=args.frames_per_batch//args.steps,
        training_parameters=vars(args).copy(),
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            "RL/dispatch_training.py", "MILP/cplex_backend.py", "MILP/operation_model.py", "dispatch_operations.py", "dispatch_candidates.py",
            "delay_events.py", "LLM/dispatch_coordination.py", "scenarios/prepare_dispatch.py",
            "topology/passenger_flow.py", "metro_agent_v2.py", "metro_world.py", "metro_scenario_v2.py", "metro_environment.py",
            "dispatch_objective.py", "route_selection.py", "dispatch_timing.py", "dispatch_records.py", "buildtopology.py")},
        observation="candidate-associated published state; central critic pools all groups",
        action="parameterized current-state command candidates; separate platform groups",
        reward="negative shared interval and terminal costs divided by reward_scale; repair penalty times fraction of changed groups",
        reward_scale=args.reward_scale, rejection_penalty=args.rejection_penalty, discount=args.gamma,
        thresholds="no trigger training", training_finished=False, iterations_completed=0,
        optimizer_updates=0, max_training_hours=max_hours,
        physical_frames=0, tensorboard_log_dir=writer.log_dir if writer else None,
        initial_passenger_seconds=recipe["tick_seconds"], passenger_input="current physical intensity and OD shares per interval; no future arrivals",
        passenger_bookkeeping="float64 OD queues and loads; onboard total derived from destination loads",
        sampling="balanced context interleaving, full cycle without replacement; explicit episode counter",
        capability_validated=False, visited_episode_ids=[],
        loss_reporting="sample-weighted minibatch means over all PPO epochs per iteration",
        time_budget_check="complete training iteration; no episode or PPO update is truncated")
    if args.prepare_only:
        policy.save(output/"initial.pt", metadata)
        (output/"preparation.json").write_text(json.dumps(metadata, indent=2)+"\n")
        return metadata
    if writer:
        writer.add_text("Run/configuration", json.dumps(dict(source_topology=recipe["topology"],
            scenario_recipe=recipe, training_parameters=vars(args)), indent=2), 0)
    logs = []
    episode_cursor = 0
    episode_order = balanced_episodes(entries, args.seed)
    started = monotonic()
    budget_seconds = max_hours * 3600 if max_hours is not None else float("inf")
    stop_reason = "iteration_limit"
    for iteration in range(args.n_iters):
        iteration_started = monotonic()
        if iteration_started - started >= budget_seconds:
            stop_reason = "time_budget"
            break
        trajectories = []
        frame_count = 0
        logs = []
        iteration_log_start = 0
        episode_returns = []
        while frame_count < args.frames_per_batch:
            if episode_cursor and episode_cursor % len(entries) == 0:
                episode_order = balanced_episodes(entries, args.seed+episode_cursor//len(entries))
            entry = episode_order[episode_cursor % len(entries)]
            episode_cursor += 1
            world = make_world(recipe["topology"], fleet=args.n_agents, seed=entry["seed"],
                events=events_for(entry, fleet=args.n_agents), tick=recipe["tick_seconds"],
                initial_passenger_seconds=recipe["tick_seconds"])
            if "initial_stock" not in metadata:
                metadata["initial_stock"] = copy.deepcopy(world.dispatch.config["metadata"]["initial_stock"])
                metadata["objective_definition"] = world.dispatch.objective.manifest()
            metadata.setdefault("initial_stock_by_episode", {})[entry["case_id"]] = copy.deepcopy(
                world.dispatch.config["metadata"]["initial_stock"])
            initial_passengers = sum(world.dispatch.objective.passenger_state(world)[k] for k in ("inside", "outside", "onboard"))
            arrival_total, last_boarded, last_alighted = 0., 0., 0.
            records, rewards, values = [], [], []
            for stage in range(args.steps):
                observed_types = sorted({e["type"] for e in world.get_published_events() if e.get("active", True)})
                arrivals = current_arrivals(world)
                before = {a.name: int(a.state.position[0]) for a in world.agents}
                log_start = len(world.dispatch.log)
                decision_started = time.perf_counter()
                model = JointDispatchModel(world, 1)
                solution, td, info = policy.propose(world, model=model)
                counts = action_counts(model, td, solution, info)
                committed = apply_dispatch_step(world, solution)
                if not committed["accepted"]:
                    raise RuntimeError(committed)
                decision_seconds = time.perf_counter()-decision_started
                world.step()
                arrivals = world.last_arrival_inputs
                audit = audit_dispatch_step(world, solution)
                execution = observe_execution(world, before, log_start, counts)
                arrived = sum(sum(values.values()) for values in arrivals.values())
                arrival_total += arrived
                people = execution["passengers"]
                balance = sum(people[k] for k in ("inside", "outside", "onboard")) + execution["alighted_total"] - initial_passengers - arrival_total
                if not math.isclose(balance, 0., abs_tol=1e-2+1e-5*(initial_passengers+arrival_total)):
                    raise AssertionError(f"Training passenger conservation failed: {balance}")
                physical = world.dispatch.interval_cost["total"]
                terminal = stage+1 == args.steps
                # Complete finite episodes retain unfinished journeys and tasks.
                if terminal:
                    physical += world.dispatch.objective.terminal(world)["total"]
                service_reward = -physical / args.reward_scale
                repair_penalty = args.rejection_penalty*info["repair_fraction"]
                reward = service_reward - repair_penalty
                records.append(td.detach())
                rewards.append(reward)
                values.append(float(td["state_value"].mean()))
                logs.append(dict(iteration=iteration, case_id=entry["case_id"], stage=stage,
                    simulation_seconds=float(world.sim_time[0]),
                    observed_event_types=observed_types,
                    physical_cost=physical, reward=reward, proposal_rejected=info["rejected"],
                    service_reward=service_reward, repair_penalty=repair_penalty,
                    repair_fraction=info["repair_fraction"], repair=info["repair"],
                    rejection_reason=(info["rejection"] or {}).get("reason"), decision_time_s=decision_seconds,
                    arrived=arrived, boarded=execution["boarded_total"]-last_boarded,
                    alighted=execution["alighted_total"]-last_alighted,
                    passenger_balance_error=balance, execution_matched=audit["matched"],
                    state_mismatches=audit["state_mismatches"],
                    positions_match=audit["positions_match"], **execution,
                    fallback_corrections=info["fallback_corrections"],
                    actual_constraint_interventions=audit["constraint_interventions"]))
                last_boarded, last_alighted = execution["boarded_total"], execution["alighted_total"]
                with (output/"training.jsonl").open("a") as stream:
                    stream.write(json.dumps(logs[-1], allow_nan=False)+"\n")
            episode_returns.append(sum(rewards))
            reward_tensor = torch.tensor(rewards).unsqueeze(-1)
            value_tensor = torch.tensor(values).unsqueeze(-1)
            done = torch.zeros_like(reward_tensor, dtype=torch.bool)
            done[-1] = True
            next_value = torch.cat((value_tensor[1:], torch.zeros_like(value_tensor[:1])))
            adv, target = generalized_advantage_estimate(args.gamma, .95, value_tensor, next_value,
                reward_tensor, done, terminated=done, time_dim=0)
            for k, td in enumerate(records):
                td["advantage"] = adv[k].expand(len(td), 1).clone()
                td["value_target"] = target[k].expand(len(td), 1).clone()
            trajectories.extend(records)
            frame_count += len(records)
        width = max(td["features"].shape[-2] for td in trajectories)
        for td in trajectories:
            padding = width - td["features"].shape[-2]
            td["features"] = nn.functional.pad(td["features"], (0, 0, 0, padding))
            td["mask"] = nn.functional.pad(td["mask"], (0, padding), value=False)
        keys = ["features", "mask", "central", "action", "action_log_prob", "advantage", "value_target", "state_value"]
        batch = torch.cat([td.select(*keys) for td in trajectories], 0).to(policy.device)
        if len(batch) > 1:
            batch["advantage"] = (batch["advantage"]-batch["advantage"].mean()) / batch["advantage"].std().clamp_min(1e-6)
        loss_names = ("policy_loss", "value_loss", "entropy_loss", "total_loss")
        loss_sums = torch.zeros(4, device=policy.device)
        loss_samples = 0
        for _ in range(args.num_epochs):
            for indices in torch.randperm(len(batch), device=policy.device).split(args.minibatch_size):
                result = loss(batch[indices])
                total = result["loss_objective"] + result["loss_critic"] + result["loss_entropy"]
                optimizer.zero_grad()
                total.backward()
                gradient_norm = nn.utils.clip_grad_norm_(loss.parameters(), 1.)
                optimizer.step()
                metadata["optimizer_updates"] += 1
                values = torch.stack((result["loss_objective"], result["loss_critic"],
                                      result["loss_entropy"], total)).detach()
                loss_sums += values * len(indices)
                loss_samples += len(indices)
                if writer:
                    for name, value in zip(loss_names, values.cpu().tolist()):
                        writer.add_scalar(f"OptimizerLosses/{name}", value, metadata["optimizer_updates"])
                    writer.add_scalar("Optimization/gradient_norm", float(gradient_norm), metadata["optimizer_updates"])
                    writer.add_scalar("Optimization/minibatch_samples", len(indices), metadata["optimizer_updates"])
        # Synchronize CUDA before the budget check; report all updates, not only
        # the last minibatch (which may be smaller than the others).
        mean_losses = (loss_sums / loss_samples).cpu().tolist()
        loss_value = mean_losses[-1]
        elapsed = monotonic() - started
        time_limit_reached = elapsed >= budget_seconds
        metadata["physical_frames"] += frame_count
        metadata.update(iterations_completed=iteration+1, elapsed_training_seconds=elapsed,
                        training_finished=time_limit_reached or iteration+1 == args.n_iters)
        metadata["coverage"] = summarize_coverage(logs, metadata.get("coverage"))
        metadata["visited_episode_ids"] = sorted(metadata["coverage"]["visited_episode_counts"])
        metadata["source_episode_ids"] = list(metadata["visited_episode_ids"])
        if writer:
            step = iteration+1
            for name, value in zip(loss_names, mean_losses):
                writer.add_scalar(f"Losses/{name}", value, step)
            rows = logs[iteration_log_start:]
            scalars = {
                "Rewards/step_reward_mean": sum(row["reward"] for row in rows) / frame_count,
                "Rewards/episode_reward_mean": sum(episode_returns) / len(episode_returns),
                "Training/physical_cost_mean": sum(row["physical_cost"] for row in rows) / frame_count,
                "Training/proposal_rejection_rate": sum(row["proposal_rejected"] for row in rows) / frame_count,
                "Training/repaired_group_fraction": sum(row["repair_fraction"] for row in rows) / frame_count,
                "Rewards/service_reward_mean": sum(row["service_reward"] for row in rows) / frame_count,
                "Rewards/repair_penalty_mean": sum(row["repair_penalty"] for row in rows) / frame_count,
                "Service/arrivals": sum(row["arrived"] for row in rows),
                "Service/boarded": sum(row["boarded"] for row in rows),
                "Service/alighted": sum(row["alighted"] for row in rows),
                "Service/inside_queue_mean": sum(row["passengers"]["inside"] for row in rows)/frame_count,
                "Service/outside_queue_mean": sum(row["passengers"]["outside"] for row in rows)/frame_count,
                "Service/moved_vehicles": sum(row["moved_vehicles"] for row in rows),
                "Service/max_passenger_balance_error": max(abs(row["passenger_balance_error"]) for row in rows),
                "Coverage/visited_episodes": len(metadata["visited_episode_ids"]),
                "Coverage/observed_event_families": len(metadata["coverage"]["observed_event_steps"]),
                "Coverage/completed_action_families": len(FAMILIES)-len(metadata["coverage"]["missing_completed_actions"]),
                "Training/fallback_correction_rate": sum(bool(row["fallback_corrections"]) for row in rows) / frame_count,
                "Training/constraint_interventions": sum(len(row["actual_constraint_interventions"]) for row in rows),
                "Training/learning_rate": optimizer.param_groups[0]["lr"],
                "Training/physical_frames": metadata["physical_frames"],
                "Training/optimizer_updates": metadata["optimizer_updates"],
                "Timing/iteration_seconds": started + elapsed - iteration_started,
                "Timing/elapsed_training_seconds": elapsed,
            }
            for name, value in scalars.items():
                writer.add_scalar(name, value, step)
            for family, values in summarize_coverage(rows)["action_counts"].items():
                for key in ("applicable", "proposed", "committed", "started", "completed", "failed"):
                    writer.add_scalar(f"Actions/{family}/{key}", values.get(key, 0), step)
            writer.flush()
        policy.save(output/"latest.pt", metadata)
        print(json.dumps(dict(iteration=iteration+1, frames=frame_count, loss=loss_value,
                              elapsed_training_seconds=elapsed), allow_nan=False), flush=True)
        if time_limit_reached:
            stop_reason = "time_budget"
            break
    metadata.update(training_finished=True, stop_reason=stop_reason, elapsed_training_seconds=monotonic()-started)
    if writer:
        writer.add_text("Run/stop_reason", stop_reason, metadata["iterations_completed"])
    policy.save(output/"latest.pt", metadata)
    (output/"training_summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(stop_reason=stop_reason, iterations_completed=metadata["iterations_completed"],
                          optimizer_updates=metadata["optimizer_updates"],
                          elapsed_training_seconds=metadata["elapsed_training_seconds"])), flush=True)
    return metadata


def evaluate(args):
    if not args.checkpoint:
        raise ValueError("Frozen evaluation requires --checkpoint")
    policy = NativeDispatchPolicy.load(args.checkpoint, args.device)
    torch.set_num_threads(getattr(args, "torch_threads", 1))
    recipe, entries = scenarios_from_args(args, args.split)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in entries[:args.episodes]:
        world = make_world(recipe["topology"], fleet=args.n_agents, seed=entry["seed"],
            events=events_for(entry, fleet=args.n_agents), tick=recipe["tick_seconds"],
            initial_passenger_seconds=recipe["tick_seconds"])
        results = EpisodeResults(output/entry["case_id"], world, method="marl", case_id=entry["case_id"],
            requested_stages=args.steps, entry=entry, recipe=recipe,
            configuration=dict(checkpoint=str(args.checkpoint), device=args.device),
            source_sha256={"RL/dispatch_training.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        costs, rejections, interventions, latencies = [], 0, [], []
        step_records = []
        initial_passengers = sum(world.dispatch.objective.passenger_state(world)[k] for k in ("inside", "outside", "onboard"))
        arrival_total, last_boarded, last_alighted = 0., 0., 0.
        error = None
        for stage in range(args.steps):
            try:
                started = time.perf_counter()
                observed = sorted({e["type"] for e in world.get_published_events() if e.get("active", True)})
                arrivals = current_arrivals(world)
                before = {a.name: int(a.state.position[0]) for a in world.agents}
                log_start = len(world.dispatch.log)
                model = JointDispatchModel(world, 1)
                solution, td, info = policy.propose(world, deterministic=True, model=model)
                counts = action_counts(model, td, solution, info)
                committed = apply_dispatch_step(world, solution)
                latencies.append(time.perf_counter()-started)
                if not committed["accepted"]:
                    raise RuntimeError(committed)
                world.step()
                arrivals = world.last_arrival_inputs
                audit = audit_dispatch_step(world, solution)
                execution = observe_execution(world, before, log_start, counts)
                arrived = sum(sum(values.values()) for values in arrivals.values())
                arrival_total += arrived
                balance = sum(execution["passengers"][k] for k in ("inside", "outside", "onboard")) + execution["alighted_total"] - initial_passengers - arrival_total
                if not math.isclose(balance, 0., abs_tol=1e-2+1e-5*(initial_passengers+arrival_total)):
                    raise AssertionError(f"Evaluation passenger conservation failed: {balance}")
                step_records.append(dict(case_id=entry["case_id"], stage=stage,
                    observed_event_types=observed, repair=info["repair"], repair_fraction=info["repair_fraction"],
                    arrived=arrived, boarded=execution["boarded_total"]-last_boarded,
                    alighted=execution["alighted_total"]-last_alighted, passenger_balance_error=balance,
                    state_mismatches=audit["state_mismatches"],
                    execution_matched=audit["matched"], positions_match=audit["positions_match"], **execution))
                last_boarded, last_alighted = execution["boarded_total"], execution["alighted_total"]
                costs.append(dict(world.dispatch.interval_cost))
                rejections += int(info["rejected"])
                interventions.append(audit["constraint_interventions"])
                results.annotate(decision_s=latencies[-1], metadata=info, audit=audit,
                                 fallback=bool(info["repair"].get("used_full_reference")))
            except Exception as exc:
                error = str(exc)
                break
        terminal = world.dispatch.objective.terminal(world)
        gamma = world.dispatch.objective.weights["discount"]
        waiting = sum(c["waiting_person_seconds"] for c in costs)
        onboard = sum(c["onboard_person_seconds"] for c in costs)
        rows.append(dict(case_id=entry["case_id"], completed_stages=len(costs), requested_stages=args.steps,
            complete=error is None, error=error, interval_costs=costs, terminal=terminal,
            cost=sum(gamma**k*c["total"] for k, c in enumerate(costs))+gamma**len(costs)*terminal["total"],
            waiting_person_seconds=waiting, onboard_person_seconds=onboard, passenger_person_seconds=waiting+onboard,
            objective_definition=world.dispatch.objective.manifest(),
            rejected_proposals=rejections, actual_constraint_interventions=interventions,
            step_records=step_records, coverage=summarize_coverage(step_records),
            initial_stock=copy.deepcopy(world.dispatch.config["metadata"]["initial_stock"])))
        rows[-1]["decision_times_s"] = latencies
        results.finish(status="error" if error else "complete", error=error, diagnostics=rows[-1],
                       computation_time_s=sum(latencies))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        source_model_metadata=policy.metadata, target_topology=recipe["topology"], split=args.split,
        scenario_recipe=recipe, scenario_sha256=recipe["sha256"],
        training_updates=0, cases=rows,
        coverage=summarize_coverage([step for row in rows for step in row["step_records"]]),
        capability_validated=False)
    write_json(output/"diagnostics"/"evaluation.json", result)
    print(json.dumps(dict(cases=len(rows), completed=sum(r["complete"] for r in rows), training_updates=0)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_curriculum_args(parser)
    parser.add_argument("--output", default="saves/dispatch_sz20")
    parser.add_argument("--tensorboard-dir", default="runs", help="TensorBoard root; each line/run gets a separate subdirectory")
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument("--max-training-hours", type=float, default=None,
                        help="Optional wall-time budget, checked between complete training iterations; whichever limit is reached first")
    parser.add_argument("--steps", type=int, default=EPISODE_STAGES, help="Physical stages per episode, from 06:00; default 240 stages cover 06:00--08:00")
    parser.add_argument("--frames-per-batch", type=int, default=2*EPISODE_STAGES, help="Physical stages per iteration; must be a multiple of steps")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--reward-scale", type=float, default=10000.)
    parser.add_argument("--rejection-penalty", type=float, default=.01,
                        help="Auxiliary penalty per unit fraction of safety-filtered decision groups")
    parser.add_argument("--torch-threads", type=int, default=1, help="CPU threads per training/evaluation process")
    parser.add_argument("--n-agents", type=int, default=None, help="Smoke-test override; otherwise use line fleet")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--episodes", type=int, default=None, help="Evaluation episode limit; default evaluates the entire selected split")
    parser.add_argument("--scenarios", nargs="+", default=None, help="Optional exact scenario names from the Python generator")
    args = parser.parse_args(argv)
    if args.reward_scale <= 0 or args.rejection_penalty < 0 or not 0 < args.gamma <= 1:
        parser.error("Invalid reward scaling/penalty")
    evaluate(args) if args.evaluate else train(args)


if __name__ == "__main__":
    main()
