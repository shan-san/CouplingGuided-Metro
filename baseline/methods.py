"""Comparable rolling solvers on JointDispatchModel, not a second metro model.

Bandit ALNS and the safe-DSL LLM-LNS are metro adaptations, not reproductions of
the complete Balans/LLM-LNS software. Neighborhood bounds are never global bounds.
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
import time

from dispatch_candidates import applicable_options, public_dispatch_state, solution_from_evaluation, stock_start_options
from dispatch_plan import DispatchCommand
from LLM.coupling_trigger import option_footprints, event_relevant
from LLM.dispatch_coordination import close_focus
from baseline.guidance import LearnedGuidance, RetrievalGuidance, file_hash
from MILP.operation_model import JointDispatchModel, EvaluationTimeout
from topology.metro_lines import LAYOUT_REVISION
from baseline.heuristics import PURE_METHODS, PureHeuristicController, rule_rank, ranked_options

MILP_METHODS = ("rh_milp", "alns_mip", "bandit_alns", "rule_guided", "rule_guided_marl",
                "retrieval_guided", "mlp_guided", "llm_lns")
METHODS = (*MILP_METHODS, *PURE_METHODS)
OPERATORS = ("random", "hotspot", "event", "shared_resource")
SCORE_KEYS = ("queue", "events", "coupling", "load", "busy", "platform")


@dataclass(frozen=True)
class BaselineSettings:
    horizon: int = 3
    budget_s: float = 10.
    repair_s: float = 2.
    iterations: int = 20
    release_fraction: float = .35
    confidence: float = .9
    threads: int = 1
    seed: int = 111
    max_evaluations: int | None = None
    command_stages: tuple[int, ...] = (0,)
    adaptive_rule_config: bool = False
    max_rule_horizon: int = 6
    population_size: int = 20
    horizon_policy: str = "operation_aware"
    max_operation_horizon: int = 12
    minimum_repair_s: float = 0.

    def __post_init__(self):
        if (self.horizon < 1 or self.iterations < 1 or self.threads < 0
                or self.horizon_policy not in {"fixed", "operation_aware"} or self.max_operation_horizon < 1
                or not math.isfinite(self.budget_s) or self.budget_s <= 0
                or not math.isfinite(self.repair_s) or self.repair_s <= 0
                or not math.isfinite(self.minimum_repair_s) or self.minimum_repair_s < 0
                or not 0 < self.release_fraction <= 1 or not 0 <= self.confidence <= 1
                or self.max_evaluations is not None and self.max_evaluations < 1
                or self.adaptive_rule_config and self.max_rule_horizon < self.horizon
                or not self.command_stages
                or type(self.population_size) is not int or self.population_size < 2
                or any(type(k) is not int or not 0 <= k < self.horizon for k in self.command_stages)):
            raise ValueError("Invalid rolling baseline settings")


def repair_allowance(settings, remaining, portion, evaluation_floor):
    """A requested repair minimum cannot extend the overall request deadline."""
    preferred = min(settings.repair_s, max(.001, settings.budget_s*portion))
    return max(0., min(remaining, max(settings.minimum_repair_s, evaluation_floor, preferred)))


def rule_configuration(world, settings):
    state = public_dispatch_state(world)
    active = any(e.get("active", True) for e in state.get("disruptions", []))
    queues = list(state.get("waiting_passengers", {}).values())
    pressure = max(queues, default=0.) > 1000.
    horizon = min(settings.max_rule_horizon, settings.horizon*(2 if active or pressure else 1))
    fraction = 1. if active else .8 if pressure else .6
    return replace(settings, horizon=horizon, budget_s=settings.budget_s*fraction), dict(
        active_published_event=active, queue_pressure=pressure, queue_threshold=1000.,
        planning_horizon=horizon, request_budget_fraction=fraction,
        basis="source-frozen deterministic rule; not learned or proved optimal")


def validate_strategy(value):
    if not isinstance(value, dict) or set(value) != {"weights", "release_fraction", "repair_fraction", "temperature"}:
        raise ValueError("LLM strategy must use the declared numerical DSL only")
    weights = value["weights"]
    if not isinstance(weights, dict) or set(weights) != set(SCORE_KEYS):
        raise ValueError("Unknown LLM strategy feature; Python code is not accepted")
    if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 10 for v in weights.values()):
        raise ValueError("Invalid strategy weight")
    for key, lower, upper in (("release_fraction", .05, 1.), ("repair_fraction", .05, 1.), ("temperature", 0., 1.)):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or not lower <= value[key] <= upper:
            raise ValueError(f"Invalid strategy field: {key}")
    return copy.deepcopy(value)


class OperatorSelector:
    """Segmented roulette ALNS or UCB1 over operator/size/budget arms."""
    def __init__(self, bandit=False):
        self.bandit = bandit
        self.arms = ([(o, f, b) for o in OPERATORS for f in (.25, .5) for b in (.15, .35)]
                     if bandit else [(o, .35, .25) for o in OPERATORS])
        self.counts = [0]*len(self.arms)
        self.values = [0.]*len(self.arms)
        self.weights = [1.]*len(self.arms)

    def choose(self, rng):
        if not self.bandit:
            return rng.choices(range(len(self.arms)), weights=self.weights, k=1)[0]
        if 0 in self.counts:
            return self.counts.index(0)
        total = sum(self.counts)
        return max(range(len(self.arms)), key=lambda i: self.values[i]+math.sqrt(2*math.log(total)/self.counts[i]))

    def update(self, index, reward):
        self.counts[index] += 1
        self.values[index] += (reward-self.values[index])/self.counts[index]
        self.weights[index] = max(.05, .8*self.weights[index]+.2*reward)

    def report(self):
        return [dict(operator=a[0], fraction=a[1], repair_fraction=a[2], count=n, mean_reward=v, weight=w)
                for a, n, v, w in zip(self.arms, self.counts, self.values, self.weights)]


def contexts(problem):
    world = problem.base
    footprints, result = {}, {}
    events = [e for e in world.get_published_events() if e.get("active", True)]
    for g, items in option_footprints(problem).items():
        fp = {key: set().union(*(v[key] for v in items)) for key in items[0]}
        if g in world.dispatch.records:
            fp["vehicles"].add(g)
            pos = world.dispatch.position(g)
            fp["nodes"].add(pos)
            if pos in world.platform_nodes:
                fp["platforms"].add(pos)
        footprints[g] = fp
        # Potential withdrawal/reassignment paths can span a whole line. They
        # describe dependencies, not the current queue at every decision owner.
        local = {world.dispatch.position(g)} & set(world.platform_nodes) if g in world.dispatch.records else (
            {int(g.removeprefix("platform_"))} if g.startswith("platform_") else set())
        result[g] = dict(queue=min(10., sum(world.waiting_at(p) for p in local)/1000.),
            events=float(sum(event_relevant(e, fp, world) for e in events)),
            load=min(10., sum(float(world.dispatch.records[v].agent.passenger_count[0]) for v in fp["vehicles"])/2520.),
            busy=float(any(world.dispatch.operation_for(v) for v in fp["vehicles"])),
            platform=float(g.startswith("platform_")), coupling=0.)
    graph = {g: set() for g in problem.groups}
    owners = {}
    for g, fp in footprints.items():
        for key in ("nodes", "vehicles", "depots", "tasks", "resources"):
            for resource in fp[key]:
                owners.setdefault((key, resource), set()).add(g)
    for related in owners.values():
        for g in related:
            graph[g].update(related-{g})
    for g in graph:
        result[g]["coupling"] = len(graph[g])/max(1, len(graph)-1)
    return result, graph, footprints


def neighborhood(problem, context, graph, operator, fraction, rng, strategy=None):
    groups = list(problem.groups)
    count = max(1, math.ceil(len(groups)*fraction))
    if strategy:
        ranked = sorted(groups, key=lambda g: (-sum(context[g][k]*strategy["weights"][k] for k in SCORE_KEYS), g))
    elif operator == "random":
        ranked = rng.sample(groups, len(groups))
    else:
        key = {"hotspot": "queue", "event": "events", "shared_resource": "coupling"}[operator]
        ranked = sorted(groups, key=lambda g: (-context[g][key], -context[g]["queue"], g))
        if operator == "shared_resource":
            related = graph[ranked[0]]
            ranked = [ranked[0]]+[g for g in ranked[1:] if g in related]+[g for g in ranked[1:] if g not in related]
    released = set(ranked[:count])
    # A rescue pair is a physical commitment, not two independent neighborhoods.
    changed = True
    while changed:
        previous = set(released)
        for operation in problem.base.dispatch.operations.values():
            if operation.participants() & released:
                released |= operation.participants() & set(groups)
        changed = previous != released
    return released


def fixes_for(problem, anchor, released):
    return {f"command_{gi}_{oi}": int(oi == anchor[gi])
            for gi, (g, options) in enumerate(problem.groups.items()) if g not in released
            for oi in range(len(options))}


def checked_seed(problem):
    initial = tuple(next((i for i, o in enumerate(options) if not o.commands and not o.actions and not o.pauses), 0)
                    for options in problem.groups.values())
    value = problem.evaluate(initial)
    if not value["feasible"]:
        initial = tuple(next((i for i, o in enumerate(options) if o.id in ("wait_window", "defer_movement")), initial[j])
                        for j, options in enumerate(problem.groups.values()))
        value = problem.evaluate(initial)
    if not value["feasible"]:
        raise RuntimeError("No checked common initial solution in the declared candidate domain")
    problem.evaluated[initial] = value
    return initial, solution_from_evaluation(problem, value, initial, status="checked_common_seed")


def repair_seeds(problem, anchor, released, masks, context, graph, rng, *, operator="random",
                 proposed=None, limit=12):
    """Feasibility dives before a flat point-cut master chooses dense assignments.

    Masks only order trials; they never remove options from the MIP domain.
    Each trial respects the destroyed neighborhood and still needs exact replay.
    """
    groups = list(problem.groups)
    options = []
    tried = Counter((gi, oi) for selection in problem.evaluated for gi, oi in enumerate(selection)
                    if oi != anchor[gi])
    for gi, (g, choices) in enumerate(problem.groups.items()):
        if g not in released:
            continue
        for oi, op in enumerate(choices):
            if oi == anchor[gi] or not masks[g][oi]:
                continue
            priority = float(proposed is not None and proposed[gi] == oi) * 1000.
            if operator == "rule":
                priority += rule_rank(problem, g, op)
            elif operator in ("hotspot", "event", "shared_resource", "llm_scores"):
                key = {"hotspot": "queue", "event": "events", "shared_resource": "coupling",
                       "llm_scores": "coupling"}[operator]
                priority += context[g][key]
            options.append((gi, oi, priority, rng.random()))
    options.sort(key=lambda item: (tried[item[:2]], -item[2], item[3]))
    result, seen_groups = [], set()

    def add(changes):
        selection = list(anchor)
        for gi, oi in changes:
            selection[gi] = oi
        key = tuple(selection)
        if key == tuple(anchor) or key in problem.evaluated or key in result:
            return
        claims = Counter()
        for g, oi in zip(groups, key):
            for raw in problem.groups[g][oi].commands:
                command = DispatchCommand.from_dict(raw, problem.base.tick_seconds)
                claims.update((command.effective_at, resource) for resource in
                              [*(("vehicle", v) for v in command.participants()), *command.resources()])
        if max(claims.values(), default=0) <= 1:
            result.append(key)

    # Visit distinct owners, then linked pairs and block moves. Repeated
    # repairs skip completed trials rather than retrying their first option.
    singles = []
    for gi, oi, _, _ in options:
        if gi not in seen_groups:
            singles.append((gi, oi))
            seen_groups.add(gi)
    section_trials = []
    for gi, oi, _, _ in options:
        op = problem.groups[groups[gi]][oi]
        if any(c["action"] == "reassign_service" and problem.base.dispatch.services[
                c["parameters"]["service_id"]].get("pattern") for c in op.commands):
            section_trials.append((gi, oi))
    stock_trials = [change for change in stock_start_options(problem, masks)
                    if groups[change[0]] in released and change[1] != anchor[change[0]]]
    # Interleave stock and bypass alternatives: two expensive section trials
    # must not consume the whole request before any insertion is evaluated.
    for changes in (stock_trials[:1], section_trials[:1], stock_trials[1:2], section_trials[1:2]):
        for change in changes:
            add([change])
    # The all-fleet wait remains in the finite domain, but is not repeatedly
    # evaluated ahead of sparse interventions in every short repair budget.
    for change in singles[:2]:
        add([change])
    for change in singles[2:max(2, limit//3)]:
        add([change])
    if proposed is not None:
        changes = [(gi, proposed[gi]) for gi, g in enumerate(groups) if g in released
                   and proposed[gi] != anchor[gi] and masks[g][proposed[gi]]]
        add(changes)
        for start in range(0, min(len(changes), 6), 2):
            add(changes[start:start+2])
    for i, left in enumerate(singles):
        for right in singles[i+1:]:
            if groups[right[0]] in graph.get(groups[left[0]], ()):
                add([left, right])
                break
        if len(result) >= max(1, 2*limit//3):
            break
    for gi, oi, _, _ in options:
        if len(result) >= limit:
            break
        add([(gi, oi)])
    return result[:limit]


def repair(problem, anchor, released, deadline, settings, *, warm=None, priorities=None, parameters=None,
           seeds=()):
    # Physical forecasts/cuts belong to a model. Do not reuse solver expressions
    # or a restricted neighborhood's lower bound in a different master.
    child = problem.fork(deadline=deadline, reuse_evaluations=True)
    # Dense actor/retrieval starts commonly contain one bad decision that
    # rejects the whole plan. Start with a sparse neighborhood proposal instead.
    if seeds:
        warm = {f"command_{gi}_{oi}": 1 for gi, oi in enumerate(seeds[0]) if oi != anchor[gi]}
    guidance = dict(variable_fixing=fixes_for(problem, anchor, released), warm_start=warm or {},
                    branching_priority=priorities or {})
    try:
        model, result = child.solve(max_evaluations=settings.max_evaluations, guidance=guidance,
                                   initial_selection=anchor, solver_parameters=parameters, threads=settings.threads,
                                   seed_selections=seeds, adaptive_seed_completion=True)
    finally:
        # Each exact cost/no-good is valid only for this unchanged snapshot and
        # catalog. Rebuild every neighborhood master and its bound independently.
        problem.evaluated.update(child.evaluated)
        for key, seconds in child.timings.items():
            problem.timings[key] += seconds
    model.dispose()
    selection = (tuple(next(i for i, o in enumerate(options) if o.id == result["selected_options"][g])
                       for g, options in problem.groups.items()) if result.get("milp_incumbent") else None)
    return selection, result


def shifted_selection(problem, previous):
    selection, matches = [], 0
    future = [c for packet in (previous or {}).get("dispatch_packets", [])[1:] for c in packet["commands"]
              if c["effective_at"] >= problem.now]
    def signature(c):
        return c["action"], c["vehicle_id"], c["parameters"], c["effective_at"]
    for group, options in problem.groups.items():
        desired = [c for c in future if c["vehicle_id"] == group or c["vehicle_id"] is None
                   and group == f"platform_{c['parameters'].get('platform_id')}"]
        index = next((i for i, o in enumerate(options) if desired and
                      [signature(c) for c in o.commands] == [signature(c) for c in desired]), 0)
        if desired and index:
            matches += 1
        selection.append(index)
    return tuple(selection), matches


class BaselineController:
    def __init__(self, method, settings=None, *, source_topology=101, retrieval_data=None,
                 mlp_checkpoint=None, strategy_path=None, marl_checkpoint=None):
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        self.method, self.settings = method, settings or BaselineSettings()
        self.rng = random.Random(self.settings.seed)
        self.previous = None
        self.selector = OperatorSelector(method == "bandit_alns")
        self.strategy, self.provider, self.marl = None, None, None
        self.provenance = dict(method=method, source_topology=source_topology, pretrained=False,
            solver_backend="cplex",
            search_protocol="masked_neighborhood_dives_v3_checked_waits",
            adaptation="native finite-command physical oracle; not an unmodified external implementation")
        from MILP.parallel_evaluation import worker_count
        self.provenance["candidate_workers"] = worker_count()
        self.pure = PureHeuristicController(method, self.settings) if method in PURE_METHODS else None
        if self.pure:
            self.provenance.update(self.pure.provenance)
            self.pure.provenance = self.provenance
        required = {"retrieval_guided": retrieval_data, "mlp_guided": mlp_checkpoint,
                    "llm_lns": strategy_path, "rule_guided_marl": marl_checkpoint}
        if method in required and not required[method]:
            raise ValueError(f"{method} requires its source-line data/model/strategy artifact")
        if method == "retrieval_guided":
            self.provider = RetrievalGuidance(retrieval_data, source_topology)
        if method == "mlp_guided":
            self.provider = LearnedGuidance(mlp_checkpoint, source_topology)
        if self.provider:
            self.provenance.update(self.provider.provenance)
            self.provenance["pretrained"] = bool(self.provider.provenance.get("trained"))
        if method == "llm_lns":
            artifact = json.loads(Path(strategy_path).read_text(encoding="utf-8"))
            if (artifact.get("kind") != "metro_llm_lns_dsl_v1" or artifact.get("source_topology") != source_topology
                    or artifact.get("layout_revision") != LAYOUT_REVISION
                    or artifact.get("origin") != "llm_api" or not artifact.get("validation_complete")
                    or not artifact.get("calls") or not artifact.get("train_snapshot_sha256")):
                raise ValueError("LLM-LNS requires an offline-generated, source-validated strategy artifact")
            self.strategy = validate_strategy(artifact["strategy"])
            self.provenance.update(artifact_sha256=file_hash(strategy_path), origin=artifact["origin"],
                                   offline_preparation=True, original_llm_lns_reproduction=False)
        if method == "rule_guided_marl":
            from RL.dispatch_training import NativeDispatchPolicy
            self.marl = NativeDispatchPolicy.load(marl_checkpoint)
            if (self.marl.metadata.get("source_topology") != source_topology
                    or self.marl.metadata.get("iterations_completed", 0) < 1):
                raise ValueError("G05 requires a trained checkpoint from the declared source line")
            self.provenance.update(pretrained=True, checkpoint_sha256=file_hash(marl_checkpoint))

    def decide(self, world, *, candidates=None, passenger_inputs=None):
        if self.pure:
            return self.pure.decide(world, candidates=candidates, passenger_inputs=passenger_inputs)
        cfg, started = self.settings, time.perf_counter()
        configuration = {}
        if self.method.startswith("rule_guided") and cfg.adaptive_rule_config:
            if candidates is not None or passenger_inputs is not None:
                raise ValueError("Adaptive G02 must regenerate the common native catalog and inputs for its horizon")
            cfg, configuration["adaptive_rule"] = rule_configuration(world, cfg)
        from baseline.heuristics import operation_aware_settings
        cfg, configuration["operation_horizon"] = operation_aware_settings(
            world, cfg, candidates=candidates, passenger_inputs=passenger_inputs)
        deadline = started+cfg.budget_s
        problem = JointDispatchModel(world, cfg.horizon, candidates=candidates, passenger_inputs=passenger_inputs,
                                     start_stages=cfg.command_stages, deadline=deadline)
        anchor, best = checked_seed(problem)
        initial_cost, current, current_cost = best["cost"], anchor, best["cost"]
        seed_evaluations = len(problem.evaluated)
        initial_checked = set(problem.evaluated)
        trace = []
        masks, _ = applicable_options(problem)
        context, graph, footprints = ({}, {}, {}) if self.method == "rh_milp" else contexts(problem)
        state = public_dispatch_state(problem.base) if self.provider else None
        all_groups = set(problem.groups)
        scores = [1.]*len(anchor)
        proposed = anchor
        if self.provider:
            proposed, scores, details = self.provider.propose(problem, state)
            configuration["guidance"] = details
        elif self.method.startswith("rule_guided"):
            proposed, matched = shifted_selection(problem, self.previous)
            if not self.marl:
                ranked = ranked_options(problem, masks)
                proposed = tuple(proposed[gi] if proposed[gi] != anchor[gi] and masks[g][proposed[gi]]
                    else next((oi for oi in ranked[g] if rule_rank(problem, g, problem.groups[g][oi]) > 0),
                              anchor[gi]) for gi, g in enumerate(problem.groups))
            configuration["reuse"] = dict(future_command_groups_matched=matched,
                                           committed_operations=len(problem.base.dispatch.operations))
            if self.marl:
                from RL.dispatch_training import candidate_observation
                import torch
                td, _ = candidate_observation(problem)
                with torch.no_grad():
                    td = self.marl.actor(td.to(self.marl.device))
                proposed = tuple(td["logits"].masked_fill(~td["mask"], -math.inf).argmax(-1).cpu().tolist())
        warm = {f"command_{gi}_{oi}": int(oi == proposed[gi])
                for gi, options in enumerate(problem.groups.values()) for oi in range(len(options))}
        priorities = ({f"command_{gi}_{oi}": min(100, int(10*context[g]["queue"]+20*context[g]["coupling"]))
                      for gi, (g, options) in enumerate(problem.groups.items()) for oi in range(len(options))}
                      if self.method.startswith("rule_guided") else {})
        parameters = None
        released = set(all_groups)
        if self.method.startswith("rule_guided"):
            affected = {g for g in problem.groups if context[g]["events"] > 0 or context[g]["queue"] > 1}
            vehicles = sorted(g for g in affected if g in problem.base.dispatch.records)
            platforms = {int(g.removeprefix("platform_")) for g in affected if g.startswith("platform_")}
            vehicles += [v for v in problem.base.dispatch.records
                         if problem.base.dispatch.position(v) in platforms and v not in vehicles]
            # Without an active hotspot keep the ordinary full-domain solve;
            # hypothetical paths do not recursively enlarge a local rule scope.
            vehicles = vehicles or sorted(problem.base.dispatch.records)
            indices, closure = close_focus(problem.base, vehicles, horizon=cfg.horizon)
            names = {problem.base.agents[i].name for i in indices}
            platforms.update(closure["focus_platforms"])
            released = {g for g in all_groups if g in names or
                        g.startswith("platform_") and int(g.removeprefix("platform_")) in platforms}
            parameters = dict(MIPFocus=1 if initial_cost > 0 else 2, Heuristics=.2,
                              RINS=10 if len(all_groups) > 20 else -1, Seed=cfg.seed)
            configuration.update(rule="published events, queue pressure, eligible operation/task dependency closure",
                focus=closure, planning_horizon=cfg.horizon, repair_budget_s=min(cfg.repair_s, cfg.budget_s),
                paired_horizon_fixed=not cfg.adaptive_rule_config, solver_parameters=parameters)
        preparation_s = time.perf_counter()-started
        # A repair shorter than one whole-window transition cannot inspect any
        # new feasible plan. Borrow from later repairs, never the request budget.
        evaluation_floor = 1.5*problem.timings["reference_construction_seconds"]
        try:
            if self.method in ("rh_milp", "rule_guided", "rule_guided_marl", "retrieval_guided", "mlp_guided"):
                fixed = set()
                if self.provider:
                    fixed = {g for i, g in enumerate(problem.groups) if scores[i] >= cfg.confidence}
                    # A confidence score is a heuristic, not a feasibility proof.
                    released = all_groups-fixed
                    if not released:
                        released.add(min(fixed, key=lambda g: scores[list(problem.groups).index(g)]))
                    current = proposed
                elif self.method != "rh_milp":
                    current = proposed
                while time.perf_counter() < deadline:
                    before = time.perf_counter()
                    if released != all_groups and deadline-before < 2*evaluation_floor:
                        # A restricted attempt must leave time to release harmful
                        # fixes; otherwise use guidance only as a search hint.
                        released = set(all_groups)
                        configuration["guidance_release"] = "insufficient_budget_for_restricted_and_full_repair"
                    # Leave time to undo harmful guidance/scope restrictions.
                    local_deadline = deadline if released == all_groups else min(deadline, before+max(cfg.repair_s, evaluation_floor))
                    fixed_anchor = proposed if self.provider else anchor
                    seed_anchor = tuple(anchor[i] if g in released else fixed_anchor[i]
                                        for i, g in enumerate(problem.groups))
                    seeds = repair_seeds(problem, seed_anchor, released, masks, context, graph, self.rng,
                        operator="rule" if self.method.startswith("rule_guided") else "random",
                        proposed=proposed if self.method != "rh_milp" else None)
                    choice, result = repair(problem, seed_anchor, released, local_deadline, cfg,
                        warm=warm if self.method != "rh_milp" else {},
                        priorities=priorities if self.method.startswith("rule_guided") else {}, parameters=parameters,
                        seeds=seeds)
                    trace.append(dict(kind="guided_repair", released=sorted(released), status=result["status"],
                                      cost=result.get("cost"), elapsed_s=time.perf_counter()-before,
                                      bound_scope="this restricted repair only", evaluations=result["evaluations"],
                                      physical_evaluations=result["physical_evaluations"],
                                      shared_evaluations=result["shared_reference_evaluations"],
                                      feasible_evaluations=result["feasible_candidate_evaluations"],
                                      infeasible_reasons=result["infeasible_candidate_reasons"],
                                      seed_trials=result["seed_trials"],
                                      parallel_candidate_evaluations=result.get("parallel_candidate_evaluations", []),
                                      termination=result["termination"]))
                    if choice is not None and result["cost"] < best["cost"]:
                        anchor, best = choice, result
                    if released == all_groups:
                        break
                    remaining = sorted(all_groups-released, key=lambda g: (scores[list(problem.groups).index(g)], g))
                    released.update(remaining[:max(1, len(remaining)//2)])
                    current = anchor
            else:
                for iteration in range(cfg.iterations):
                    if time.perf_counter() >= deadline:
                        break
                    if self.strategy:
                        operator, fraction, portion = "llm_scores", self.strategy["release_fraction"], self.strategy["repair_fraction"]
                        arm = None
                    else:
                        arm = self.selector.choose(self.rng)
                        operator, fraction, portion = self.selector.arms[arm]
                        if self.method == "alns_mip":
                            fraction = cfg.release_fraction
                    released = neighborhood(problem, context, graph, operator, fraction, self.rng, self.strategy)
                    before, old_best = time.perf_counter(), best["cost"]
                    repair_budget = repair_allowance(cfg, deadline-before, portion, evaluation_floor)
                    seeds = repair_seeds(problem, current, released, masks, context, graph, self.rng,
                                         operator=operator)
                    choice, result = repair(problem, current, released, min(deadline, before+repair_budget), cfg,
                                            seeds=seeds)
                    accepted = False
                    if choice is not None:
                        delta = (result["cost"]-current_cost)/max(1., abs(current_cost))
                        temperature = (self.strategy["temperature"] if self.strategy else .03)*(.95**iteration)
                        accepted = choice != current and (delta <= 0 or temperature > 0 and
                            self.rng.random() < math.exp(-min(745., delta/temperature)))
                        if accepted:
                            current, current_cost = choice, result["cost"]
                        if result["cost"] < best["cost"]:
                            anchor, best = choice, result
                    elapsed = time.perf_counter()-before
                    reward = max(0., (old_best-best["cost"])/max(1., abs(old_best)))
                    if arm is not None:
                        self.selector.update(arm, min(1., reward/max(.001, elapsed/repair_budget)) if self.method == "bandit_alns"
                                             else 5. if reward > 0 else 1. if accepted else .1)
                    trace.append(dict(kind="lns", iteration=iteration, operator=operator, released=sorted(released),
                        fraction=fraction, repair_budget_s=repair_budget, accepted=accepted, reward=reward,
                        status=result["status"], cost=result.get("cost"), best_cost=best["cost"],
                        elapsed_s=elapsed, evaluations=result["evaluations"],
                        physical_evaluations=result["physical_evaluations"],
                        shared_evaluations=result["shared_reference_evaluations"],
                        feasible_evaluations=result["feasible_candidate_evaluations"],
                        infeasible_reasons=result["infeasible_candidate_reasons"], seed_trials=result["seed_trials"],
                        parallel_candidate_evaluations=result.get("parallel_candidate_evaluations", []),
                        termination=result["termination"], bound_scope="this neighborhood only"))
        except EvaluationTimeout:
            trace.append(dict(kind="request_budget_exhausted"))
        self.previous = dict(dispatch_packets=[dict(commands=copy.deepcopy(p["commands"]))
                                               for p in best["dispatch_packets"]])
        metadata = dict(method=self.method, initial_cost=initial_cost, cost=best["cost"],
            decision_s=time.perf_counter()-started, budget_s=self.settings.budget_s,
            effective_request_budget_s=cfg.budget_s,
            initial_solution="same checked route/committed-operation continuation before guidance",
            configuration=configuration, search_trace=trace, selector=self.selector.report(),
            provenance=self.provenance, settings=asdict(self.settings), effective_settings=asdict(cfg),
            candidate_catalog={g: [o.id for o in opts] for g, opts in problem.groups.items()},
            global_optimality_proven=False, global_lower_bound=None,
            scope="online finite-catalog rolling coordination under published inputs; not the full-horizon offline baseline")
        metadata["budget_exceeded"] = metadata["decision_s"] > self.settings.budget_s
        novel = len(problem.evaluated)-seed_evaluations
        improved = best["cost"] < initial_cost-1e-6
        novel_values = [v for key, v in problem.evaluated.items() if key not in initial_checked]
        metadata.update(preparation_s=preparation_s, search_s=max(0., metadata["decision_s"]-preparation_s),
            timings_s=dict(problem.timings), novel_candidate_evaluations=novel,
            feasible_novel_evaluations=sum(v["feasible"] for v in novel_values),
            infeasible_candidate_reasons=dict(Counter(v.get("reason", "unspecified")
                for v in novel_values if not v["feasible"])),
            executed_selected_options=best.get("selected_options", {}),
            strictly_improved=improved, seed_only=not improved,
            search_starved=novel == 0 and problem.domain_size > 1,
            request_status="improved" if improved else "seed_only_no_new_evaluation" if novel == 0 else "searched_no_improvement")
        return best, metadata
