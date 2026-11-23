"""Solver-free native dispatch policies; only the common physical oracle is used."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import math
import random
import time

from dispatch_candidates import applicable_options, solution_from_evaluation, stock_start_options
from MILP.operation_model import EvaluationTimeout, JointDispatchModel, nominal_action, planning_copy

PURE_METHODS = ("greedy", "ga", "pso", "rule_based")


class SearchLimit(Exception):
    pass


def operation_aware_settings(world, settings, *, candidates=None, passenger_inputs=None):
    """Use one declared window for all trials, including the reference."""
    if settings.horizon_policy == "fixed" or candidates is not None or passenger_inputs is not None:
        return settings, dict(horizon=settings.horizon, reasons=["fixed_or_caller_declared_window"])
    from MILP.collection_protocol import choose_horizon
    info = choose_horizon(world, base=settings.horizon,
        maximum=max(settings.horizon, settings.max_operation_horizon), tail_stages=1, recovery_work_only=True)
    return replace(settings, horizon=info["horizon"],
                   max_rule_horizon=max(settings.max_rule_horizon, info["horizon"])), info


def evolution_engine():
    try:
        import pymoo
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.algorithms.soo.nonconvex.pso import PSO
        from pymoo.core.problem import Problem
    except ImportError as exc:
        raise ImportError("GA/PSO require pymoo: pip install 'pymoo>=0.6.1,<0.7'") from exc
    return pymoo.__version__, GA, PSO, Problem


class SearchSession:
    """Cache scalar fitness, retaining a full trajectory only for the incumbent."""
    def __init__(self, problem, limit):
        self.problem, self.limit = problem, limit
        self.cache, self.trace = {}, []
        self.best_selection = self.best_value = None
        self.last_selection = self.last_value = None

    def evaluate(self, selection):
        self.problem._check_time()
        key = tuple(int(i) for i in selection)
        if key in self.cache:
            return self.cache[key]
        if self.limit is not None and len(self.cache) >= self.limit:
            raise SearchLimit()
        value = self.problem.evaluate(key)
        return self._record(key, value)

    def _record(self, key, value):
        self.last_selection, self.last_value = key, value
        feasible = bool(value["feasible"])
        record = dict(feasible=feasible, cost=value.get("cost"), reason=value.get("reason"),
                      vehicle=value.get("vehicle"), vehicles=value.get("vehicles", []),
                      details=value.get("details", {}))
        self.cache[key] = record
        improved = feasible and (self.best_value is None or value["cost"] < self.best_value["cost"]-1e-6)
        if improved:
            self.best_selection, self.best_value = key, value
        self.trace.append(dict(evaluation=len(self.cache), feasible=feasible, cost=record["cost"],
                               improved=improved, reason=record["reason"]))
        return record

    def evaluate_many(self, selections):
        requested = [tuple(int(i) for i in selection) for selection in selections]
        keys = list(dict.fromkeys(requested))
        pending = [key for key in keys if key not in self.cache]
        available = None if self.limit is None else max(0, self.limit-len(self.cache))
        limited = available is not None and len(pending) > available
        trials = pending if available is None else pending[:available]
        values = (self.problem.evaluate_many(trials) if hasattr(self.problem, "evaluate_many") else
                  {key: self.problem.evaluate(key) for key in trials})
        for key, value in values.items():
            self._record(key, value)
        if len(values) != len(trials):
            raise EvaluationTimeout()
        if limited:
            raise SearchLimit()
        return [self.cache[key] for key in requested]


def continuation(problem, wait=False):
    result = []
    for options in problem.groups.values():
        default = next((i for i, o in enumerate(options) if not o.commands and not o.actions and not o.pauses), 0)
        result.append(next((i for i, o in enumerate(options) if o.id in ("wait_window", "defer_movement")), default)
                      if wait else default)
    return tuple(result)


def reserve_order(problem):
    cached = getattr(problem, "_rule_reserve_order", None)
    if cached is not None and cached[0] == problem.catalog_fingerprint:
        return cached[1]
    world, distances = problem.base, {}
    masks = applicable_options(problem)[0]
    for group, options in problem.groups.items():
        for option, legal in zip(options, masks[group]):
            if not legal:
                continue
            for command in option.commands:
                if command["action"] not in {"reserve_insert", "reenter"}:
                    continue
                name = command["vehicle_id"]
                v = world.dispatch.records[name]
                if v.role != "reserve" or v.depot is None:
                    continue
                target = command["parameters"].get("exit_node_id", world.dispatch.depots[v.depot]["exit_node_id"])
                path = world.dispatch.available_path(world.dispatch.position(name), target,
                                                     allow_depot=v.depot, vehicle_id=name)
                if path is not None:
                    distances[name] = min(distances.get(name, math.inf), len(path))
    order = {name: i for i, name in enumerate(sorted(distances, key=lambda name: (distances[name], name)))}
    problem._rule_reserve_order = (problem.catalog_fingerprint, order)
    return order


def rule_rank(problem, group, option):
    """Fixed current-state rules, not fitted weights or an objective prediction."""
    world = problem.base
    if not option.commands:
        return 0. if not option.actions and not option.pauses else -1.
    scores = []
    for c in option.commands:
        action, name, p = c["action"], c["vehicle_id"], c["parameters"]
        if name is None:
            platform = p["platform_id"]
            inside = world.waiting_at(platform)
            outside = world.platform_nodes[platform].passenger_flow.outside_count(0)
            capacity = world.event_view().queue_capacity(platform, world.safe_platform_queue_capacity)
            if action == "admission_control":
                desired = max(0., (capacity-inside)/problem.tick)
                if not math.isfinite(desired):
                    desired = world.boarding_rate_persons_per_second
                current = world.dispatch.admission_rates.get(platform, math.inf)
                useful = inside >= .9*capacity or outside > 0 or current < desired
                score = 500.-abs(p["rate"]-desired)/max(1., desired) if useful and p["rate"] != current else -1.
            else:
                target = world.agents[0].headway_tracker.targets.get(platform, world.agents[0].headway_seconds)
                score = 100. if inside > 0 and p["headway_seconds"] < target else -1.
        else:
            vehicle = world.dispatch.records[name]
            pos = world.dispatch.position(name)
            load = float(vehicle.agent.passenger_count[0])
            queue = world.waiting_at(pos) if pos in world.platform_nodes else 0.
            blocked = any((info := world.event_view().get_segment_disruption_info(s.id, name))
                          and info["type"] == "BLOCK" for s in world.base_segments
                          if s.start_platform.id == pos)
            score = -1.
            if action == "rescue_tow":
                score = 1000.
            elif action == "evacuate" and vehicle.failed and load > 0:
                score = 950.
            elif action == "inspect_repair" and vehicle.needs_inspection:
                score = 900.
            elif action == "withdraw" and (vehicle.needs_inspection or world.dispatch.service_impairments(name)):
                score = 850. if not vehicle.failed else -1.
            elif action == "cancel_service" and vehicle.failed and load == 0 and vehicle.service_id:
                score = 800.
            elif action == "reassign_service":
                task = world.dispatch.services[p["service_id"]]
                if vehicle.service_id is None:
                    score = 700.+min(queue/1000., 1.)
                elif (task.get("purpose") == "section_service"
                      and world.dispatch.section_need(p["service_id"], pos)):
                    route = world.dispatch.service_route(p["service_id"], pos)
                    served = sum(world.waiting_at(p) for p in set(route[0])) if route else 0.
                    score = 460.+min(served/1000., 10.)
            elif action in ("reserve_insert", "reenter"):
                if vehicle.role == "reserve":
                    need = world.dispatch.reserve_need()["needed_trains"]
                    rank = reserve_order(problem).get(name, math.inf)
                    score = 600.-rank if rank < need else -1.
                elif action == "reenter" and any(world.waiting_at(n) > 0 for n in world.platform_nodes):
                    score = 300.
            elif action in ("reroute", "short_turn") and blocked:
                score = 450. if action == "reroute" else 400.
            elif action == "skip_stop" and world.event_view().active("platform_closure", "platform", p["target_node"]):
                score = 350.
            elif action == "hold":
                remaining = max(0., float(vehicle.agent.state.dwell_min_limit[0])-float(vehicle.agent.state.dwell_time[0]))
                if blocked or remaining > 0:
                    score = 200.-abs(p["seconds"]-max(remaining, problem.tick))/max(1., problem.tick)
            elif action == "run_profile":
                old = world.dispatch.profile_selections.get(name)
                factor = world.dispatch.profiles[p["profile_id"]]
                if old and old[0] == p["segment_id"] and world.dispatch.profiles[old[1]] > factor:
                    score = 150.-factor
            elif action == "adjust_departure" and not blocked:
                score = 50.
        scores.append(score)
    return min(scores, default=-1.)


def ranked_options(problem, masks):
    return {g: sorted((i for i, legal in enumerate(masks[g]) if legal),
                      key=lambda i: (-rule_rank(problem, g, options[i]), i))
            for g, options in problem.groups.items()}


def filter_rules(session, proposed, anchor):
    """Undo conflicting rule choices; never call a MIP-based repair."""
    problem = session.problem
    groups = list(problem.groups)
    selection, changes = list(proposed), []
    for _ in range(len(groups)+1):
        value = session.evaluate(selection)
        if value["feasible"]:
            return tuple(selection), value, changes
        implicated = {value.get("vehicle")} | set(value.get("vehicles", []))
        commands = set(value.get("details", {}))
        differing = [i for i in range(len(groups)) if selection[i] != anchor[i]]
        if not differing:
            return anchor, session.evaluate(anchor), changes
        affected = []
        for i in differing:
            option = problem.groups[groups[i]][selection[i]]
            participants = {groups[i]} | set(option.actions) | set(option.pauses)
            for c in option.commands:
                participants |= {c["vehicle_id"], c["parameters"].get("failed")}
            if participants & implicated or any(c["command_id"] in commands for c in option.commands):
                affected.append(i)
        for i in affected or differing:
            changes.append(dict(group=groups[i], proposed=selection[i], executed=anchor[i], reason=value["reason"]))
            selection[i] = anchor[i]
    raise RuntimeError("Rule feasibility filtering did not converge")


class PureHeuristicController:
    def __init__(self, method, settings):
        if method not in PURE_METHODS:
            raise ValueError("Unknown solver-free method")
        self.method, self.settings = method, settings
        self.rng = random.Random(settings.seed)
        self.evolution_statistics = {}
        self.provenance = dict(method=method, pretrained=False, solver_backend="none", uses_milp=False,
            physical_oracle="JointDispatchModel.evaluate only; no solve/master/CPLEX calls",
            candidate_domain="same native finite command catalog, including platform controls")
        if method in ("ga", "pso"):
            self.provenance.update(engine="pymoo", engine_version=evolution_engine()[0],
                encoding="random keys decoded to individually applicable categorical options",
                population_policy="configured maximum, budgeted from measured physical reference cost")

    def _fallback(self, world, started, reason, problem=None, session=None, passenger_inputs=None):
        from LLM.dispatch_coordination import checked_reference
        visible = problem.base if problem is not None else planning_copy(world)
        actions = {a.name: nominal_action(visible, a) for a in visible.agents}
        inputs = problem.inputs[:1] if problem is not None else passenger_inputs[:1] if passenger_inputs is not None else None
        solution = checked_reference(world, actions, 1, inputs)
        elapsed = time.perf_counter()-started
        return solution, dict(method=self.method, fallback=True, request_status=reason,
            decision_s=elapsed, budget_s=self.settings.budget_s, budget_exceeded=elapsed>self.settings.budget_s,
            provenance=self.provenance, solver_backend="none", uses_milp=False,
            evolution=dict(self.evolution_statistics),
            settings=asdict(self.settings), global_optimality_proven=False,
            search_trace=session.trace if session else [], evaluations=len(session.cache) if session else 0)

    def decide(self, world, *, candidates=None, passenger_inputs=None):
        started, cfg = time.perf_counter(), self.settings
        self.evolution_statistics = {}
        cfg, horizon_policy = operation_aware_settings(world, cfg, candidates=candidates, passenger_inputs=passenger_inputs)
        problem = session = None
        initial_checked = set()
        reason, corrections, selected, selected_value = "iterations", [], None, None
        try:
            problem = JointDispatchModel(world, cfg.horizon, candidates=candidates, passenger_inputs=passenger_inputs,
                start_stages=cfg.command_stages, deadline=started+cfg.budget_s, defer_reference=True)
            session = SearchSession(problem, cfg.max_evaluations)
            anchor = continuation(problem)
            value = session.evaluate(anchor)
            if not value["feasible"]:
                anchor = continuation(problem, wait=True)
                value = session.evaluate(anchor)
            if not value["feasible"]:
                return self._fallback(world, started, "no_checked_seed", problem, session)
            initial_cost = value["cost"]
            initial_checked = set(session.cache)
            masks, _ = applicable_options(problem)
            ranked = ranked_options(problem, masks)
            proposed = tuple(indices[0] if indices else anchor[i] for i, indices in enumerate(ranked.values()))
            if self.method == "rule_based":
                selected, selected_value, corrections = filter_rules(session, proposed, anchor)
                if selected == session.best_selection:
                    selected_value = session.best_value
                elif selected == session.last_selection:
                    selected_value = session.last_value
                elif selected_value["feasible"]:
                    selected_value = problem.evaluate(selected)
                reason = "fixed_rules_and_feasibility_filter"
            elif self.method == "greedy":
                self._greedy(session, ranked, cfg.iterations)
            else:
                self._evolution(session, ranked, proposed)
        except (EvaluationTimeout, SearchLimit) as exc:
            reason = "time_limit" if isinstance(exc, EvaluationTimeout) else "evaluation_limit"
        if session is None or session.best_value is None:
            return self._fallback(world, started, reason, problem, session, passenger_inputs)
        if selected_value is None or not selected_value.get("packets"):
            selected, selected_value = session.best_selection, session.best_value
        solution = solution_from_evaluation(problem, selected_value, selected, status="checked_"+self.method)
        solution["corrections"] = list(solution.get("corrections", [])) + corrections
        elapsed = time.perf_counter()-started
        novel = [v for key, v in session.cache.items() if key not in initial_checked]
        improved = solution["cost"] < initial_cost-1e-6
        return solution, dict(method=self.method, initial_cost=initial_cost, cost=solution["cost"],
            horizon_policy=horizon_policy, effective_settings=asdict(cfg),
            decision_s=elapsed, budget_s=cfg.budget_s, budget_exceeded=elapsed>cfg.budget_s,
            termination=reason, evaluations=len(session.cache), provenance=self.provenance,
            solver_backend="none", uses_milp=False, settings=asdict(cfg), global_optimality_proven=False,
            search_trace=session.trace, proposal_corrections=corrections,
            evolution=dict(self.evolution_statistics),
            novel_candidate_evaluations=len(novel),
            feasible_novel_evaluations=sum(v["feasible"] for v in novel),
            infeasible_candidate_reasons=dict(Counter(v.get("reason", "unspecified") for v in novel if not v["feasible"])),
            strictly_improved=improved, seed_only=selected == anchor,
            search_starved=not novel and problem.domain_size > 1,
            request_status=("fixed_rules" if self.method == "rule_based" else
                            "improved" if improved else "searched_no_improvement" if novel else "seed_only_no_new_evaluation"),
            timings_s=dict(problem.timings), continuation_mode=problem.continuation_mode,
            selected_families=dict(Counter(c["action"] for p in solution["dispatch_packets"] for c in p["commands"])),
            candidate_catalog={g:[o.id for o in opts] for g,opts in problem.groups.items()},
            parallel_candidate_evaluations=list(problem.parallel_evaluation_stats),
            scope="full native fleet and platform catalog; shared published-input physical evaluation")

    @staticmethod
    def _greedy(session, ranked, iterations):
        from MILP.parallel_evaluation import worker_count
        for _ in range(iterations):
            previous = session.best_value["cost"]
            # Visit every group before deeper alternatives, avoiding first-group starvation.
            for rank in range(max(map(len, ranked.values()), default=0)):
                batch = []
                for gi, indices in enumerate(ranked.values()):
                    if rank >= len(indices):
                        continue
                    choice = list(session.best_selection)
                    choice[gi] = indices[rank]
                    if worker_count() == 1:
                        session.evaluate(choice)
                    else:
                        batch.append(choice)
                if batch:
                    session.evaluate_many(batch)
            if session.best_value["cost"] >= previous-1e-6:
                break

    def _evolution(self, session, ranked, proposed):
        import numpy as np
        problem = session.problem
        _, GA, PSO, Problem = evolution_engine()
        options = [indices or [session.best_selection[i]] for i,indices in enumerate(ranked.values())]
        dimensions = [i for i,v in enumerate(options) if len(v)>1]
        if not dimensions:
            return
        anchor, cfg = session.best_selection, self.settings

        def encode(selection):
            return np.array([(options[i].index(selection[i])+.5)/len(options[i]) for i in dimensions])

        def decode(vector):
            choice = list(anchor)
            for i,x in zip(dimensions,vector):
                choice[i] = options[i][min(len(options[i])-1, max(0, int(float(x)*len(options[i]))))]
            return tuple(choice)

        class NativeProblem(Problem):
            def __init__(self):
                super().__init__(n_var=len(dimensions), n_obj=1, n_ieq_constr=1, xl=0., xu=1.)

            def _evaluate(self, vectors, out, *args, **kwargs):
                values = session.evaluate_many([decode(v) for v in vectors])
                out["F"] = np.array([[v["cost"] if v["feasible"] else 0.] for v in values])
                out["G"] = np.array([[0. if v["feasible"] else 1.] for v in values])

        # A dense first population changed nearly every train/platform at once;
        # a request expired before finding even a second feasible individual.
        # Sparse initialization changes no domain or later mutation/crossover.
        stock_moves = stock_start_options(problem, applicable_options(problem)[0])
        moves = [(i, choice) for rank in range(max(map(len, options))) for i in dimensions
                 if rank < len(options[i]) and (choice := options[i][rank]) != anchor[i]]
        moves = list(dict.fromkeys([*(move for move in stock_moves if move[0] in dimensions
                                     and move[1] != anchor[move[0]]), *moves]))
        reference_s = problem.timings["reference_construction_seconds"]
        evaluation_estimate = max(.001, reference_s*(2 if problem.reviewed_continuation else 1))
        remaining = max(0., problem.deadline-time.perf_counter())
        from MILP.parallel_evaluation import worker_count
        population_size = (cfg.population_size if worker_count() != 1 or not math.isfinite(remaining) else
                           max(2, min(cfg.population_size, int(remaining/(2*evaluation_estimate)))))
        population, seen = [encode(anchor)], {tuple(anchor)}
        for index in range(1, population_size*16+1):
            if len(population) >= population_size:
                break
            selection = list(anchor)
            gi, oi = moves[(index-1) % len(moves)]
            selection[gi] = oi
            if index % 3 == 0:
                other, alternative = moves[self.rng.randrange(len(moves))]
                if other != gi:
                    selection[other] = alternative
            key = tuple(selection)
            if key not in seen:
                population.append(encode(selection))
                seen.add(key)
        if len(population) < 2:
            return
        sampling = np.array(population)
        self.evolution_statistics = dict(initial_population_size=len(population), completed_generations=0,
            configured_population_size=cfg.population_size, estimated_evaluation_s=evaluation_estimate,
            initialization="distinct_sparse_decoded_choices")
        algorithm = (GA(pop_size=len(population), sampling=sampling, eliminate_duplicates=True)
                     if self.method == "ga" else
                     PSO(pop_size=len(population), sampling=sampling, pertube_best=False))
        algorithm.setup(NativeProblem(), termination=("n_gen",cfg.iterations), seed=self.rng.randrange(2**31), verbose=False)
        while algorithm.has_next():
            session.problem._check_time()
            population = algorithm.ask()
            if population is None or len(population)==0:
                break
            algorithm.evaluator.eval(algorithm.problem, population)
            algorithm.tell(infills=population)
            self.evolution_statistics["completed_generations"] += 1
