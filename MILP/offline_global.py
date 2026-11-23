"""Certified, full-stage offline optimization over the original simulator.

A dynamically expanded state-network MILP retains an optimistic suffix arc for
EVERY unexplored subtree. Feasible complete trajectories give upper bounds;
the network relaxation gives lower bounds. No initial command catalog, fixed
MARL continuation, or truncated rolling window defines the search domain.

This is an exact but potentially expensive implicit formulation, not a compact
linearization of all simulator rules. Optimality is relative to OfflineDomain's
explicit finite operational grids, not arbitrary continuous command parameters.
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
import math
import time

from MILP import cplex_backend as mip

from MILP.offline_domain import OfflineDomain, OfflineLimit, StageControl, check_deadline
from MILP.operation_model import EvaluationTimeout, nominal_action, passenger_total, positions, set_actions
from dispatch_plan import DispatchCommand
from dispatch_timing import ArrivalWindow, ContinuationMismatch, compile_resource_waits, decision_seconds


class OfflinePrimalUnavailable(RuntimeError):
    """A heuristic failed; this says nothing about the full decision domain."""


def synchronize_physics(world):
    world.disruption_manager.update_disruptions(float(world.sim_time[0]))
    world.dispatch._faults()


def freeze_arrivals(world, stages):
    """Exogenous deterministic arrivals from this benchmark's actual event path."""
    manager = copy.deepcopy(world.disruption_manager)
    result = []
    for stage in range(stages):
        profile = []
        for second in range(int(decision_seconds(world) / float(world.tick_seconds))):
            manager.update_disruptions(stage*decision_seconds(world) + second*float(world.tick_seconds))
            profile.append({p: n.passenger_flow.interval_arrivals(manager.physical_view(), float(world.tick_seconds))
                            for p, n in world.platform_nodes.items()})
        result.append(ArrivalWindow(profile))
    return result


class NativeOfflineProblem:
    def __init__(self, world, domain, passenger_inputs=None):
        if world.dispatch is None or world.num_envs != 1 or float(world.sim_time[0]) != 0:
            raise ValueError("Offline benchmark requires one native world starting at 06:00")
        if not math.isclose(decision_seconds(world), domain.tick):
            raise ValueError("World and offline domain use different decision intervals")
        if float(world.tick_seconds) != 1:
            raise ValueError("The native offline benchmark requires one-second physical execution")
        self.root = copy.deepcopy(world)  # Intentionally NOT planning_copy().
        synchronize_physics(self.root)
        self.domain, self.horizon = domain, domain.stages
        self.gamma = world.dispatch.objective.weights["discount"]
        self.inputs = copy.deepcopy(passenger_inputs) if passenger_inputs is not None else freeze_arrivals(world, self.horizon)
        if len(self.inputs) != self.horizon:
            raise ValueError("Offline arrivals must specify the complete episode")
        if any(not math.isfinite(float(n)) or n < 0 for interval in self.inputs
               for values in interval.values() for n in values.values()):
            raise ValueError("Offline arrivals must be finite and nonnegative")
        self.arrivals = [math.fsum(float(n) for values in interval.values() for n in values.values())
                        for interval in self.inputs]
        self.physical_tick = float(world.tick_seconds)
        count = int(round(domain.tick / self.physical_tick))
        self.physical_arrivals = []
        for interval, total in zip(self.inputs, self.arrivals):
            profile = getattr(interval, "profile", None)
            self.physical_arrivals.append(
                [math.fsum(float(n) for values in second.values() for n in values.values()) for second in profile]
                if profile is not None else [0.] * (count - 1) + [total])
        if any(not math.isfinite(n) or n < 0 for n in self.arrivals):
            raise ValueError("Offline arrivals must be finite and nonnegative")
        self.boarding_ceiling = math.fsum(float(a.passenger_capacity) for a in world.agents)
        self.objective = world.dispatch.objective.manifest()
        self._preferred = None
        self.primal_seed_trials = []

    def controls(self, world, stage, deadline):
        # This is an ordering hint, not a cut: the full declared domain follows.
        self._preferred = None
        try:
            preferred, value = self.checked_nominal(world, stage, deadline)
        except OfflinePrimalUnavailable:
            preferred = None
        else:
            self._preferred = (world, stage, preferred, value)
            yield preferred
            del value
        for control in self.domain.joint_controls(world, stage, deadline=deadline):
            if control != preferred:
                yield control

    def suffix_lower_bound(self, world, stage):
        """Relaxed aggregate queue: unlimited access, instantaneous alighting.

        Arrivals are charged before boarding in world.step. At most the total
        physical fleet capacity can board per tick. Ignore paths, occupancy,
        onboard passengers, service times and operation costs to stay optimistic.
        Evacuation adds queues; no action can erase an unfinished journey.
        """
        counts = world.dispatch.objective.passenger_state(world)
        queue = counts["inside"]+counts["outside"]
        weights = world.dispatch.objective.weights
        bound = 0.
        for k in range(stage, self.horizon):
            for entering in self.physical_arrivals[k]:
                queue += entering
                bound += self.gamma**k * weights["waiting"] * self.physical_tick * queue
                queue = max(0., queue-self.boarding_ceiling)
        bound += self.gamma**self.horizon * weights["waiting"] * weights["terminal_passenger_seconds"] * queue
        return max(0., bound*(1.-1e-12)-1e-6)

    def checked_nominal(self, world, stage, deadline, *, reverse=False, commands=()):
        """Repair conflicts by trying alternate successors and single deferrals.

        This is a primal heuristic only. Active multi-vehicle operations defer
        as a unit; every returned control passes the original transition audit.
        """
        nominal = {a.name: nominal_action(world, a) for a in world.agents}
        pending, seen = deque([(nominal, frozenset())]), set()
        all_names = frozenset(nominal)
        attempts = 0
        while pending:
            check_deadline(deadline)
            if attempts < 4*len(nominal)+4:
                actions, paused = pending.popleft()
            else:
                actions = {name: 0 for name in nominal}
                paused = all_names
                pending.clear()
            signature = (tuple(sorted(actions.items())), tuple(sorted(paused)))
            if signature in seen:
                continue
            seen.add(signature)
            attempts += 1
            control = StageControl({name: 0 if name in paused else action for name, action in actions.items()},
                                   copy.deepcopy(list(commands)), sorted(paused))
            value = self.transition(world, control, stage, deadline)
            if value["feasible"]:
                return control, value
            if not value.get("conflicts"):
                raise OfflinePrimalUnavailable("Offline primal reference has no checked continuation")
            bundles = []
            for name in sorted(value["conflicts"], reverse=reverse):
                operation = value["trial"].dispatch.operation_for(name)
                bundles.append(frozenset(operation.participants() if operation else {name}))
                if operation is None and name not in paused:
                    agent = world.dispatch.records[name].agent
                    current = world.dispatch.position(name)
                    successors = sorted((world.nodes[current].getnext() or {}).items(), key=lambda item: str(item[0]))
                    for label, (_, target) in enumerate(successors, 1):
                        segment = world._step_segment.get((current, target))
                        if label == actions[name] or not world.edge_event_allowed(agent, current, target, view=world.event_view()):
                            continue
                        if any(segment and segment.end_platform.id == depot["node_id"] for depot in world.dispatch.depots.values()):
                            continue
                        alternative = dict(actions, **{name: label})
                        pending.append((alternative, paused))
            for bundle in [*bundles, frozenset().union(*bundles)]:
                candidate = paused | bundle
                deferred = {name: 0 if name in candidate else action for name, action in actions.items()}
                pending.append((deferred, candidate))
            if paused == all_names:
                break
        raise OfflinePrimalUnavailable("Offline primal deferral failed to resolve a conflict")

    def transition(self, world, control, stage, deadline=math.inf, *, execution_records=None):
        check_deadline(deadline)
        if self._preferred is not None:
            cached_world, cached_stage, cached_control, value = self._preferred
            self._preferred = None
            if execution_records is None and world is cached_world and stage == cached_stage and control == cached_control:
                return value
        proposal = self._transition_once(world, control, stage, deadline)
        if not proposal["feasible"] and not proposal.get("conflicts"):
            return proposal
        if proposal.get("conflicts"):
            trial = proposal["trial"]
            pairs = [(op.participants(), op.command_id, op.started_at)
                     for op in world.dispatch.operations.values() if op.action == "rescue_tow"]
            pairs += [(c.participants(), c.command_id, math.inf)
                      for plan in trial.dispatch.plans.values() for c in plan.commands if c.action == "rescue_tow"]
            targets, waits = compile_resource_waits(trial, pairs, stage=stage,
                                                    check_time=lambda: check_deadline(deadline))
        else:
            if execution_records is None:
                return proposal
            targets, waits = proposal["state"].last_physical_requests, []
        # Only this clean run may enter the search graph or result recorder.
        value = self._transition_once(world, control, stage, deadline, targets=targets,
                                      execution_records=execution_records)
        if value["feasible"]:
            value["planned_resource_waits"] = waits
        return value

    def _transition_once(self, world, control, stage, deadline, *, targets=None, execution_records=None):
        check_deadline(deadline)
        trial = copy.deepcopy(world)
        trial.execution_records = execution_records
        trial._planning_deadline = deadline
        trial._strict_dispatch_execution = targets is not None
        synchronize_physics(trial)
        before = passenger_total(trial) + sum(float(a.cumulative_alighted[0]) for a in trial.agents)
        commands = []
        for i, raw in enumerate(control.commands):
            command = copy.deepcopy(raw)
            offset = command.pop("offset_seconds", 0)
            if type(offset) is not int or not 0 <= offset < self.domain.tick:
                raise ValueError("Offline command offset outside the decision interval")
            command.update(command_id=f"offline_{stage}_{i}", effective_at=stage*self.domain.tick+offset)
            commands.append(command)
        if commands:
            result = trial.submit_dispatch_plan(dict(plan_id=f"offline_stage_{stage}", snapshot_basis="physical",
                snapshot_version=trial.dispatch.snapshot(published=False)["version"],
                created_at=stage*self.domain.tick, commands=commands))
            if not result["accepted"]:
                return dict(feasible=False, reason=result["reason"], conflicts=[])
        trial.dispatch._activate()
        held = set(control.pauses) | {name for name,action in control.actions.items()
                                     if action == 0 and trial.dispatch.operation_for(name) is None}
        if any(not trial.dispatch.discretionary_hold_allowed(name) for name in held):
            return dict(feasible=False,reason="discretionary_hold_requires_safe_platform",conflicts=[])
        set_actions(trial, control.actions)
        if targets is None:
            targets = {name: trial.dispatch.position(name) for name in control.pauses}
        # Unexpected execution exceptions are NOT infeasibility certificates.
        try:
            trial.step(arrival_inputs=self.inputs[stage], dispatch_targets=targets)
        except EvaluationTimeout:
            raise OfflineLimit("time_limit") from None
        except ContinuationMismatch as exc:
            return dict(feasible=False, reason="compiled_target_not_executable", vehicle=exc.vehicle, conflicts=[])
        rejected = [trial.dispatch.command_results[c["command_id"]] for c in commands
                    if trial.dispatch.command_results[c["command_id"]]["status"] in ("rejected", "failed")]
        conflicts = [name for name, row in trial.last_transition_audit.items() if row["conflict"]]
        if rejected or conflicts or trial.last_execution_deviations:
            return dict(feasible=False, reason="command_precondition" if rejected else
                        "shared_resource_conflict" if conflicts else "execution_deviation",
                        conflicts=conflicts, trial=trial)
        arrived = sum(sum(values.values()) for values in self.inputs[stage].values())
        after = passenger_total(trial) + sum(float(a.cumulative_alighted[0]) for a in trial.agents)
        if not math.isclose(after, before+arrived, abs_tol=1e-3, rel_tol=1e-5):
            raise AssertionError("Offline transition violated passenger conservation")
        cost = float(trial.dispatch.interval_cost["total"])*self.gamma**stage
        if not math.isfinite(cost) or cost < 0:
            raise AssertionError("Nonnegative finite costs are required for the global lower bound")
        synchronize_physics(trial)
        return dict(feasible=True, state=trial, cost=cost,
                    passenger_balance_error=after-before-arrived)

    def terminal_cost(self, world):
        return self.gamma**self.horizon * world.dispatch.objective.terminal(world)["total"]

    def greedy(self, deadline, *, reverse=False, startup_commands=(), progress=None, seed_name="routing"):
        """Primal seed only; all changes are recorded as benchmark decisions."""
        world, cost, controls = copy.deepcopy(self.root), 0., []
        last_report = time.perf_counter()
        for stage in range(self.horizon):
            check_deadline(deadline)
            control, value = self.checked_nominal(world, stage, deadline, reverse=reverse,
                                                  commands=startup_commands if stage == 0 else ())
            world = value["state"]
            cost += value["cost"]
            controls.append(control)
            if progress and time.perf_counter()-last_report >= 10.:
                progress("primal_seed_progress", seed=seed_name, seed_stage=stage+1,
                         seed_prefix_cost=cost)
                last_report = time.perf_counter()
        return cost+self.terminal_cost(world), controls

    def primal_seeds(self, deadline, progress=None):
        started = time.perf_counter()
        value = self.greedy(deadline, progress=progress)
        self.primal_seed_trials.append(dict(name="routing", status="complete", cost=value[0],
                                           elapsed_s=time.perf_counter()-started))
        yield value
        # A second primal proposal explores dispatch parameters and stock use.
        # These commands belong to the declared domain; they do not fix search.
        minimum = min(self.domain.headway_values(self.root))
        commands = [dict(action="adjust_departure", vehicle_id=None,
                         parameters=dict(platform_id=p, headway_seconds=minimum))
                    for p in sorted(self.root.platform_nodes)]
        commands.extend(dict(action="reserve_insert", vehicle_id=v.id, parameters={})
                        for v in self.root.dispatch.records.values()
                        if v.depot is not None and v.role == "reserve" and not v.has_served
                        and self.root.dispatch.reserve_need()["service_need_detected"]
                        and not v.failed and not v.needs_inspection and v.status == "DEPOT_READY")
        started = time.perf_counter()
        seed_deadline = started+.25*max(0., deadline-started)
        try:
            value = self.greedy(seed_deadline, startup_commands=commands, reverse=True, progress=progress,
                                seed_name="headways_and_reserves")
        except OfflineLimit:
            self.primal_seed_trials.append(dict(name="headways_and_reserves", status="seed_budget",
                                               elapsed_s=time.perf_counter()-started))
            check_deadline(deadline)
            if progress:
                progress("primal_seed_limit", seed="headways_and_reserves")
        except OfflinePrimalUnavailable as exc:
            self.primal_seed_trials.append(dict(name="headways_and_reserves", status="no_checked_seed",
                reason=str(exc), elapsed_s=time.perf_counter()-started))
            if progress:
                progress("primal_seed_rejected", seed="headways_and_reserves", reason=str(exc))
        else:
            self.primal_seed_trials.append(dict(name="headways_and_reserves", status="complete", cost=value[0],
                                               elapsed_s=time.perf_counter()-started))
            yield value

    def replay(self, controls, on_stage=None, *, deadline=math.inf, record_directory=None, result_writer=None):
        if len(controls) != self.horizon:
            raise ValueError("A baseline plan must contain the complete episode")
        world, total, maximum_error = copy.deepcopy(self.root), 0., 0.
        waiting = onboard = 0.
        from dispatch_records import enable_execution_records
        records = enable_execution_records(world) if record_directory is not None else None
        if result_writer is not None:
            result_writer.attach(world)
            records = world.execution_records
        event_families, active_events = set(), set()
        for stage, control in enumerate(controls):
            for event in world.event_view().active():
                event_families.add(event["type"])
                active_events.add(event["event_id"])
            value = self.transition(world, control, stage, deadline, execution_records=records)
            if not value["feasible"]:
                raise AssertionError(f"Offline plan replay failed at stage {stage}: {value['reason']}")
            world = value["state"]
            total += value["cost"]
            waiting += world.dispatch.interval_cost["waiting_person_seconds"]
            onboard += world.dispatch.interval_cost["onboard_person_seconds"]
            maximum_error = max(maximum_error, abs(value["passenger_balance_error"]))
            if on_stage:
                on_stage(stage, world, control)
        if records is not None:
            records.write(record_directory, world)
        return dict(cost=total+self.terminal_cost(world),
            waiting_person_seconds=waiting, onboard_person_seconds=onboard,
            passenger_person_seconds=waiting+onboard,
            terminal=world.dispatch.objective.terminal(world),
            passenger_balance_error_max=maximum_error,
            completed_passengers=sum(float(a.cumulative_alighted[0]) for a in world.agents),
            end_seconds=float(world.sim_time[0]), hard_constraint_interventions=0,
            observed_physical_event_families=sorted(event_families),
            observed_physical_event_ids=sorted(active_events),
            verified_stages=self.horizon)


@dataclass
class SearchNode:
    id: int
    stage: int
    prefix_cost: float
    world: object
    parent: int | None = None
    control: StageControl | None = None
    iterator: object = None
    frontier: object = None
    balance: object = None
    suffix_bound: float = 0.
    attempts: int = 0


class OfflineGlobalSolver:
    """Exact state expansion with a valid MILP frontier relaxation.

    Removing a frontier requires exhaustion or a nonnegative-cost bound proving
    that its descendants cannot improve the incumbent. Limits and execution
    errors leave the frontier intact. Preferred chains find complete plans;
    rotation revisits earlier frontiers instead of exhausting the last stage.
    """
    def __init__(self, problem, *, time_limit=3600., max_nodes=None,
                 max_transitions=None, threads=0, absolute_gap=1e-6, relative_gap=0., progress=None,
                 on_incumbent=None, initial_controls=None, log_path=None):
        if not math.isfinite(time_limit) or time_limit < 0 or type(threads) is not int or threads < 0:
            raise ValueError("Finite nonnegative time budget and nonnegative integer threads required")
        if any(v is not None and v < 1 for v in (max_nodes, max_transitions)):
            raise ValueError("Search limits must be positive or omitted")
        if not math.isfinite(absolute_gap) or not math.isfinite(relative_gap) or min(absolute_gap, relative_gap) < 0:
            raise ValueError("Finite nonnegative optimality tolerances required")
        self.problem, self.time_limit, self.max_nodes = problem, time_limit, max_nodes
        self.max_transitions, self.threads = max_transitions, threads
        self.absolute_gap, self.relative_gap, self.progress = absolute_gap, relative_gap, progress
        self.best_cost, self.best_controls, self.lower = math.inf, None, 0.
        self.nodes, self.transitions, self.rejections = [], 0, 0
        self.errors = []
        self.on_incumbent, self.initial_controls, self.log_path = on_incumbent, initial_controls, log_path
        self.deepest_stage, self.incumbents = 0, 0
        self.frontier_rotations = 0

    def suffix_bound(self, world, stage):
        method = getattr(self.problem, "suffix_lower_bound", None)
        value = float(method(world, stage)) if method else 0.
        if not math.isfinite(value) or value < 0:
            raise AssertionError("Suffix bound must be finite and nonnegative")
        return value

    def emit(self, event, **details):
        if self.progress:
            self.progress(dict(event=event, elapsed_s=time.perf_counter()-self.started,
                explored_states=len(self.nodes), transitions=self.transitions,
                deepest_stage=self.deepest_stage, incumbents=self.incumbents,
                frontier_rotations=self.frontier_rotations,
                lower_bound=self.lower, upper_bound=self.best_cost if self.best_controls is not None else None,
                **details))

    def edge(self, origin, target, cost, name):
        return self.model.addVar(vtype=mip.BINARY, obj=cost, name=name,
            column=mip.Column([1., -1.], [origin, target]))

    def add_node(self, world, stage, prefix, parent=None, control=None, edge_cost=0.):
        if self.max_nodes is not None and len(self.nodes) >= self.max_nodes:
            raise OfflineLimit("state_limit")
        ident = len(self.nodes)
        balance = self.source if parent is None else self.model.addConstr(self.model.linear_expr() == 0., name=f"flow_{ident}")
        suffix = self.suffix_bound(world, stage)
        node = SearchNode(ident, stage, prefix, world, parent, control, balance=balance, suffix_bound=suffix)
        if parent is not None:
            self.edge(self.nodes[parent].balance, balance, edge_cost, f"transition_{ident}")
        node.frontier = self.edge(balance, self.sink, suffix, f"unexplored_{ident}")
        self.nodes.append(node)
        self.deepest_stage = max(self.deepest_stage, stage)
        return node

    def close(self, node):
        self.model.remove(node.frontier)
        node.frontier = node.iterator = node.world = None

    def incumbent(self, cost, controls):
        if not math.isfinite(cost) or cost < 0:
            raise AssertionError("Invalid full-horizon incumbent cost")
        if cost < self.best_cost:
            self.best_cost, self.best_controls = float(cost), list(controls)
            self.incumbent_edge.ub = 1.
            self.model.set_variable_objective(self.incumbent_edge, self.best_cost)
            self.incumbents += 1
            if self.on_incumbent:
                self.on_incumbent(self.best_cost, self.best_controls,
                                  dict(elapsed_s=time.perf_counter()-self.started, number=self.incumbents))
            self.emit("incumbent")

    def path(self, node):
        result = []
        while node.parent is not None:
            result.append(node.control)
            node = self.nodes[node.parent]
        return list(reversed(result))

    def bound(self):
        self.model.update()
        remaining = self.deadline-time.perf_counter()
        if remaining <= 0:
            return
        self.model.Params.TimeLimit = remaining
        self.model.optimize()
        if self.model.Status == mip.INFEASIBLE:
            if self.best_controls is not None:
                raise AssertionError("Frontier model excluded a verified incumbent")
            return
        if self.model.Status not in (mip.OPTIMAL, mip.TIME_LIMIT, mip.INTERRUPTED, mip.SUBOPTIMAL):
            raise RuntimeError(f"Frontier MILP status {self.model.Status}")
        value = float(self.model.ObjBound)
        if math.isfinite(value):
            self.lower = max(self.lower, max(0., value))
        if self.best_controls is not None and self.lower > self.best_cost + self.absolute_gap:
            raise AssertionError("Offline lower bound exceeded its upper bound")

    def gap_closed(self):
        if self.best_controls is None:
            return False
        tolerance = max(self.absolute_gap, self.relative_gap*abs(self.best_cost))
        return self.best_cost-self.lower <= tolerance

    def solve(self, *, model_path=None):
        self.started = time.perf_counter()
        self.last_progress = self.started
        self.deadline = self.started+self.time_limit
        self.model = mip.Model("offline_full_horizon_state_network")
        self.model.Params.OutputFlag = int(self.log_path is not None)
        if self.log_path is not None:
            self.model.Params.LogToConsole = 0
            self.model.Params.LogFile = str(self.log_path)
        self.model.Params.Threads = self.threads
        self.model.Params.MIPGap = 0.
        self.model.Params.MIPGapAbs = 0.
        self.source = self.model.addConstr(self.model.linear_expr() == 1., name="source")
        self.sink = self.model.addConstr(self.model.linear_expr() == -1., name="sink")
        self.incumbent_edge = self.edge(self.source, self.sink, 0., "verified_complete_trajectory")
        self.incumbent_edge.ub = 0.
        root = self.add_node(copy.deepcopy(self.problem.root), 0, 0.)
        self.lower = root.suffix_bound
        stack, termination = [root], "search_exhausted"
        try:
            try:
                if self.initial_controls is not None:
                    verification = self.problem.replay(self.initial_controls, deadline=self.deadline)
                    self.incumbent(verification["cost"], self.initial_controls)
                seeds = (self.problem.primal_seeds(self.deadline, progress=self.emit) if hasattr(self.problem, "primal_seeds")
                         else iter([self.problem.greedy(self.deadline)]))
                for cost, controls in seeds:
                    self.incumbent(cost, controls)
            except OfflineLimit:
                raise
            except Exception as exc:
                # A failed seed does not justify cutting any search subtree.
                self.errors.append(dict(stage="primal_seed", error=repr(exc)))
            self.bound()
            while stack and not self.gap_closed():
                check_deadline(self.deadline)
                if self.max_transitions is not None and self.transitions >= self.max_transitions:
                    raise OfflineLimit("transition_limit")
                node = stack[-1]
                if self.best_controls is not None and node.prefix_cost+node.suffix_bound >= self.best_cost:
                    self.close(node)
                    stack.pop()
                    continue
                if node.iterator is None:
                    node.iterator = iter(self.problem.controls(node.world, node.stage, self.deadline))
                try:
                    control = next(node.iterator)
                except StopIteration:
                    self.close(node)
                    stack.pop()
                    if not stack:
                        break
                    continue
                self.transitions += 1
                node.attempts += 1
                value = self.problem.transition(node.world, control, node.stage, self.deadline)
                if self.transitions % 100 == 0 or time.perf_counter()-self.last_progress >= 10.:
                    self.bound()
                    self.emit("search")
                    self.last_progress = time.perf_counter()
                if not value["feasible"]:
                    self.rejections += 1
                    if node.attempts % 32 == 0 and len(stack) > 1:
                        stack.append(stack.pop(0))
                        self.frontier_rotations += 1
                    continue
                prefix = node.prefix_cost+value["cost"]
                if node.stage+1 == self.problem.horizon:
                    self.incumbent(prefix+self.problem.terminal_cost(value["state"]), self.path(node)+[control])
                    if len(stack) > 1:
                        stack.append(stack.pop(0))
                        self.frontier_rotations += 1
                elif prefix < self.best_cost:
                    child = self.add_node(value["state"], node.stage+1, prefix, node.id, control, value["cost"])
                    stack.append(child)
                elif node.attempts % 32 == 0 and len(stack) > 1:
                    stack.append(stack.pop(0))
                    self.frontier_rotations += 1
            if self.gap_closed():
                termination = "optimality_gap"
            elif not stack:
                # Only exhaustion/pruning can empty this stack.
                if self.best_controls is not None:
                    self.lower = self.best_cost
                termination = "search_exhausted"
        except OfflineLimit as exc:
            termination = str(exc)
        except KeyboardInterrupt:
            termination = "interrupted"
        except Exception as exc:
            termination = "model_error"
            self.errors.append(dict(stage="exact_search", error=repr(exc)))
        self.model.update()
        if model_path:
            self.model.write(str(model_path))
        valid_certificate = termination != "model_error"
        optimal = valid_certificate and self.gap_closed()
        infeasible = valid_certificate and termination == "search_exhausted" and self.best_controls is None
        status = "model_error" if not valid_certificate else "optimal_discrete_model" if optimal else "infeasible_discrete_model" if infeasible else (
            "feasible_limit" if self.best_controls is not None else "no_incumbent_limit")
        upper = self.best_cost if self.best_controls is not None else None
        lower = min(self.lower, upper) if upper is not None else self.lower
        gap = upper-lower if upper is not None else None
        result = dict(status=status, termination=termination, optimality_proven=optimal,
            objective_direction="minimize", lower_bound=lower,
            upper_bound=upper, absolute_gap=gap,
            relative_gap=gap/max(abs(upper), 1e-12) if upper is not None else None,
            absolute_gap_tolerance=self.absolute_gap, relative_gap_tolerance=self.relative_gap,
            elapsed_search_s=time.perf_counter()-self.started, time_limit_s=self.time_limit,
            state_limit=self.max_nodes, transition_limit=self.max_transitions,
            solver_backend="cplex", solver_threads_requested=self.threads,
            cplex_status=self.model.cplex_status,
            solver_version=self.model.native_model.get_cplex().get_version(),
            # Deprecated alias retained for existing result consumers.
            gurobi_threads_requested=self.threads,
            explored_states=len(self.nodes), tested_transitions=self.transitions,
            rejected_transitions=self.rejections, remaining_frontiers=sum(n.frontier is not None for n in self.nodes),
            deepest_stage=self.deepest_stage, incumbent_count=self.incumbents,
            frontier_rotations=self.frontier_rotations,
            errors=self.errors, domain=self.problem.domain.manifest(),
            primal_seed_trials=getattr(self.problem, "primal_seed_trials", []),
            certificate_scope="complete episode and all reachable stage decisions in the declared discrete operational model",
            bound_method="state-network MILP with optimistic aggregate-queue suffix bounds (zero if unavailable)",
            information="offline oracle: full physical event path and complete exogenous arrival inputs",
            fixed_marl_continuation=False, rolling_horizon=False,
            budget_scope="warm-start revalidation, primal seeds and exact search; final independent replay is timed separately",
            model_export="partially expanded state network; unexplored suffix arcs are relaxations, not feasible schedules")
        self.model.dispose()
        self.emit("finished")
        return result, self.best_controls
