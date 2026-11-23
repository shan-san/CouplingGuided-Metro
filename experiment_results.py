"""One executed-episode result contract for every native benchmark method."""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from dispatch_records import clock, enable_execution_records
from dispatch_timing import decision_seconds, timing_manifest

RESULT_SCHEMA = "metro_episode_v1"
VEHICLE_FIELDS = ("stage", "seconds", "clock", "vehicle", "node", "station", "status",
                  "service_id", "onboard", "operation", "phase")
METRIC_FIELDS = ("method", "case_id", "line", "topology", "episode", "status", "complete",
    "completed_stages", "requested_stages", "simulation_seconds", "cost", "realized_interval_cost",
    "discounted_terminal_cost", "waiting_person_seconds", "inside_waiting_person_seconds",
    "outside_waiting_person_seconds", "onboard_person_seconds", "passenger_person_seconds",
    "completed_passengers", "boarded_passengers", "unfinished_passengers", "headway_deviation_seconds",
    "departure_headway_count", "departure_headway_mean_seconds", "departure_headway_std_seconds",
    "operation_units", "decision_count", "average_decision_s", "total_decision_s", "p95_decision_s",
    "max_decision_s", "computation_time_s", "elapsed_episode_s", "fallbacks", "budget_exceedances",
    "resource_interventions", "execution_guard_mismatches", "strictly_improved_decisions",
    "lower_bound", "upper_bound", "absolute_gap", "relative_gap", "optimality_proven", "verified",
    "source_changed")
CORE_FILES = ("manifest.json", "summary.json", "metrics.csv", "decisions.jsonl", "vehicle_states.csv",
    "timetable.csv", "station_events.csv", "operation_events.jsonl", "resource_occupancy.csv",
    "scenario_events.json", "timing.json", "operational_health.json", "progress.json")
PHYSICAL_SOURCES = ("experiment_results.py", "dispatch_records.py", "dispatch_timing.py", "dispatch_operations.py",
    "dispatch_objective.py", "dispatch_plan.py", "delay_events.py", "metro_world.py", "metro_agent_v2.py",
    "MILP/operation_model.py", "scenarios/prepare_dispatch.py", "topology/metro_lines.py",
    "topology/passenger_flow.py", "buildtopology.py", "linedata.py", "route_selection.py")
ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _state_hash(world):
    from MILP.operation_model import state_guard
    return _hash(state_guard(world))


def _ledger(world, initial, arrivals):
    counts = world.dispatch.objective.passenger_state(world)
    completed = sum(float(a.cumulative_alighted[0]) for a in world.agents)
    remaining = sum(counts[k] for k in ("inside", "outside", "onboard"))
    residual = initial + arrivals - completed - remaining
    return dict(initial_passengers=initial, arrivals=arrivals, completed=completed, remaining=remaining,
                residual=residual, conserved=math.isclose(initial+arrivals, completed+remaining, rel_tol=1e-7, abs_tol=1e-3))


class EpisodeResults:
    """Stream identical executed controls and compute metrics without reward reconstruction."""

    def __init__(self, directory, world, *, method, case_id, requested_stages, episode=0,
                 entry=None, recipe=None, configuration=None, source_sha256=None, online=True):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any((self.directory/name).exists() for name in CORE_FILES):
            raise FileExistsError(f"Refusing to overwrite episode results: {self.directory}")
        self.started = time.perf_counter()
        self.online = bool(online)
        sources = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in PHYSICAL_SOURCES}
        sources.update(source_sha256 or {})
        topology = getattr(world, "topology", None)
        from topology.metro_lines import LINE_SPECS
        line = LINE_SPECS.get(topology, {}).get("key")
        self.manifest = dict(result_schema=RESULT_SCHEMA, method=method, case_id=case_id, line=line,
            topology=topology, episode=episode, requested_stages=requested_stages, online=self.online,
            fleet=len(world.agents), entry=copy.deepcopy(entry), recipe=copy.deepcopy(recipe),
            configuration=copy.deepcopy(configuration or {}), source_sha256=sources,
            initial_state_sha256=_state_hash(world), timing=timing_manifest(world),
            objective_definition=world.dispatch.objective.manifest(),
            metric_definitions=dict(cost="discounted executed interval costs plus terminal cost; minimize",
                passenger_time="person-seconds; inside waiting + outside waiting + all onboard time",
                headways="actual passenger-service departures, grouped by directional platform; population standard deviation",
                decision_time="online policy invocation through dispatch commitment, excluding physical simulation; offline null",
                missing="null means not applicable or not measured, never a zero substitute"),
            files={name: name for name in CORE_FILES}, diagnostics="diagnostics/")
        self.costs, self.latencies = [], []
        self.inside_waiting = self.outside_waiting = self.arrivals = 0.
        self.pending = None
        self.counts = dict(fallbacks=0, budget_exceedances=0, resource_interventions=0,
                           execution_guard_mismatches=0, strictly_improved_decisions=0)
        initial = _ledger(world, 0., 0.)
        self.initial_mass = initial["completed"] + initial["remaining"]
        self.initial_completed = sum(float(a.cumulative_alighted[0]) for a in world.agents)
        self.initial_boarded = sum(float(a.cumulative_boarded[0]) for a in world.agents)
        self.attach(world)
        (self.directory/"decisions.jsonl").touch()
        with (self.directory/"vehicle_states.csv").open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, VEHICLE_FIELDS).writeheader()
        self._vehicle_states(-1, world)
        write_json(self.directory/"scenario_events.json", copy.deepcopy(getattr(world, "_configured_events", None) or []))
        write_json(self.directory/"manifest.json", self.manifest)
        self.progress(phase="running" if self.online else "searching")

    def attach(self, world):
        if _state_hash(world) != self.manifest["initial_state_sha256"]:
            raise ValueError("Result recorder initial state differs")
        self.world = world
        records = enable_execution_records(world)
        records.result_writer = self
        self.plan_ids = set(world.dispatch.plans)

    def accrue(self, counts, tick):
        self.inside_waiting += counts["inside"] * tick
        self.outside_waiting += counts["outside"] * tick

    def _vehicle_states(self, stage, world):
        now = float(world.sim_time[0])
        with (self.directory/"vehicle_states.csv").open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, VEHICLE_FIELDS)
            for v in world.dispatch.records.values():
                operation = world.dispatch.operation_for(v.id)
                node = world.dispatch.position(v.id)
                writer.writerow(dict(stage=stage, seconds=now, clock=clock(now, world.clock_start_seconds),
                    vehicle=v.id, node=node, station=getattr(world.nodes[node], "station_name", ""), status=v.status,
                    service_id=v.service_id or "", onboard=float(v.agent.passenger_count[0]),
                    operation=operation.action if operation else "", phase=operation.phase if operation else ""))

    def record_window(self, world, initial_actions):
        self.flush()
        self.world = world
        stage = len(self.costs)
        cost = copy.deepcopy(world.dispatch.interval_cost)
        self.costs.append(cost)
        arrivals = world.last_arrival_inputs
        self.arrivals += sum(sum(values.values()) for values in arrivals.values())
        plans = [asdict(plan) for ident, plan in world.dispatch.plans.items() if ident not in self.plan_ids]
        self.plan_ids.update(world.dispatch.plans)
        trace_hash = _hash(world.last_physical_audits)
        interventions = sum(bool(row["conflict"]) for tick in world.last_physical_audits for row in tick.values())
        self.counts["resource_interventions"] += interventions
        now = float(world.sim_time[0])
        self.pending = dict(result_schema=RESULT_SCHEMA, stage=stage,
            start_seconds=now-decision_seconds(world), seconds=now, decision_s=None, budget_s=None,
            budget_exceeded=None, metadata={}, audit={}, interval_cost=cost,
            audit_classification=None, conditional_inputs_changed=None,
            passenger_ledger=_ledger(world, self.initial_mass, self.arrivals),
            control=dict(plans=plans, commands=[c for plan in plans for c in plan["commands"]],
                         actions=dict(initial_actions), targets=copy.deepcopy(world.last_physical_requests)),
            planned_targets=None, execution_deviations=copy.deepcopy(world.last_execution_deviations),
            actual_arrivals=dict(arrivals), actual_arrival_profile=copy.deepcopy(arrivals.profile),
            actual_positions={v.id: world.dispatch.position(v.id) for v in world.dispatch.records.values()},
            actual_execution_sha256=trace_hash, actual_state_sha256=_state_hash(world),
            passenger_state=world.dispatch.objective.passenger_state(world))
        self._vehicle_states(stage, world)
        self.progress()

    def progress(self, *, phase="running", search=None):
        gamma = self.world.dispatch.objective.weights["discount"]
        interval = sum(gamma**stage*row["total"] for stage, row in enumerate(self.costs))
        terminal = self.world.dispatch.objective.terminal(self.world)["total"]
        write_json(self.directory/"progress.json", dict(result_schema=RESULT_SCHEMA,
            method=self.manifest["method"], case_id=self.manifest["case_id"], phase=phase,
            completed_stages=len(self.costs), requested_stages=self.manifest["requested_stages"],
            simulation_seconds=float(self.world.sim_time[0]), elapsed_s=time.perf_counter()-self.started,
            realized_interval_cost=interval if self.costs else None,
            cost_if_stopped_now=interval+gamma**len(self.costs)*terminal if self.costs else None,
            search=search, scope="progress only; not a complete-episode result"))

    def annotate(self, *, decision_s=None, budget_s=None, metadata=None, audit=None,
                 audit_classification=None, inputs_changed=None, planned_targets=None,
                 fallback=False, strictly_improved=False):
        if self.pending is None:
            raise ValueError("No completed physical decision window to annotate")
        if decision_s is not None:
            if not self.online or not math.isfinite(decision_s) or decision_s < 0:
                raise ValueError("Decision latency must be a nonnegative online measurement")
            self.latencies.append(float(decision_s))
        exceeded = None if decision_s is None or budget_s is None else decision_s > budget_s
        self.counts["fallbacks"] += int(fallback)
        self.counts["strictly_improved_decisions"] += int(strictly_improved)
        self.counts["budget_exceedances"] += int(bool(exceeded))
        self.counts["execution_guard_mismatches"] += int(bool(audit) and not audit.get("matched", False))
        self.pending.update(decision_s=decision_s, budget_s=budget_s, budget_exceeded=exceeded,
            metadata=copy.deepcopy(metadata or {}), audit=copy.deepcopy(audit or {}),
            audit_classification=audit_classification, conditional_inputs_changed=inputs_changed,
            planned_targets=copy.deepcopy(planned_targets))
        self.flush()

    def flush(self):
        if self.pending is not None:
            with (self.directory/"decisions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(self.pending, ensure_ascii=False, allow_nan=False)+"\n")
            self.pending = None

    def finish(self, *, status="complete", error=None, diagnostics=None, computation_time_s=None,
               verified=False, bounds=None, source_changed=None):
        self.flush()
        world = self.world
        world.execution_records.write(self.directory, world)
        stages = len(self.costs)
        physical = stages > 0
        complete = status == "complete" and stages == self.manifest["requested_stages"]
        if status == "complete" and not complete:
            status = "partial"
        objective = world.dispatch.objective
        terminal = objective.terminal(world) if physical else None
        gamma = objective.weights["discount"]
        interval = sum(gamma**stage*row["total"] for stage, row in enumerate(self.costs)) if physical else None
        terminal_cost = gamma**stages*terminal["total"] if physical else None
        totals = {key: sum(row[key] for row in self.costs) if physical else None for key in
                  ("waiting_person_seconds", "onboard_person_seconds", "passenger_person_seconds",
                   "headway_deviation_seconds", "operation_units")}
        departures = {}
        for visit in world.execution_records.visits:
            if visit["departure_seconds"] is not None and visit.get("departure_passenger_service", visit["passenger_service"]):
                departures.setdefault(visit["platform"], []).append(visit["departure_seconds"])
        headways = [b-a for times in departures.values() for a, b in zip(sorted(times), sorted(times)[1:])]
        bounds = dict(bounds or {})
        if source_changed is None:
            source_changed = any(not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest
                                 for name, digest in self.manifest["source_sha256"].items())
        summary = dict(result_schema=RESULT_SCHEMA,
            **{key: self.manifest[key] for key in ("method", "case_id", "line", "topology", "episode", "requested_stages")},
            status=status, complete=complete, error=error, completed_stages=stages,
            simulation_seconds=float(world.sim_time[0]),
            cost=None if interval is None else interval+terminal_cost,
            realized_interval_cost=interval, discounted_terminal_cost=terminal_cost,
            **totals, inside_waiting_person_seconds=self.inside_waiting if physical else None,
            outside_waiting_person_seconds=self.outside_waiting if physical else None,
            completed_passengers=sum(float(a.cumulative_alighted[0]) for a in world.agents)-self.initial_completed if physical else None,
            boarded_passengers=sum(float(a.cumulative_boarded[0]) for a in world.agents)-self.initial_boarded if physical else None,
            unfinished_passengers=terminal["unfinished_passengers"] if terminal else None,
            terminal=terminal, passenger_ledger=_ledger(world, self.initial_mass, self.arrivals) if physical else None,
            departure_headway_count=len(headways), departure_headway_mean_seconds=statistics.mean(headways) if headways else None,
            departure_headway_std_seconds=statistics.pstdev(headways) if headways else None,
            decision_count=len(self.latencies) if self.online else None,
            average_decision_s=statistics.mean(self.latencies) if self.latencies else None,
            total_decision_s=sum(self.latencies) if self.latencies else None,
            p95_decision_s=sorted(self.latencies)[math.ceil(.95*len(self.latencies))-1] if self.latencies else None,
            max_decision_s=max(self.latencies) if self.latencies else None,
            computation_time_s=computation_time_s, elapsed_episode_s=time.perf_counter()-self.started,
            **self.counts, lower_bound=bounds.get("lower_bound"), upper_bound=bounds.get("upper_bound"),
            relative_gap=bounds.get("relative_gap"), absolute_gap=bounds.get("absolute_gap"),
            optimality_proven=bool(bounds.get("optimality_proven", False)), verified=bool(verified),
            objective_source="DispatchObjective interval costs plus final terminal cost",
            objective_definition=objective.manifest(), timing=timing_manifest(world),
            source_sha256=self.manifest["source_sha256"], source_changed=bool(source_changed),
            episode_seed=(self.manifest["entry"] or {}).get("seed"),
            scenario_sha256=(self.manifest["recipe"] or {}).get("sha256"),
            initial_state_sha256=self.manifest["initial_state_sha256"],
            experiment_configuration=self.manifest["configuration"],
            metrics_scope="complete executed episode" if complete else "executed prefix; not a complete-episode comparison",
            diagnostics="diagnostics/method.json")
        write_json(self.directory/"diagnostics"/"method.json", diagnostics or {})
        write_json(self.directory/"summary.json", summary)
        self.progress(phase=status)
        with (self.directory/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, METRIC_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(summary)
        return summary


def replay_episode(manifest_path):
    """One policy-free replay reader for all methods, including the offline plan."""
    from MILP.operation_model import set_actions
    from scenarios.prepare_dispatch import make_world
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("result_schema") != RESULT_SCHEMA:
        raise ValueError("Not a common episode manifest")
    for name, digest in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Replay source changed: {name}")
    entry = manifest["entry"] or {}
    if "seed" not in entry:
        raise ValueError("Manifest lacks the initial episode seed; exact replay is unavailable")
    events = json.loads((manifest_path.parent/"scenario_events.json").read_text(encoding="utf-8"))
    world = make_world(manifest["topology"], fleet=manifest["fleet"], seed=entry["seed"], events=events,
        initial_passenger_seconds=manifest["timing"]["decision_seconds"])
    if _state_hash(world) != manifest["initial_state_sha256"]:
        raise ValueError("Replay initial physical state differs")
    cost, count = 0., 0
    gamma = world.dispatch.objective.weights["discount"]
    with (manifest_path.parent/"decisions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["result_schema"] != RESULT_SCHEMA or row["stage"] != count:
                raise ValueError("Replay stages or schema differ")
            for raw in row["control"]["plans"]:
                raw["commands"] = list(raw["commands"])
                raw["snapshot_version"] = world.dispatch.snapshot(published=raw["snapshot_basis"] == "published")["version"]
                result = world.submit_dispatch_plan(raw)
                if not result["accepted"]:
                    raise ValueError(f"Replay command rejected: {result}")
            set_actions(world, row["control"]["actions"])
            profile = [{int(p): {int(d): q for d, q in values.items()} for p, values in tick.items()}
                       for tick in row["actual_arrival_profile"]]
            world.step(arrival_inputs=profile, dispatch_targets=row["control"]["targets"])
            if _state_hash(world) != row["actual_state_sha256"] or _hash(world.last_physical_audits) != row["actual_execution_sha256"]:
                raise ValueError(f"Replay physical state or second-resolution trace differs at stage {count}")
            if not math.isclose(world.dispatch.interval_cost["total"], row["interval_cost"]["total"], abs_tol=1e-5):
                raise ValueError("Replay interval objective differs")
            cost += gamma**count*world.dispatch.interval_cost["total"]
            count += 1
    if not count:
        raise ValueError("No executed decisions to replay")
    cost += gamma**count*world.dispatch.objective.terminal(world)["total"]
    summary = json.loads((manifest_path.parent/"summary.json").read_text(encoding="utf-8"))
    if not math.isclose(cost, summary["cost"], rel_tol=1e-9, abs_tol=1e-4):
        raise ValueError("Replay episode objective differs")
    return dict(verified=True, completed_stages=count, complete=summary["complete"], cost=cost)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, help="Path to any native method's manifest.json")
    print(json.dumps(replay_episode(parser.parse_args().replay), ensure_ascii=False))
