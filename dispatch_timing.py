"""Shared second-resolution execution for the native dispatch model."""
import copy
import math
import time


TIME_SCHEMA = "native_seconds_v1"


class ContinuationMismatch(ValueError):
    """An exact target became unreachable; retain the owner for scope repair."""
    def __init__(self, vehicle, origin, target, second):
        super().__init__("Conditional second-resolution continuation no longer follows a directed edge")
        self.vehicle, self.origin, self.target, self.second = vehicle, origin, target, second


class ArrivalWindow(dict):
    """Interval totals plus exact physical arrivals used for certified replay."""
    def __init__(self, profile):
        super().__init__()
        self.profile = copy.deepcopy(profile)
        for second in profile:
            for platform, values in second.items():
                row = self.setdefault(platform, {})
                for destination, count in values.items():
                    row[destination] = row.get(destination, 0.) + count


def decision_seconds(world):
    return float(getattr(world, "decision_seconds", world.tick_seconds) or 1)


def timing_manifest(world):
    return dict(schema=TIME_SCHEMA if float(world.tick_seconds) == 1 else "legacy_discrete_dispatch",
                physical_tick_seconds=float(world.tick_seconds),
                decision_seconds=decision_seconds(world),
                execution="integer-second physical transitions; no interpolated station events")


def compile_resource_waits(world, tow_pairs=(), *, start_seconds=None, stage=0, check_time=None):
    """Compile private arbitration into explicit targets for a clean replay."""
    audits = world.last_physical_audits if world.second_precision else [world.last_transition_audit]
    start = float(world.sim_time[0])-len(audits)*float(world.tick_seconds) if start_seconds is None else start_seconds
    targets, waits = [], []
    for second, audit in enumerate(audits):
        if check_time is not None:
            check_time()
        stopped = {name for name, row in audit.items() if row["conflict"]}
        now = start+second*float(world.tick_seconds)
        for pair, command_id, fallback_start in tow_pairs:
            result = world.dispatch.command_results.get(command_id, {})
            begun = result.get("started_at", fallback_start)
            begun = fallback_start if begun is None else begun
            finished = result.get("finished_at")
            if begun <= now and (finished is None or now < finished) and pair & stopped:
                stopped |= pair
        targets.append({name: row["origin"] if name in stopped else row["requested"]
                        for name, row in audit.items()})
        waits.extend(dict(kind="reviewed_continuation_wait", stage=stage, second=second,
            vehicle_id=name, origin=audit[name]["origin"], requested_target=audit[name]["requested"],
            reason="shared_resource_conflict" if audit[name]["conflict"] else "coupled_tow_wait")
            for name in sorted(stopped) if audit[name]["requested"] != audit[name]["origin"])
    return targets if world.second_precision else targets[0], waits


def advance_window(world, arrival_inputs=None, dispatch_targets=None):
    with world.disruption_manager.cached_queries():
        return _advance_window(world, arrival_inputs, dispatch_targets)


def _advance_window(world, arrival_inputs=None, dispatch_targets=None):
    """Run the same physical transitions in planning, execution and replay.

    Primitive route/hold decisions persist for a decision interval. Committed
    operation paths and future commands are reconsidered at every physical tick.
    A target mapping defers a vehicle for the interval; a target sequence is an
    exact per-second continuation checked by the planning oracle.
    """
    from MILP.operation_model import nominal_action, set_actions
    tick, interval = float(world.tick_seconds), decision_seconds(world)
    count = int(round(interval / tick))
    if count < 1 or not math.isclose(count * tick, interval):
        raise ValueError("Decision interval must contain whole physical seconds")
    targets = dispatch_targets
    if targets is None:
        targets = getattr(world, "_dispatch_stage_targets", None)
    world._dispatch_stage_targets = None
    expected = getattr(world, "_dispatch_expected_positions", None)
    world._dispatch_expected_positions = None
    if isinstance(targets, list) and len(targets) != count:
        raise ValueError("A second-resolution continuation needs one target map per physical tick")
    initial_actions = {a.name: int(a.state.action[0]) for a in world.agents}
    initial_positions = {a.name: int(a.state.position[0]) for a in world.agents}
    profile = getattr(arrival_inputs, "profile", None)
    if isinstance(arrival_inputs, list):
        profile = arrival_inputs
    if profile is not None and len(profile) != count:
        raise ValueError("Arrival profile must contain one mapping per physical second")
    arrivals = arrival_inputs or {}
    per_second = ({p: {d: n / count for d, n in values.items()} for p, values in arrivals.items()}
                  if isinstance(arrivals, dict) else {})
    totals = world.dispatch.objective.interval(0., 0., [])
    audits, requests, delivered, deviations = [], [], [], []
    platform_waiting = 0.
    deferred = set()
    deadline = getattr(world, "_planning_deadline", math.inf)
    for second in range(count):
        if deadline != math.inf and time.perf_counter() >= deadline:
            from MILP.operation_model import EvaluationTimeout
            raise EvaluationTimeout()
        if second:
            set_actions(world, {a.name: nominal_action(world, a) if initial_actions[a.name] else 0
                                for a in world.agents})
        if isinstance(targets, list):
            override = dict(targets[second])
            for name, target in list(override.items()):
                current = world.dispatch.position(name)
                if target != current and target not in world._has_edge.get(current, ()):
                    if getattr(world, "_strict_dispatch_execution", False):
                        raise ContinuationMismatch(name, current, target, second)
                    op = world.dispatch.operation_for(name)
                    deferred |= op.participants() if op else {name}
                    row = dict(vehicle_id=name, planned_target=target, actual_origin=current,
                               reason="physical_state_diverged_from_conditional_plan")
                    deviations.append(dict(time=float(world.sim_time[0]), **row))
                    world.dispatch.event("plan_execution_deviation", **row)
            override.update({name: world.dispatch.position(name) for name in deferred})
        else:
            override = {name: node for name, node in (targets or {}).items()
                        if second == 0 or node == initial_positions[name]}
        if profile is not None:
            entering = profile[second]
        elif arrival_inputs is None:
            world.disruption_manager.update_disruptions(float(world.sim_time[0]))
            entering = {p: n.passenger_flow.interval_arrivals(world.event_view(), tick)
                        for p, n in world.platform_nodes.items()}
        else:
            entering = per_second
        world._step_physical(arrival_inputs=entering, dispatch_targets=override)
        platform_waiting += world.last_platform_waiting_person_seconds
        delivered.append(copy.deepcopy(entering))
        audits.append(copy.deepcopy(world.last_transition_audit))
        requests.append({name: row["requested"] for name, row in world.last_transition_audit.items()})
        for key in totals:
            totals[key] += world.dispatch.interval_cost[key]
    world.dispatch.interval_cost = totals
    world.last_physical_audits = audits
    world.last_physical_requests = requests
    world.last_arrival_inputs = ArrivalWindow(delivered)
    world.last_platform_waiting_person_seconds = platform_waiting
    world.last_execution_deviations = deviations
    world.last_transition_audit = {name: dict(rows[-1][name],
        conflict=any(row[name]["conflict"] for row in rows)) for name in initial_actions for rows in [audits]}
    actual = {a.name: int(a.state.position[0]) for a in world.agents}
    world.last_dispatch_execution = dict(positions_match=expected is None or expected == actual,
        mismatches={name: dict(expected=node, actual=actual[name]) for name, node in (expected or {}).items()
                    if actual[name] != node}, physical_steps=count, deviations=deviations,
        constraint_interventions=[name for name, row in world.last_transition_audit.items() if row["conflict"]])
    recorder = getattr(getattr(world, "execution_records", None), "result_writer", None)
    if recorder is not None:
        recorder.record_window(world, initial_actions)

