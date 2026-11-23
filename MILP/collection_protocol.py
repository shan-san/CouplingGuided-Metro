"""Published-only sampling and conditional collection-window estimates.

This module has no solver or learning-library dependencies. Neither API advances
the world. A window is a duration estimate, not a completion certificate: resource
competition and the unknown clearance of published restrictions can add delay.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import math


__all__ = ["StratifiedSampler", "choose_horizon"]

_INFORMATION_EVENTS = {"observation_delay", "recovery_prediction_revision"}
_EVENT_FIELDS = ("event_id", "type", "target", "parameters", "reported_at",
                 "active", "predicted_end_seconds", "information_version")
_QUALIFICATION_FIELDS = ("role", "status", "service_id", "depot", "failed",
                         "needs_inspection", "has_served")
_PHYSICAL_FIELDS = ("position", "passenger_count", "passengers_by_destination",
                    "passenger_waiting_mass", "travel_delay_steps", "pending_edge", "profile")
_CONDITIONAL = "conditional duration estimate; competition may add delay; completion not guaranteed"


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _seconds(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _key(value):
    # Snapshot destination dictionaries may mix integer and string keys.
    if isinstance(value, dict):
        return tuple(sorted((str(k), _key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_key(v) for v in value)
    return value


def _events(world):
    return sorted(({k: e[k] for k in _EVENT_FIELDS if k in e}
                   for e in world.get_published_events()), key=lambda e: str(e["event_id"]))


def _phase(time_s):
    if time_s < 900:
        return "normal"
    if time_s < 2700:
        return "event_onsets"
    if time_s < 4500:
        return "partial_clearance"
    return "service_recovery"


def _qualifications(snapshot):
    vehicles = {}
    for key, row in snapshot.get("vehicles", {}).items():
        data = {k: row.get(k) for k in _QUALIFICATION_FIELDS}
        # Holding does not change passenger-service or insertion qualifications.
        if data["status"] == "HOLDING":
            data["status"] = "IN_SERVICE"
        vehicles[key] = data
    services = {key: dict(vehicle_id=row.get("vehicle_id"),
                          reserved_vehicle_id=row.get("reserved_vehicle_id"),
                          started=row.get("started_at") is not None,
                          ended=row.get("ended_at") is not None,
                          completed=row.get("completed", False), stops=row.get("stops"))
                for key, row in snapshot.get("services", {}).items()}
    return dict(vehicles=vehicles, services=services)


def _operation_phases(snapshot):
    # IDs, deadlines, countdowns and path progress are not new operation phases.
    return sorted((_key({k: op.get(k) for k in (
        "vehicle_id", "action", "phase", "partner", "depot", "target", "service_id", "wait_reason")})
        for op in snapshot.get("operations", {}).values() if op.get("action") != "hold"), key=repr)


def _busy_vehicles(snapshot):
    return {key for op in snapshot.get("operations", {}).values()
            for key in (op.get("vehicle_id"), op.get("partner")) if key is not None}


def _active_failures(events):
    return {e["target"]["id"] for e in events
            if e["type"] == "vehicle_breakdown" and e.get("active", True)}


def _depot_exit_allowed(depot, events):
    return not any(e["type"] == "depot_access_restriction" and e.get("active", True)
                   and e["target"]["id"] == depot and not e["parameters"]["exit_allowed"] for e in events)


def _insertion_action(key, vehicle, busy, failures, events):
    if (key in busy or key in failures or vehicle.get("depot") is None
            or vehicle.get("status") != "DEPOT_READY" or vehicle.get("failed")
            or vehicle.get("needs_inspection") or not _depot_exit_allowed(vehicle["depot"], events)):
        return None
    if vehicle.get("has_served"):
        return "reenter"
    return "reserve_insert" if vehicle.get("role") == "reserve" else None


def _opportunities(snapshot, events, cleared_breakdowns):
    """Public eligibility evidence, never command masks or improvement labels."""
    vehicles = snapshot.get("vehicles", {})
    busy, failures = _busy_vehicles(snapshot), _active_failures(events)
    opportunities = {}
    capacity_pressure = any(e.get("active", True) and e["type"] in
        {"vehicle_breakdown", "vehicle_capacity_loss", "demand_surge"} for e in events)
    for key, vehicle in vehicles.items():
        if key in busy:
            continue
        if (vehicle.get("depot") is not None and vehicle.get("needs_inspection")
                and key in cleared_breakdowns and key not in failures):
            opportunities["inspect_repair", key] = dict(action="inspect_repair", vehicle_id=key,
                basis="stored_needs_inspection_and_published_failure_clearance")
        insertion = _insertion_action(key, vehicle, busy, failures, events)
        if insertion == "reenter":
            opportunities["reenter", key] = dict(action="reenter", vehicle_id=key,
                basis="depot_ready_healthy_and_has_served")
        if insertion == "reserve_insert" and capacity_pressure:
            item = opportunities.setdefault(("reserve_insert", vehicle["depot"]),
                dict(action="reserve_insert", depot=vehicle["depot"], vehicle_ids=[],
                     basis="public_capacity_pressure_and_qualified_depot_stock"))
            item["vehicle_ids"].append(key)
    for task_id, task in snapshot.get("services", {}).items():
        stops = task.get("stops")
        if (not stops or len(stops) < 2 or task.get("vehicle_id") is not None
                or task.get("reserved_vehicle_id") is not None or task.get("started_at") is not None
                or task.get("ended_at") is not None or task.get("completed")):
            continue
        if task.get("purpose") == "section_service" and not any(e.get("active", True)
                and e["type"] in {"track_blockage", "switch_failure", "power_restriction", "turnback_capacity_loss"}
                for e in events):
            continue
        if any(e["type"] == "platform_closure" and e.get("active", True)
               and e["target"]["id"] == stops[0] for e in events):
            continue
        qualified = sorted(key for key, vehicle in vehicles.items() if key not in busy
            and key not in failures and vehicle.get("depot") is None and not vehicle.get("failed")
            and not vehicle.get("needs_inspection") and vehicle.get("position") in
                (stops[:-1] if task.get("pattern") else [stops[0]]))
        if qualified:
            opportunities["reassign_service", task_id] = dict(action="reassign_service", task_id=task_id,
                vehicle_ids=qualified, origin=stops[0], basis="unbound_unreserved_task_with_qualified_vehicle_at_origin")
    rescue_targets = {key: None for key, vehicle in vehicles.items()
                      if vehicle.get("failed") and vehicle.get("depot") is None}
    for event in events:
        if event["type"] != "vehicle_breakdown":
            continue
        key = event["target"]["id"]
        if key in rescue_targets or (event.get("active", True)
                and event["parameters"].get("mode") == "immobilized"):
            rescue_targets[key] = event["event_id"]
    for key, event_id in rescue_targets.items():
        vehicle = vehicles.get(key)
        if vehicle is None or key in busy or vehicle.get("depot") is not None:
            continue
        rescuers = sorted(other for other, row in vehicles.items() if other != key
            and other not in busy and other not in failures and not row.get("failed")
            and not row.get("needs_inspection") and (row.get("depot") is None
                or _depot_exit_allowed(row["depot"], events)))
        if rescuers:
            opportunities["rescue_tow", key] = dict(action="rescue_tow", vehicle_id=key,
                event_id=event_id, rescuer_ids=rescuers,
                basis=("public_snapshot_failed_track_vehicle_with_publicly_qualified_rescuer" if vehicle.get("failed")
                       else "published_immobilized_track_vehicle_with_publicly_qualified_rescuer"))
    return opportunities


def _recovery_context(snapshot, events):
    return (any(e["type"] not in _INFORMATION_EVENTS for e in events)
            or any(v.get("failed") or v.get("needs_inspection") for v in snapshot.get("vehicles", {}).values())
            or any(op.get("action") in {"inspect_repair", "rescue_tow", "withdraw", "reenter", "reserve_insert"}
                   for op in snapshot.get("operations", {}).values())
            or any(task.get("ended_at") is not None for task in snapshot.get("services", {}).values()))


class StratifiedSampler:
    """One pickleable sampler per episode; call observe once per decision stage.

    interval/min_gap are decision-stage counts, not physical ticks. tick_seconds
    is only the fallback clock when a fake/legacy snapshot has no ``time``.
    Fixed experiment ``phase`` labels are independent of the observed stratum.
    ``phase_limit`` is a per-stratum episode quota, not a time-band quota.
    Call reset() before reusing an instance for another episode.

    Periodic sampling requires a distinct public physical state. Time, version,
    command IDs, operation countdowns and routine hold commands alone do not
    create new states. Changes suppressed by min_gap remain pending.

    opportunity_limit (0..4, default 0) reserves extra episode samples after a
    stratum's regular quota is exhausted. Only first-seen public inspect_repair,
    reenter, reassign_service and rescue_tow qualifications can use this quota.
    A qualification already covered by a regular sample cannot use it again.
    Pending qualifications survive min_gap only while still currently eligible.
    Metadata is eligibility evidence, not a solver feasibility/benefit label.
    Public snapshot failed track vehicles remain rescue opportunities after
    their external breakdown report clears; the vehicle identity deduplicates
    that same qualification across report revisions and clearance.
    """

    def __init__(self, tick_seconds=30, interval=20, normal_limit=2, phase_limit=4, min_gap=1,
                 opportunity_limit=0, successful_quota=False, attempt_multiplier=2,
                 combination_quota=False):
        self.tick_seconds = _seconds(tick_seconds, "tick_seconds", positive=True)
        self.interval = _integer(interval, "interval", 1)
        self.normal_limit = _integer(normal_limit, "normal_limit")
        self.phase_limit = _integer(phase_limit, "phase_limit")
        self.min_gap = _integer(min_gap, "min_gap")
        self.opportunity_limit = _integer(opportunity_limit, "opportunity_limit")
        self.successful_quota = bool(successful_quota)
        self.attempt_multiplier = _integer(attempt_multiplier, "attempt_multiplier", 1)
        self.combination_quota = bool(combination_quota)
        if self.opportunity_limit > 4:
            raise ValueError("opportunity_limit must not exceed 4 extra episode samples")
        self.reset()

    def reset(self):
        self.counts = Counter()
        self.accepted_counts = Counter()
        self.combination_attempts, self.combination_successes = Counter(), Counter()
        self._last_stage = self._sample_stage = None
        self._previous = None
        self._seen_events, self._cleared_events = set(), set()
        self._active_ids = set()
        self._sampled_states, self._pending = set(), set()
        self.opportunity_count = 0
        self._seen_opportunities, self._cleared_breakdowns = set(), set()
        self._pending_opportunities = {}

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Older collection checkpoints predate the opt-in opportunity quota.
        for key, default in (("opportunity_limit", 0), ("opportunity_count", 0),
                             ("_seen_opportunities", set()), ("_cleared_breakdowns", set()),
                             ("_pending_opportunities", {})):
            self.__dict__.setdefault(key, default)
        self.__dict__.setdefault("successful_quota", False)
        self.__dict__.setdefault("attempt_multiplier", 2)
        self.__dict__.setdefault("accepted_counts", Counter())
        self.__dict__.setdefault("combination_quota", False)
        self.__dict__.setdefault("combination_attempts", Counter())
        self.__dict__.setdefault("combination_successes", Counter())

    def feedback(self, sampling, accepted, *, coupled_joint=False):
        if accepted:
            self.accepted_counts[sampling["stratum"]] += 1
        goal_met = accepted and (sampling.get("coverage_goal") != "coupled_joint" or coupled_joint)
        if self.combination_quota and goal_met:
            self.combination_successes[tuple(sampling["quota_key"])] += 1
        if not goal_met and self.successful_quota and sampling["stratum"] != "normal":
            self._pending.add("quality_retry")

    def observe(self, world, stage):
        """Return sampling metadata from public events, stock and observed queues."""
        _integer(stage, "stage")
        if self._last_stage is not None:
            if stage < self._last_stage:
                raise ValueError("Stages must increase; reset the sampler for a new episode")
            if stage == self._last_stage:
                return None
        snapshot, events = world.dispatch_snapshot(), _events(world)
        time_s = _seconds(snapshot.get("time", stage * self.tick_seconds), "snapshot time")
        physical = [e for e in events if e["type"] not in _INFORMATION_EVENTS]
        if self.opportunity_limit:
            self._cleared_breakdowns.update(e["target"]["id"] for e in physical
                if e["type"] == "vehicle_breakdown" and not e.get("active", True))
            eligible = _opportunities(snapshot, events, self._cleared_breakdowns)
            self._pending_opportunities = {key: value for key, value in eligible.items()
                if key in self._pending_opportunities or key not in self._seen_opportunities}
            self._seen_opportunities.update(eligible)
        active_ids = {e["event_id"] for e in physical if e.get("active", True)}
        self._seen_events.update(e["event_id"] for e in physical)
        self._cleared_events.update(e["event_id"] for e in physical if not e.get("active", True))
        self._cleared_events.update(self._active_ids - active_ids)
        self._active_ids = active_ids
        if active_ids:
            stratum = ("partial_recovery" if self._cleared_events - active_ids else
                       "overlap" if len(active_ids) > 1 else "single_event")
        else:
            stratum = "recovery" if self._seen_events else "normal"
        phase = _phase(time_s)
        qualifications, operations = _qualifications(snapshot), _operation_phases(snapshot)
        current = dict(published_events=_key(events), operation_phase=_key(operations),
                       eligibility=_key(qualifications), stratum=stratum, phase=phase)
        if self._previous is None:
            self._pending.add("initial")
        else:
            self._pending.update(k + "_changed" for k in current if current[k] != self._previous[k])
        self._previous, self._last_stage = current, stage
        physical_state = {key: {k: v.get(k) for k in _PHYSICAL_FIELDS}
                          for key, v in snapshot.get("vehicles", {}).items()}
        state = _key(dict(current, physical=physical_state,
                          queues={str(p): round(world.waiting_at(p), 1)
                                  for p in getattr(world, "platform_nodes", {})},
                          paths={str(op.get("vehicle_id")): list(op.get("path", ()))
                                 for op in snapshot.get("operations", {}).values()
                                 if op.get("action") != "hold"},
                          admission_rates=snapshot.get("admission_rates", {})))
        limit = self.normal_limit if stratum == "normal" else self.phase_limit
        active_types = sorted({e["type"] for e in physical if e.get("active", True)})
        quota_key = (stratum, *active_types)
        if stratum == "normal" and limit == 0:
            return None
        if self.combination_quota and stratum != "normal":
            extra = (self.combination_successes[quota_key] >= limit
                     or self.combination_attempts[quota_key] >= limit*self.attempt_multiplier)
        elif self.successful_quota and stratum != "normal":
            extra = (self.accepted_counts[stratum] >= limit
                     or self.counts[stratum] >= limit*self.attempt_multiplier)
        else:
            extra = self.counts[stratum] >= limit
        if state in self._sampled_states and not (self.successful_quota and "quality_retry" in self._pending):
            return None
        if extra and (not self._pending_opportunities or self.opportunity_count >= self.opportunity_limit):
            return None
        if self._sample_stage is not None and stage - self._sample_stage < self.min_gap:
            return None
        periodic = self._sample_stage is not None and stage - self._sample_stage >= self.interval
        if not self._pending and not periodic and not self._pending_opportunities:
            return None
        reasons = sorted(self._pending)
        if periodic:
            reasons.append("periodic")
        opportunities = [self._pending_opportunities[key] for key in sorted(self._pending_opportunities)]
        if opportunities:
            reasons.insert(0, "opportunity")
        self._pending.clear()
        self._sample_stage = stage
        self._sampled_states.add(state)
        if extra:
            self.opportunity_count += 1
        else:
            self.counts[stratum] += 1
            self.combination_attempts[quota_key] += 1
        self._pending_opportunities.clear()
        result = dict(stratum=stratum, reason=reasons[0], reasons=reasons, phase=phase,
                    time_s=time_s, stage=stage, active_event_ids=sorted(active_ids),
                    active_event_types=active_types,
                    stratum_count=self.counts[stratum])
        if self.combination_quota:
            result.update(quota_key=list(quota_key), coverage_goal=(
                "coupled_joint" if len(active_types) > 1 else "improved_guidance"))
        if self.opportunity_limit:
            result.update(quota="opportunity" if extra else "regular", opportunity_count=self.opportunity_count,
                          opportunities=opportunities,
                          opportunity_scope="public_eligibility_only_not_joint_feasibility_or_benefit_labels")
        return result


class _DurationEstimate:
    """Static directed paths plus published state; never consult execution internals."""

    def __init__(self, world, snapshot, events, tick):
        self.dispatch = world.dispatch
        self.config = getattr(world, "dispatch_config", None)
        if self.config is None:
            self.config = self.dispatch.config
        self.snapshot = snapshot
        self.vehicles = snapshot.get("vehicles", {})
        self.operations = list(snapshot.get("operations", {}).values())
        self.events = [e for e in events if e.get("active", True)]
        self.depots = self.config.get("depots", {})
        self.tick = tick
        self.reasons, self.unknown = [], False
        self.clearance_rate = self.config.get("alighting_rate_persons_per_second")
        self.boarding_rate_fallback = self.clearance_rate is None
        if self.boarding_rate_fallback:
            self.clearance_rate = getattr(world, "boarding_rate_persons_per_second", None)
        self.platforms = getattr(world, "platform_nodes", None)
        self.edge_segments = {}
        for segment, nodes in (getattr(world, "segment2nodes", None) or {}).items():
            ids = [getattr(node, "id", node) for node in nodes]
            for origin, target in zip(ids, ids[1:]):
                self.edge_segments[origin, target] = (str(getattr(segment, "id", segment)), len(ids) - 1)
        from delay_events import EventView
        self.view = EventView(self.events, getattr(world, "event_resources", {}))
        self.turnback_segments = {segment for spec in self.view.resources.get("turnbacks", {}).values()
                                  for segment in spec["segments"]}
        self.entry_routes = {}
        nodes = getattr(world, "nodes", {})
        entries = {node for spec in self.view.resources.get("switches", {}).values()
                   for node in [spec.get("node_id"), *spec.get("route_entry_nodes", {}).values()]}
        for origin in entries:
            if origin in nodes:
                for route, target in (nodes[origin].getnext() or {}).items():
                    self.entry_routes.setdefault((origin, target), []).append(route)

    def duration(self, seconds):
        return math.ceil(_seconds(seconds, "duration") / self.tick) * self.tick

    def path(self, start, target, depot=None, *, open_only=False):
        if open_only and hasattr(self.dispatch, "available_path"):
            return self.dispatch.available_path(start, target, allow_depot=depot, view=self.view)
        return self.dispatch.path(start, target, allow_depot=depot)

    def travel(self, path, *, start=None, vehicle_id=None, tow=False, allowances=None):
        if path is None:
            return None
        if not path:
            return 0.
        # Use network-wide published slowdown allowances, without assuming that
        # a restriction will end at its predicted recovery time. Round per edge.
        factor = max([1., self.config.get("tow_multiplier", 2.) if tow else 1.]
                     + list(self.config.get("run_profiles", {"normal": 1., "slow": 1.2, "very_slow": 1.5}).values())
                     + [e["parameters"].get("travel_time_multiplier", 1.) for e in self.events])
        extra = self.running_extra(path, start, vehicle_id, {} if allowances is None else allowances)
        return len(path) * self.duration(self.tick * factor) + extra

    def running_extra(self, path, start, vehicle_id, allowances):
        edges = list(zip([start, *path], path))
        mapped = start is not None and all(edge in self.edge_segments for edge in edges)
        covered, sizes = Counter(), {}
        if mapped:
            for edge in edges:
                segment, size = self.edge_segments[edge]
                covered[segment] += 1
                sizes[segment] = size
        shares = {segment: Fraction(count, sizes[segment]) for segment, count in covered.items()}
        extra = 0.
        for event in self.events:
            if event["type"] != "running_time_variation":
                continue
            target = event["target"]
            if target["kind"] == "vehicle" and target["id"] != vehicle_id:
                continue
            seconds = event["parameters"]["additional_seconds"]
            if not seconds:
                continue
            share = (shares.get(str(target["id"]), 0.) if target["kind"] == "segment"
                     else sum(shares.values())) if mapped else 0
            if mapped and not share:
                continue
            if mapped:
                self.reasons.append("runtime_segment_share_allowance")
            else:
                self.reasons.extend(("runtime_global_once_fallback",
                    "runtime_fallback_assumption:each_unmapped_event_once_per_operation_not_per_edge"))
            # Share one ledger across approach/tow/rear legs. An unmapped event
            # contributes one full allowance, not another copy on every leg.
            old_share, fallback = allowances.get(event["event_id"], (0, False))
            new_share, new_fallback = old_share + share, fallback or not mapped
            extra += (self.duration(seconds * float(max(new_share, int(new_fallback))))
                      - self.duration(seconds * float(max(old_share, int(fallback)))))
            allowances[event["event_id"]] = new_share, new_fallback
        return extra

    def dwell(self, vehicle_id):
        vehicle = self.vehicles[vehicle_id]
        if (vehicle.get("depot") is not None or vehicle.get("failed")
                or vehicle.get("status") not in {"IN_SERVICE", "HOLDING"}):
            return 0.
        position = vehicle.get("position")
        at_platform = (position in self.platforms if self.platforms is not None
                       else vehicle.get("at_platform"))
        if at_platform is False:
            return 0.
        events = [e for e in self.events if e["type"] == "dwell_extension" and
                  (e["target"]["kind"], e["target"]["id"]) in
                  (("vehicle", vehicle_id), ("platform", position))]
        if not events:
            return 0.
        groups = {}
        for event in events:
            parameters = event["parameters"]
            group = parameters.get("effect_group", event["event_id"])
            groups[group] = max(groups.get(group, 0.), parameters["additional_seconds"])
        extension = sum(groups.values())
        if "dwell_remaining_seconds" in vehicle:
            # This optional public value already includes published extensions.
            remaining = vehicle["dwell_remaining_seconds"]
        elif "base_dwell_seconds" in vehicle and "dwell_elapsed_seconds" in vehicle:
            remaining = max(0., vehicle["base_dwell_seconds"] + extension - vehicle["dwell_elapsed_seconds"])
        elif "dwell_min_limit" in vehicle and "dwell_time" in vehicle:
            remaining = max(0., (vehicle["dwell_min_limit"] - vehicle["dwell_time"]) * self.tick + extension)
        else:
            # Native dispatch snapshots currently omit dwell timers. Do not
            # reach into live agents; allow this vehicle's full extension once.
            remaining = extension
            self.reasons.append("dwell_remaining_unknown:full_matching_extension_once")
        if at_platform is None:
            self.reasons.append("dwell_location_unknown:assume_current_platform_only")
        return self.duration(remaining)

    def exit_time(self, depot, vehicle_id=None):
        spec = self.depots.get(depot)
        if spec is None:
            return None
        paths = [self.path(spec["node_id"], target, depot, open_only=True)
                 for target in spec.get("exit_node_ids", [spec["exit_node_id"]])]
        durations = [self.travel(path, start=spec["node_id"], vehicle_id=vehicle_id)
                     for path in paths if path is not None]
        # Allow any declared reachable exit, rather than assuming the fastest.
        return max(durations) if durations else None

    def clearance(self, vehicle):
        seconds = max(self.config.get("clearance_seconds", 30.), vehicle.get("dwell_remaining_seconds", 0.))
        load = vehicle.get("passenger_count", 0.)
        rate = self.clearance_rate
        if load and rate is None:
            self.unknown = True
            self.reasons.append("passenger_clearance_rate_unknown")
        elif rate is not None:
            seconds = max(seconds, load / _seconds(rate, "alighting rate", positive=True))
            if load and self.boarding_rate_fallback:
                self.reasons.append("passenger_clearance_rate:public_boarding_default")
        return self.duration(seconds)

    def rescue_route_open(self, start, path):
        edges = list(zip([start, *path], path))
        for origin, target in edges:
            segment = self.edge_segments.get((origin, target), (None,))[0]
            if not self.view.edge_allowed("", origin, target, segment,
                    self.entry_routes.get((origin, target), ()), allow_immobilized=True):
                return False
        if any(edge not in self.edge_segments for edge in edges) and any(
                e["type"] == "track_blockage" or e["type"] == "power_restriction"
                and not e["parameters"]["entry_allowed"] for e in self.events):
            self.reasons.append("rescue_route_unverified_under_published_blockage")
            return False
        return True

    def section_duration(self, vehicle_id, path, *, phase="CLEARING", remaining=0., turnback_entry=None):
        vehicle = self.vehicles[vehicle_id]
        start = vehicle["position"]
        seconds = self.travel(path, start=start, vehicle_id=vehicle_id)
        if seconds is None:
            return None
        seconds += self.duration(remaining)
        if phase == "SERVICE_CLEARING":
            seconds += self.duration(self.config.get("turnback_seconds", 60.))
        # Account for both passenger stops and the clearance/turning phases of
        # the finite round trip. Future boarding and competition remain unknown.
        for i, (origin, target) in enumerate(zip([start, *path], path)):
            if origin not in (self.platforms or {}):
                continue
            node = self.platforms[origin]
            baseline = getattr(node, "dwell_seconds", None) or self.config.get("clearance_seconds", 30.)
            dwell = self.duration(baseline + self.view.dwell_addition(vehicle_id, origin))
            segment = self.edge_segments.get((origin, target), (None,))[0]
            if segment in self.turnback_segments:
                if i == 0 and turnback_entry == origin:
                    continue
                clear = self.clearance(dict(vehicle, dwell_remaining_seconds=dwell))
                seconds += clear + self.duration(self.config.get("turnback_seconds", 60.))
            else:
                seconds += (self.duration(vehicle.get("dwell_remaining_seconds", dwell))
                            if i == 0 and phase == "RUNNING" else dwell)
        return seconds

    def prospective_section(self, vehicle_id):
        vehicle = self.vehicles[vehicle_id]
        if (vehicle.get("depot") is not None or vehicle.get("failed") or vehicle.get("needs_inspection")
                or vehicle.get("position") not in (self.platforms or {})):
            return None
        choices = []
        for sid, task in self.snapshot.get("services", {}).items():
            stops = task.get("stops")
            if (task.get("purpose") != "section_service" or not stops
                    or vehicle["position"] not in (stops[:-1] if task.get("pattern") else [stops[0]])
                    or task.get("vehicle_id") is not None or task.get("reserved_vehicle_id") is not None
                    or task.get("started_at") is not None or task.get("ended_at") is not None or task.get("completed")):
                continue
            if task.get("pattern"):
                cycle = list(stops[:-1])
                index = cycle.index(vehicle["position"])
                stops = cycle[index:] + cycle[:index] + [vehicle["position"]]
            path = []
            for origin, target in zip(stops, stops[1:]):
                part = self.path(origin, target)
                if not part:
                    break
                path.extend(part)
            else:
                if self.rescue_route_open(vehicle["position"], path):
                    seconds = self.section_duration(vehicle_id, path, remaining=self.clearance(vehicle))
                    if seconds is not None:
                        choices.append((seconds, sid))
        return min(choices) if choices else None

    def ongoing(self, op):
        phase, action = op["phase"], op["action"]
        vehicle = self.vehicles[op["vehicle_id"]]
        remaining = self.duration(op.get("remaining_seconds", 0.))
        if action == "inspect_repair":
            spec = self.depots[op["depot"]]
            path = self.exit_time(op["depot"], op["vehicle_id"])
            return None if path is None else remaining + self.duration(spec["preparation_seconds"]) + path
        if (action == "reassign_service" and
                self.snapshot.get("services", {}).get(op.get("service_id"), {}).get("purpose") == "section_service"):
            return self.section_duration(op["vehicle_id"], op.get("path", ()), phase=phase,
                                         remaining=remaining, turnback_entry=op.get("turnback_entry"))
        if action != "rescue_tow":
            path = self.travel(op.get("path", ()), start=vehicle["position"], vehicle_id=op["vehicle_id"])
            dwell = self.dwell(op["vehicle_id"]) if op.get("path") else 0.
            delay = self.duration(vehicle.get("travel_delay_steps", 0.) * self.tick) if op.get("path") else 0.
            turning = self.duration(self.config.get("turnback_seconds", 60.)) if (
                action == "short_turn" and phase == "CLEARING") else 0.
            return max(remaining, dwell) + path + delay + turning
        failed = self.vehicles[op["partner"]]
        allowances = {}
        tow_path = op.get("continuation", ()) if phase in {"CLEARING_RESCUER", "APPROACH", "COUPLING"} else op.get("path", ())
        approach = self.travel(op.get("path", ()), start=vehicle["position"], vehicle_id=op["vehicle_id"],
                               allowances=allowances) if phase in {"CLEARING_RESCUER", "APPROACH"} else 0.
        coupling = self.duration(self.config.get("coupling_seconds", 60.)) if phase in {"CLEARING_RESCUER", "APPROACH"} else 0.
        clearing = self.clearance(failed) if phase in {"CLEARING_RESCUER", "APPROACH", "COUPLING", "TOW"} and failed.get("passenger_count", 0.) else 0.
        delay = self.duration(max(vehicle.get("travel_delay_steps", 0.),
                                  failed.get("travel_delay_steps", 0.)) * self.tick)
        # One extra towing edge lets the rescuer's rear footprint enter storage.
        tow_time = self.travel(tow_path, start=failed["position"], vehicle_id=op["partner"],
                               tow=True, allowances=allowances)
        rear_start = tow_path[-2] if len(tow_path) > 1 else failed["position"]
        if phase == "CLEAR_REAR":
            rear_start = vehicle["position"]
        rear = self.travel([self.depots[op["depot"]]["node_id"]], start=rear_start,
                           vehicle_id=op["vehicle_id"], tow=True, allowances=allowances)
        return remaining + approach + coupling + clearing + delay + tow_time + rear

    def rescue(self, failed_id):
        failed = self.vehicles.get(failed_id)
        if failed is None:
            return None
        busy = {key for op in self.operations for key in (op.get("vehicle_id"), op.get("partner"))}
        reported_failed = {e["target"]["id"] for e in self.events if e["type"] == "vehicle_breakdown"}
        candidates = []
        for key, vehicle in self.vehicles.items():
            if key == failed_id or key in busy or key in reported_failed or vehicle.get("failed") or vehicle.get("needs_inspection"):
                continue
            origin_depot = vehicle.get("depot")
            if origin_depot and any(e["type"] == "depot_access_restriction"
                    and e["target"]["id"] == origin_depot and not e["parameters"]["exit_allowed"] for e in self.events):
                continue
            approach = self.path(vehicle["position"], failed["position"], origin_depot)
            if not approach or not self.rescue_route_open(vehicle["position"], approach):
                continue
            corridor = set(approach) - {failed["position"]}
            if any(other.get("depot") is None and other["position"] in corridor
                   for other_id, other in self.vehicles.items() if other_id not in {key, failed_id}):
                continue
            for depot, spec in self.depots.items():
                stock = set(self.snapshot.get("inventory", {}).get(depot, ()))
                stock.update(self.snapshot.get("depot_bookings", {}).get(depot, ()))
                if len(stock | {key, failed_id}) > spec["capacity"]:
                    continue
                if any(e["type"] == "depot_access_restriction" and e["target"]["id"] == depot
                       and not e["parameters"]["entry_allowed"] for e in self.events):
                    continue
                tow = self.path(failed["position"], spec["node_id"], depot)
                if not tow or not self.rescue_route_open(failed["position"], tow):
                    continue
                preparation = (self.duration(self.depots[origin_depot]["preparation_seconds"])
                               if origin_depot else self.clearance(vehicle))
                allowances = {}
                approach_time = self.travel(list(approach)[:-1], start=vehicle["position"],
                                            vehicle_id=key, allowances=allowances)
                tow_time = self.travel(tow, start=failed["position"], vehicle_id=failed_id,
                                       tow=True, allowances=allowances)
                rear_start = tow[-2] if len(tow) > 1 else failed["position"]
                rear = self.travel([spec["node_id"]], start=rear_start, vehicle_id=key,
                                   tow=True, allowances=allowances)
                candidates.append(preparation + approach_time
                    + self.clearance(failed) + self.duration(self.config.get("coupling_seconds", 60.))
                    + tow_time + rear)
        # This is a conditional duration allowance over currently qualified
        # candidates, not a claim that any joint rescue is immediately feasible.
        return max(candidates) if candidates else None


def choose_horizon(world, *, base=6, maximum=60, tail_stages=6, remaining_stages=None,
                   recovery_work_only=False, prospective_stock_actions=None, focus_vehicles=None):
    """Return a capped, published-only conditional window in decision stages.

    Static reads are limited to dispatch_config (or dispatch.config),
    dispatch.path, public static segment2nodes/platform_nodes/nodes/event_resources, tick_seconds and
    decision_seconds and the public static boarding_rate_persons_per_second
    (the executor's default alighting rate). Paths contain destinations, excluding the origin, as in
    DispatchOperations.path. Durations are rounded
    to physical ticks before conversion to decision stages. Ordinary healthy
    depot stock does not extend the window.
    Healthy reserve/reentry stock does include preparation, directed exit and
    service tail when published events (active or cleared), recovery operations,
    failed/inspection stock or ended public services establish recovery context.
    An unstarted inspection cannot extend the window while that vehicle still
    has an active published breakdown. Waiting for public clearance is censored,
    not an assumed future repair start. Already committed inspections continue
    to use their public remaining duration.
    A public snapshot failed vehicle still on track remains a rescue target even
    after its external breakdown report clears; report clearance is not repair.

    ``requested_stages``/``required_seconds`` are pre-cap estimates. They are
    None when a committed operation's route/duration or an available candidate's
    duration is unknown; then use the largest allowed window and mark censored.
    A reported failure without a current qualified/reachable rescue option does
    NOT request the cap: retain the base/other actionable work window, mark
    censored and report no_current_rescue_option. Do not extend for prospective
    insertion solely in this blocked rescue context. Known ongoing/inspection
    work and available rescues still retain their long duration estimates.
    Otherwise censored means the estimate exceeded maximum (default 60) or the
    episode remainder. Published blocking restrictions never expire by forecast.
    Running-time additions use the traversed fraction of each directed segment;
    without a complete static edge map, each matching event is allowed once per
    operation, with explicit fallback reasons. This assumes at most one affected
    segment per unmapped event, not an exact multi-segment timing reconstruction.
    Dwell extensions are separate current-vehicle waits, never edge travel time.
    If public dwell timers are missing, allow only its matching full extension
    once and report the missing-timer assumption rather than reading live agents.
    Even an uncensored result never guarantees completion under competition.
    With ``recovery_work_only``, ordinary movement/hold commitments do not
    enlarge the window, and spare-stock allowances need a current loss of
    vehicle capacity, failure or demand surge rather than any historical event.
    A feasible finite section-service cycle under current infrastructure
    restrictions includes both turns and passenger stops; ordinary full-line
    reassignment does not request an entire commercial journey. Prospective
    cycles use the shortest currently open choice for each focused vehicle.
    Such commitments remain in the transition and terminal objective.
    An explicit ``prospective_stock_actions`` limits uncommitted stock allowances
    to sampled public opportunities. Committed work is never filtered by it.
    ``focus_vehicles`` restricts duration estimates to the requested decision
    owners and their committed operation partners, not the modeled fleet.
    """
    base = max(6, _integer(base, "base", 1))
    maximum = _integer(maximum, "maximum", 1)
    _integer(tail_stages, "tail_stages")
    if remaining_stages is not None:
        _integer(remaining_stages, "remaining_stages")
    tick = _seconds(world.tick_seconds, "tick_seconds", positive=True)
    decision = _seconds(getattr(world, "decision_seconds", tick), "decision_seconds", positive=True)
    if not math.isclose(decision / tick, round(decision / tick)) or decision < tick:
        raise ValueError("A decision stage must contain whole physical ticks")
    snapshot, events = world.dispatch_snapshot(), _events(world)
    selected = set(snapshot["vehicles"]) if focus_vehicles is None else set(focus_vehicles)
    if not selected <= set(snapshot["vehicles"]):
        raise ValueError("Unknown horizon focus vehicle")
    estimate = _DurationEstimate(world, snapshot, events, tick)
    work, reasons, no_rescue_option = 0., ["base"], False
    active_failures = _active_failures(events)
    inspection_unavailable = False

    def include(seconds, reason):
        nonlocal work
        reasons.append(reason)
        if seconds is None:
            estimate.unknown = True
            reasons.append("directed_path_or_duration_unknown")
        else:
            work = max(work, seconds + tail_stages * decision)

    participants = set()
    for op in estimate.operations:
        if not {op.get("vehicle_id"), op.get("partner")} & selected:
            continue
        participants.update((op.get("vehicle_id"), op.get("partner")))
        if recovery_work_only and op["action"] in {"adjust_departure", "run_profile", "hold", "reroute", "skip_stop"}:
            continue
        if recovery_work_only and op["action"] == "reassign_service":
            task = snapshot.get("services", {}).get(op.get("service_id"), {})
            if task.get("purpose") == "section_service":
                include(estimate.ongoing(op), "ongoing_section_service:" + op["vehicle_id"])
            elif op["phase"] != "RUNNING":
                include(estimate.duration(op.get("remaining_seconds", 0.)),
                        "service_activation:reassign_service:" + op["phase"])
            # Ordinary commercial continuations remain modeled without forcing
            # their full duration into every online coordination request.
            continue
        include(estimate.ongoing(op), "ongoing_operation:" + op["action"] + ":" + op["phase"])
    for key, vehicle in estimate.vehicles.items():
        if key not in selected or key in participants or not vehicle.get("needs_inspection") or vehicle.get("depot") is None:
            continue
        if key in active_failures:
            inspection_unavailable = True
            reasons.append("inspection_not_currently_applicable:active_published_breakdown:" + key)
            continue
        depot = vehicle["depot"]
        spec, exit_seconds = estimate.depots[depot], estimate.exit_time(depot, key)
        include(None if exit_seconds is None else estimate.duration(spec["inspection_seconds"])
                + estimate.duration(spec["preparation_seconds"]) + exit_seconds, "inspection:" + key)
    rescue_targets = active_failures | {key for key, vehicle in estimate.vehicles.items()
                                       if vehicle.get("failed") and vehicle.get("depot") is None}
    for key in sorted(rescue_targets):
        if key not in selected or key in participants:
            continue
        vehicle = estimate.vehicles.get(key, {})
        if vehicle.get("depot") is None:
            seconds = estimate.rescue(key)
            if seconds is None:
                no_rescue_option = True
                reasons.extend(("no_current_rescue_option", "no_current_rescue_option:" + str(key)))
            else:
                reason = "reported_vehicle_breakdown:" if key in active_failures else "unrecovered_track_vehicle:"
                include(seconds, reason + str(key))
        else:
            reasons.append("awaiting_published_failure_clearance:" + str(key))
    stock_needed = (not recovery_work_only or any(
        e.get("active", True) and (e["type"] in {"vehicle_breakdown", "vehicle_capacity_loss", "demand_surge"}
            or e["type"] in {"dwell_extension", "running_time_variation"} and e["target"]["kind"] == "vehicle")
        for e in events) or any(v.get("failed") or v.get("needs_inspection")
                              for v in estimate.vehicles.values()))
    if not no_rescue_option and stock_needed and _recovery_context(snapshot, events):
        busy, failures = _busy_vehicles(snapshot), _active_failures(events)
        insertion_durations = []
        for key, vehicle in estimate.vehicles.items():
            if key not in selected:
                continue
            action = _insertion_action(key, vehicle, busy, failures, events)
            if action is None or prospective_stock_actions is not None and action not in prospective_stock_actions:
                continue
            depot = vehicle["depot"]
            exit_seconds = estimate.exit_time(depot, key)
            duration = (None if exit_seconds is None else
                        estimate.duration(estimate.depots[depot]["preparation_seconds"]) + exit_seconds)
            if recovery_work_only:
                insertion_durations.append((duration, action, key))
            else:
                include(duration, "contextual_insertion:" + action + ":" + key)
        if insertion_durations:
            known = [row for row in insertion_durations if row[0] is not None]
            duration, action, key = min(known, key=lambda row: row[0]) if known else insertion_durations[0]
            include(duration, "shortest_contextual_insertion:" + action + ":" + key)
    for key in estimate.vehicles:
        if key in selected and key not in participants:
            vehicle = estimate.vehicles[key]
            if any(e["type"] in {"track_blockage", "switch_failure", "power_restriction", "turnback_capacity_loss"}
                   for e in estimate.events):
                section = estimate.prospective_section(key)
                if section is not None:
                    seconds, sid = section
                    include(seconds, "prospective_section_service:" + key + ":" + sid)
            impaired = any(e["target"]["kind"] == "vehicle" and e["target"]["id"] == key
                and e["type"] in {"dwell_extension", "running_time_variation", "vehicle_capacity_loss"}
                for e in estimate.events)
            if (impaired and vehicle.get("depot") is None and not vehicle.get("failed")
                    and vehicle.get("position") in (estimate.platforms or ())):
                durations = []
                for depot, spec in estimate.depots.items():
                    if any(e["type"] == "depot_access_restriction" and e["target"]["id"] == depot
                           and not e["parameters"]["entry_allowed"] for e in estimate.events):
                        continue
                    path = estimate.path(vehicle["position"], spec["node_id"], depot, open_only=True)
                    if path and estimate.rescue_route_open(vehicle["position"], path):
                        duration = estimate.travel(path, start=vehicle["position"], vehicle_id=key)
                        if duration is not None:
                            durations.append(estimate.clearance(vehicle) + duration)
                if durations:
                    include(min(durations), "prospective_withdrawal:" + key)
            dwell = estimate.dwell(key)
            if dwell:
                include(dwell, "current_dwell:" + key)
    required = max(base * decision, work)
    requested = math.ceil(required / decision)
    cap = min(maximum, remaining_stages) if remaining_stages is not None else maximum
    horizon = cap if estimate.unknown else min(requested, cap)
    if requested > maximum or estimate.unknown:
        reasons.append("maximum_cap")
    if remaining_stages is not None and (requested > remaining_stages or estimate.unknown):
        reasons.append("episode_remaining_cap")
    return dict(horizon=horizon, requested_stages=None if estimate.unknown else requested,
                required_seconds=None if estimate.unknown else required,
                reasons=list(dict.fromkeys(reasons + estimate.reasons + [_CONDITIONAL])),
                censored=estimate.unknown or no_rescue_option or inspection_unavailable or horizon < requested)
