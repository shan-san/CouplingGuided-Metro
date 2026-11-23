"""Independent delay records and effective inputs for the existing MetroRL model.

The simulator queries physical records. Online controllers receive an immutable
published projection, which contains neither private clearance times nor future
reports. This module changes data/constraints, not the scheduling architecture.
"""
from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


EVENT_TARGETS = {
    "dwell_extension": {"platform", "vehicle"},
    "running_time_variation": {"segment", "vehicle"},
    "speed_restriction": {"segment"},
    "track_blockage": {"segment"},
    "platform_closure": {"platform"},
    "station_capacity_reduction": {"platform"},
    "switch_failure": {"switch"},
    "turnback_capacity_loss": {"turnback"},
    "signal_degradation": {"segment"},
    "power_restriction": {"segment"},
    "vehicle_capacity_loss": {"vehicle"},
    "vehicle_breakdown": {"vehicle"},
    "depot_access_restriction": {"depot"},
    "demand_surge": {"passenger_origin"},
    "od_shift": {"passenger_origin"},
    "boarding_rate_reduction": {"platform"},
    "recovery_prediction_revision": {"segment", "vehicle", "depot", "platform", "switch", "turnback", "passenger_origin"},
    "observation_delay": {"segment", "vehicle", "depot", "platform", "switch", "turnback", "passenger_origin"},
}
INFORMATION_EVENTS = {"observation_delay", "recovery_prediction_revision"}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_events(events):
    """Validate a complete batch, including references between event instances."""
    lookup = {}
    for e in events:
        eid, kind = e.get("event_id"), e.get("type")
        _require(isinstance(eid, str) and bool(eid) and eid not in lookup, "Missing or duplicate event ID")
        _require(kind in EVENT_TARGETS, f"Unknown event type: {kind}")
        target, p = e.get("target", {}), e.get("parameters", {})
        _require(target.get("kind") in EVENT_TARGETS[kind] and isinstance(target.get("id"), (str, int))
                 and not isinstance(target.get("id"), bool), f"Invalid target for {eid}")
        _require(isinstance(p, dict), f"Invalid parameters for {eid}")
        start, end, report = (e.get(k) for k in ("start_seconds", "actual_end_seconds", "first_report_seconds"))
        _require(all(_number(v) for v in (start, end, report)) and 0 <= start < end and report >= start,
                 f"Invalid event interval/report for {eid}")
        if "effect_group" in p:
            _require(isinstance(p["effect_group"], str) and bool(p["effect_group"]), "Invalid effect_group")
        def bounded(key, low, high=math.inf, integer=False):
            v = p.get(key)
            _require(_number(v) and low <= v <= high and (not integer or isinstance(v, int)), f"Invalid {key} in {eid}")
        if kind in {"dwell_extension", "running_time_variation"}:
            bounded("additional_seconds", 0)
        if kind in {"speed_restriction", "signal_degradation", "power_restriction"}:
            bounded("travel_time_multiplier", 1)
        if kind == "signal_degradation":
            bounded("minimum_safety_headway_seconds", 0)
        if kind == "power_restriction":
            _require(isinstance(p.get("entry_allowed"), bool), f"Missing power entry permission in {eid}")
        if kind == "station_capacity_reduction":
            bounded("safe_queue_capacity_persons", 0, integer=True)
        if kind == "vehicle_capacity_loss":
            bounded("passenger_capacity", 0, integer=True)
        if kind == "turnback_capacity_loss":
            bounded("capacity", 0, integer=True)
        if kind == "boarding_rate_reduction":
            bounded("boarding_rate_multiplier", 0, 1)
        if kind == "demand_surge":
            bounded("arrival_intensity_multiplier", 1)
        if kind == "od_shift":
            ds = p.get("preferred_destination_ids")
            _require(isinstance(ds, list) and bool(ds) and len(ds) == len(set(ds)) and target["id"] not in ds,
                     f"Invalid OD destinations in {eid}")
            bounded("weight_multiplier", 0)
        if kind == "vehicle_breakdown":
            _require(p.get("mode") in {"immobilized", "can_limp_to_depot", "unavailable_in_depot"}, f"Invalid failure mode in {eid}")
        if kind == "switch_failure":
            _require(isinstance(p.get("disabled_route_ids"), list) and bool(p["disabled_route_ids"]), f"Missing disabled routes in {eid}")
        if kind == "depot_access_restriction":
            _require(all(isinstance(p.get(k), bool) for k in ("entry_allowed", "exit_allowed")), f"Invalid depot permissions in {eid}")
        if kind in INFORMATION_EVENTS:
            _require(isinstance(p.get("references_event_id"), str), f"Missing event reference in {eid}")
        if kind == "observation_delay":
            bounded("delay_seconds", 0)
        previous = report
        for r in e.get("recovery_reports", []):
            t, end_estimate = r.get("published_seconds"), r.get("predicted_end_seconds")
            _require(_number(t) and _number(end_estimate) and t >= previous and end_estimate >= t, f"Invalid recovery report in {eid}")
            previous = t
        lookup[eid] = e
    for e in events:
        if e["type"] not in INFORMATION_EVENTS:
            continue
        ref = lookup.get(e["parameters"]["references_event_id"])
        _require(ref is not None and ref["type"] not in INFORMATION_EVENTS and ref["target"] == e["target"],
                 "Information event must reference a matching physical event")
        _require(e["first_report_seconds"] >= ref["first_report_seconds"], "Information metadata would reveal an unreported event")
        if e["type"] == "observation_delay":
            _require(ref["first_report_seconds"] == ref["start_seconds"] + e["parameters"]["delay_seconds"],
                     "Observation delay contradicts the physical report time")


class EventView:
    """Effective parameter algebra over a frozen set of currently known events."""
    def __init__(self, records=(), resources=None):
        self.records = copy.deepcopy(list(records))
        self.resources = copy.deepcopy(resources or {})

    def active(self, kind=None, target_kind=None, target=None):
        return [e for e in self.records if e.get("active", True)
                and (kind is None or e["type"] == kind)
                and (target_kind is None or e["target"]["kind"] == target_kind)
                and (target is None or e["target"]["id"] == target)]

    @staticmethod
    def _independent_sum(records, key):
        # Repeated measurements of one effect use a common effect_group.
        # Different groups represent independent incremental process delays.
        groups = defaultdict(float)
        for e in records:
            group = e["parameters"].get("effect_group", e["event_id"])
            groups[group] = max(groups[group], e["parameters"][key])
        return sum(groups.values())

    def get_segment_disruption_info(self, segment_id, vehicle_id=None):
        es = self.active(target_kind="segment", target=str(segment_id))
        runtime = [e for e in es if e["type"] == "running_time_variation"]
        runtime += self.active("running_time_variation", "vehicle", vehicle_id) if vehicle_id else []
        blocked = any(e["type"] == "track_blockage" or e["type"] == "power_restriction"
                      and not e["parameters"]["entry_allowed"] for e in es)
        factor = max([1.] + [e["parameters"]["travel_time_multiplier"] for e in es
                             if e["type"] in {"speed_restriction", "signal_degradation", "power_restriction"}])
        if not es and not runtime:
            return None
        reports = [e.get("reported_at") for e in es if e.get("reported_at") is not None]
        estimates = [e.get("predicted_end_seconds") for e in es if e.get("predicted_end_seconds") is not None]
        return dict(segment_id=str(segment_id), type="BLOCK" if blocked else "SPEED_LIMIT", factor=factor,
                    running_time_add_seconds=self._independent_sum(runtime, "additional_seconds"),
                    event_ids=sorted({e["event_id"] for e in es + runtime}), is_active=True,
                    reported_at=min(reports) if reports else None,
                    predicted_end_seconds=max(estimates) if estimates else None)

    def dwell_addition(self, vehicle, platform):
        es = [e for e in self.active("dwell_extension") if (e["target"]["kind"], e["target"]["id"])
              in (("vehicle", vehicle), ("platform", platform))]
        return self._independent_sum(es, "additional_seconds")

    def queue_capacity(self, platform, base=math.inf):
        return min([base] + [e["parameters"]["safe_queue_capacity_persons"] for e in self.active("station_capacity_reduction", "platform", platform)])

    def vehicle_capacity(self, vehicle, base):
        return min([base] + [e["parameters"]["passenger_capacity"] for e in self.active("vehicle_capacity_loss", "vehicle", vehicle)])

    def boarding_multiplier(self, platform):
        return min([1.] + [e["parameters"]["boarding_rate_multiplier"] for e in self.active("boarding_rate_reduction", "platform", platform)])

    def boarding_allowed(self, platform, vehicle=None):
        return not self.active("platform_closure", "platform", platform) and (vehicle is None or not self.active("vehicle_breakdown", "vehicle", vehicle))

    def vehicle_operable(self, vehicle):
        return all(e["parameters"]["mode"] == "can_limp_to_depot" for e in self.active("vehicle_breakdown", "vehicle", vehicle))

    def demand_multiplier(self, platform):
        groups = defaultdict(lambda: 1.)
        for e in self.active("demand_surge", "passenger_origin", platform):
            group = e["parameters"].get("effect_group", e["event_id"])
            groups[group] = max(groups[group], e["parameters"]["arrival_intensity_multiplier"])
        return math.prod(groups.values())

    def od_weights(self, platform, base):
        weights = dict(base)
        for e in self.active("od_shift", "passenger_origin", platform):
            for d in e["parameters"]["preferred_destination_ids"]:
                if d in weights:
                    weights[d] *= e["parameters"]["weight_multiplier"]
        total = sum(weights.values())
        if total <= 0:
            weights, total = dict(base), sum(base.values())
        return {d: w / total for d, w in weights.items()} if total > 0 else {}

    def safety_headway(self, segment, base=30.):
        return max([base] + [e["parameters"]["minimum_safety_headway_seconds"] for e in self.active("signal_degradation", "segment", segment)])

    def turnback_capacity(self, resource, base=1):
        return min([base] + [e["parameters"]["capacity"] for e in self.active("turnback_capacity_loss", "turnback", resource)])

    def edge_allowed(self, vehicle, origin, dest, segment, route_ids=(), *, allow_immobilized=False):
        if (not allow_immobilized and not self.vehicle_operable(vehicle)) or self.active("platform_closure", "platform", dest):
            return False
        info = self.get_segment_disruption_info(segment, vehicle) if segment else None
        if info and info["type"] == "BLOCK":
            return False
        disabled_here = set()
        for e in self.active("switch_failure"):
            spec = self.resources.get("switches", {}).get(e["target"]["id"], {})
            entries = spec.get("route_entry_nodes", {})
            disabled_here.update(rid for rid in e["parameters"]["disabled_route_ids"]
                                 if origin == entries.get(str(rid), spec.get("node_id")))
        if route_ids and set(route_ids) <= disabled_here:
            return False
        for rid, spec in self.resources.get("turnbacks", {}).items():
            if (segment in spec["segments"] and self.turnback_capacity(rid, spec.get("capacity", 1)) == 0
                    and origin not in spec.get("interior_nodes", [])):
                return False
        for e in self.active("depot_access_restriction"):
            spec = self.resources.get("depots", {}).get(e["target"]["id"], {})
            if segment in spec.get("entry_segments", []) and not e["parameters"]["entry_allowed"]:
                return False
            if segment in spec.get("exit_segments", []) and not e["parameters"]["exit_allowed"]:
                return False
        return True


class DelayEvents:
    """Backward-compatible LineDisruption storage, keyed by event ID, not object."""
    def __init__(self):
        self.disruptions = {}
        self.current_time = 0.
        self.resources = {}
        self._serial = 0
        self._query_cache = None

    @contextmanager
    def cached_queries(self):
        """Reuse read-only views within execution, invalidated at every clock update."""
        previous = self._query_cache
        self._query_cache = {}
        try:
            yield
        finally:
            if previous is not None:
                previous.clear()
            self._query_cache = previous

    def _view(self, kind, records):
        cache = self._query_cache
        if cache is None:
            return EventView(records(), self.resources)
        key = (kind, self.current_time, id(self.resources))
        if key not in cache:
            cache[key] = EventView(records(), self.resources)
        return cache[key]

    def _invalidate_views(self):
        if self._query_cache is not None:
            self._query_cache.clear()

    @staticmethod
    def _segment_key(segment):
        return str(getattr(segment, "id", segment))

    def clear(self):
        self.disruptions.clear()
        self.current_time = 0.
        self._invalidate_views()

    def add_events(self, events):
        batch = copy.deepcopy(list(events))
        validate_events(list(self.disruptions.values()) + batch)
        self.disruptions.update({e["event_id"]: e for e in batch})
        self._invalidate_views()

    def add_event(self, **event):
        self.add_events([event])
        return event["event_id"]

    def add_disruption(self, segment_id, start_time, end_time, disruption_type="BLOCK", factor=None):
        _require(disruption_type in {"BLOCK", "SPEED_LIMIT"}, "Use add_event for typed events")
        self._serial += 1
        eid = f"legacy_{self._serial}"
        while eid in self.disruptions:
            self._serial += 1
            eid = f"legacy_{self._serial}"
        self.add_event(event_id=eid, type="track_blockage" if disruption_type == "BLOCK" else "speed_restriction",
                       target=dict(kind="segment", id=self._segment_key(segment_id)),
                       start_seconds=start_time, actual_end_seconds=end_time + 1,
                       first_report_seconds=start_time,
                       parameters={} if disruption_type == "BLOCK" else {"travel_time_multiplier": factor}, recovery_reports=[])
        return eid

    def update_disruptions(self, current_time):
        _require(_number(current_time) and current_time >= 0, "Invalid event clock")
        self.current_time = float(current_time)
        self._invalidate_views()

    def physical_view(self):
        return self._view("physical", lambda: [e for e in self.disruptions.values()
                          if e["type"] not in INFORMATION_EVENTS and e["start_seconds"] <= self.current_time < e["actual_end_seconds"]])

    def published_records(self):
        now, result = self.current_time, {}
        messages = []
        for e in self.disruptions.values():
            report = e["first_report_seconds"]
            if report > now:
                continue
            item = {k: copy.deepcopy(e[k]) for k in ("event_id", "type", "target", "parameters")}
            item.update(reported_at=report, active=now < e["actual_end_seconds"] + report - e["start_seconds"],
                        predicted_end_seconds=None, information_version=0)
            result[e["event_id"]] = item
            for r in e.get("recovery_reports", []):
                if r["published_seconds"] <= now:
                    messages.append((r["published_seconds"], e["event_id"], r))
        for _, eid, r in sorted(messages, key=lambda m: (m[0], m[1])):
            recipients = [eid]
            if result[eid]["type"] == "recovery_prediction_revision":
                recipients.append(result[eid]["parameters"]["references_event_id"])
            for rid in recipients:
                if rid in result:
                    result[rid]["predicted_end_seconds"] = r["predicted_end_seconds"]
                    result[rid]["information_version"] += 1
        return list(result.values())

    def published_view(self):
        # An estimate does not confirm restoration. Hold each active reported
        # restriction over the horizon; a later publication causes replanning.
        return self._view("published", self.published_records)

    def get_segment_disruption_info(self, segment_id):
        return self.physical_view().get_segment_disruption_info(self._segment_key(segment_id))

    def is_segment_disrupted(self, segment_id):
        return self.get_segment_disruption_info(segment_id) is not None

    def get_disrupted_segments(self):
        view = self.physical_view()
        ids = {e["target"]["id"] for e in view.records if e["target"]["kind"] == "segment"}
        return {sid: view.get_segment_disruption_info(sid) for sid in sorted(ids)}


def read_event_file(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            _require(k not in result, f"Duplicate JSON key: {k}")
            result[k] = v
        return result
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    data = json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)
    validate_events(data["events"])
    return data
