"""Executed station visits, operation events and half-open resource intervals."""
import copy
import csv
import json
from pathlib import Path

from dispatch_timing import timing_manifest


def clock(seconds, start=21600):
    if seconds is None:
        return ""
    value = int(seconds) + int(start)
    return f"{value//3600:02d}:{value//60%60:02d}:{value%60:02d}"


class ExecutionRecords:
    def __deepcopy__(self, memo):
        # Planning/search clones must not record hypothetical events as executed.
        return None

    def __init__(self, world):
        self.visits, self.open_visits, self.resource_intervals = [], {}, []
        self.resources = {}
        self.log_index = len(world.dispatch.log)
        self.operations = []
        self.timing = timing_manifest(world)
        self.start_clock = world.clock_start_seconds
        self.previous = {v.id: world.dispatch.position(v.id) for v in world.dispatch.records.values()}
        for v in world.dispatch.records.values():
            if v.depot is None and self.previous[v.id] in world.platform_nodes:
                self.arrive(world, v, self.previous[v.id], float(world.sim_time[0]), initial=True)
        self.observe_resources(world)

    def arrive(self, world, vehicle, node, time, initial=False):
        visit = dict(vehicle=vehicle.id, service_id=vehicle.service_id or "", platform=node,
                     station=getattr(world.nodes[node], "station_name", ""), arrival_seconds=time,
                     departure_seconds=None, initial_state=initial,
                     passenger_service=world.dispatch.passenger_service(vehicle.agent, node))
        self.visits.append(visit)
        self.open_visits[vehicle.id] = visit

    def observe(self, world):
        now, tick = float(world.sim_time[0]), float(world.tick_seconds)
        for vehicle in world.dispatch.records.values():
            old, new = self.previous[vehicle.id], world.dispatch.position(vehicle.id)
            if old != new:
                visit = self.open_visits.pop(vehicle.id, None)
                if visit is not None:
                    visit["departure_seconds"] = now - tick
                    visit["departure_service_id"] = vehicle.service_id or ""
                    visit["departure_passenger_service"] = world.dispatch.passenger_service(vehicle.agent, old)
                if new in world.platform_nodes and vehicle.depot is None:
                    self.arrive(world, vehicle, new, now)
            self.previous[vehicle.id] = new
        self.operations.extend(copy.deepcopy(world.dispatch.log[self.log_index:]))
        self.log_index = len(world.dispatch.log)
        self.observe_resources(world)

    def observe_resources(self, world):
        now = float(world.sim_time[0])
        active = {}
        for v in world.dispatch.records.values():
            if v.depot is not None:
                claims = [("depot", v.depot)]
            else:
                node = world.dispatch.position(v.id)
                claims = [("node", str(node))]
                claims += [("reservation", str(n)) for n in world._reservations[v.agent][0]]
                claims += [("turnback_receiver", str(platform))
                           for platform, owner in world.dispatch.turnback_receivers.items() if owner == v.id]
                for segment in world.base_segments:
                    if node in world._segment_interiors_for_records[segment.id]:
                        claims.append(("segment", segment.id))
                        claims.extend(("turnback", key) for key, spec in world.event_resources.get("turnbacks", {}).items()
                                      if segment.id in spec["segments"])
                        claims.extend(("depot_access", key) for key, spec in world.event_resources.get("shared_routes", {}).items()
                                      if segment.id in spec["segments"])
            for kind, resource in claims:
                active[v.id, kind, resource] = v.service_id or ""
        for key in set(self.resources) - set(active):
            row = self.resources.pop(key)
            if now > row["start_seconds"]:
                self.resource_intervals.append(dict(row, end_seconds=now))
        for key, service in active.items():
            if key in self.resources and self.resources[key]["service_id"] != service:
                row = self.resources.pop(key)
                if now > row["start_seconds"]:
                    self.resource_intervals.append(dict(row, end_seconds=now))
            if key not in self.resources:
                vehicle, kind, resource = key
                self.resources[key] = dict(vehicle=vehicle, service_id=service, resource_type=kind,
                                           resource_id=resource, start_seconds=now)

    def write(self, directory, world):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        rows = [dict(v, arrival_clock=clock(v["arrival_seconds"], self.start_clock),
                     departure_clock=clock(v["departure_seconds"], self.start_clock)) for v in self.visits]
        fields = ("vehicle", "service_id", "platform", "station", "arrival_seconds", "departure_seconds",
                  "arrival_clock", "departure_clock", "initial_state", "passenger_service",
                  "departure_service_id", "departure_passenger_service")
        for name in ("station_events.csv", "timetable.csv"):
            with (directory/name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fields)
                writer.writeheader()
                writer.writerows(rows)
        now = float(world.sim_time[0])
        resources = self.resource_intervals + [dict(row, end_seconds=now) for row in self.resources.values()
                                               if now > row["start_seconds"]]
        for row in resources:
            kind = "node" if row["resource_type"] == "reservation" else row["resource_type"]
            row["physical_resource_id"] = f"{kind}:{row['resource_id']}"
        with (directory/"resource_occupancy.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, ("vehicle", "service_id", "resource_type", "resource_id", "physical_resource_id", "start_seconds", "end_seconds"))
            writer.writeheader()
            writer.writerows(resources)
        with (directory/"operation_events.jsonl").open("w", encoding="utf-8") as stream:
            for row in self.operations:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
        # Successful replay does not imply that passenger service was provided.
        from dispatch_diagnostics import service_diagnostics
        events = getattr(world, "_configured_events", None) or []
        report_times = [e["first_report_seconds"] for e in events
                        if e["type"] not in {"observation_delay", "recovery_prediction_revision"}]
        health = service_diagnostics(rows, self.operations, end_seconds=now,
                                     first_report_seconds=min(report_times, default=None))
        (directory/"operational_health.json").write_text(json.dumps(health, ensure_ascii=False, indent=2)+"\n")
        (directory/"timing.json").write_text(json.dumps(dict(self.timing, end_seconds=now,
            interval_convention="[start_seconds, end_seconds)",
            initial_arrival="initial-state boundary, not an observed arrival", censored_departure="blank at episode end",
            resource_interpretation="physical footprints and actual forward/rear reservation cells; not a full-section mutex"), indent=2)+"\n")


def enable_execution_records(world):
    world._segment_interiors_for_records = {s.id: {n.id for n in world.segment2nodes[s][1:-1]}
                                            for s in world.base_segments}
    world.execution_records = ExecutionRecords(world)
    return world.execution_records
