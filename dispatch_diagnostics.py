"""Common read-only diagnostics of executed service, not a feasibility test."""
import csv
import gzip
from collections import defaultdict


def service_diagnostics(visits, operations, *, end_seconds, first_report_seconds=None):
    by_vehicle = defaultdict(list)
    for row in visits:
        by_vehicle[row["vehicle"]].append(row)
    departures = [float(row["departure_seconds"]) for row in visits
                  if row.get("departure_seconds") not in (None, "")]
    early_reserves = [row for row in operations
                     if row.get("event") == "command_started" and row.get("action") == "reserve_insert"
                     and first_report_seconds is not None and float(row["time"]) < first_report_seconds]
    no_departures = sorted(vehicle for vehicle, rows in by_vehicle.items()
                           if not any(row.get("departure_seconds") not in (None, "") for row in rows))
    return dict(schema="executed_service_diagnostics_v1", end_seconds=float(end_seconds),
                recorded_vehicles=len(by_vehicle), station_visits=len(visits),
                departure_events=len(departures), first_departure_seconds=min(departures, default=None),
                vehicles_without_recorded_departure=no_departures,
                zero_departures=not departures, first_report_seconds=first_report_seconds,
                reserve_commands_before_first_report=early_reserves,
                interpretation="Historical observations only; an early reserve is a policy review flag, not proof of a safety violation")


def read_node_trace(path):
    result = defaultdict(list)
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            start, end = float(row["start_seconds"]), float(row["end_seconds"])
            if not start.is_integer() or not end.is_integer() or not 0 <= start < end:
                raise ValueError("Node intervals must use positive integer-second extents")
            result[row["vehicle"]].append((int(start), int(end), int(row["node"]), row["service_id"]))
    for vehicle, rows in result.items():
        rows.sort()
        if any(a[1] > b[0] for a, b in zip(rows, rows[1:])):
            raise ValueError(f"Overlapping physical node intervals for {vehicle}")
    return dict(result)


def trace_segments(rows, geometry, *, end_seconds):
    """Recover dwell and movement from half-open occupancy, without smoothing.

    A node interval ending at t contains its last move during [t-1,t].
    A service-id change at the same node is not a physical departure.
    """
    result = []
    for index, (start, end, node, service) in enumerate(rows):
        if not 0 <= start < end <= end_seconds:
            raise ValueError("A node interval extends beyond the executed episode")
        point = geometry.get(str(node))
        if point is None:
            continue
        following = rows[index + 1] if index + 1 < len(rows) else None
        changes_node = following is not None and following[0] == end and following[2] != node
        stop_end = end - 1 if changes_node else end
        if stop_end > start:
            result.append(dict(start=start, end=stop_end, y0=point["y"], y1=point["y"],
                               direction=point["direction"], node=node, kind="station" if point["platform"] else "section_stop"))
        if not changes_node:
            continue
        successor = following[2]
        next_point = geometry.get(str(successor))
        if successor not in point["successors"]:
            raise ValueError(f"Recorded movement is not a directed edge: {node}->{successor}")
        if next_point is not None:
            result.append(dict(start=end-1, end=end, y0=point["y"], y1=next_point["y"],
                               direction=point["direction"], node=node, kind="move"))
    return result


def long_station_links(visits, traces=None, *, threshold_seconds=600):
    by_vehicle = defaultdict(list)
    for row in visits:
        by_vehicle[row["vehicle"]].append(row)
    result = []
    for vehicle, rows in by_vehicle.items():
        rows.sort(key=lambda row: float(row["arrival_seconds"]))
        for before, after in zip(rows, rows[1:]):
            if before.get("departure_seconds") in (None, "") or before["service_id"] != after["service_id"]:
                continue
            start, end = float(before["departure_seconds"]), float(after["arrival_seconds"])
            if end-start <= threshold_seconds or before["platform"] == after["platform"]:
                continue
            stops = [max(0, min(end, b-1)-max(start, a)) for a, b, _, _ in (traces or {}).get(vehicle, ())]
            result.append(dict(vehicle=vehicle, origin=before["station"], destination=after["station"],
                               origin_platform=int(before["platform"]), destination_platform=int(after["platform"]),
                               departure_seconds=start, arrival_seconds=end, elapsed_seconds=end-start,
                               longest_node_stop_seconds=max(stops, default=None),
                               stop_seconds=sum(stops) if stops else None))
    return sorted(result, key=lambda row: (-row["elapsed_seconds"], row["vehicle"]))
