from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from topology.node_type import NodeType
from topology.route import Route, RouteType
from tr.linesegment import LineSegment
from logger import logger


@dataclass(frozen=True)
class TimetableStop:
    platform_id: int
    arrival_time: int
    departure_time: int
    platform_name: Optional[str] = None
    segment: Optional[LineSegment] = None

    @property
    def dwell_time(self) -> int:
        return max(0, self.departure_time - self.arrival_time)

    def shifted(self, delta: int) -> "TimetableStop":
        return TimetableStop(
            platform_id=self.platform_id,
            arrival_time=self.arrival_time + delta,
            departure_time=self.departure_time + delta,
            platform_name=self.platform_name,
            segment=self.segment,
        )


@dataclass
class TimetableRun:
    run_id: Union[int, str]
    start_time: int
    stops: List[TimetableStop]
    cycle_time: int

    def duration(self) -> int:
        return self.cycle_time

    def next_departure_after(self, timestamp: int) -> Optional[TimetableStop]:
        if not self.stops or self.cycle_time <= 0:
            return None

        if timestamp <= self.stops[-1].departure_time:
            for stop in self.stops:
                if timestamp <= stop.departure_time:
                    return stop

        elapsed = max(0, timestamp - self.start_time)
        cycles_completed, remainder = divmod(elapsed, self.cycle_time)
        for stop in self.stops:
            offset = stop.departure_time - self.start_time
            if remainder <= offset:
                return stop.shifted(cycles_completed * self.cycle_time)

        return self.stops[0].shifted((cycles_completed + 1) * self.cycle_time)


@dataclass
class Timetable:
    route_id: Union[int, str]
    runs: List[TimetableRun]

    def all_departures(self) -> Iterable[Tuple[Union[int, str], TimetableStop]]:
        for run in self.runs:
            for stop in run.stops:
                yield run.run_id, stop

    def next_departure_after(self, timestamp: int) -> Optional[Tuple[Union[int, str], TimetableStop]]:
        best: Optional[Tuple[Union[int, str], TimetableStop]] = None
        for run in self.runs:
            stop = run.next_departure_after(timestamp)
            if stop is None:
                continue
            if best is None or stop.departure_time < best[1].departure_time:
                best = (run.run_id, stop)
        return best


def _platform_sequence(route: Route) -> List:
    if route.route_type != RouteType.CIRCULAR:
        raise ValueError(f"Route {route.id} must be circular.")
    if route.node_list:
        return list(route.node_list)
    if getattr(route, "nodes", None):
        platforms = []
        for node in route.nodes:
            node_type = getattr(node, "node_type", None)
            if node_type is None:
                continue
            if node_type == NodeType.PLATFORM:
                platforms.append(node)
        return platforms
    raise ValueError(f"Route {route.id} has no platforms defined.")


def _travel_time_lookup(line_segments: Sequence[LineSegment]) -> Dict[Tuple[int, int], int]:
    lookup: Dict[Tuple[int, int], int] = {}
    for segment in line_segments:
        start_id = getattr(segment.start_platform, "id", segment.start_platform)
        end_id = getattr(segment.end_platform, "id", segment.end_platform)
        lookup[(start_id, end_id)] = segment.weight
    return lookup


def _segment_lookup(line_segments: Sequence[LineSegment]) -> Dict[Tuple[int, int], LineSegment]:
    lookup: Dict[Tuple[int, int], LineSegment] = {}
    for segment in line_segments:
        start_id = getattr(segment.start_platform, "id", segment.start_platform)
        end_id = getattr(segment.end_platform, "id", segment.end_platform)
        lookup[(start_id, end_id)] = segment
    return lookup


def build_single_run(
    route: Route,
    line_segments: Sequence[LineSegment],
    start_time: int,
    *,
    dwell_overrides: Optional[Dict[int, int]] = None,
    default_dwell: int = 30,
    run_id: Union[int, str] = 1,
    loops: int = 1,
) -> TimetableRun:
    dwell_overrides = dwell_overrides or {}
    platforms = _platform_sequence(route)
    travel = _travel_time_lookup(line_segments)
    segments = _segment_lookup(line_segments)

    if loops <= 0:
        raise ValueError("loops must be a positive integer")

    loop_stops: List[TimetableStop] = []
    current_time = start_time
    total = len(platforms)

    def resolve(raw):
        return getattr(raw, "id", raw), getattr(raw, "name", None), getattr(raw, "dwell", None)

    for idx, raw in enumerate(platforms):
        pid, pname, pdwell = resolve(raw)
        dwell = dwell_overrides.get(pid, pdwell if pdwell is not None else default_dwell)
        arrival = current_time
        departure = arrival + max(0, dwell)
        next_pid = resolve(platforms[(idx + 1) % total])[0]
        segment = segments.get((pid, next_pid))
        if segment is None:
            raise ValueError(f"Missing segment for {pid}->{next_pid}.")

        travel_time = travel.get((pid, next_pid))
        if travel_time is None:
            raise ValueError(f"Missing travel time for segment {pid}->{next_pid}.")

        loop_stops.append(
            TimetableStop(
                pid,
                arrival,
                departure,
                platform_name=pname,
                segment=segment,
            )
        )

        current_time = departure + travel_time

    cycle_time = current_time - start_time
    if cycle_time <= 0:
        raise ValueError("Computed cycle time must be positive for circular routes.")

    run_stops: List[TimetableStop] = []
    for loop_index in range(loops):
        offset = loop_index * cycle_time
        for stop in loop_stops:
            if offset == 0:
                run_stops.append(stop)
            else:
                run_stops.append(stop.shifted(offset))

    total_cycle_time = cycle_time * loops

    return TimetableRun(
        run_id=run_id,
        start_time=start_time,
        stops=run_stops,
        cycle_time=total_cycle_time,
    )


def build_timetable(
    route: Route,
    line_segments: Sequence[LineSegment],
    *,
    start_times: Optional[Sequence[int]] = None,
    first_start_time: Optional[int] = None,
    headway: Optional[int] = None,
    num_runs: Optional[int] = None,
    dwell_overrides: Optional[Dict[int, int]] = None,
    default_dwell: int = 30,
    loops_per_run: Union[int, Sequence[int]] = 1,
) -> Timetable:
    if start_times is None:
        if first_start_time is None or headway is None:
            raise ValueError("Provide start_times or first_start_time/headway(+optional num_runs).")
        count = num_runs if num_runs is not None else 1
        start_times = [first_start_time + i * headway for i in range(count)]

    if isinstance(loops_per_run, int):
        loops_list = [loops_per_run] * len(start_times)
    elif isinstance(loops_per_run, (list, tuple)):
        loops_list = list(loops_per_run)
        if len(loops_list) != len(start_times):
            raise ValueError("loops_per_run length must match number of runs")
    else:
        raise TypeError("loops_per_run must be an int or a sequence of ints")

    runs = [
        build_single_run(
            route=route,
            line_segments=line_segments,
            start_time=start,
            dwell_overrides=dwell_overrides,
            default_dwell=default_dwell,
            run_id=index,
            loops=int(loops_list[index - 1]),
        )
        for index, start in enumerate(start_times, start=1)
    ]

    if logger:
        logger.info(
            "Timetable created for route %s: %d runs, loops per run=%s",
            getattr(route, "id", None),
            len(runs),
            loops_list,
        )
        for run in runs:
            stops_desc = ", ".join(
                f"{stop.platform_id}(arr={stop.arrival_time},dep={stop.departure_time})"
                for stop in run.stops
            )
            logger.info(
                "  Run %s start=%s cycle=%s stops=[%s]",
                run.run_id,
                run.start_time,
                run.cycle_time,
                stops_desc,
            )
    return Timetable(route_id=getattr(route, "id", None), runs=runs)
