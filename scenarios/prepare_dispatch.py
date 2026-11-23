"""Generate reproducible scenarios in memory; preview them as readable tables."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from delay_events import EVENT_TARGETS, validate_events
from topology.metro_lines import LINE_SPECS, line_id, dispatch_configuration, event_resources


# Edit combinations and sampling ranges here, not in per-episode data files.
TICK_SECONDS = 30
PHYSICAL_TICK_SECONDS = 1
EPISODE_SECONDS = 7200
EPISODE_STAGES = EPISODE_SECONDS // TICK_SECONDS
TIMELINE = {
    "normal": (0, 900),
    "event_onsets": (900, 2700),
    "partial_clearance": (2700, 4500),
    "service_recovery": (4500, EPISODE_SECONDS),
}
START_SECONDS = (900, 1080, 1260, 1440, 1620)
CLEARANCE_SECONDS = (2700, 3060, 3420, 3780, 4140, 4500)
ONSET_GAPS_SECONDS = (180, 240, 300)
SPLIT_OFFSETS = {"train": 0, "validation": 100000, "test": 200000}
HELDOUT_COMBINATIONS = ("B", "D", "F")
COMBINATIONS = {
    "A": ("track_blockage", "speed_restriction", "platform_closure"),
    "B": ("switch_failure", "speed_restriction", "dwell_extension"),
    "C": ("demand_surge", "od_shift", "station_capacity_reduction", "boarding_rate_reduction"),
    "D": ("power_restriction", "depot_access_restriction", "speed_restriction"),
    "E": ("signal_degradation", "running_time_variation", "dwell_extension"),
    "F": ("vehicle_capacity_loss", "demand_surge", "dwell_extension"),
    "G": ("vehicle_breakdown", "speed_restriction", "depot_access_restriction"),
}
EVENT_LABELS = {
    "dwell_extension": "停站延长", "running_time_variation": "运行时间变化",
    "speed_restriction": "限速", "track_blockage": "区间封锁", "platform_closure": "站台关闭",
    "station_capacity_reduction": "站台候车容量降低", "switch_failure": "道岔失效",
    "turnback_capacity_loss": "折返能力下降", "signal_degradation": "信号能力下降",
    "power_restriction": "供电受限", "vehicle_capacity_loss": "车辆载客能力降低",
    "vehicle_breakdown": "车辆故障", "depot_access_restriction": "车辆段出入受限",
    "demand_surge": "客流突增", "od_shift": "OD分布变化", "boarding_rate_reduction": "上车速率下降",
    "recovery_prediction_revision": "恢复信息更新", "observation_delay": "事件信息延迟",
}


def curriculum(tid, seed, *, start=None, duration=None, fleet=None):
    rng = random.Random(seed)
    n = len(LINE_SPECS[tid]["stations"])
    p = rng.randrange(1, n)
    custom_timing = start is not None or duration is not None
    start = rng.choice(START_SECONDS) if start is None else start
    duration = rng.choice(CLEARANCE_SECONDS) - start if duration is None else duration
    segment = f"{p}-{p+1}"
    resources = event_resources(tid)
    switch = next(iter(resources["switches"]))
    turnback = next(iter(resources["turnbacks"]))
    specs = {
        "dwell_extension": ("vehicle", "agent_0", {"additional_seconds": rng.choice((30, 60, 90))}),
        "running_time_variation": ("segment", segment, {"additional_seconds": 60}),
        "speed_restriction": ("segment", segment, {"travel_time_multiplier": rng.choice((1.5, 2., 2.5, 3.))}),
        "track_blockage": ("segment", segment, {}),
        "platform_closure": ("platform", p+1, {}),
        "station_capacity_reduction": ("platform", p, {"safe_queue_capacity_persons": 100}),
        "switch_failure": ("switch", switch, {"disabled_route_ids": resources["switches"][switch]["route_ids"]}),
        "turnback_capacity_loss": ("turnback", turnback, {"capacity": 0}),
        "signal_degradation": ("segment", segment, {"travel_time_multiplier": 1.5, "minimum_safety_headway_seconds": 60}),
        "power_restriction": ("segment", segment, {"travel_time_multiplier": 2., "entry_allowed": False}),
        "vehicle_capacity_loss": ("vehicle", "agent_0", {"passenger_capacity": 200}),
        "vehicle_breakdown": ("vehicle", "agent_0", {"mode": "immobilized"}),
        "depot_access_restriction": ("depot", "D0", {"entry_allowed": True, "exit_allowed": False}),
        "demand_surge": ("passenger_origin", p, {"arrival_intensity_multiplier": 2.}),
        "od_shift": ("passenger_origin", p, {"preferred_destination_ids": [n], "weight_multiplier": 3.}),
        "boarding_rate_reduction": ("platform", p, {"boarding_rate_multiplier": .3}),
    }
    def record(kind, eid=None):
        target, ident, parameters = specs[kind]
        return dict(event_id=eid or kind, type=kind, target=dict(kind=target, id=ident),
                    parameters=copy.deepcopy(parameters), start_seconds=start,
                    actual_end_seconds=start+duration, first_report_seconds=start, recovery_reports=[])
    cases = {"normal": []}
    for kind in specs:
        cases[f"single_{kind}"] = [record(kind)]
    base = record("track_blockage", "reported_blockage")
    info = dict(event_id="revision", type="recovery_prediction_revision", target=copy.deepcopy(base["target"]),
        parameters=dict(references_event_id=base["event_id"]), start_seconds=start,
        actual_end_seconds=start+duration, first_report_seconds=start+30,
        recovery_reports=[dict(published_seconds=start+30, predicted_end_seconds=start+duration+120)])
    cases["single_recovery_prediction_revision"] = [base, info]
    base = record("track_blockage", "late_blockage")
    base["first_report_seconds"] += 60
    info = dict(event_id="delay", type="observation_delay", target=copy.deepcopy(base["target"]),
        parameters=dict(references_event_id=base["event_id"], delay_seconds=60), start_seconds=start,
        actual_end_seconds=start+duration, first_report_seconds=start+60, recovery_reports=[])
    cases["single_observation_delay"] = [base, info]
    reserve = f"agent_{(fleet or LINE_SPECS[tid]['fleet']) - 2}"
    cases["operation_depot_recovery"] = [dict(event_id="depot_vehicle_failure", type="vehicle_breakdown",
        # The initial reserve may have entered service before this later failure.
        target=dict(kind="vehicle", id=reserve), parameters=dict(mode="immobilized"),
        start_seconds=start, actual_end_seconds=start+duration, first_report_seconds=start, recovery_reports=[])]
    for key, kinds in COMBINATIONS.items():
        records = [record(k) for k in kinds]
        if custom_timing:
            # Explicit overrides keep short, start-at-zero regression fixtures possible.
            for index, item in enumerate(records):
                item["start_seconds"] += TICK_SECONDS*index
                item["first_report_seconds"] = item["start_seconds"]
                item["actual_end_seconds"] += TICK_SECONDS*(len(records)-index)
        else:
            gap = rng.choice(ONSET_GAPS_SECONDS)
            onset_order = rng.sample(range(len(records)), len(records))
            clearance_order = rng.sample(range(len(records)), len(records))
            first_clearance = rng.choice((2700, 2880, 3060))
            last_clearance = rng.choice((4140, 4320, 4500))
            for index, event_index in enumerate(onset_order):
                records[event_index]["start_seconds"] = start+gap*index
                records[event_index]["first_report_seconds"] = start+gap*index
            for index, event_index in enumerate(clearance_order):
                step = (last_clearance-first_clearance)*index // ((len(records)-1)*TICK_SECONDS)
                records[event_index]["actual_end_seconds"] = first_clearance+step*TICK_SECONDS
        # Different affected sections, rather than redundant speed inside a blockage.
        if key == "A":
            records[1]["target"]["id"] = f"{n+2}-{n+1}"
        if key == "D":
            adjacent = p+1 if p < n-1 else p-1
            records[2]["target"]["id"] = f"{adjacent}-{adjacent+1}"
        cases[f"combination_{key}"] = records
    for records in cases.values():
        validate_events(records)
    return cases


def current_arrivals(world, *, published=False):
    """Current-interval inputs; never advance events or read future arrivals."""
    view = world.event_view(published=published)
    from dispatch_timing import decision_seconds
    return {p: n.passenger_flow.interval_arrivals(view, decision_seconds(world))
            for p, n in world.platform_nodes.items()}


def initialize_stock(world):
    """Keep the full fleet, but do not start trains in overlapping protection regions."""
    config = copy.deepcopy(world.dispatch.config)
    n = len(LINE_SPECS[world.topology]["stations"])
    occupied = set()
    for name, spec in config["vehicles"].items():
        pos = spec.get("initial_service_node", spec.get("initial_node"))
        if spec.get("role", "service") != "service" or pos is None:
            continue
        agent = world.dispatch.records[name].agent
        route = 1 if pos <= n else 2
        choices = world.nodes[pos].getnext() or {}
        if route not in choices:
            routes = [rid for rid, target in choices.items() if (pos, target) in world._step_segment
                      and world._step_segment[(pos, target)].end_platform.id
                          not in {d["node_id"] for d in world.dispatch.depots.values()}]
            route = min(routes) if routes else route
        agent.route_adapter._apply_route(agent, 0, route)
        agent.potential_next_pos[0] = pos
        region = world._compute_reservation(agent, 0, pos, route)
        successor = world._get_next_on_route(pos, route)
        if successor is not None:
            region |= world._compute_reservation(agent, 0, successor, route)
        if region & occupied:
            # Layover stock uses an existing reachable depot, including lines
            # with one shared external facility rather than two end depots.
            choices = [(len(path), key) for key, depot in world.dispatch.depots.items()
                       if (path := world.dispatch.path(pos, depot["node_id"], allow_depot=key)) is not None]
            if not choices:
                raise ValueError(f"No reachable layover depot for {name}")
            depot = min(choices)[1]
            spec.pop("initial_node", None)
            spec.update(role="service", depot=depot, has_served=True, initial_service_node=pos)
        else:
            spec.pop("depot", None)
            spec.update(initial_node=pos, initial_service_node=pos)
            occupied |= region
    config["metadata"]["initial_stock"] = dict(total=len(world.agents),
        in_service=sum(v.get("depot") is None for v in config["vehicles"].values()),
        layover=sum(v.get("depot") is not None and v.get("role", "service") == "service" for v in config["vehicles"].values()),
        unused_reserves=sum(v.get("role") == "reserve" for v in config["vehicles"].values()),
        rule="nonoverlapping current and one-step protection regions; excess service stock waits in depots",
        prior_service="stored service vehicles are qualified layover stock; two unused reserve vehicles remain distinct")
    world.dispatch = None
    world.configure_dispatch(config)
    world.refresh_reservations()


def initialize_passengers(world, initial_seconds=0):
    for node in world.platform_nodes.values():
        flow = node.passenger_flow
        flow._init_destination_distribution(world.platform_nodes)
        if initial_seconds:
            flow.add_arrivals(flow.interval_arrivals(world.event_view(), initial_seconds))


def make_world(tid, *, fleet=None, seed=7, events=(), tick=30, initial_passenger_seconds=0):
    import torch
    import linedata
    from metro_scenario_v2 import MetroScenarioV2
    if initial_passenger_seconds < 0:
        raise ValueError("Initial passenger interval must be nonnegative")
    # Scenario seeds must not reset the actor's exploration random stream.
    state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            linedata.load_topology(tid)
            config = dispatch_configuration(tid, fleet)
            scenario = MetroScenarioV2(n_agents=len(config["vehicles"]), use_float_boarding=True)
            world = scenario.env_make_world(1, torch.device("cpu"), topology=tid, default_interval=PHYSICAL_TICK_SECONDS,
                decision_seconds=tick, second_precision=True,
                dispatch_config=config, event_resources=event_resources(tid),
                enable_random_blockages=False, enable_random_speed_limits=False,
                enable_simulation_blockages=False, enable_simulation_speed_limits=False)
            world.load_events(list(events), resources=event_resources(tid))
            initialize_passengers(world, initial_passenger_seconds)
    finally:
        random.setstate(state)
    return world


def scenario_names(split):
    if split not in SPLIT_OFFSETS:
        raise ValueError(f"Unknown scenario split: {split}")
    names = ["normal"] + [f"single_{kind}" for kind in EVENT_TARGETS]
    names += ["operation_depot_recovery"]
    names += [f"combination_{key}" for key in COMBINATIONS
              if split == "test" or key not in HELDOUT_COMBINATIONS]
    return names


def read_scenarios(line, split, scenarios=None, *, repeats=3, seed=111):
    """Return a small generation recipe and descriptors, never event files."""
    tid = line_id(line)
    if not isinstance(repeats, int) or not 1 <= repeats <= 100000:
        raise ValueError("scenario_repeats must be between 1 and 100000 (disjoint split seeds)")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("scenario_seed must be a nonnegative integer")
    names = scenario_names(split)
    unknown = set(scenarios or ()) - set(names)
    if unknown:
        raise ValueError(f"Scenarios unavailable in {split}: {sorted(unknown)}")
    names = [name for name in names if not scenarios or name in scenarios]
    from topology.metro_lines import LAYOUT_REVISION, depot_sites
    recipe = dict(schema="metro_dispatch_seeded_v5_seconds", topology=tid, line=copy.deepcopy(LINE_SPECS[tid]),
        layout_revision=LAYOUT_REVISION, depot_sites=depot_sites(tid),
        physical_tick_seconds=PHYSICAL_TICK_SECONDS, decision_seconds=TICK_SECONDS,
        seed=seed, repeats=repeats, split_offsets=dict(SPLIT_OFFSETS), tick_seconds=TICK_SECONDS, clock_start="06:00:00",
        episode_seconds=EPISODE_SECONDS, episode_stages=EPISODE_STAGES, timeline=copy.deepcopy(TIMELINE),
        event_start_seconds=list(START_SECONDS), single_clearance_seconds=list(CLEARANCE_SECONDS),
        combination_onset_gaps_seconds=list(ONSET_GAPS_SECONDS),
        event_families=sorted(EVENT_TARGETS), heldout_combinations=list(HELDOUT_COMBINATIONS),
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            "scenarios/prepare_dispatch.py", "topology/metro_lines.py", "delay_events.py")})
    recipe["sha256"] = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    entries = [dict(case_id=f"{LINE_SPECS[tid]['key']}_{split}_{repeat}_{name}", topology=tid,
                    split=split, repeat=repeat, seed=seed+SPLIT_OFFSETS[split]+repeat, scenario=name)
               for repeat in range(repeats) for name in names]
    return recipe, entries


def events_for(entry, *, fleet=None):
    """Fresh records per episode; callers cannot mutate a later regeneration."""
    if entry["scenario"] not in scenario_names(entry["split"]):
        raise ValueError(f"Scenario unavailable in {entry['split']}: {entry['scenario']}")
    return curriculum(entry["topology"], entry["seed"], fleet=fleet)[entry["scenario"]]


def add_curriculum_args(parser):
    parser.add_argument("--line", required=True, help="Named line: sz20, sz1, sz10, sz6b or sh16")
    parser.add_argument("--scenario-seed", type=int, default=111, help="Fixed scenario seed, separate from policy training seed")
    parser.add_argument("--scenario-repeats", type=int, default=3, help="Seed realizations per scenario in each split")


def scenarios_from_args(args, split):
    return read_scenarios(args.line, split, args.scenarios, repeats=args.scenario_repeats, seed=args.scenario_seed)


def runtime_scenario(line, name, *, split="test", repeat=0, seed=111):
    if repeat < 0:
        raise ValueError("scenario_repeat must be nonnegative")
    recipe, entries = read_scenarios(line, split, [name], repeats=repeat+1, seed=seed)
    entry = entries[repeat]
    return dict(events=events_for(entry), event_resources=event_resources(recipe["topology"]),
                clock_start_seconds=21600, tick_seconds=recipe["tick_seconds"],
                scenario_name=entry["case_id"], scenario_recipe=recipe)


def preview(line, *, split="test", repeat=0, seed=111, scenarios=None, repeats=3, details=False):
    recipe, entries = read_scenarios(line, split, scenarios, repeats=repeats, seed=seed)
    if not 0 <= repeat < repeats:
        raise ValueError("repeat must be within scenario_repeats")
    print(f"\n{recipe['line']['name']} | {split} | 基础种子 {seed} | {len(entries)} 个情形 | 起点 06:00:00")
    print("默认时段：06:00--06:15无事件，06:15--06:45错峰发生，06:45--07:15分批解除，07:15--08:00观察恢复。")
    print("| 场景 | 事件类型 |")
    print("| --- | --- |")
    selected = [entry for entry in entries if entry["repeat"] == repeat]
    for entry in selected:
        records = events_for(entry)
        labels = "、".join(EVENT_LABELS[e["type"]] for e in records) or "正常运行"
        print(f"| {entry['scenario']} | {labels} |")
    if not details:
        return
    def clock(seconds):
        total = 21600+int(seconds)
        return f"{total//3600:02d}:{total//60%60:02d}:{total%60:02d}"
    for entry in selected:
        print(f"\n{entry['case_id']} | 实际种子 {entry['seed']}")
        print("| 事件 | 对象 | 发生 | 首次报告 | 仿真解除 | 参数 |")
        print("| --- | --- | --- | --- | --- | --- |")
        for event in events_for(entry):
            target = event["target"]
            params = "; ".join(f"{key}={value}" for key, value in event["parameters"].items()) or "无"
            print(f"| {EVENT_LABELS[event['type']]} ({event['event_id']}) | {target['kind']} {target['id']} | "
                  f"{clock(event['start_seconds'])} | {clock(event['first_report_seconds'])} | "
                  f"{clock(event['actual_end_seconds'])} | {params} |")
            for report in event["recovery_reports"]:
                print(f"  恢复信息于 {clock(report['published_seconds'])} 发布，预计解除 {clock(report['predicted_end_seconds'])}。")
    print("仿真解除时刻仅用于人工核查，不作为控制器提前已知的信息。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", nargs="+", default=[s["key"] for s in LINE_SPECS.values()])
    parser.add_argument("--scenario-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--split", choices=list(SPLIT_OFFSETS), default="test")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--details", action="store_true", help="Print event targets, times and parameters; no files are written")
    args = parser.parse_args()
    for line in args.lines:
        preview(line, split=args.split, repeat=args.repeat, seed=args.seed, scenarios=args.scenarios,
                repeats=args.scenario_repeats, details=args.details)


if __name__ == "__main__":
    main()
