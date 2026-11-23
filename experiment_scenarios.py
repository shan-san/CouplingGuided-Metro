"""Shared scenario overrides for benchmark and generalization experiments."""

from __future__ import annotations

import argparse
import copy
import json
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_TOPOLOGY16_DISRUPTION_SHIFT = {
    "blockages": [
        {"segment_id": "2-3", "start_time": 1200, "end_time": 2600},
        {"segment_id": "16-17", "start_time": 3200, "end_time": 4300},
    ],
    "speed_limits": [
        {"segment_id": "22-23", "start_time": 5000, "end_time": 6200, "factor": 2.5},
    ],
}

DEFAULT_TOPOLOGY16_DEMAND_SHIFT = {
    "passenger_demand_scale": 1.35,
    "passenger_od_variant": "east_heavy",
}


def add_scenario_args(parser: argparse.ArgumentParser) -> None:
    """Add common scenario override flags to a benchmark parser."""
    parser.add_argument(
        "--scenario-name",
        type=str,
        default=None,
        help="场景名称，用于标记输出；支持 topology16_disruption_shift/topology16_demand_shift。",
    )
    parser.add_argument(
        "--blockage",
        action="append",
        default=[],
        metavar="SEGMENT:START:END",
        help="覆盖阻塞事件，例如 2-3:1200:2600；可重复。",
    )
    parser.add_argument(
        "--speed-limit",
        action="append",
        default=[],
        metavar="SEGMENT:START:END:FACTOR",
        help="覆盖限速事件，例如 22-23:5000:6200:2.5；可重复。",
    )
    parser.add_argument(
        "--scenario-config",
        type=str,
        default=None,
        help="JSON场景配置文件；字段可包含 blockages/speed_limits/passenger_demand_scale/passenger_od_variant。",
    )
    parser.add_argument(
        "--passenger-demand-scale",
        type=float,
        default=None,
        help="客流强度倍率；1.0为默认客流。",
    )
    parser.add_argument(
        "--passenger-od-variant",
        type=str,
        default=None,
        choices=["baseline", "reverse_tidal", "flatten", "center_heavy", "west_heavy", "east_heavy"],
        help="OD分布变体。",
    )


def _parse_blockage_spec(spec: str) -> Dict[str, Any]:
    parts = str(spec).split(":")
    if len(parts) != 3:
        raise ValueError(f"阻塞格式应为 SEGMENT:START:END，收到: {spec}")
    segment_id, start_time, end_time = parts
    return {
        "segment_id": segment_id.strip(),
        "start_time": int(float(start_time)),
        "end_time": int(float(end_time)),
    }


def _parse_speed_limit_spec(spec: str) -> Dict[str, Any]:
    parts = str(spec).split(":")
    if len(parts) != 4:
        raise ValueError(f"限速格式应为 SEGMENT:START:END:FACTOR，收到: {spec}")
    segment_id, start_time, end_time, factor = parts
    return {
        "segment_id": segment_id.strip(),
        "start_time": int(float(start_time)),
        "end_time": int(float(end_time)),
        "factor": float(factor),
    }


def _load_json_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"场景配置文件必须是JSON对象: {path}")
    return data


def scenario_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a normalized scenario config from CLI args."""
    config: Dict[str, Any] = {}
    scenario_name = getattr(args, "scenario_name", None)

    if scenario_name == "topology16_disruption_shift":
        config.update(copy.deepcopy(DEFAULT_TOPOLOGY16_DISRUPTION_SHIFT))
    elif scenario_name == "topology16_demand_shift":
        config.update(copy.deepcopy(DEFAULT_TOPOLOGY16_DEMAND_SHIFT))

    file_config = _load_json_config(getattr(args, "scenario_config", None))
    config.update(file_config)

    blockages = [_parse_blockage_spec(spec) for spec in getattr(args, "blockage", [])]
    speed_limits = [_parse_speed_limit_spec(spec) for spec in getattr(args, "speed_limit", [])]
    if blockages:
        config["blockages"] = blockages
    if speed_limits:
        config["speed_limits"] = speed_limits

    demand_scale = getattr(args, "passenger_demand_scale", None)
    if demand_scale is not None:
        config["passenger_demand_scale"] = float(demand_scale)

    od_variant = getattr(args, "passenger_od_variant", None)
    if od_variant:
        config["passenger_od_variant"] = od_variant

    config["scenario_name"] = scenario_name or config.get("scenario_name")
    return config


def scenario_kwargs(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert normalized config into kwargs accepted by MetroWorldV1."""
    if not config:
        return {}
    kwargs: Dict[str, Any] = {}
    if "blockages" in config or "speed_limits" in config:
        kwargs["custom_disruptions"] = {
            "blockages": copy.deepcopy(config.get("blockages", [])),
            "speed_limits": copy.deepcopy(config.get("speed_limits", [])),
        }
    if "passenger_demand_scale" in config:
        kwargs["passenger_demand_scale"] = float(config["passenger_demand_scale"])
    if config.get("passenger_od_variant"):
        kwargs["passenger_od_variant"] = str(config["passenger_od_variant"])
    if config.get("scenario_name"):
        kwargs["scenario_name"] = str(config["scenario_name"])
    for key in ("dispatch_config", "events", "event_resources", "clock_start_seconds",
                "boarding_rate_persons_per_second", "safe_platform_queue_capacity"):
        if key in config:
            kwargs[key] = copy.deepcopy(config[key])
    return kwargs


def _station_key(node: Any) -> str:
    name = (
        getattr(node, "station_name", None)
        or getattr(node, "platform_name", None)
        or getattr(node, "name", None)
        or str(getattr(node, "id", ""))
    )
    return str(name).replace("上行", "").replace("下行", "").strip()


def apply_passenger_variant(world: Any) -> None:
    """Apply passenger demand and OD variants to already-built platform nodes."""
    scale = float(getattr(world, "passenger_demand_scale", 1.0) or 1.0)
    variant = str(getattr(world, "passenger_od_variant", "baseline") or "baseline")
    if abs(scale - 1.0) < 1e-9 and variant == "baseline":
        return

    platform_nodes = getattr(world, "platform_nodes", {}) or {}
    for node in platform_nodes.values():
        od_profile = getattr(node, "od_profile", None)
        if not isinstance(od_profile, dict):
            continue

        updated = copy.deepcopy(od_profile)
        station_index = int(updated.get("index", 0) or 0)
        station_count = 30
        local_scale = scale

        if variant == "reverse_tidal":
            role = updated.get("role", "Residential")
            if role == "Residential":
                updated["role"] = "Employment"
            elif role == "Employment":
                updated["role"] = "Residential"
        elif variant == "flatten":
            updated["role"] = "Hub"
            updated["od_matrix_ref"] = {}
        elif variant == "center_heavy":
            if 10 <= station_index <= 23:
                local_scale *= 1.5
            else:
                local_scale *= 0.75
        elif variant == "west_heavy":
            local_scale *= 1.45 if station_index < station_count // 2 else 0.8
        elif variant == "east_heavy":
            local_scale *= 1.45 if station_index >= station_count // 2 else 0.8

        updated["base_rate_offpeak"] = float(updated.get("base_rate_offpeak", 1000.0)) * local_scale

        if variant in {"center_heavy", "west_heavy", "east_heavy"}:
            od_matrix = copy.deepcopy(updated.get("od_matrix_ref", {}))
            origin_key = _station_key(node)
            row = od_matrix.get(origin_key)
            if isinstance(row, dict):
                for dest_key in list(row.keys()):
                    try:
                        dest_index = _infer_station_index_from_matrix_key(platform_nodes.values(), dest_key)
                    except ValueError:
                        continue
                    if variant == "center_heavy" and 10 <= dest_index <= 23:
                        row[dest_key] = float(row[dest_key]) * 1.6
                    elif variant == "west_heavy" and dest_index < station_count // 2:
                        row[dest_key] = float(row[dest_key]) * 1.5
                    elif variant == "east_heavy" and dest_index >= station_count // 2:
                        row[dest_key] = float(row[dest_key]) * 1.5
                od_matrix[origin_key] = row
                updated["od_matrix_ref"] = od_matrix

        node.od_profile = updated
        if hasattr(node, "passenger_destinations"):
            node.passenger_destinations = {}


def _infer_station_index_from_matrix_key(nodes: Iterable[Any], station: str) -> int:
    station = str(station).strip()
    for node in nodes:
        profile = getattr(node, "od_profile", None)
        if not isinstance(profile, dict):
            continue
        node_station = str(profile.get("station", "")).strip()
        if node_station == station:
            return int(profile.get("index", 0) or 0)
    raise ValueError(station)


def custom_disruptions_for_world(config: Optional[Dict[str, Any]]) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    kwargs = scenario_kwargs(config)
    return kwargs.get("custom_disruptions")


def custom_disruptions_for_milp(config: Optional[Dict[str, Any]]) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    custom = custom_disruptions_for_world(config)
    if not custom:
        return None
    return {
        "blockages": [_event_for_milp(item) for item in custom.get("blockages", [])],
        "speed_limits": [_event_for_milp(item, include_factor=True) for item in custom.get("speed_limits", [])],
    }


def _event_for_milp(item: Dict[str, Any], include_factor: bool = False) -> Dict[str, Any]:
    segment_id = str(item.get("segment_id", "")).strip()
    parts = segment_id.split("-")
    if len(parts) != 2:
        raise ValueError(f"MILP场景只支持 u-v 形式的segment_id，收到: {segment_id}")
    event = {
        "segment": (int(parts[0]), int(parts[1])),
        "segment_id": segment_id,
        "start_time": int(item["start_time"]),
        "end_time": int(item["end_time"]),
    }
    if include_factor:
        event["factor"] = float(item.get("factor", 2.0))
    return event


def scenario_suffix(config: Optional[Dict[str, Any]]) -> str:
    if not config:
        return ""
    name = config.get("scenario_name")
    if name:
        return f"_{name}"
    parts: List[str] = []
    if config.get("blockages") or config.get("speed_limits"):
        parts.append("custom_disruption")
    if config.get("passenger_demand_scale") is not None or config.get("passenger_od_variant"):
        parts.append("custom_demand")
    return "_" + "_".join(parts) if parts else ""
