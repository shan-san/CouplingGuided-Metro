"""Solver-independent plan contract. Times are absolute simulation seconds."""
from dataclasses import dataclass, field
import math


PARAMETERS = {
    "reserve_insert": ({"exit_node_id": int}, set()),
    "reenter": ({"exit_node_id": int}, set()),
    "withdraw": ({"depot": str, "path_node_ids": list}, set()),
    "short_turn": ({"target_node": int}, {"target_node"}),
    "rescue_tow": ({"failed": str, "depot": str, "approach_node_ids": list, "tow_node_ids": list}, {"failed"}),
    "inspect_repair": ({}, set()),
    "hold": ({"seconds": float}, {"seconds"}),
    "adjust_departure": ({"target_node": int, "headway_seconds": float, "platform_id": int, "departure_time": float}, set()),
    "reroute": ({"target_node": int, "path_node_ids": list}, {"target_node"}),
    "run_profile": ({"segment_id": str, "profile_id": str}, {"segment_id", "profile_id"}),
    "skip_stop": ({"target_node": int, "path_node_ids": list}, {"target_node"}),
    "cancel_service": ({}, set()),
    "reassign_service": ({"service_id": str}, {"service_id"}),
    "evacuate": ({}, set()),
    "admission_control": ({"platform_id": int, "rate": float}, {"platform_id", "rate"}),
}


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class DispatchCommand:
    command_id: str
    action: str
    vehicle_id: str | None
    effective_at: float
    parameters: dict = field(default_factory=dict)
    expires_at: float | None = None
    expected_node: int | None = None
    expected_service_id: str | None = None
    locked: bool = False

    @classmethod
    def from_dict(cls, data, tick_seconds=1):
        if not isinstance(data, dict) or set(data) - set(cls.__dataclass_fields__):
            raise ValueError("Invalid command fields")
        action=data.get("action")
        if action not in PARAMETERS:
            raise ValueError("Unsupported plan action")
        types,required=PARAMETERS[action]
        params=data.get("parameters", {})
        if not isinstance(params,dict) or set(params)-set(types) or not required<=set(params):
            raise ValueError("Invalid action parameters")
        for name,value in params.items():
            kind=types[name]
            if kind is float:
                if number(value,name)<0:
                    raise ValueError("Negative operation duration")
                if name == "departure_time" and tick_seconds == 1 and value != int(value):
                    raise ValueError("Departure time must be an integer simulation second")
            elif kind is int:
                if not isinstance(value,int) or isinstance(value,bool):
                    raise ValueError("Node IDs must be integers")
            elif kind is list:
                if not isinstance(value, list) or not value or any(
                        not isinstance(n, int) or isinstance(n, bool) for n in value):
                    raise ValueError("A planned path must be a nonempty list of integer node IDs")
            else:
                identifier(value,name)
        effective=number(data.get("effective_at"),"effective_at")
        if effective<0 or not math.isclose(effective/tick_seconds,round(effective/tick_seconds),abs_tol=1e-8):
            raise ValueError("Command time must align with physical ticks")
        expires=data.get("expires_at")
        if expires is not None and number(expires,"expires_at")<effective:
            raise ValueError("Command expires before it becomes effective")
        expected=data.get("expected_node")
        if expected is not None and (not isinstance(expected,int) or isinstance(expected,bool)):
            raise ValueError("expected_node must be an integer")
        locked=data.get("locked",False)
        if not isinstance(locked,bool):
            raise ValueError("locked must be boolean")
        expected_service=data.get("expected_service_id")
        if expected_service is not None:
            identifier(expected_service,"expected_service_id")
        vehicle=data.get('vehicle_id')
        platform_headway = action == 'adjust_departure' and vehicle is None
        if platform_headway:
            if set(params) != {'platform_id', 'headway_seconds'}:
                raise ValueError('Platform headway control needs platform_id and headway_seconds')
        elif action == 'adjust_departure':
            if 'target_node' not in params or 'platform_id' in params:
                raise ValueError('Vehicle departure needs target_node, without platform_id')
        if action=='admission_control' or platform_headway:
            if vehicle is not None or expected is not None or expected_service is not None:
                raise ValueError('Platform control cannot include vehicle preconditions')
        else: vehicle=identifier(vehicle,'vehicle_id')
        return cls(identifier(data.get("command_id"),"command_id"),action,
                   vehicle,effective,
                   {k: list(v) if isinstance(v, list) else v for k,v in params.items()},
                   expires,expected,expected_service,locked)

    def participants(self):
        if self.vehicle_id is None: return set()
        return {self.vehicle_id, self.parameters["failed"]} if self.action=="rescue_tow" else {self.vehicle_id}

    def resources(self):
        if self.action == 'admission_control':
            return {('admission', self.parameters['platform_id'])}
        if self.action == 'adjust_departure' and self.vehicle_id is None:
            return {('headway', self.parameters['platform_id'])}
        return set()


@dataclass(frozen=True)
class DispatchPlan:
    plan_id: str
    snapshot_version: int
    created_at: float
    commands: tuple[DispatchCommand, ...]
    replace_future: bool = False
    snapshot_basis: str = 'published'

    @classmethod
    def from_dict(cls,data,tick_seconds=1):
        if not isinstance(data,dict) or set(data)-set(cls.__dataclass_fields__):
            raise ValueError("Invalid plan fields")
        version=data.get("snapshot_version")
        if not isinstance(version,int) or isinstance(version,bool) or version<0:
            raise ValueError("Invalid snapshot version")
        created=number(data.get("created_at"),"created_at")
        if created<0:
            raise ValueError("Negative plan creation time")
        commands=data.get("commands")
        if not isinstance(commands,list) or not commands:
            raise ValueError("Plan needs commands")
        commands=tuple(DispatchCommand.from_dict(c,tick_seconds) for c in commands)
        if len({c.command_id for c in commands})!=len(commands):
            raise ValueError("Duplicate command IDs")
        for i,c in enumerate(commands):
            if c.effective_at<created:
                raise ValueError("Command precedes plan creation")
            for other in commands[i+1:]:
                if c.effective_at==other.effective_at and (c.participants() & other.participants() or c.resources() & other.resources()):
                    raise ValueError("Simultaneous commands reuse a physical vehicle")
        replace=data.get("replace_future",False)
        if not isinstance(replace,bool):
            raise ValueError("replace_future must be boolean")
        basis=data.get('snapshot_basis','published')
        if basis not in ('physical','published'):
            raise ValueError('Invalid snapshot version basis')
        return cls(identifier(data.get("plan_id"),"plan_id"),version,created,commands,replace,basis)
