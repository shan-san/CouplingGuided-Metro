"""Shared configuration for assigning timetable runs at reset."""

from dataclasses import dataclass
from typing import Dict, Optional, Union


@dataclass
class TrainRunConfig:
    """Legacy entry-point configuration; native dispatch owns its own state."""

    route_id: Optional[Union[int, str]] = None
    headway: int = 240
    loops_per_run: int = 1
    first_start_time: int = 0
    assignments: Optional[Dict[int, Union[int, str]]] = None
