"""Definitions for metro topology node types."""

from __future__ import annotations

from enum import Enum


class NodeType(Enum):
    """Type labels for nodes within the metro topology."""

    PLATFORM = "platform"
    POSITION = "position"
    SWITCH = "switch"

