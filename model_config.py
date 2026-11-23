from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PolicyNetConfig:
    """Configuration for the policy network architecture.

    This intentionally mirrors the training setup so evaluation can rebuild the
    exact same network without inspecting checkpoint internals.
    """

    # Core architecture
    policy_depth: int = 6
    policy_hidden_dim: int = 32
    share_parameters_policy: bool = True

    # Environment-related
    n_agents: int = 14

    # Action space (per agent)
    # 动作空间: 0=停留, 1-3=不同方向移动 (共4个动作)
    n_action_dim: int = 4

    @staticmethod
    def from_hparams(hparams: Dict[str, Any]) -> "PolicyNetConfig":
        return PolicyNetConfig(
            policy_depth=int(hparams.get("policy_depth", 7)),
            policy_hidden_dim=int(hparams.get("policy_hidden_dim", 192)),
            share_parameters_policy=bool(hparams.get("share_parameters_policy", True)),
            n_agents=int(hparams.get("n_agents", 14)),
            n_action_dim=int(hparams.get("n_action_dim", 4)),  # 4个动作：停留+3个方向
        )

    @staticmethod
    def from_args(args: Any) -> "PolicyNetConfig":
        return PolicyNetConfig(
            policy_depth=int(getattr(args, "policy_depth", 7)),
            policy_hidden_dim=int(getattr(args, "policy_hidden_dim", 192)),
            share_parameters_policy=bool(getattr(args, "share_parameters_policy", True)),
            n_agents=int(getattr(args, "n_agents", 14)),
            n_action_dim=int(getattr(args, "n_action_dim", 4) or 4),  # 4个动作：停留+3个方向
        )

    def to_hparams(self) -> Dict[str, Any]:
        return {
            "policy_depth": int(self.policy_depth),
            "policy_hidden_dim": int(self.policy_hidden_dim),
            "share_parameters_policy": bool(self.share_parameters_policy),
            "n_agents": int(self.n_agents),
            "n_action_dim": int(self.n_action_dim),
        }


def get_default_policy_config() -> PolicyNetConfig:
    """Default policy configuration matching training defaults."""
    return PolicyNetConfig()
