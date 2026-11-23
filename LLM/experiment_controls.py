"""Explicit controls for native dispatch experiments, without changing physics."""
from __future__ import annotations

import math


def validate_experiment_controls(config):
    choices = {
        "reference_policy": ("marl", "rule"),
        "trigger_mode": ("coupling", "legacy", "periodic", "event"),
        "coordination_scope": ("coupling", "full_fleet", "nearest"),
        "optimization_mode": ("joint", "staged"),
        "coordination_horizon_policy": ("adaptive", "fixed"),
    }
    defaults = dict(reference_policy="marl", trigger_mode="coupling",
                    coordination_scope="coupling", optimization_mode="joint", coordination_horizon_policy="adaptive")
    for field, options in choices.items():
        if getattr(config, field, defaults[field]) not in options:
            raise ValueError(f"Invalid {field}; expected one of {options}")
    for field, default in (("coordination_period", 10), ("nearest_vehicles", 6)):
        value = getattr(config, field, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    for field in ("fixed_planning_horizon", "scope_window_closure"):
        if type(getattr(config, field, False)) is not bool:
            raise ValueError(f"{field} must be boolean")
    for field, default, minimum in (("coordination_max_horizon", 24, 1), ("coordination_service_tail", 3, 0),
                                    ("scope_expansion_agents", 2, 0), ("scope_window_vehicles", 12, 1)):
        value = getattr(config, field, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")


def experiment_configuration(config):
    """A credential-free record of the controls actually used by the runtime."""
    validate_experiment_controls(config)
    fields = dict(reference_policy="marl", trigger_mode="coupling", coordination_period=10,
                  coordination_scope="coupling", nearest_vehicles=6, optimization_mode="joint",
                  fixed_planning_horizon=False, intervention_cooldown=10, cached_plan_max_steps=0,
                  milp_time_horizon=None, milp_time_limit=None, coordination_time_budget_s=None,
                  allowed_decision_delay_s=None, joint_guidance_mode="full", reference_quality_guard=True,
                  use_rag=True, use_small_model_guidance=True, always_hierarchical=False)
    fields.update(coordination_horizon_policy="adaptive", coordination_max_horizon=24,
                  coordination_service_tail=3, scope_expansion_agents=2,
                  scope_window_closure=False, scope_window_vehicles=12,
                  coupling_request_policy="regional_cooldown_with_pending_age")
    fields["small_model_request_timeout"] = 30.
    result = {key: getattr(config, key, default) for key, default in fields.items()}
    if result["trigger_mode"] != "coupling":
        result["coupling_request_policy"] = "not_used:" + result["trigger_mode"]
    return {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
            for key, value in result.items()}


def component_checks(policy, config):
    """Intentional ablations do not relax requirements on retained components."""
    mode = getattr(config, "joint_guidance_mode", "full")
    checks = {}
    if getattr(config, "reference_policy", "marl") == "marl":
        checks["native_marl"] = bool(policy.rl_available and policy.native_rl_policy is not None)
    if mode in ("full", "no_slm"):
        checks["llm"] = bool(policy.llm_available)
    if mode in ("full", "no_llm"):
        slm = policy.small_model
        checks["slm"] = bool(getattr(config, "use_small_model_guidance", True) and slm and
                             (getattr(slm, "_model", None) is not None or
                              getattr(slm, "_llm_client", None) is not None))
    if getattr(config, "use_rag", True) and mode not in ("no_guidance", "rules"):
        manager = policy.rag_manager
        cases = manager.case_library.cases if manager else []
        checks["rag"] = bool(cases and all(case.get("metadata", {}).get("native_dispatch") for case in cases))
    return checks
