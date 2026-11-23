"""
MARL Policy Performance Benchmark

This script runs a headless simulation to benchmark the performance of the MARL policy.
It specifically measures and records the decision time and rewards when the MARL
is triggered (e.g., during disruptions).

The results, including step-wise rewards, decision times, and decision sources,
are saved to a CSV file for later analysis.

Usage:
    python benchmark_marl.py [--steps N]
    python benchmark_marl.py --topology=1 --nogui --steps N # Run topology 1 without GUI for N steps
"""

import argparse
import copy
import logging
import math
import os
import sys
import time
import random
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from torchrl.envs import TransformedEnv, RewardSum

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Params import (
    simu_parser,
    rl_parser,
    apply_topology_defaults,
    get_default_tick_seconds,
    get_default_n_agents,
)
from envwrapper import MetroRLEnv
from linedata import load_topology, get_platform_by_id
from main_marl import load_rl_policy
from metro_rl_env import MetroRLEnvV2
from train_run_config import TrainRunConfig
from experiment_scenarios import add_scenario_args, scenario_config_from_args, scenario_kwargs, scenario_suffix

logger = logging.getLogger(__name__)


class StationEventRecorder:
    """Tracks per-agent station arrival and departure events during simulation."""

    VALID_EVENT_TYPES = {"arrival", "departure"}

    def __init__(self, base_env, policy_label: str) -> None:
        self.base_env = base_env
        self.policy_label = policy_label
        self.metro_env = getattr(base_env, "_env", base_env)
        self.world = getattr(self.metro_env, "world", None)
        self.tick_seconds = float(getattr(self.world, "tick_seconds", 1) or 1) if self.world is not None else 1.0
        self.topology = getattr(base_env, "topology", None)
        self.prev_platforms: Dict[Tuple[str, int], int] = {}
        self.events: List[Dict[str, object]] = []
        self._native_seen = set()
        if self.world is not None and getattr(self.world, "second_precision", False):
            from dispatch_records import enable_execution_records
            if getattr(self.world, "execution_records", None) is None:
                enable_execution_records(self.world)
        self._bootstrap()

    def _bootstrap(self) -> None:
        if self.world is None:
            return
        num_envs = getattr(self.world, "num_envs", 0) or 0
        for env_index in range(num_envs):
            for agent in self.world.agents:
                key = (agent.name, env_index)
                self.prev_platforms[key] = self._get_platform(agent, env_index)

    @staticmethod
    def _tensor_at(value, index: int) -> Optional[float]:
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        if value.ndim == 0:
            return float(value.item())
        if index < 0 or index >= value.shape[0]:
            return None
        return float(value[index].item())

    def _get_platform(self, agent, env_index: int) -> int:
        platform_tensor = getattr(agent.state, "current_platform", None)
        value = self._tensor_at(platform_tensor, env_index)
        if value is None:
            return -1
        return int(value)

    def _get_last_departure_time(self, agent, env_index: int) -> Optional[float]:
        last_departure_tensor = getattr(agent.state, "last_departure_time", None)
        value = self._tensor_at(last_departure_tensor, env_index)
        return value

    def _get_sim_time(self, env_index: int) -> Optional[float]:
        if self.world is None:
            return None
        sim_time_tensor = getattr(self.world, "sim_time", None)
        return self._tensor_at(sim_time_tensor, env_index)

    def _resolve_station_label(self, station_id: int) -> str:
        platform = get_platform_by_id(station_id)
        name = getattr(platform, "name", None)
        if name:
            return str(name)
        return f"s{station_id}"

    def _build_event(
        self,
        *,
        agent_name: str,
        env_index: int,
        station_id: int,
        event_type: str,
        sim_time: float,
        step_index: int,
    ) -> Dict[str, object]:
        safe_event = event_type if event_type in self.VALID_EVENT_TYPES else "unknown"
        return {
            "policy": self.policy_label,
            "topology": self.topology,
            "env_index": env_index,
            "agent": agent_name,
            "station": self._resolve_station_label(station_id),
            "event_type": safe_event,
            "step": step_index,
            "sim_time_tick": sim_time,
            "tick_seconds": self.tick_seconds,
            "sim_time_seconds": sim_time * self.tick_seconds,
        }

    def record(self, step_index: int) -> List[Dict[str, object]]:
        if self.world is None:
            return []
        if getattr(self.world, "second_precision", False):
            recorded = []
            for index, visit in enumerate(self.world.execution_records.visits):
                for kind, field in (("arrival", "arrival_seconds"), ("departure", "departure_seconds")):
                    if visit[field] is None or (kind == "arrival" and visit["initial_state"]) or (index, kind) in self._native_seen:
                        continue
                    recorded.append(self._build_event(agent_name=visit["vehicle"], env_index=0,
                        station_id=visit["platform"], event_type=kind, sim_time=visit[field], step_index=step_index))
                    self._native_seen.add((index, kind))
            self.events.extend(recorded)
            return recorded

        recorded: List[Dict[str, object]] = []
        num_envs = getattr(self.world, "num_envs", 0) or 0

        for env_index in range(num_envs):
            sim_time = self._get_sim_time(env_index)
            if sim_time is None:
                continue

            for agent in self.world.agents:
                key = (agent.name, env_index)
                current_platform = self._get_platform(agent, env_index)
                previous_platform = self.prev_platforms.get(key)

                if previous_platform is None:
                    self.prev_platforms[key] = current_platform
                    continue

                if current_platform == previous_platform:
                    continue

                if previous_platform >= 0 and current_platform != previous_platform:
                    departure_time = self._get_last_departure_time(agent, env_index)
                    if departure_time is None or not math.isfinite(departure_time):
                        departure_time = sim_time
                    recorded.append(
                        self._build_event(
                            agent_name=agent.name,
                            env_index=env_index,
                            station_id=previous_platform,
                            event_type="departure",
                            sim_time=departure_time,
                            step_index=step_index,
                        )
                    )

                if current_platform >= 0:
                    recorded.append(
                        self._build_event(
                            agent_name=agent.name,
                            env_index=env_index,
                            station_id=current_platform,
                            event_type="arrival",
                            sim_time=sim_time,
                            step_index=step_index,
                        )
                    )

                self.prev_platforms[key] = current_platform

        if recorded:
            self.events.extend(recorded)
        return recorded

    def to_frame(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(
                columns=
                [
                    "policy",
                    "topology",
                    "env_index",
                    "agent",
                    "station",
                    "event_type",
                    "step",
                    "sim_time_tick",
                    "tick_seconds",
                    "sim_time_seconds",
                ]
            )
        return pd.DataFrame(self.events)

    def get_headway_std(self) -> float:
        if not self.events:
            return 0.0
        
        df = pd.DataFrame(self.events)
        # Filter for arrival events
        arrivals = df[df["event_type"] == "arrival"].copy()
        
        if arrivals.empty:
            return 0.0
            
        # Sort by station and time
        arrivals.sort_values(by=["station", "sim_time_seconds"], inplace=True)
        
        # Calculate headways
        headway_stds = []
        for station, group in arrivals.groupby("station"):
            if len(group) < 2:
                continue
            
            # Calculate differences between consecutive arrival times
            times = group["sim_time_seconds"].values
            headways = np.diff(times)
            
            if len(headways) > 0:
                headway_stds.append(np.std(headways))
                
        if not headway_stds:
            return 0.0
            
        return float(np.mean(headway_stds))


class AgentLoadRecorder:
    """Tracks passenger load for each agent at each step."""
    def __init__(self, env, policy_label: str):
        self.env = env
        self.policy_label = policy_label
        self.base_env = getattr(env, "base_env", env)
        self.metro_env = getattr(self.base_env, "_env", self.base_env)
        self.world = getattr(self.metro_env, "world", None)
        self.topology = getattr(env, "topology", None)
        self.records: List[Dict[str, object]] = []
    
    def record(self, step_index: int) -> List[Dict[str, object]]:
        """Record passenger load for all agents at the current step."""
        if self.world is None:
            return []
        
        recorded: List[Dict[str, object]] = []
        num_envs = getattr(self.world, "num_envs", 0) or 0
        
        for env_index in range(num_envs):
            for agent in self.world.agents:
                # Get passenger count from agent
                # Try multiple ways to access passenger count
                load = 0.0
                
                # Method 1: Direct property access
                if hasattr(agent, "passenger_count"):
                    load_tensor = agent.passenger_count
                    if isinstance(load_tensor, torch.Tensor):
                        if load_tensor.ndim == 0:
                            load = float(load_tensor.item())
                        elif env_index < load_tensor.shape[0]:
                            load = float(load_tensor[env_index].item())
                        else:
                            load = 0.0
                    else:
                        load = float(load_tensor)
                # Method 2: State attribute access
                elif hasattr(agent, "state") and hasattr(agent.state, "passenger_count"):
                    load_tensor = agent.state.passenger_count
                    if isinstance(load_tensor, torch.Tensor):
                        if load_tensor.ndim == 0:
                            load = float(load_tensor.item())
                        elif env_index < load_tensor.shape[0]:
                            load = float(load_tensor[env_index].item())
                        else:
                            load = 0.0
                    else:
                        load = float(load_tensor)
                
                record = {
                    "policy": self.policy_label,
                    "topology": self.topology,
                    "env_index": env_index,
                    "step": step_index,
                    "agent": agent.name,
                    "passenger_load": load,
                }
                recorded.append(record)
        
        if recorded:
            self.records.extend(recorded)
        return recorded
    
    def to_frame(self) -> pd.DataFrame:
        """Convert records to DataFrame."""
        if not self.records:
            return pd.DataFrame(
                columns=["policy", "topology", "env_index", "step", "agent", "passenger_load"]
            )
        return pd.DataFrame(self.records)


class GlobalMetricRecorder:
    """Tracks global metrics like average waiting time."""
    def __init__(self, env):
        self.env = env
        self.total_waiting_time = 0.0
        self.tick_seconds = env.base_env.default_interval
        
        # Initialize previous boarded count
        self.prev_total_boarded = 0.0
        if hasattr(env.base_env, "agents"):
             for agent in env.base_env.agents:
                self.prev_total_boarded += agent.cumulative_boarded.sum().item()

    def update(self) -> Dict[str, float]:
        step_metrics = {
            "waiting_count": 0.0,
            "boarded_count": 0.0,
        }

        # 1. Waiting Time
        current_waiting = 0
        for node in self.env.base_env.world.platform_nodes.values():
            current_waiting += node.waiting_passengers.sum().item()
        world = self.env.base_env.world
        if getattr(world, "second_precision", False):
            self.total_waiting_time += world.last_platform_waiting_person_seconds
        else:
            self.total_waiting_time += current_waiting * self.tick_seconds
        step_metrics["waiting_count"] = current_waiting

        # 2. Boarded Passengers (Delta)
        current_total_boarded = 0
        for agent in self.env.base_env.agents:
            current_total_boarded += agent.cumulative_boarded.sum().item()
        
        step_boarded = current_total_boarded - self.prev_total_boarded
        self.prev_total_boarded = current_total_boarded
        step_metrics["boarded_count"] = step_boarded

        return step_metrics

    def get_metrics(self):
        total_boarded = 0
        for agent in self.env.base_env.agents:
            total_boarded += agent.cumulative_boarded.sum().item()
        
        avg_waiting_time = self.total_waiting_time / total_boarded if total_boarded > 0 else 0.0
        
        return {
            "total_waiting_time": self.total_waiting_time,
            "total_boarded": total_boarded,
            "avg_waiting_time": avg_waiting_time,
        }


def _build_env(sim_args, rl_args, device: torch.device, horizon: int, seed: int, scenario_config: Optional[Dict[str, object]] = None):
    resolved_args = copy.deepcopy(rl_args)
    apply_topology_defaults(resolved_args, sim_args.topology)

    tick_seconds = get_default_tick_seconds(sim_args.topology) or 1
    os.environ["METRO_TOPOLOGY"] = str(sim_args.topology)
    load_topology(sim_args.topology)

    start_steps = resolved_args.start_steps
    max_steps = start_steps + horizon
    episode_length = getattr(resolved_args, "max_steps", horizon)
    n_agents = get_default_n_agents(sim_args.topology)

    headway = getattr(resolved_args, "train_headway", 240)
    loops_per_run = getattr(resolved_args, "train_loops_per_run", 7)
    first_start_time = getattr(resolved_args, "train_first_start_time", 0)
    assignments = getattr(resolved_args, "train_assignments", None)

    train_run_config = TrainRunConfig(
        route_id=None,
        headway=headway,
        loops_per_run=loops_per_run,
        first_start_time=first_start_time,
        assignments=assignments,
    )

    extra_kwargs = scenario_kwargs(scenario_config)

    base_env = MetroRLEnvV2(
        num_envs=1,
        max_steps=max_steps,
        device=device,
        n_agents=n_agents,
        train_run_config=train_run_config,
        topology=sim_args.topology,
        start_step=start_steps,
        episode_length=episode_length,
        use_float_boarding=True,
        enable_random_blockages=False,
        enable_random_speed_limits=False,
        enable_simulation_blockages=True,
        enable_simulation_speed_limits=True,
        default_interval=tick_seconds,
        seed=seed,
        **extra_kwargs,
    )
    env = TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    return env, resolved_args


def _resolve_model_path(topology: int) -> str:
    if topology == 1:
        filename = "metro_mappo_best.pt"
    elif topology == 2:
        filename = "metro_topology_mappo_best.pt"
    elif topology == 3:
        filename = "metro_shenzhenline1_mappo_best.pt"
    elif topology == 4:
        filename = "metro_topology_4_mappo_best.pt"
    elif topology == 5:
        filename = "metro_topology_5_mappo_best.pt"
    elif topology == 6:
        filename = "metro_topology_6_mappo_best.pt"
    elif topology == 7:
        filename = "metro_topology_7_mappo_best.pt"
    elif topology == 8:
        filename = "metro_topology_8_mappo_best.pt"
    elif topology == 9:
        filename = "metro_topology_9_mappo_best.pt"
    elif topology == 10:
        filename = "metro_topology_10_mappo_best.pt"
    elif topology == 11:
        filename = "metro_topology_11_mappo_best.pt"
    elif topology == 12:
        filename = "metro_topology_12_mappo_best.pt"
    elif topology == 13:
        filename = "metro_topology_13_mappo_best.pt"
    elif topology == 14:
        filename = "metro_topology_14_mappo_best.pt"
    elif topology == 15:
        filename = "metro_topology_15_mappo_best.pt"
    elif topology == 16:
        filename = "metro_topology_16_mappo_best.pt"
    else:
        filename = "metro_mappo_best.pt"
        logger.warning("未知拓扑，回退至默认模型。")
    return os.path.join(project_root, "MARL_model", filename)


def run_marl_benchmark(
    *,
    num_steps: int,
    sim_args,
    rl_args,
    device: torch.device,
    model_path: Optional[str] = None,
    policy_label: str = "Pure_MARL",
    seed: int = 123,
    scenario_config: Optional[Dict[str, object]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    env, resolved_rl_args = _build_env(sim_args, rl_args, device, num_steps, seed, scenario_config)
    n_agents = resolved_rl_args.n_agents

    if model_path is None:
        model_path = _resolve_model_path(sim_args.topology)

    try:
        policy = load_rl_policy(model_path, env, device, resolved_rl_args)
    except Exception as exc:
        env.close()
        raise RuntimeError(f"MARL 策略加载失败: {exc}") from exc

    results = []
    tensordict_obs = env.reset()
    station_recorder = StationEventRecorder(env.base_env, policy_label)
    agent_load_recorder = AgentLoadRecorder(env, policy_label)
    metric_recorder = GlobalMetricRecorder(env)
    pbar = tqdm(range(num_steps), desc=f"Benchmark {policy_label}")

    try:
        for step in pbar:
            decision_start = time.perf_counter()
            with torch.no_grad():
                action_td = policy(tensordict_obs.to(device))
            decision_time = time.perf_counter() - decision_start

            tensordict_result = env.step(action_td)
            reward = tensordict_result.get(("next", "agents", "reward")).mean().item()
            done = tensordict_result.get(("next", "done"), torch.tensor(False)).any().item()

            step_metrics = metric_recorder.update()
            station_recorder.record(step)
            agent_load_recorder.record(step)

            results.append({
                "policy": policy_label,
                "step": step,
                "reward": reward,
                "decision_time_sec": decision_time,
                "decision_source": "MARL",
                "waiting_passengers": step_metrics["waiting_count"],
                "boarded_passengers": step_metrics["boarded_count"],
            })

            tensordict_obs = tensordict_result.get("next").clone()
            if done:
                logger.info("回合在第 %s 步结束。", step)
                break
    finally:
        env.close()

    metrics = metric_recorder.get_metrics()
    metrics["avg_headway_std"] = station_recorder.get_headway_std()
    
    logger.info("Benchmark Metrics:")
    logger.info(f"  Average Passenger Waiting Time: {metrics['avg_waiting_time']:.2f} seconds")
    logger.info(f"  Total Boarded Passengers: {metrics['total_boarded']}")
    logger.info(f"  Average Headway Standard Deviation: {metrics['avg_headway_std']:.2f} seconds")

    # Add metrics to results dataframe (repeated for all rows or just summary)
    # Or return separate metrics
    
    return pd.DataFrame(results), station_recorder.to_frame(), agent_load_recorder.to_frame(), metrics


def _parse_args():
    sim_args, remaining = simu_parser.parse_known_args()
    parser = argparse.ArgumentParser(description="Benchmark MARL policy.", add_help=True)
    parser.add_argument("--steps", type=int, default=7200, help="模拟步数")
    parser.add_argument("--output", type=str, default=None, help="结果CSV输出路径")
    parser.add_argument("--station-output", type=str, default=None, help="站点事件CSV输出路径")
    parser.add_argument("--agent-load-output", type=str, default=None, help="Agent载客数CSV输出路径")
    parser.add_argument("--results-dir", type=str, default=None, help="默认结果输出目录")
    parser.add_argument("--seed", type=int, default=111, help="随机种子")
    add_scenario_args(parser)
    bench_args = parser.parse_args(remaining)
    return sim_args, bench_args


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    sim_args, bench_args = _parse_args()

    seed = bench_args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rl_args = rl_parser.parse_args([])
    apply_topology_defaults(rl_args, sim_args.topology)
    scenario_config = scenario_config_from_args(bench_args)

    device = torch.device("cpu")
    logger.info(f"MARL 基准测试 | 拓扑: {sim_args.topology} | 步数: {bench_args.steps} | 种子: {seed}")

    df, station_df, agent_load_df, metrics = run_marl_benchmark(
        num_steps=bench_args.steps,
        sim_args=sim_args,
        rl_args=rl_args,
        device=device,
        seed=seed,
        scenario_config=scenario_config,
    )

    topology_suffix = f"_topology{sim_args.topology}"
    suffix = scenario_suffix(scenario_config)
    results_dir = bench_args.results_dir or os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = bench_args.output or os.path.join(results_dir, f"marl{topology_suffix}{suffix}_benchmark_results.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("MARL 基准结果已保存至 %s", output_path)

    station_output_path = bench_args.station_output or os.path.join(results_dir, f"marl{topology_suffix}{suffix}_station_events.csv")
    station_df.to_csv(station_output_path, index=False, encoding="utf-8-sig")
    logger.info("站点事件记录已保存至 %s", station_output_path)
    
    agent_load_output_path = bench_args.agent_load_output or os.path.join(results_dir, f"marl{topology_suffix}{suffix}_agent_loads.csv")
    agent_load_df.to_csv(agent_load_output_path, index=False, encoding="utf-8-sig")
    logger.info("Agent载客数记录已保存至 %s", agent_load_output_path)
    
    # Save metrics to a separate file
    metrics_output_path = os.path.join(results_dir, f"marl{topology_suffix}{suffix}_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_output_path, index=False, encoding="utf-8-sig")
    logger.info("全局指标已保存至 %s", metrics_output_path)


if __name__ == "__main__":
    main()
