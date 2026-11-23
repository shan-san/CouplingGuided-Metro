"""Passenger flow utilities for metro topology nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional
import math

import torch


if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from topology.node import Node

from topology.node_type import NodeType


def _destination_tensor(queues, destination, like):
    value = queues.get(destination)
    if value is None:
        value = queues[destination] = torch.zeros_like(like)
    return value


def _queue_total(queues, like):
    return torch.stack(tuple(queues.values())).sum(0) if queues else torch.zeros_like(like)


@dataclass
class PassengerFlow:
    """Encapsulates passenger-related state and behaviour for a node."""

    node: "Node"
    arrival_rate: float = 1.0
    waiting_passengers: Optional[torch.Tensor] = None
    destinations: Dict[int, float] = field(default_factory=dict)
    waiting_by_destination: Dict[int, torch.Tensor] = field(default_factory=dict)
    outside_by_destination: Dict[int, torch.Tensor] = field(default_factory=dict)
    waiting_mass_by_destination: Dict[int, torch.Tensor] = field(default_factory=dict)
    outside_mass_by_destination: Dict[int, torch.Tensor] = field(default_factory=dict)

    def init_passenger_tensor(self, num_envs: int, device: torch.device | str | int) -> None:
        """Initialise the waiting passenger tensor for vectorised environments."""
        self.waiting_passengers = torch.zeros(num_envs, device=device, dtype=torch.float64)
        self.waiting_by_destination.clear()
        self.outside_by_destination.clear()
        self.waiting_mass_by_destination.clear()
        self.outside_mass_by_destination.clear()

    def ensure_od_queues(self):
        """Assign only previously unclassified legacy demand; never redraw old OD."""
        if self.waiting_passengers is None:
            raise RuntimeError("Passenger tensor not initialised")
        total = _queue_total(self.waiting_by_destination, self.waiting_passengers)
        for idx in range(len(total)):
            actual = float(self.waiting_passengers[idx])
            old = float(total[idx])
            if actual < old - 1e-5:
                factor = max(0., actual / old) if old else 0.
                for q in self.waiting_by_destination.values():
                    q[idx] *= factor
                for mass in self.waiting_mass_by_destination.values():
                    mass[idx] *= factor
            elif actual > old + 1e-5:
                weights = self.destinations
                if not weights:
                    raise ValueError("Waiting passengers need a destination distribution")
                for dest, weight in weights.items():
                    q = _destination_tensor(self.waiting_by_destination, dest, total)
                    q[idx] += (actual - old) * weight

    def outside_count(self, idx=0):
        return sum(float(q[idx]) for q in self.outside_by_destination.values())

    def sync_waiting(self):
        self.waiting_passengers.copy_(_queue_total(self.waiting_by_destination, self.waiting_passengers))

    def admit(self, idx, *, events=None):
        world = getattr(self, "world", None)
        events = events if events is not None else (world.event_view() if world else None)
        if events and not events.boarding_allowed(self.node.id):
            return
        cap = events.queue_capacity(self.node.id, world.safe_platform_queue_capacity) if events else math.inf
        space = max(0., cap - float(self.waiting_passengers[idx]))
        outside = self.outside_count(idx)
        allowance = min(space, outside)
        if world and world.dispatch:
            allowance = world.dispatch.admission_budget(self.node.id, allowance)
        if outside <= 0 or allowance <= 0:
            return
        fraction = min(1., allowance / outside) if outside else 0.
        for dest, q in self.outside_by_destination.items():
            inside = _destination_tensor(self.waiting_by_destination, dest, q)
            mass = _destination_tensor(self.outside_mass_by_destination, dest, q)
            inside_mass = _destination_tensor(self.waiting_mass_by_destination, dest, q)
            if q.numel() == 1:
                inside.add_(q, alpha=fraction)
                q.mul_(1-fraction)
                inside_mass.add_(mass, alpha=fraction)
                mass.mul_(1-fraction)
            else:
                amount = float(q[idx]) * fraction
                inside[idx] += amount
                q[idx] -= amount
                inside_mass[idx] += mass[idx] * fraction
                mass[idx] *= 1-fraction
        self.sync_waiting()

    def take_boarding(self, idx, amount, *, with_waiting=False, excluded_destinations=()):
        self.ensure_od_queues()
        total = sum(float(q[idx]) for d, q in self.waiting_by_destination.items() if d not in excluded_destinations)
        fraction = min(1., amount / total) if total else 0.
        result, masses = {}, {}
        for dest, q in self.waiting_by_destination.items():
            if dest in excluded_destinations:
                continue
            result[dest] = float(q[idx]) * fraction
            q[idx] -= result[dest]
            mass = _destination_tensor(self.waiting_mass_by_destination, dest, q)
            masses[dest] = float(mass[idx]) * fraction
            mass[idx] *= 1 - fraction
        self.sync_waiting()
        return (result, masses) if with_waiting else result

    def accrue_waiting(self, seconds):
        """Person-seconds accrued once per physical interval, including outside."""
        self.ensure_od_queues()
        for queues, masses in ((self.waiting_by_destination, self.waiting_mass_by_destination),
                               (self.outside_by_destination, self.outside_mass_by_destination)):
            for dest, count in queues.items():
                mass = _destination_tensor(masses, dest, count)
                mass.add_(count, alpha=seconds)

    def receive_vehicle(self, agent, idx, *, destinations=None):
        """Transfer unfinished journeys without erasing destinations or past waiting."""
        self.ensure_od_queues()
        for dest, onboard in agent.passengers_by_destination.items():
            if destinations is not None and dest not in destinations:
                continue
            count = float(onboard[idx])
            mass = agent.passenger_waiting_mass.setdefault(dest, torch.zeros_like(onboard))
            if dest == self.node.id:
                agent.cumulative_alighted[idx] += count
                agent.completed_waiting_mass[idx] += mass[idx]
            else:
                q = self.outside_by_destination.setdefault(dest, torch.zeros_like(onboard))
                m = self.outside_mass_by_destination.setdefault(dest, torch.zeros_like(onboard))
                q[idx] += count
                m[idx] += mass[idx]
            onboard[idx] = 0
            mass[idx] = 0
        agent.passenger_count[idx] = math.fsum(float(q[idx]) for q in agent.passengers_by_destination.values())
        self.admit(idx)

    def interval_arrivals(self, events, seconds, *, sample=False):
        """Current declared intensity and OD shares, not a forecast or future sample."""
        world = getattr(self, "world", None)
        if world and world.dispatch and self.node.id in {d["node_id"] for d in world.dispatch.depots.values()}:
            return {}
        rate = float(self.arrival_rate)
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("Passenger arrival intensity must be nonnegative and finite")
        mean = rate * seconds * events.demand_multiplier(self.node.id)
        count = float(torch.poisson(torch.tensor(mean, device=self.waiting_passengers.device))) if sample else mean
        if count == 0:
            return {}
        return {d: count * p for d, p in events.od_weights(self.node.id, self.destinations).items()}

    def add_arrivals(self, arrivals, idx=0):
        self.ensure_od_queues()
        for destination, count in arrivals.items():
            if (isinstance(count, bool) or not isinstance(count, (float, int)) or not math.isfinite(count) or count < 0
                    or destination not in self.world.platform_nodes or destination == self.node.id):
                raise ValueError("Invalid fresh OD arrivals")
            if count == 0:
                continue
            q = _destination_tensor(self.outside_by_destination, destination, self.waiting_passengers)
            q[idx] += count
        self.admit(idx)

    def generate_passengers(
        self,
        env_index: Optional[int] = None,
        all_platform_nodes: Optional[Dict[int, "Node"]] = None,
        current_time: Optional[float] = None,
    ) -> None:
        """Generate passengers with realistic tidal (Shenzhen-style) flow logic."""
        node = self.node
        if node.type is not NodeType.PLATFORM:
            return

        if current_time is None:
            raise ValueError("current_time must be provided")

        world = getattr(self, "world", None)
        if world and world.dispatch and node.id in {d["node_id"] for d in world.dispatch.depots.values()}:
            return
        current_time += float(getattr(world, "clock_start_seconds", 0))

        seconds_in_day = 24 * 60 * 60
        seconds_since_midnight = current_time % seconds_in_day
        hour = int((seconds_since_midnight // 3600) % 24)

        od_profile = getattr(node, "od_profile", None)
        if isinstance(od_profile, dict) and od_profile.get("line") == "shenzhen_line1":
            rate_offpeak = float(od_profile.get("base_rate_offpeak", 1000.0))
            role = od_profile.get("role", "Residential")
            
            # --- 1. 动态峰平比 (Realistic Peak-Off Ratios) ---
            # 居住区峰值极其陡峭(4-5x)，商务区次之，Hub最稳
            role_factors = {"Residential": 4.5, "Employment": 3.2, "Hub": 2.2}
            factor = role_factors.get(role, 2.5)
            
            if 7 <= hour < 9: # 早高峰：居住区大规模流出
                peak_factor = factor if role == "Residential" else (factor * 0.7)
            elif 17 <= hour < 19: # 晚高峰：商务区大规模流出
                peak_factor = factor if role == "Employment" else (factor * 0.7)
            elif 11 <= hour < 14:
                peak_factor = 1.6
            else:
                peak_factor = 1.0
            
            # 转换为每步生成的 lambda
            hourly_rate = rate_offpeak * peak_factor
            rate = max(0.01, hourly_rate / 720.0)
            low, high = rate * 0.85, rate * 1.15
        else:
            # 降级逻辑
            if 7 <= hour < 9 or 17 <= hour < 19:
                low, high = 1.0, 1.5
            else:
                low, high = 0.3, 0.5

        if self.waiting_passengers is None:
            raise RuntimeError("Passenger tensor not initialised")

        device = self.waiting_passengers.device
        if all_platform_nodes and getattr(self, "_last_distribution_hour", None) != hour:
            self._init_destination_distribution_tidal(all_platform_nodes, hour)
            self._last_distribution_hour = hour
        if not self.destinations and all_platform_nodes:
            self._init_destination_distribution_tidal(all_platform_nodes, hour)
        self.ensure_od_queues()
        events = world.event_view() if world else None
        multiplier = events.demand_multiplier(node.id) if events else 1.
        weights = events.od_weights(node.id, self.destinations) if events else dict(self.destinations)
        indices = range(len(self.waiting_passengers)) if env_index is None else [env_index]
        for idx in indices:
            count = float(torch.poisson(torch.empty(1, device=device).uniform_(low, high) * multiplier))
            for dest, weight in weights.items():
                queue = self.outside_by_destination.setdefault(dest, torch.zeros_like(self.waiting_passengers))
                queue[idx] += count * weight
            self.admit(idx, events=events)

        # --- 2. 潮汐目的地分配 (Dynamic Destination Bias) ---
        # 如果当前小时改变，重新计算目的地分布以模拟潮汐

    def _init_destination_distribution(self, all_platform_nodes):
        """Compatibility with Node's original initialization entry point."""
        world = getattr(self, "world", None)
        seconds = float(world.sim_time[0]) + world.clock_start_seconds if world else 0.
        self._init_destination_distribution_tidal(all_platform_nodes, int(seconds // 3600) % 24)

    def _init_destination_distribution_tidal(self, all_platform_nodes: Dict[int, "Node"], hour: int) -> None:
        """引入朝向偏移（早高峰进城，晚高峰回城）"""
        total_weight = 0.0
        self.destinations = {}
        
        od_profile = getattr(self.node, "od_profile", None)
        origin_role = od_profile.get("role", "Residential") if od_profile else "Residential"
        od_matrix_ref = od_profile.get("od_matrix_ref", {}) if od_profile else {}
        origin_name = (
            getattr(self.node, "station_name", None)
            or getattr(self.node, "platform_name", None)
            or getattr(self.node, "name", None)
            or str(self.node.id)
        )
        origin_key = str(origin_name).replace("上行", "").replace("下行", "").strip()
        
        for node_id, node in all_platform_nodes.items():
            if node_id == self.node.id or node.type is not NodeType.PLATFORM:
                continue

            dest_name = (
                getattr(node, "station_name", None)
                or getattr(node, "platform_name", None)
                or getattr(node, "name", None)
                or str(node_id)
            )
            matching_name = str(dest_name).replace("上行", "").replace("下行", "").strip()
            dest_role = ""
            # 从外部字典或属性获取目的地角色（此处简化处理）
            # 实际中应从 node.od_profile 获取
            dest_profile = getattr(node, "od_profile", None)
            dest_role = dest_profile.get("role", "Residential") if dest_profile else "Residential"
            
            weight = float(od_matrix_ref.get(origin_key, {}).get(matching_name, 1.0))
            
            # --- 潮汐偏置 (Tidal Bias) & 反向噪声 (Counter-flow Noise) ---
            # 用户要求反向 OD 提到正向的 10-15%
            primary_boost = 3.5
            counter_noise_ratio = 0.12 # 目标反向占正向的 12%
            counter_flow_factor = primary_boost * counter_noise_ratio # 约 0.42
            
            if 7 <= hour < 9: # 早高峰：倾向于流向 Employment
                if dest_role == "Employment": weight *= primary_boost
                if dest_role == "Residential": weight *= counter_flow_factor
            elif 17 <= hour < 19: # 晚高峰：倾向于流向 Residential
                if dest_role == "Residential": weight *= primary_boost
                if dest_role == "Employment": weight *= counter_flow_factor

            # --- 3. 打破整齐感 (Random Perturbation) ---
            # 同层站点加 ±15-25% 随机扰动，避免矩阵过于“完美”
            noise = 1.0 + (torch.rand(1).item() * 0.4 - 0.2) # ±20% 扰动
            weight *= noise
            
            # 非线性距离衰减修正：对远距离 CBD 保持吸引力
            # (这里因为已有 weight 矩阵，若矩阵已是重力模型，我们通过 role 修正它)

            if weight > 0:
                self.destinations[node_id] = weight
                total_weight += weight

        if total_weight > 0:
            for node_id in self.destinations:
                self.destinations[node_id] /= total_weight
