
import random
from typing import Optional

import torch

from envwrapper import MetroRLEnv, EnvWrapper
from metro_environment import MetroEnv
from metro_scenario_v2 import MetroScenarioV2
from linedata import load_topology


class MetroRLEnvV2(MetroRLEnv):
    """MetroRLEnv variant wired to MetroScenarioV2/MetroAgentV2."""

    def _build_env(
        self,
        num_envs: int,
        max_steps: Optional[int],
        seed: Optional[int],
        **scenario_kwargs,
    ) -> MetroEnv:
        canonical_seed = scenario_kwargs.pop("canonical_dispatch_seed", None)
        load_topology(self.topology)

        merged_kwargs = {
            **scenario_kwargs,
            "topology": self.topology,
            "enable_random_blockages": self.enable_random_blockages,
            "enable_random_speed_limits": self.enable_random_speed_limits,
            "enable_simulation_blockages": self.enable_simulation_blockages,
            "enable_simulation_speed_limits": self.enable_simulation_speed_limits,
            "default_interval": self.default_interval,
        }

        scenario_config = dict(merged_kwargs)
        episode_length = scenario_config.pop("episode_length", None)
        start_step = scenario_config.pop("start_step", None)
        if max_steps is not None and "max_steps" not in scenario_config:
            scenario_config["max_steps"] = max_steps
        scenario_config.setdefault("n_agents", scenario_config)

        self.scenario = MetroScenarioV2(**scenario_config)

        env_kwargs = dict(merged_kwargs)
        env_kwargs.pop("episode_length", None)
        env_kwargs.pop("start_step", None)

        reset_world_factory = None
        if canonical_seed is not None:
            if num_envs != 1 or torch.device(self.device).type != "cpu":
                raise ValueError("Canonical native dispatch benchmarks require one CPU physical world")
            from scenarios.prepare_dispatch import make_world

            def reset_world_factory(episode_seed):
                world = make_world(self.topology, fleet=self.scenario.n_agents,
                    seed=canonical_seed if episode_seed is None else episode_seed,
                    events=merged_kwargs.get("events", ()), tick=self.default_interval,
                    initial_passenger_seconds=self.default_interval)
                self.scenario.world = world
                self.scenario.num_envs = world.num_envs
                self.scenario.device = world.device
                self.scenario._headway_tracker = world.agents[0].headway_tracker
                self.scenario._position_scale = float(max(world.nodes)) or 1.
                self.scenario._max_congestion = float(max(1, len(world.agents)-1))
                return world

        env = MetroEnv(
            scenario=self.scenario,
            num_envs=num_envs,
            max_steps=max_steps,
            seed=seed,
            episode_length=episode_length,
            start_step=start_step,
            reset_world_factory=reset_world_factory,
            **env_kwargs,
        )

        original_reset_world_at = self.scenario.env_reset_world_at

        def custom_reset_world_at(env_index: Optional[int] = None) -> None:
            # During initial bootstrap we randomize positions while forcing platform starts.
            if getattr(self.scenario, "_run_ready", False):
                original_reset_world_at(env_index)
                return

            self.scenario.world.init_random_weights()
            for agent in self.scenario.world.agents:
                agent.reset_state(env_index)

            if env_index is None:
                random_positions = []
                for env_idx in range(self.scenario.num_envs):
                    if self.init_at_platforms:
                        platform_nodes = list(self.scenario.world.platform_nodes.keys())
                        if len(platform_nodes) >= self.scenario.n_agents:
                            candidates = platform_nodes
                        else:
                            candidates = list(self.scenario.world.nodes.keys())
                    else:
                        candidates = list(self.scenario.world.nodes.keys())
                    random_positions.append(random.sample(candidates, self.scenario.n_agents))

                for agent_idx, agent in enumerate(self.scenario.world.agents):
                    positions = torch.zeros(
                        self.scenario.num_envs,
                        device=self.scenario.device,
                    )
                    for env_idx in range(self.scenario.num_envs):
                        positions[env_idx] = random_positions[env_idx][agent_idx]
                    agent.init_position(positions)
            else:
                original_reset_world_at(env_index)

        # self.scenario.env_reset_world_at = custom_reset_world_at

        world = getattr(self.scenario, "world", None)
        if world is not None and not getattr(world, "second_precision", False):
            # Native dispatch already separates its one-second physical clock
            # from the decision interval; never replace that physical clock.
            setattr(world, "tick_seconds", self.default_interval)

        return EnvWrapper._build_env(self, env=env)
