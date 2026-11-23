"""Route selection policies and adapters for train agents."""

from typing import Optional
import random


class RouteSelectionPolicy:
    """Base policy for selecting routes at switch nodes."""

    def choose_route(self, env_index: int, current_node_id: int, route_keys, agent) -> Optional[object]:
        raise NotImplementedError


class RandomRouteSelectionPolicy(RouteSelectionPolicy):
    """Randomly choose among the available route options, with optional default bias."""

    def __init__(self, prefer_default: bool = True, default_bias: float = 0.6) -> None:
        self.prefer_default = bool(prefer_default)
        self.default_bias = float(max(0.0, min(default_bias, 1.0)))

    def choose_route(self, env_index: int, current_node_id: int, route_keys, agent) -> Optional[object]:
        if not route_keys:
            return None
        if len(route_keys) == 1:
            return route_keys[0]

        route_keys = list(route_keys)
        default_key = None
        route_obj = getattr(agent, "route", None)
        if route_obj is not None:
            default_key = getattr(route_obj, "id", route_obj)
        if default_key is None:
            adapter = getattr(agent, "route_adapter", None)
            default_key = getattr(adapter, "default_route_key", None)

        if (
            self.prefer_default
            and default_key in route_keys
            and random.random() < self.default_bias
        ):
            return default_key

        candidates = [key for key in route_keys if key != default_key]
        if candidates:
            return random.choice(candidates)
        return random.choice(route_keys)


class RouteSelectionAdapter:
    """Adapter that applies a route-selection policy and keeps per-env route state."""

    def __init__(self, policy: RouteSelectionPolicy, num_envs: int, default_route_key=None):
        self.policy = policy
        self.default_route_key = default_route_key
        self._current_route_key = [default_route_key] * num_envs
        self._route_key_to_index = {}

    def _should_force_random(self, agent, env_index: int) -> bool:
        if getattr(getattr(agent, "world", None), "dispatch", None):
            return False
        action_tensor = getattr(agent, "action", None)
        if action_tensor is None:
            return False

        try:
            current_action = int(action_tensor[env_index].item())
        except (IndexError, AttributeError, TypeError, ValueError):
            return False

        default_value = getattr(agent, "_default_move_action_value", None)
        if default_value is None:
            action_enum = getattr(agent, "Action", None)
            if action_enum is not None:
                default_value = getattr(action_enum, "MOVE_DEFAULT", None)

        if default_value is None:
            default_value = getattr(self, "_default_move_action_value", None)

        if default_value is None:
            default_value = 1

        return current_action == default_value

    def _ensure_size(self, agent) -> None:
        num_envs = getattr(agent, "num_envs", len(self._current_route_key))
        current_length = len(self._current_route_key)
        if current_length < num_envs:
            fill_value = self.default_route_key
            self._current_route_key.extend([fill_value] * (num_envs - current_length))
        elif current_length > num_envs:
            self._current_route_key = self._current_route_key[:num_envs]

    def _apply_route(self, agent, env_index: int, route_key) -> None:
        self._ensure_size(agent)
        if env_index >= len(self._current_route_key):
            return

        self._current_route_key[env_index] = route_key

        state = getattr(agent, "state", None)
        if state is None:
            return

        if route_key is None:
            state.current_route[env_index] = -1
            return

        if isinstance(route_key, (int, float)):
            state.current_route[env_index] = int(route_key)
            return

        index = self._route_key_to_index.get(route_key)
        if index is None:
            index = len(self._route_key_to_index) + 1
            self._route_key_to_index[route_key] = index
        state.current_route[env_index] = index

    def reset_all(self, agent) -> None:
        self._route_key_to_index.clear()
        self._current_route_key = [self.default_route_key] * getattr(agent, "num_envs", 0)
        for env_index in range(len(self._current_route_key)):
            self._apply_route(agent, env_index, self.default_route_key)

    def reset_env(self, agent, env_index: int) -> None:
        self._apply_route(agent, env_index, self.default_route_key)

    def ensure_route(self, agent, env_index: int, current_node_id: int):
        self._ensure_size(agent)

        world = getattr(agent, "world", None)
        if world is None:
            self._apply_route(agent, env_index, None)
            return None

        node = world.get_node(current_node_id)
        route_map = node.getnext() or {}
        available = {key: value for key, value in route_map.items() if value is not None}

        if not available:
            self._apply_route(agent, env_index, None)
            return None

        if self._should_force_random(agent, env_index):
            route_keys = list(available.keys())
            if not route_keys:
                self._apply_route(agent, env_index, None)
                return None

            if len(route_keys) == 1:
                chosen_key = route_keys[0]
            else:
                chosen_key = self.policy.choose_route(env_index, current_node_id, route_keys, agent)
                if chosen_key not in available:
                    chosen_key = route_keys[0]

            self._apply_route(agent, env_index, chosen_key)
            return chosen_key

        current_route = None
        if env_index < len(self._current_route_key):
            current_route = self._current_route_key[env_index]

        if current_route in available:
            self._apply_route(agent, env_index, current_route)
            return current_route

        if self.default_route_key in available:
            chosen_key = self.default_route_key
        else:
            route_keys = list(available.keys())
            if getattr(world, "dispatch", None):
                # A checked dispatch replay must not depend on global RNG calls
                # made while evaluating other candidate plans.
                chosen_key = sorted(route_keys, key=str)[0]
            else:
                chosen_key = self.policy.choose_route(env_index, current_node_id, route_keys, agent)
            if chosen_key not in available:
                chosen_key = route_keys[0]

        self._apply_route(agent, env_index, chosen_key)
        return chosen_key

    def next_node(self, agent, env_index: int, current_node_id: int):
        chosen_route = self.ensure_route(agent, env_index, current_node_id)
        if chosen_route is None:
            return None

        world = getattr(agent, "world", None)
        if world is None:
            return None

        node = world.get_node(current_node_id)
        route_map = node.getnext() or {}
        return route_map.get(chosen_route)

    def next_node_by_direction(self, agent, env_index: int, current_node_id: int, direction_index: int = 0):
        """Get next node based on explicit direction index.
        
        Args:
            agent: The train agent
            env_index: Environment index
            current_node_id: Current node ID
            direction_index: 0-based index of direction (0 = first/default direction)
        
        Returns:
            Next node ID if available, None otherwise.
            
        If the requested direction_index is not available, falls back to the first
        available direction (direction_index=0).
        """
        world = getattr(agent, "world", None)
        if world is None:
            return None

        node = world.get_node(current_node_id)
        route_map = node.getnext() or {}
        available = {key: value for key, value in route_map.items() if value is not None}

        if not available:
            return None

        # 获取所有可用方向的route keys（排序以保证一致性）
        route_keys = sorted(available.keys(), key=lambda x: str(x))
        
        # 如果请求的方向不存在，回退到第一个方向
        if direction_index < 0 or direction_index >= len(route_keys):
            direction_index = 0
        
        chosen_key = route_keys[direction_index]
        
        # 更新当前路由状态
        self._apply_route(agent, env_index, chosen_key)
        
        return available.get(chosen_key)
    
    def get_available_directions(self, agent, current_node_id: int) -> int:
        """Get the number of available movement directions from current node.
        
        Returns:
            Number of available directions (0 if no movement possible).
        """
        world = getattr(agent, "world", None)
        if world is None:
            return 0

        node = world.get_node(current_node_id)
        route_map = node.getnext() or {}
        available = {key: value for key, value in route_map.items() if value is not None}
        return len(available)
