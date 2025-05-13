import torch
import logging
from tqdm import trange
from typing import override
from dataclasses import dataclass
from flipper_training.engine.engine_state import PhysicsState
from flipper_training.rl_objectives import BaseObjective


@dataclass
class MixedObjective(BaseObjective):
    classes: list[type[BaseObjective]]
    opts: list[dict]
    cache_size: int = 0
    _cache_cursor: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.env.terrain_cfg is None) or ("typelist" not in self.env.terrain_cfg.grid_extras):
            raise ValueError("Terrain config must contain a typelist specifying origin of each heightmap.")
        self.objectives = [cls(env=self.env, rng=self.rng, **opts) for cls, opts in zip(self.classes, self.opts)]
        self._init_cache()

    def state_dict(self):
        return {
            "cache_cursor": self._cache_cursor,
        }

    def load_state_dict(self, state_dict):
        self._cache_cursor = state_dict["cache_cursor"]

    def _init_cache(self) -> None:
        B = self.physics_config.num_robots
        total_needed_per_robot = self.cache_size  # Number of cache entries per robot
        # Initialize cache tensors
        start_pos_cache = torch.empty((self.cache_size, B, 3), dtype=torch.float32)
        goal_pos_cache = torch.empty((self.cache_size, B, 3), dtype=torch.float32)
        step_limits_cache = torch.empty((self.cache_size, B), dtype=torch.int32)
        orientation_cache = torch.empty((self.cache_size, B, 4), dtype=torch.float32)
        joint_angles_cache = torch.empty((self.cache_size, B, self.robot_model.num_driving_parts), dtype=torch.float32)
        # Process each robot's terrain separately
        for b in trange(B, desc="Initializing start/goal position cache"):
            objective_index = self.env.terrain_cfg.grid_extras["indexlist"][b].item()
            objective = self.objectives[objective_index]
            if not hasattr(objective, "_generate_cache_states"):
                raise ValueError(f"Objective {objective.__class__.__name__} does not implement _generate_cache_states.")
            start_pos_cache[:, b], goal_pos_cache[:, b] = objective._generate_cache_states(b, total_needed_per_robot)
            if not hasattr(objective, "_compute_step_limits"):
                raise ValueError(f"Objective {objective.__class__.__name__} does not implement _compute_step_limits.")
            step_limits_cache[:, b] = objective._compute_step_limits(start_pos_cache[:, b], goal_pos_cache[:, b])
            joint_angles_cache[:, b] = objective._get_initial_joint_angles(self.cache_size)
            if not hasattr(objective, "_get_initial_orientation_quat"):
                raise ValueError(f"Objective {objective.__class__.__name__} does not implement _get_initial_orientation_quat.")
            orientation_cache[:, b] = objective._get_initial_orientation_quat(start_pos_cache[:, b], goal_pos_cache[:, b])
        # Store the cache
        self.cache = {
            "start": start_pos_cache,
            "goal": goal_pos_cache,
            "ori": orientation_cache,
            "step_limits": step_limits_cache,
            "joint_angles": joint_angles_cache,
        }
        self._cache_cursor = 0

    @override
    def generate_start_goal_states(self) -> tuple[PhysicsState, PhysicsState, torch.IntTensor | torch.LongTensor]:
        """
        Generates start/goal positions for the robots in the environment.

        The world configuration must contain a suitability mask that indicates which parts of the world are suitable for start/goal positions.

        Returns:
        - A tuple of PhysicsState objects containing the start and goal positions for the robots.
        - A tensor containing the iteration limits for the robots.
        """
        if self._cache_cursor < self.cache_size:
            start_state, goal_state = self._construct_full_start_goal_states(
                self.cache["start"][self._cache_cursor],
                self.cache["goal"][self._cache_cursor],
                self.cache["ori"][self._cache_cursor],
                self.cache["joint_angles"][self._cache_cursor],
            )
            step_limits = self.cache["step_limits"][self._cache_cursor].to(self.device)
            self._cache_cursor += 1
            return start_state, goal_state, step_limits
        else:
            logging.warning("Start/goal cache exhausted, doubling size and generating new start/goal positions")
            self.cache_size *= 2
            self._init_cache()
            return self.generate_start_goal_states()

    def _construct_full_start_goal_states(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        orientation: torch.Tensor,
        joint_angles: torch.Tensor,
    ) -> tuple[PhysicsState, PhysicsState]:
        """
        Constructs full start and goal states for the robots in the environment.

        Args:
            start_pos (torch.Tensor): The starting positions of the robots.
            goal_pos (torch.Tensor): The goal positions of the robots.
            orientation (torch.Tensor): The orientations of the robots.
            joint_angles (torch.Tensor): The joint angles of the robots.

        Returns:
            tuple[PhysicsState, PhysicsState]: A tuple containing the start and goal states.
        """
        B = self.physics_config.num_robots
        start_pos = start_pos.to(self.device)
        goal_pos = goal_pos.to(self.device)
        orientation = orientation.to(self.device)
        joint_angles = joint_angles.to(self.device)
        start_state = PhysicsState.dummy(robot_model=self.robot_model, batch_size=B, device=self.device)
        goal_state = PhysicsState.dummy(robot_model=self.robot_model, batch_size=B, device=self.device)
        for i, objective in enumerate(self.objectives):  # Iterate over each objective
            m = self.env.terrain_cfg.grid_extras["indexlist"] == i
            start_state[m], goal_state[m] = objective._construct_full_start_goal_states(
                start_pos[m],
                goal_pos[m],
                orientation[m],
                joint_angles[m],
            )
        return start_state, goal_state

    @torch.compile
    def check_reached_goal(self, prev_state: PhysicsState, state: PhysicsState, goal: PhysicsState) -> torch.BoolTensor:
        m = torch.zeros_like(state.x[:, 0], dtype=torch.bool)
        for i, objective in enumerate(self.objectives):
            m_i = self.env.terrain_cfg.grid_extras["indexlist"] == i
            m |= objective.check_reached_goal(prev_state, state, goal) & m_i
        return m

    @torch.compile
    def check_terminated_wrong(self, prev_state: PhysicsState, state: PhysicsState, goal: PhysicsState) -> torch.BoolTensor:
        m = torch.zeros_like(state.x[:, 0], dtype=torch.bool)
        for i, objective in enumerate(self.objectives):
            m_i = self.env.terrain_cfg.grid_extras["indexlist"] == i
            m |= objective.check_reached_goal(prev_state, state, goal) & m_i
        return m

    def start_goal_to_simview(self, start: PhysicsState, goal: PhysicsState):
        try:
            from simview import SimViewStaticObject, BodyShapeType
        except ImportError:
            logging.warning("SimView is not installed. Cannot visualize start/goal positions.")
            return {}

        # start
        pos = start.x
        start_object = SimViewStaticObject.create_batched(
            name="Start",
            shape_type=BodyShapeType.POINTCLOUD,
            shapes_kwargs=[
                {
                    "points": pos[i, None],
                    "color": "#ff0000",
                }
                for i in range(pos.shape[0])
            ],
        )
        # goal
        pos = goal.x
        goal_object = SimViewStaticObject.create_batched(
            name="Goal",
            shape_type=BodyShapeType.POINTCLOUD,
            shapes_kwargs=[
                {
                    "points": pos[i, None],
                    "color": "#0000ff",
                }
                for i in range(pos.shape[0])
            ],
        )
        return [start_object, goal_object]
