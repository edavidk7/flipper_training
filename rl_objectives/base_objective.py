import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch

from flipper_training.configs import PhysicsEngineConfig, RobotModelConfig, WorldConfig
from flipper_training.engine.engine_state import PhysicsState


@dataclass
class BaseObjective(ABC):
    """
    Base class for an objective that manages the generation of start/goal positions for the robots in the environment,
    as well as the reward function and termination condition.

    Attributes:
        num_robots: Number of robots in the environment.
        robot_model: RobotModelConfig object containing the configuration of the robot.
        world_config: WorldConfig object containing the configuration of the world.
        rng: Random number generator.
        _supports_cache: Boolean indicating whether the objective supports caching of the start/goal positions.
        _cache_capacity: Maximum number of start/goal positions to cache.
        _supports_compile: Boolean indicating whether the objective supports compilation of its functions.
    """

    device: torch.device | str
    physics_config: PhysicsEngineConfig
    robot_model: RobotModelConfig
    world_config: WorldConfig
    rng: torch.Generator
    _supports_cache: ClassVar[bool]
    _supports_compile: ClassVar[bool]

    def __post_init__(self) -> None:
        if self._supports_cache:
            self._init_cache()

    def compile(self, **compile_opts) -> None:
        """
        Compiles the functions of the objective for faster inference.
        """
        if not self._supports_compile:
            warnings.warn(
                "This objective doesn't support compilation. The compile_functions will do nothing.", stacklevel=1
            )
            return
        self.generate_start_goal_states = torch.compile(self.generate_start_goal_states, **compile_opts)
        self.check_reached_goal = torch.compile(self.check_reached_goal, **compile_opts)
        self.check_terminated_wrong = torch.compile(self.check_terminated_wrong, **compile_opts)

    def _init_cache(self):
        """
        Initializes the cache for the start/goal positions.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_start_goal_states(self) -> tuple[PhysicsState, PhysicsState, torch.IntTensor | torch.LongTensor]:
        """
        Generates start/goal positions for the robots in the environment.

        The world configuration must contain a suitability mask that indicates which parts of the world are suitable for start/goal positions.

        Returns:
        - A tuple of PhysicsState objects containing the start and goal positions for the robots.
        - A tensor containing the iteration limits for the robots.

        """

    @abstractmethod
    def check_reached_goal(self, state: PhysicsState, goal: PhysicsState) -> torch.BoolTensor:
        """
        Check if the robots have reached the goal.

        This function should be as efficient as possible, as it is called every iteration.

        Args:
        - state: PhysicsState object containing the current state of the robot.
        - goal: PhysicsState object containing the goal state of the robot.

        Returns:
        - Tensor containing a boolean indicating whether each
        """

    @abstractmethod
    def check_terminated_wrong(self, state: PhysicsState, goal: PhysicsState) -> torch.BoolTensor:
        """
        Check if the robots have terminated due to reaching an infeasible/illegal state.

        This function should be as efficient as possible, as it is called every iteration.

        Args:
        - state: PhysicsState object containing the current state of the robot.
        - goal: PhysicsState object containing the goal state of the robot.

        Returns:
        - Tensor containing a boolean indicating whether each robot has terminated.
        """
