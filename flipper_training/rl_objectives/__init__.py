import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
from functools import wraps

if TYPE_CHECKING:
    from flipper_training.environment.env import Env
    from flipper_training.engine.engine_state import PhysicsState
    from flipper_training.configs.terrain_config import TerrainConfig
    from flipper_training.configs.engine_config import PhysicsEngineConfig

    try:
        from simview import SimViewStaticObject
    except ImportError:
        SimViewStaticObject = None


@dataclass(kw_only=True)
class BaseObjective(ABC):
    """
    Base class for an objective that manages the generation of start/goal positions for the robots in the environment,
    as well as the reward function and termination condition.

    Attributes:
    - env: The environment in which the robots operate.
    - rng: Random number generator.
    Methods:
    - generate_start_goal_states: Generates start/goal positions for the robots in the environment.
    - check_reached_goal: Checks if the robots have reached the goal.
    - check_terminated_wrong: Checks if the robots have terminated due to reaching an infeasible/illegal state.

    """

    env: "Env"
    rng: torch.Generator
    terrain_config: "TerrainConfig | None" = None
    physics_config: "PhysicsEngineConfig | None" = None

    def __post_init__(self):
        """
        Post-initialization method to unpack the environment configuration.
        """
        self.device = self.env.device
        self.robot_model = self.env.robot_cfg
        if self.physics_config is None:
            self.physics_config = self.env.phys_cfg
        if self.terrain_config is None:
            self.terrain_config = self.env.terrain_cfg

    @property
    def name(self) -> str:
        """
        Returns the name of the objective.
        """
        return self.__class__.__name__

    @classmethod
    def make_factory(cls, **opts):
        """
        Factory method to create a reward function with the given options.
        """

        @wraps(cls)
        def factory(env: "Env"):
            return cls(env=env, **opts)

        return factory

    @abstractmethod
    def generate_start_goal_states(self) -> tuple["PhysicsState", "PhysicsState", torch.IntTensor | torch.LongTensor]:
        """
        Generates start/goal positions for the robots in the environment.

        The world configuration must contain a suitability mask that indicates which parts of the world are suitable for start/goal positions.

        Returns:
        - A tuple of PhysicsState objects containing the start and goal positions for the robots.
        - A tensor containing the step limits for the robots.
        """

    @abstractmethod
    def check_reached_goal(self, prev_state: "PhysicsState", state: "PhysicsState", goal: "PhysicsState") -> torch.BoolTensor:
        """
        Check if the robots have reached the goal.

        This function should be as efficient as possible, as it is called every iteration.

        Args:
        - prev_state: PhysicsState object containing the previous state of the robot.
        - state: PhysicsState object containing the current state of the robot.
        - goal: PhysicsState object containing the goal state of the robot.

        Returns:
        - Tensor containing a boolean indicating whether each
        """

    def state_dict(self):
        """
        Returns the state dictionary of the objective. Useful for saving curriculum progress.
        """
        return {}

    def load_state_dict(self, state_dict):
        """
        Loads the state dictionary of the objective. Useful for loading curriculum progress.
        """
        return None

    @abstractmethod
    def check_terminated_wrong(self, prev_state: "PhysicsState", state: "PhysicsState", goal: "PhysicsState") -> torch.BoolTensor:
        """
        Check if the robots have terminated due to reaching an infeasible/illegal state.

        This function should be as efficient as possible, as it is called every iteration.

        Args:
        - prev_state: PhysicsState object containing the previous state of the robot.
        - state: PhysicsState object containing the current state of the robot.
        - goal: PhysicsState object containing the goal state of the robot.

        Returns:
        - Tensor containing a boolean indicating whether each robot has terminated.
        """

    def start_goal_to_simview(self, start: "PhysicsState", goal: "PhysicsState") -> list["SimViewStaticObject"]:
        """
        Converts the start and goal states to a dictionary of body states for visualization.

        Args:
        - start: PhysicsState object containing the start state of the robot.
        - goal: PhysicsState object containing the goal state of the robot.

        Returns:
        - A dictionary containing the objects to be visualized in SimView.
        """
        raise NotImplementedError("Not implemented for this objective")

    def reset(self, reset_mask: torch.Tensor, training: bool) -> None:
        """
        Reset the internal state of the objective. Should be called when the environment is reset.
        Args:
        - reset_mask: A tensor indicating which robots should be reset.
        - training: A boolean indicating whether the environment is in training mode.
        """
        return None


ObjectiveFactory = Callable[["Env"], BaseObjective]
