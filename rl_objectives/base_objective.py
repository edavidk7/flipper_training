from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from flipper_training.configs import PhysicsEngineConfig, RobotModelConfig, WorldConfig
from flipper_training.engine.engine_state import PhysicsState


@dataclass
class BaseObjective(ABC):
    """
    Base class for an objective that manages the generation of start/goal positions for the robots in the environment,
    as well as the reward function and termination condition.
    """

    device: torch.device | str
    physics_config: PhysicsEngineConfig
    robot_model: RobotModelConfig
    world_config: WorldConfig
    rng: torch.Generator

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
