import torch
from dataclasses import dataclass
from typing import ClassVar, Callable
from abc import ABC, abstractmethod
from flipper_training.configs import RobotModelConfig, WorldConfig
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer


@dataclass
class BaseObjective(ABC):
    """
    Base class for an objective that manages the generation of start/goal positions for the robots in the environment, as well as the reward function and termination condition.

    Attributes:
    - min_dist_to_goal: float - minimum distance from start to goal in meters
    - max_dist_to_goal: float - maximum distance from start to goal in meters
    - max_goal_generation_attempts: int - maximum number of attempts to generate a goal
    - feasability_thresh: float - threshold for feasibility check of start/goal positions, used to avoid floating point errors
    """

    min_dist_to_goal: float = 1.  # meters
    max_dist_to_goal: float = torch.inf  # meters
    max_goal_generation_attempts: ClassVar[int] = 1000
    feasability_thresh: float = 1 - 1e-6  # threshold for feasibility check, used to avoid floating point errors

    def generate_start_goal_states(self, world_config: WorldConfig, robot_model: RobotModelConfig, rng: torch.Generator | None = None, skip_mask: torch.Tensor | None = None) -> tuple[PhysicsState, PhysicsState]:
        """
            Generates start/goal positions for the robots in the environment.

            The world configuration must contain a suitability mask that indicates which parts of the world are suitable for start/goal positions.

            Args:
            - world_config: WorldConfig object containing the configuration of the world.
            - robot_model: RobotModelConfig object containing the configuration of the robot.
            - rng: Random number generator.
            - skip_mask: Tensor containing a mask which robots to skip (for example if they haven't reached the previous goal yet).

            Returns:
            - A tuple of PhysicsState objects containing the start and goal positions for the robots.
            """
        B = world_config.z_grid.shape[0]
        device = world_config.z_grid.device
        starts = torch.zeros((B, 3), device=device)
        goals = torch.zeros((B, 3), device=device)
        for i in range(B):
            if skip_mask is not None and skip_mask[i]:
                continue
            while True:
                start = self._generate_start(i, world_config, robot_model, device, rng)
                goal, succ = self._generate_goal(start, i, world_config, robot_model, device, rng)
                if succ:
                    starts[i] = start
                    goals[i] = goal
                    break
        start_state = PhysicsState.dummy(x=starts, robot_model=robot_model)
        goal_state = PhysicsState.dummy(x=goals, robot_model=robot_model)
        return start_state, goal_state

    @abstractmethod
    def check_reached_goal_or_terminate(self, state: PhysicsState, goal: PhysicsState) -> torch.Tensor:
        """
        Check if the robots have reached the goal or if the episode should be terminated due to an illegal state.

        This function should be as efficient as possible, as it is called every iteration.

        Args:
        - state: PhysicsState object containing the current state of the robot.
        - goal: PhysicsState object containing the goal state of the robot.

        Returns:
        - Tensor containing a boolean indicating whether each
        """
        raise NotImplementedError

    @abstractmethod
    def _generate_start(self, robot_idx: int, world_config: WorldConfig, robot_model: RobotModelConfig, device: torch.device | str, rng: torch.Generator | None = None) -> torch.Tensor:
        """
        Generates the start position for the robot in the environment.

        Args:
        - robot_idx: Index of the robot.
        - world_config: WorldConfig object containing the configuration of the world.
        - robot_model: RobotModelConfig object containing the configuration of the robot.
        - device: Device on which to generate the start.
        - rng: Random number generator.

        Returns:
        - tensor containing the start position.
        """
        raise NotImplementedError

    @abstractmethod
    def _generate_goal(self, start: torch.Tensor, robot_idx: int, world_config: WorldConfig, robot_model: RobotModelConfig, device: torch.device | str, rng: torch.Generator | None = None) -> tuple[torch.Tensor, bool]:
        """
        Generates the goal position for the robot in the environment.

        Args:
        - start: tensor containing the start position.
        - robot_idx: Index of the robot.
        - world_config: WorldConfig object containing the configuration of the world.
        - robot_model: RobotModelConfig object containing the configuration of the robot.
        - device: Device on which to generate the goal.
        - rng: Random number generator.

        Returns:
        - tuple containing the goal position and a boolean indicating whether the goal is feasible.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_iteration_limits(self, start_state: PhysicsState, goal_state: PhysicsState, robot_model: RobotModelConfig, dt: float) -> torch.Tensor:
        """
        Compute the iteration limits for the robots in the environment.

        Args:
        - start_state: PhysicsState object containing the start state of the robots.
        - goal_state: PhysicsState object containing the goal state of the robots.
        - robot_model: RobotModelConfig object containing the configuration of the robot.
        - dt: Time step.

        Returns:
        - Tensor containing the iteration limits for the robots.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_reward(self,
                       prev_state: PhysicsState,
                       action: torch.Tensor,
                       curr_state: PhysicsState,
                       goal: PhysicsState,
                       ) -> torch.Tensor:
        """
        Compute the reward for the given state and goal.

        Args:
        - prev_state: PhysicsState object containing the previous state of the robots.
        - action: Tensor containing the action taken by the robots.
        - curr_state: PhysicsState object containing the current state of the robots.
        - goal: PhysicsState object containing the goal state of the robots

        Returns:
            The reward for the given state and goal.
        """
        raise NotImplementedError
