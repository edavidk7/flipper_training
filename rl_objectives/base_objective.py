import torch
from dataclasses import dataclass
from typing import TypeVar, Generic, Union, List, ClassVar
from abc import ABC, abstractmethod
from flipper_training.configs import RobotModelConfig, WorldConfig
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer


ObjectiveLike = TypeVar("ObjectiveLike", bound=Union[torch.Tensor, PhysicsState])


@dataclass
class BaseObjective(ABC, Generic[ObjectiveLike]):
    """
    Base class for an objective that manages the generation of start/goal positions for the robots in the environment, as well as the reward function and termination condition.

    The class is generic over the type of the start/goal positions used internally by the generators.

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
        starts = []
        goals = []
        dummy = self._get_dummy_objective_like(device)
        for i in range(B):
            if skip_mask is not None and skip_mask[i]:
                starts.append(dummy)
                goals.append(dummy)
                continue
            while True:
                start = self._generate_start(i, world_config, robot_model, device, rng)
                goal, succ = self._generate_goal(start, i, world_config, robot_model, device, rng)
                if succ:
                    starts.append(start)
                    goals.append(goal)
                    break
        s, g = self._construct_full_start_goal_states(starts, goals, robot_model)
        s = s.to(device)
        g = g.to(device)
        return s, g

    @abstractmethod
    def _get_dummy_objective_like(self, device: str | torch.device) -> ObjectiveLike:
        """
        Returns a dummy objective-like object that can be used to skip the generation of start/goal states for a robot.

        Args:
        - device: Device on which to generate the dummy object. 
        Returns:
        - A dummy objective-like object.
        """
        raise NotImplementedError

    @abstractmethod
    def _construct_full_start_goal_states(self, partial_starts: List[ObjectiveLike], partial_goals: List[ObjectiveLike], robot_model: RobotModelConfig) -> tuple[PhysicsState, PhysicsState]:
        """
        Constructs full start/goal states from partial start/goal states. 
        This is for e.g. computing the rotation matrices to orient the robot towards the goal.

        Args:
        - partial_starts: List of partial start states.
        - partial_goals: List of partial goal states.
        - robot_model: RobotModelConfig object containing the configuration of the robot.

        Returns:
        - A tuple of PhysicsState objects containing the full start and goal states.
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    @abstractmethod
    def _generate_start(self, robot_idx: int, world_config: WorldConfig, robot_model: RobotModelConfig, device: torch.device | str, rng: torch.Generator | None = None) -> ObjectiveLike:
        """
            Generates the start state for the robot in the environment.

            Args:
            - robot_idx: Index of the robot.
            - world_config: WorldConfig object containing the configuration of the world.
            - robot_model: RobotModelConfig object containing the configuration of the robot.
            - device: Device on which to generate the start.
            - rng: Random number generator.

            Returns:
            - The start state for the robot.
            """
        raise NotImplementedError

    @abstractmethod
    def _generate_goal(self, start: ObjectiveLike, robot_idx: int, world_config: WorldConfig, robot_model: RobotModelConfig, device: torch.device | str, rng: torch.Generator | None = None) -> tuple[ObjectiveLike, bool]:
        """
        Generates the goal position for the robot in the environment.

        Args:
        - start: Start state of the robot.
        - robot_idx: Index of the robot.
        - world_config: WorldConfig object containing the configuration of the world.
        - robot_model: RobotModelConfig object containing the configuration of the robot.
        - device: Device on which to generate the goal.
        - rng: Random number generator.

        Returns:
        - tuple containing the goal state and a boolean indicating whether the goal is feasible.
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
