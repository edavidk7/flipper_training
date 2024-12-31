import torch
from dataclasses import dataclass
from typing import ClassVar, NamedTuple
from abc import ABC, abstractmethod
from flipper_training.configs import RobotModelConfig, WorldConfig
from flipper_training.utils.environment import interpolate_grid
from flipper_training.engine.engine_state import PhysicsState


@dataclass
class BaseObjectiveManager(ABC):
    """
    Base class for an objective manager that generates start/goal positions for the robots in the environment.

    Attributes:
    - min_dist_to_goal: float - minimum distance from start to goal in meters
    - max_dist_to_goal: float - maximum distance from start to goal in meters
    """

    min_dist_to_goal: float = 1.  # meters
    max_dist_to_goal: float = torch.inf  # meters
    max_goal_attempts: ClassVar[int] = 1000
    feasability_thresh: float = 1 - 1e-6  # threshold for feasibility check, used to avoid floating point errors

    def generate_start_goal_states(self, world_config: WorldConfig, robot_model: RobotModelConfig, rng: torch.Generator | None = None) -> tuple[PhysicsState, PhysicsState]:
        """
            Generates start/goal positions for the robots in the environment.

            The world configuration must contain a suitability mask that indicates which parts of the world are suitable for start/goal positions.

            Args:
            - world_config: WorldConfig object containing the configuration of the world.
            - robot_model: RobotModelConfig object containing the configuration of the robot.
            - rng: Random number generator.

            Returns:
            - A tuple of PhysicsState objects containing the start and goal positions for the robots.
            """
        B = world_config.z_grid.shape[0]
        device = world_config.z_grid.device
        starts = torch.zeros((B, 3), device=device)
        goals = torch.zeros((B, 3), device=device)
        for i in range(B):
            while True:
                start = self._generate_start(i, world_config, robot_model, device, rng)
                goal, succ = self._generate_goal(start, i, world_config, robot_model, device, rng)
                if succ:
                    starts[i] = start
                    goals[i] = goal
                    break
        start_state = PhysicsState.empty_dummy(x=starts)
        goal_state = PhysicsState.empty_dummy(x=goals)
        return start_state, goal_state

    @abstractmethod
    def check_reached_goal(self, state: PhysicsState, goal: PhysicsState) -> torch.Tensor:
        """
        Check if the robots have reached the goal.

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


@dataclass
class HeightAwareObjectiveManager(BaseObjectiveManager):
    """
    Objective manager that generates goals with awareness of the height difference between the start and goal positions such that the robot can reach the goal (it must not be much higher than the start position).
    """

    higher_allowed: float = 0.5  # meters
    min_dist_to_goal: float = 0.1  # meters
    start_drop: float = 0.1  # meters

    def _generate_start(self, robot_idx: int,
                        world_config: WorldConfig,
                        robot_model: RobotModelConfig,
                        device: torch.device | str,
                        rng: torch.Generator | None = None) -> torch.Tensor:
        assert world_config.suitable_mask is not None, "WorldConfig must contain a suitability mask."
        max_feasible_coord = world_config.max_coord - robot_model.radius
        while True:
            start = torch.rand((1, 1, 2), generator=rng, device=device) * 2 * max_feasible_coord - max_feasible_coord  # interval [-max_feasible_coord, max_feasible_coord]
            feasibility = interpolate_grid(world_config.suitable_mask[None, robot_idx], start, world_config.max_coord)
            if feasibility >= self.feasability_thresh:
                z = interpolate_grid(world_config.z_grid[None, robot_idx], start, world_config.max_coord)
                z += abs(robot_model.robot_points[..., 2].min()) + self.start_drop
                break
        return torch.cat([start, z], dim=-1).squeeze()

    def _generate_goal(self, start: torch.Tensor,
                       robot_idx: int,
                       world_config: WorldConfig,
                       robot_model: RobotModelConfig,
                       device: torch.device | str,
                       rng: torch.Generator | None = None) -> tuple[torch.Tensor, bool]:
        assert world_config.suitable_mask is not None, "WorldConfig must contain a suitability mask."
        max_feasible_coord = world_config.max_coord - robot_model.radius
        for _ in range(self.max_goal_attempts):
            goal = torch.rand((1, 1, 2), generator=rng, device=device) * 2 * max_feasible_coord - max_feasible_coord  # interval [-max_feasible_coord, max_feasible_coord]
            feasibility = interpolate_grid(world_config.suitable_mask[None, robot_idx], goal, world_config.max_coord).item()
            if feasibility >= self.feasability_thresh and self.min_dist_to_goal <= torch.norm(goal - start[..., :2]) <= self.max_dist_to_goal:
                z = interpolate_grid(world_config.z_grid[None, robot_idx], goal, world_config.max_coord)
                if z - start[2] <= self.higher_allowed:
                    return torch.cat([goal, z], dim=-1).squeeze(), True
        return torch.zeros(3), False

    def check_reached_goal(self, state: PhysicsState, goal: PhysicsState) -> torch.Tensor:
        return torch.norm(state.x - goal.x, dim=-1) <= self.min_dist_to_goal
