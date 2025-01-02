import torch
from dataclasses import dataclass
from flipper_training.configs import RobotModelConfig, WorldConfig
from flipper_training.utils.environment import interpolate_grid
from flipper_training.engine.engine_state import PhysicsState
from flipper_training.rl_objectives.base_objective import BaseObjective


@dataclass
class SimpleDrivingObjective(BaseObjective):
    """
    Objective manager that generates start/goal positions for the robots in the environment. The start position is generated randomly within the suitable area of the world, while the goal position is generated randomly within the suitable area of the world and at a minimum distance from the start position, but not further than the maximum distance. It is ensured that the goal position is not too high above the start position (to avoid unreachable goals).
    """

    higher_allowed: float = 0.5  # meters
    min_dist_to_goal: float = 0.1  # meters
    start_drop: float = 0.1  # meters
    iteration_limit_factor: float = 2.

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
        for _ in range(self.max_goal_generation_attempts):
            goal = torch.rand((1, 1, 2), generator=rng, device=device) * 2 * max_feasible_coord - max_feasible_coord  # interval [-max_feasible_coord, max_feasible_coord]
            feasibility = interpolate_grid(world_config.suitable_mask[None, robot_idx], goal, world_config.max_coord).item()
            if feasibility >= self.feasability_thresh and self.min_dist_to_goal <= torch.norm(goal - start[..., :2]) <= self.max_dist_to_goal:
                z = interpolate_grid(world_config.z_grid[None, robot_idx], goal, world_config.max_coord)
                if z - start[2] <= self.higher_allowed:
                    return torch.cat([goal, z], dim=-1).squeeze(), True
        return torch.zeros(3), False

    def check_reached_goal(self, state: PhysicsState, goal: PhysicsState) -> torch.Tensor:
        return torch.norm(state.x - goal.x, dim=-1) <= self.min_dist_to_goal

    def compute_iteration_limits(self, start_state: PhysicsState, goal_state: PhysicsState, dt: float) -> torch.Tensor:
        """
        Compute the iteration limits for the robots in the environment. In this case, the iteration limits are computed based on the L1 distance between the start and goal positions (manhattan distance), times self.iteration_limit_factor. This should allow the robot to reach the goal in a reasonable number of iterations.

        Args:
        - start_state: PhysicsState object containing the start state of the robots.
        - goal_state: PhysicsState object containing the goal state of the robots.
        - dt: Time step.

        Returns:
        - Tensor containing the iteration limits for the robots.

        """
        return torch.ceil(self.iteration_limit_factor * torch.norm(goal_state.x - start_state.x, dim=-1, p=1) / dt).long()
