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

    Attributes:
    - higher_allowed: Maximum height difference between the start and goal positions.
    - min_dist_to_goal: Minimum distance between the start and goal positions.
    - start_drop: Height of the start position above the ground.
    - iteration_limit_factor: Factor to multiply the L1 distance between the start and goal positions to compute the iteration limits.
    - goal_reward: Reward for reaching the goal.
    - distance_weight: Weight for the distance reward component.
    - roll_rate_weight: Weight for the roll rate reward component.
    """

    higher_allowed: float = 0.5  # meters
    min_dist_to_goal: float = 0.1  # meters
    start_drop: float = 0.1  # meters
    iteration_limit_factor: float = 2.
    goal_reward: float = 1000.0
    distance_weight: float = 1.0
    roll_rate_weight: float = 0.0
    pitch_rate_weight: float = 0.0

    def _generate_start(self, robot_idx: int,
                        world_config: WorldConfig,
                        robot_model: RobotModelConfig,
                        device: torch.device | str,
                        rng: torch.Generator | None = None) -> torch.Tensor:
        assert world_config.suitable_mask is not None, "WorldConfig must contain a suitable mask for start/goal generation."
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
        assert world_config.suitable_mask is not None, "WorldConfig must contain a suitable mask for start/goal generation."
        max_feasible_coord = world_config.max_coord - robot_model.radius
        for _ in range(self.max_goal_generation_attempts):
            goal = torch.rand((1, 1, 2), generator=rng, device=device) * 2 * max_feasible_coord - max_feasible_coord  # interval [-max_feasible_coord, max_feasible_coord]
            feasibility = interpolate_grid(world_config.suitable_mask[None, robot_idx], goal, world_config.max_coord).item()
            if feasibility >= self.feasability_thresh and self.min_dist_to_goal <= torch.norm(goal - start[..., :2]) <= self.max_dist_to_goal:
                z = interpolate_grid(world_config.z_grid[None, robot_idx], goal, world_config.max_coord)
                if z - start[2] <= self.higher_allowed:
                    return torch.cat([goal, z], dim=-1).squeeze(), True
        return torch.zeros(3), False

    def check_reached_goal_or_terminate(self, state: PhysicsState, goal: PhysicsState) -> torch.Tensor:
        reached_goal = torch.norm(state.x - goal.x, dim=-1) <= self.min_dist_to_goal
        return reached_goal

    def compute_iteration_limits(self, start_state: PhysicsState, goal_state: PhysicsState, robot_model: RobotModelConfig, dt: float) -> torch.Tensor:
        return torch.ceil(self.iteration_limit_factor * torch.norm(goal_state.x - start_state.x, dim=-1, p=1) / (robot_model.vel_max * dt)).long()

    @torch.no_grad()
    def compute_reward(self,
                       prev_state: PhysicsState,
                       action: torch.Tensor,
                       curr_state: PhysicsState,
                       goal: PhysicsState,
                       ) -> torch.Tensor:
        """
        Compute the reward for the given state and goal.

        Args:
            prev_state: The previous state of the environment.
            action: The action taken in the previous state.
            curr_state: The current state of the environment after taking the action in the previous state.
            goal: The goal state of the environment.

        Returns:
            The reward for the given state and goal.
        """
        dist_to_goal_prev = torch.linalg.norm(prev_state.x - goal.x, dim=-1)
        dist_to_goal_curr = torch.linalg.norm(curr_state.x - goal.x, dim=-1)
        dist_reward = torch.clamp((dist_to_goal_prev - dist_to_goal_curr), min=0.) * self.distance_weight
        roll_pitch_rates_sq = curr_state.omega[..., :2].pow(2)  # shape (batch_size, 2)
        roll_pitch_penalties = torch.tensor([self.roll_rate_weight, self.pitch_rate_weight], device=roll_pitch_rates_sq.device) * roll_pitch_rates_sq
        return dist_reward - roll_pitch_penalties.sum(dim=-1)
