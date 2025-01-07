from tracemalloc import start
import torch
from typing import List
from dataclasses import dataclass
from flipper_training.configs import RobotModelConfig, WorldConfig
from flipper_training.utils.environment import interpolate_grid
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.rl_objectives.base_objective import BaseObjective
from flipper_training.utils.geometry import rot_Z, yaw_from_R


@dataclass
class SimpleStabilizationObjective(BaseObjective[torch.Tensor]):
    """
    Objective manager that generates start/goal positions for the robots in the environment. The start position is generated randomly within the suitable area of the world, while the goal position is generated randomly within the suitable area of the world and at a minimum distance from the start position, but not further than the maximum distance. It is ensured that the goal position is not too high above the start position (to avoid unreachable goals).
    The robot is rewarded for minimizing the rotational velocities and for keeping itself oriented towards the goal position.

    Attributes:
    - higher_allowed: Maximum height difference between the start and goal positions.
    - min_dist_to_goal: Minimum distance between the start and goal positions.
    - start_drop: Height of the start position above the ground.
    - roll_rate_weight: Weight for the roll rate reward component.
    - pitch_rate_weight: Weight for the pitch rate reward component.
    - orientation_weight: Weight for the orientation reward component.
    - iteration_limit_factor: Factor to multiply the time to reach the furthest goal by to get the iteration limit.
    """

    higher_allowed: float = 0.5  # meters
    min_dist_to_goal: float = 0.1  # meters
    start_drop: float = 0.1  # meters
    iteration_limit_factor: float = 1.5
    roll_rate_weight: float = 1.0
    pitch_rate_weight: float = 1.0
    orientation_weight: float = 1.0

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

    def _get_dummy_objective_like(self, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(3, device=device)

    def _construct_full_start_goal_states(self, partial_starts: List[torch.Tensor], partial_goals: List[torch.Tensor], robot_model: RobotModelConfig) -> tuple[PhysicsState, PhysicsState]:
        """This class takes in the lists of start and goal position tensors and assembles them into full PhysicsState objects. Specifically, we rotate the robot to face the goal position."""
        starts = torch.stack(partial_starts, dim=0)
        goals = torch.stack(partial_goals, dim=0)
        diff_vecs = goals[..., :2] - starts[..., :2]
        ori = torch.atan2(diff_vecs[..., 1], diff_vecs[..., 0])
        rots = rot_Z(ori)
        return PhysicsState.dummy(x=starts, R=rots, robot_model=robot_model), PhysicsState.dummy(x=goals, R=rots, robot_model=robot_model)

    def check_reached_goal_or_terminate(self, state: PhysicsState, goal: PhysicsState) -> torch.Tensor:
        reached_goal = torch.norm(state.x - goal.x, dim=-1) <= self.min_dist_to_goal
        return reached_goal

    def compute_iteration_limits(self, start_state: PhysicsState, goal_state: PhysicsState, robot_model: RobotModelConfig, dt: float) -> torch.Tensor:
        dists = torch.linalg.norm(goal_state.x - start_state.x, dim=-1)  # distances from starts to goals
        furthest = dists.max() / (robot_model.vel_max * dt)  # time to reach the furthest goal
        steps = (furthest * self.iteration_limit_factor).ceil()
        return torch.full(start_state.batch_size, steps, device=start_state.x.device, dtype=torch.int32)

    @torch.no_grad()
    def compute_reward(self,
                       prev_state: PhysicsState,
                       state_der: PhysicsStateDer,
                       curr_state: PhysicsState,
                       goal: PhysicsState,
                       ) -> torch.Tensor:
        # angular velocity penalty
        roll_pitch_rates_sq = curr_state.omega[..., :2].pow(2)  # shape (batch_size, 2)
        roll_pitch_penalties = torch.tensor([self.roll_rate_weight, self.pitch_rate_weight], device=roll_pitch_rates_sq.device) * roll_pitch_rates_sq
        # deviation from correct orientation penalty
        goal_diff = goal.x - curr_state.x
        vec_ori = torch.atan2(goal_diff[..., 1], goal_diff[..., 0])
        robot_ori = yaw_from_R(curr_state.R)
        return -roll_pitch_penalties.sum(dim=-1) - self.orientation_weight * (robot_ori - vec_ori)**2
