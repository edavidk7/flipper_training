import torch
from typing import List, Literal
from dataclasses import dataclass
from flipper_training.configs import RobotModelConfig, WorldConfig
from flipper_training.utils.environment import interpolate_grid
from flipper_training.engine.engine_state import PhysicsState
from flipper_training.rl_objectives.base_objective import BaseObjective
from flipper_training.utils.geometry import euler_to_quaternion, quaternion_to_roll


@dataclass
class SimpleStabilizationObjective(BaseObjective[torch.Tensor]):
    """
    Objective manager that generates start/goal positions for the robots in the environment. The start position is generated randomly within the suitable area of the world, while the goal position is generated randomly within the suitable area of the world and at a minimum distance from the start position, but not further than the maximum distance. It is ensured that the goal position is not too high above the start position (to avoid unreachable goals).
    The robot is rewarded for minimizing the rotational velocities and for moving towards the goal position.

    Attributes:
    - higher_allowed: float - maximum allowed height difference between start and goal positions in meters
    - min_dist_to_goal: float - minimum distance from start to goal in meters
    - max_dist_to_goal: float - maximum distance from start to goal in meters
    - goal_reached_threshold: float - distance threshold for reaching the goal
    - start_drop: float - height of the robot above the ground at the start position
    - iteration_limit_factor: float - factor to multiply the time to reach the furthest goal to get the iteration limit
    - max_feasible_roll: float - maximum allowed roll angle in radians
    - start_position_orientation: Literal["random", "towards_goal"] - orientation of the robot at the start position
    """

    higher_allowed: float = 0.5  # meters
    min_dist_to_goal: float = 0.8  # meters
    max_dist_to_goal: float = 1.0  # meters
    goal_reached_threshold: float = 0.05  # meters
    start_drop: float = 0.1  # meters
    iteration_limit_factor: float = 10
    max_feasible_roll: float = 1.2  # radians
    start_position_orientation: Literal["random", "towards_goal"] = "towards_goal"

    def _generate_start(
        self,
        robot_idx: int,
        world_config: WorldConfig,
        robot_model: RobotModelConfig,
        device: torch.device | str,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        assert world_config.suitable_mask is not None, (
            "WorldConfig must contain a suitable mask for start/goal generation."
        )
        max_feasible_coord = world_config.max_coord - robot_model.radius
        while True:
            start = (
                torch.rand((1, 1, 2), generator=rng, device=device) * 2 * max_feasible_coord
                - max_feasible_coord
            )  # interval [-max_feasible_coord, max_feasible_coord]
            feasibility = interpolate_grid(
                world_config.suitable_mask[None, robot_idx],
                start,
                world_config.max_coord,
            )
            if feasibility >= self.feasability_thresh:
                z = interpolate_grid(
                    world_config.z_grid[None, robot_idx], start, world_config.max_coord
                )
                z += abs(robot_model.robot_points[..., 2].min()) + self.start_drop
                break
        return torch.cat([start, z], dim=-1).squeeze()

    def _generate_goal(
        self,
        start: torch.Tensor,
        robot_idx: int,
        world_config: WorldConfig,
        robot_model: RobotModelConfig,
        device: torch.device | str,
        rng: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, bool]:
        assert world_config.suitable_mask is not None, (
            "WorldConfig must contain a suitable mask for start/goal generation."
        )
        max_feasible_coord = world_config.max_coord - robot_model.radius
        for _ in range(self.max_goal_generation_attempts):
            goal = (
                torch.rand((1, 1, 2), generator=rng, device=device) * 2 * max_feasible_coord
                - max_feasible_coord
            )  # interval [-max_feasible_coord, max_feasible_coord]
            feasibility = interpolate_grid(
                world_config.suitable_mask[None, robot_idx],
                goal,
                world_config.max_coord,
            ).item()
            if (
                feasibility >= self.feasability_thresh
                and self.min_dist_to_goal
                <= torch.norm(goal - start[..., :2])
                <= self.max_dist_to_goal
            ):
                z = interpolate_grid(
                    world_config.z_grid[None, robot_idx], goal, world_config.max_coord
                )
                if z - start[2] <= self.higher_allowed:
                    return torch.cat([goal, z], dim=-1).squeeze(), True
        return torch.zeros(3), False

    def _get_dummy_objective_like(self, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(3, device=device)

    def _construct_full_start_goal_states(
        self,
        partial_starts: List[torch.Tensor],
        partial_goals: List[torch.Tensor],
        robot_model: RobotModelConfig,
    ) -> tuple[PhysicsState, PhysicsState]:
        """This class takes in the lists of start and goal position tensors and assembles them into full PhysicsState objects. Specifically, we rotate the robot to face the goal position."""
        starts = torch.stack(partial_starts, dim=0)
        goals = torch.stack(partial_goals, dim=0)
        rots = self._get_initial_orientation_matrix(starts, goals)
        return PhysicsState.dummy(x=starts, q=rots, robot_model=robot_model), PhysicsState.dummy(
            x=goals, robot_model=robot_model
        )

    def _get_initial_orientation_matrix(
        self, starts: torch.Tensor, goals: torch.Tensor
    ) -> torch.Tensor:
        if self.start_position_orientation == "towards_goal":
            diff_vecs = goals[..., :2] - starts[..., :2]
            ori = torch.atan2(diff_vecs[..., 1], diff_vecs[..., 0])
        elif self.start_position_orientation == "random":
            ori = (
                torch.rand(starts.shape[0], device=starts.device) * 2 * torch.pi
            )  # random orientation
        else:
            raise ValueError(
                f"Invalid start_position_orientation: {self.start_position_orientation}"
            )
        return euler_to_quaternion(torch.zeros_like(ori), torch.zeros_like(ori), ori)

    def check_reached_goal(self, state: PhysicsState, goal: PhysicsState) -> torch.BoolTensor:
        reached_goal = torch.norm(state.x - goal.x, dim=-1) <= self.min_dist_to_goal
        return reached_goal

    def check_terminated_wrong(self, state: PhysicsState, goal: PhysicsState) -> torch.BoolTensor:
        rolls = quaternion_to_roll(state.q)
        return rolls.abs() > self.max_feasible_roll

    def compute_iteration_limits(
        self,
        start_state: PhysicsState,
        goal_state: PhysicsState,
        robot_model: RobotModelConfig,
        dt: float,
    ) -> torch.Tensor:
        dists = torch.linalg.norm(
            goal_state.x - start_state.x, dim=-1
        )  # distances from starts to goals
        furthest = dists.max() / (robot_model.vel_max * dt)  # time to reach the furthest goal
        steps = (furthest * self.iteration_limit_factor).ceil()
        return torch.full(
            start_state.batch_size,
            steps,
            device=start_state.x.device,
            dtype=torch.int32,
        )
