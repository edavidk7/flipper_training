import torch
import warnings
from tqdm import tqdm
from typing import Literal, override
from dataclasses import dataclass
from flipper_training.engine.engine_state import PhysicsState
from flipper_training.rl_objectives.base_objective import BaseObjective
from flipper_training.utils.geometry import euler_to_quaternion, quaternion_to_roll


@dataclass
class SimpleStabilizationObjective(BaseObjective):
    """
    Objective manager that generates start/goal positions for the robots in the environment. The start position is generated randomly within the suitable area of the world, while the goal position is generated randomly within the suitable area of the world and at a minimum distance from the start position, but not further than the maximum distance. It is ensured that the goal position is not too high above the start position (to avoid unreachable goals).
    The robot is rewarded for minimizing the rotational velocities and for moving towards the goal position.
    """

    higher_allowed: float
    min_dist_to_goal: float
    max_dist_to_goal: float
    goal_reached_threshold: float
    start_drop: float
    iteration_limit_factor: float
    max_feasible_roll: float
    start_position_orientation: Literal["random", "towards_goal"]
    cache_size: int
    _cache_cursor: int = 0

    def __post_init__(self) -> None:
        if self.world_config.suitable_mask is None:
            raise ValueError("World configuration must contain a suitable mask for start/goal positions.")
        self._init_cache()

    @override
    def _init_cache(self) -> None:
        """
        Initializes the cache for the start/goal positions.
        """
        B = self.physics_config.num_robots
        state_prototype = PhysicsState.dummy(robot_model=self.robot_model, batch_size=B)
        start_state_cache = state_prototype.unsqueeze(0).repeat(self.cache_size, 1)
        goal_state_cache = start_state_cache.clone()
        remaining = self.cache_size * B
        remaining_mask = torch.ones((self.cache_size, B), dtype=torch.bool)
        pbar = tqdm(total=remaining, desc=f"{self.__class__.__name__}: generating start/goal position cache")
        while remaining > 0:
            start_ij = torch.randint(0, self.world_config.grid_size, (self.cache_size, B, 2), dtype=torch.int32, generator=self.rng)
            goal_ij = torch.randint(0, self.world_config.grid_size, (self.cache_size, B, 2), dtype=torch.int32, generator=self.rng)
            start_xyz = self.world_config.ij_to_xyz(start_ij)
            goal_xyz = self.world_config.ij_to_xyz(goal_ij)
            start_suit = self.world_config.ij_to_suited_mask(start_ij).bool()
            goal_suit = self.world_config.ij_to_suited_mask(goal_ij).bool()
            copy_mask = remaining_mask & start_suit & goal_suit & self.is_start_goal_xyz_valid(start_xyz, goal_xyz)
            start_state_cache.x[copy_mask] = start_xyz[copy_mask]
            goal_state_cache.x[copy_mask] = goal_xyz[copy_mask]
            added = copy_mask.sum().item()
            remaining -= added
            remaining_mask[copy_mask] = False
            pbar.update(added)
        pbar.close()
        self.cache = {}
        start_state_cache = start_state_cache.to(self.device).view(self.cache_size, B)
        goal_state_cache = goal_state_cache.to(self.device).view(self.cache_size, B)
        start_state_cache.x = start_state_cache.x + torch.tensor([0.0, 0.0, self.start_drop], device=start_state_cache.x.device)
        oris = self._get_initial_orientation_quat(start_state_cache.x, goal_state_cache.x)
        start_state_cache.q = oris
        goal_state_cache.q = oris
        self.cache["start"] = start_state_cache
        self.cache["goal"] = goal_state_cache
        self.cache["iteration_limits"] = self._compute_iteration_limits(start_state_cache, goal_state_cache)
        self._cache_cursor = 0

    def is_start_goal_xyz_valid(self, start_xyz: torch.Tensor, goal_xyz: torch.Tensor) -> torch.BoolTensor:
        """
        Checks if the start and goal positions are valid based on the suitability mask of the world configuration.

        Args:
        - start_xyz: Tensor of shape (B, 3) representing the start positions.
        - goal_xyz: Tensor of shape (B, 3) representing the goal positions.

        Returns:
        - A boolean tensor indicating whether each start/goal pair is valid.
        """
        dist = torch.linalg.norm(start_xyz - goal_xyz, dim=-1)
        z_diff = goal_xyz[..., 2] - start_xyz[..., 2]
        return (dist >= self.min_dist_to_goal) & (dist <= self.max_dist_to_goal) & (z_diff <= self.higher_allowed)

    @override
    def generate_start_goal_states(self) -> tuple[PhysicsState, PhysicsState, torch.IntTensor | torch.LongTensor]:
        """
        Generates start/goal positions for the robots in the environment.

        The world configuration must contain a suitability mask that indicates which parts of the world are suitable for start/goal positions.

        Returns:
        - A tuple of PhysicsState objects containing the start and goal positions for the robots.
        - A tensor containing the iteration limits for the robots.
        """
        if self._cache_cursor < self.cache_size:
            start_state = self.cache["start"][self._cache_cursor]
            goal_state = self.cache["goal"][self._cache_cursor]
            iteration_limits = self.cache["iteration_limits"][self._cache_cursor]
            self._cache_cursor += 1
            return start_state, goal_state, iteration_limits
        else:
            warnings.warn(f"{self.__class__.__name__} cache exhausted, generating new start/goal positions")
            self._init_cache()
            return self.generate_start_goal_states()

    def _get_initial_orientation_quat(self, starts_x: torch.Tensor, goals_x: torch.Tensor) -> torch.Tensor:
        match self.start_position_orientation:
            case "towards_goal":
                diff_vecs = goals_x[..., :2] - starts_x[..., :2]
                ori = torch.atan2(diff_vecs[..., 1], diff_vecs[..., 0])
            case "random":
                ori = torch.rand(starts_x.shape[0], device=starts_x.device) * 2 * torch.pi  # random orientation
            case _:
                raise ValueError(f"Invalid start_position_orientation: {self.start_position_orientation}")
        return euler_to_quaternion(torch.zeros_like(ori), torch.zeros_like(ori), ori)

    def check_reached_goal(self, state: PhysicsState, goal: PhysicsState) -> torch.BoolTensor:
        return torch.linalg.norm(state.x - goal.x, dim=-1) <= self.min_dist_to_goal

    def check_terminated_wrong(self, state: PhysicsState, goal: PhysicsState) -> torch.BoolTensor:
        return quaternion_to_roll(state.q).abs() > self.max_feasible_roll

    def _compute_iteration_limits(
        self,
        start_state: PhysicsState,
        goal_state: PhysicsState,
    ) -> torch.IntTensor:
        dists = torch.linalg.norm(goal_state.x - start_state.x, dim=-1)  # distances from starts to goals
        fastest_traversal = dists / (self.robot_model.v_max * self.physics_config.dt)  # time to reach the furthest goal
        steps = (fastest_traversal * self.iteration_limit_factor).ceil()
        return steps.int()
