import torch
import logging
import math
from tqdm import trange
from typing import Literal, override
from dataclasses import dataclass
from flipper_training.engine.engine_state import PhysicsState
from flipper_training.rl_objectives import BaseObjective
from flipper_training.utils.geometry import euler_to_quaternion, quaternion_to_euler
import torch.nn.functional as F
from flipper_training.utils.logutils import get_terminal_logger


@dataclass
class StairCrossing(BaseObjective):
    goal_reached_threshold: float
    start_z_offset: float
    goal_z_offset: float
    iteration_limit_factor: float
    max_feasible_pitch: float
    max_feasible_roll: float
    start_position_orientation: Literal["random", "towards_goal"]
    init_joint_angles: torch.Tensor | Literal["max", "min", "random"]
    cache_size: int
    sampling_mode: Literal["lowest_highest", "any"] = "lowest_highest"
    min_dist_from_edge: float = 0.3  # Minimum distance of the x,y position of start/goal from the edge of the step
    resample_random_joint_angles_on_reset: bool = False
    _cache_cursor: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.terrain_config.grid_extras is None or "step_indices" not in self.terrain_config.grid_extras:
            raise ValueError("World configuration must contain 'step_indices' in grid_extras for StairCrossing.")
        self.term_logger = get_terminal_logger("stair_crossing")
        self._init_cache()

    def state_dict(self):
        return {
            "cache_cursor": self._cache_cursor,
        }

    def load_state_dict(self, state_dict):
        self._cache_cursor = state_dict["cache_cursor"]

    def _get_suitable_core_mask(self, step_indices: torch.Tensor) -> torch.BoolTensor:
        """
        Identify core cells of each step plateau by excluding any cell adjacent
        to a different index or outside (-1), and then erode by the configured
        min_dist_from_edge (in meters) using grid_res to compute radius.
        """
        si = step_indices  # (B, H, W)
        valid = si != -1
        # number of cells to erode based on configured distance
        grid_res = self.terrain_config.grid_res
        radius = int(math.ceil(self.min_dist_from_edge / grid_res))

        # first pass: remove immediate edges (4-connected neighbors)
        pad_si = F.pad(si, (1, 1, 1, 1), mode="constant", value=-1)
        up = pad_si[:, :-2, 1:-1]
        down = pad_si[:, 2:, 1:-1]
        left = pad_si[:, 1:-1, :-2]
        right = pad_si[:, 1:-1, 2:]
        core = valid & (up == si) & (down == si) & (left == si) & (right == si)

        # iterative erosion for additional radius
        if radius > 1:
            for i in range(step_indices.shape[0]):
                neigh = core[i]
                for _ in range(radius - 1):
                    pad_c = F.pad(neigh.float(), (1, 1, 1, 1), mode="constant", value=0).bool()
                    upc = pad_c[:-2, 1:-1]
                    downc = pad_c[2:, 1:-1]
                    leftc = pad_c[1:-1, :-2]
                    rightc = pad_c[1:-1, 2:]
                    neigh_next = neigh & upc & downc & leftc & rightc
                    if not neigh_next.any():
                        self.term_logger.warning(
                            f"Warning: No suitable core cells found after erosion for min_dist_from_edge={self.min_dist_from_edge} for robot {i}."
                            " Try reducing the distance or increasing grid size."
                        )
                        # if no core cells are found, fall back to the original core
                        break
                    neigh = neigh_next

                core[i] = neigh
        return core

    def _init_cache(self) -> None:
        B = self.physics_config.num_robots
        total_needed_per_robot = self.cache_size

        step_indices = self.terrain_config.grid_extras["step_indices"].cpu()  # (B, H, W)
        suitable_core_mask = self._get_suitable_core_mask(step_indices)
        # import matplotlib.pyplot as plt
        # from flipper_training.vis.static_vis import plot_grids_xyz

        # for b in range(B):
        #     plt.imshow(suitable_core_mask[b].cpu().numpy())
        #     plt.title(f"Robot {b} suitable core mask")
        #     plt.colorbar()
        #     plt.show()
        #     plot_grids_xyz(
        #         self.terrain_config.x_grid[b].cpu(),
        #         self.terrain_config.y_grid[b].cpu(),
        #         self.terrain_config.z_grid[b].cpu(),
        #         title=f"Robot {b} terrain",
        #     )
        #     plt.show()
        # Initialize cache tensors
        start_pos_cache = torch.empty((self.cache_size, B, 3), dtype=torch.float32)
        goal_pos_cache = torch.empty((self.cache_size, B, 3), dtype=torch.float32)

        for b in trange(B, desc="Initializing start/goal position cache (Stairs)"):
            indices_b = step_indices[b]
            core_mask_b = suitable_core_mask[b]

            valid_core_indices_b = torch.nonzero(core_mask_b, as_tuple=False)  # (N_valid, 2) [i, j]

            if valid_core_indices_b.shape[0] == 0:
                raise ValueError(
                    f"No suitable core cells found for robot {b} with min_dist_from_edge={self.min_dist_from_edge}."
                    "Try reducing the distance or increasing grid size."
                )

            collected = 0
            oversample_factor = 2  # Start with oversampling

            while collected < total_needed_per_robot:
                remaining = total_needed_per_robot - collected
                n_samples = int(remaining * oversample_factor)

                if self.sampling_mode == "lowest_highest":
                    valid_indices_values = indices_b[core_mask_b]
                    min_step = valid_indices_values.min()
                    max_step = valid_indices_values.max()
                    if min_step == max_step:
                        raise ValueError(
                            f"Only one valid step level ({min_step}) found for robot {b} in 'lowest_highest' mode. Cannot sample distinct start/goal."
                        )

                    lowest_indices = torch.nonzero((indices_b == min_step) & core_mask_b, as_tuple=False)
                    highest_indices = torch.nonzero((indices_b == max_step) & core_mask_b, as_tuple=False)

                    if lowest_indices.shape[0] == 0 or highest_indices.shape[0] == 0:
                        raise ValueError(f"Could not find both lowest ({min_step}) and highest ({max_step}) suitable core cells for robot {b}.")

                    n_samples = min(n_samples, lowest_indices.shape[0], highest_indices.shape[0])
                    if n_samples == 0:
                        oversample_factor *= 2  # Should not happen if checks above pass, but safety first
                        continue

                    low_idx = torch.randperm(lowest_indices.shape[0], generator=self.rng)[:n_samples]
                    high_idx = torch.randperm(highest_indices.shape[0], generator=self.rng)[:n_samples]
                    lowest_ij = lowest_indices[low_idx]
                    highest_ij = highest_indices[high_idx]

                    is_start_low_mask = torch.bernoulli(torch.full((n_samples,), 0.5), generator=self.rng).bool().unsqueeze(-1)
                    start_ij = torch.where(is_start_low_mask, lowest_ij, highest_ij)
                    goal_ij = torch.where(is_start_low_mask, highest_ij, lowest_ij)

                elif self.sampling_mode == "any":
                    n_valid_core = valid_core_indices_b.shape[0]
                    if n_valid_core < 2:
                        raise ValueError(f"Need at least 2 suitable core cells for robot {b} in 'any' mode, found {n_valid_core}.")

                    start_perm_idx = torch.randperm(n_valid_core, generator=self.rng)
                    goal_perm_idx = torch.randperm(n_valid_core, generator=self.rng)

                    start_ij_all = valid_core_indices_b[start_perm_idx]
                    goal_ij_all = valid_core_indices_b[goal_perm_idx]

                    # Ensure start and goal are not on the same step
                    start_steps = indices_b[start_ij_all[:, 0], start_ij_all[:, 1]]
                    goal_steps = indices_b[goal_ij_all[:, 0], goal_ij_all[:, 1]]
                    different_step_mask = start_steps != goal_steps

                    start_ij = start_ij_all[different_step_mask][:n_samples]
                    goal_ij = goal_ij_all[different_step_mask][:n_samples]
                    n_samples = start_ij.shape[0]  # Update n_samples based on valid pairs found

                else:
                    raise ValueError(f"Unknown sampling mode: {self.sampling_mode}")

                if n_samples == 0:
                    oversample_factor *= 2
                    if oversample_factor > 128:  # Prevent infinite loop
                        raise RuntimeError(f"Failed to find valid start/goal pairs for robot {b} after extensive oversampling. Check parameters.")
                    continue

                # Convert ij to xyz for this robot
                start_xyz = torch.stack(
                    [
                        g[b, start_ij[:, 0], start_ij[:, 1]]
                        for g in [self.terrain_config.x_grid, self.terrain_config.y_grid, self.terrain_config.z_grid]
                    ],
                    dim=-1,
                ).cpu()
                goal_xyz = torch.stack(
                    [
                        g[b, goal_ij[:, 0], goal_ij[:, 1]]
                        for g in [self.terrain_config.x_grid, self.terrain_config.y_grid, self.terrain_config.z_grid]
                    ],
                    dim=-1,
                ).cpu()

                # No extra validation needed here as suitability was checked via core_mask
                n_new = min(n_samples, remaining)
                start_pos_cache[collected : collected + n_new, b, :] = start_xyz[:n_new]
                goal_pos_cache[collected : collected + n_new, b, :] = goal_xyz[:n_new]
                collected += n_new
                oversample_factor *= 1.1  # Increase slightly if needed

        # Store the cache
        self.cache = {
            "start": start_pos_cache,
            "goal": goal_pos_cache,
            "ori": self._get_initial_orientation_quat(start_pos_cache, goal_pos_cache),
            "step_limits": self._compute_step_limits(start_pos_cache, goal_pos_cache),
            "joint_angles": torch.stack([self._get_initial_joint_angles() for _ in range(self.cache_size)], dim=0),
        }
        self._cache_cursor = 0

    def _get_initial_joint_angles(self) -> torch.Tensor:
        # Identical to TrunkCrossing
        high = self.robot_model.joint_limits[None, 1].cpu()
        low = self.robot_model.joint_limits[None, 0].cpu()
        match self.init_joint_angles:
            case "max":
                return high.repeat(self.physics_config.num_robots, 1)
            case "min":
                return low.repeat(self.physics_config.num_robots, 1)
            case "random":
                ang = torch.rand((self.physics_config.num_robots, self.robot_model.num_driving_parts), generator=self.rng) * (high - low) + low
                ang = ang.clamp(min=low, max=high)
                return ang
            case torch.Tensor():
                if len(self.init_joint_angles) != self.robot_model.num_driving_parts:
                    raise ValueError(
                        f"Invalid shape for init_joint_angles: {self.init_joint_angles.shape}. Expected {self.robot_model.num_driving_parts}."
                    )
                ang = self.init_joint_angles.repeat(self.physics_config.num_robots, 1)
                ang = ang.clamp(min=low, max=high)
                return ang
            case _:
                raise ValueError("Invalid value for init_joint_angles.")

    def _construct_full_start_goal_states(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        oris: torch.Tensor,
        thetas: torch.Tensor,
    ) -> tuple[PhysicsState, PhysicsState]:
        # Identical to TrunkCrossing
        start_state = PhysicsState.dummy(
            robot_model=self.robot_model,
            batch_size=start_pos.shape[0],
            device=self.device,
            x=start_pos.to(self.device),
            q=oris.to(self.device),
            thetas=thetas.to(self.device),
        )
        goal_state = PhysicsState.dummy(
            robot_model=self.robot_model, batch_size=goal_pos.shape[0], device=self.device, x=goal_pos.to(self.device), q=oris.to(self.device)
        )
        goal_state.x[..., 2] += self.goal_z_offset
        start_state.x[..., 2] += self.start_z_offset
        return start_state, goal_state

    @override
    def generate_start_goal_states(self) -> tuple[PhysicsState, PhysicsState, torch.IntTensor | torch.LongTensor]:
        # Identical to TrunkCrossing (uses cache)
        if self._cache_cursor < self.cache_size:
            start_state, goal_state = self._construct_full_start_goal_states(
                self.cache["start"][self._cache_cursor],
                self.cache["goal"][self._cache_cursor],
                self.cache["ori"][self._cache_cursor],
                self.cache["joint_angles"][self._cache_cursor],
            )
            step_limits = self.cache["step_limits"][self._cache_cursor].to(self.device)
            self._cache_cursor += 1
            return start_state, goal_state, step_limits
        else:
            logging.warning("Start/goal cache exhausted, doubling size and generating new start/goal positions")
            self.cache_size *= 2
            self._init_cache()
            return self.generate_start_goal_states()

    def _get_initial_orientation_quat(self, starts_x: torch.Tensor, goals_x: torch.Tensor) -> torch.Tensor:
        # Identical to TrunkCrossing
        match self.start_position_orientation:
            case "towards_goal":
                diff_vecs = goals_x[..., :2] - starts_x[..., :2]
                ori = torch.atan2(diff_vecs[..., 1], diff_vecs[..., 0])
            case "random":
                ori = torch.rand(starts_x.shape[0], device=starts_x.device, generator=self.rng) * 2 * torch.pi  # random orientation
            case _:
                raise ValueError(f"Invalid start_position_orientation: {self.start_position_orientation}")
        return euler_to_quaternion(torch.zeros_like(ori), torch.zeros_like(ori), ori)

    def check_reached_goal(self, prev_state: PhysicsState, state: PhysicsState, goal: PhysicsState) -> torch.BoolTensor:
        # Identical to TrunkCrossing
        return torch.linalg.norm(state.x - goal.x, dim=-1) <= self.goal_reached_threshold

    def check_terminated_wrong(self, prev_state: PhysicsState, state: PhysicsState, goal: PhysicsState) -> torch.BoolTensor:
        # Identical to TrunkCrossing + step‐jump check
        rolls, pitches, _ = quaternion_to_euler(state.q)
        old_fail = (pitches.abs() > self.max_feasible_pitch) | (rolls.abs() > self.max_feasible_roll)
        # new: terminate if step index changed by >1
        ti = self.terrain_config.grid_extras["step_indices"]  # (B, H, W)
        B_range = torch.arange(self.physics_config.num_robots, device=self.device)
        prev_xy = prev_state.x[..., :2]  # (B,2)
        curr_xy = state.x[..., :2]  # (B,2)
        prev_ij = self.terrain_config.xy2ij(prev_xy)  # (B,2)
        curr_ij = self.terrain_config.xy2ij(curr_xy)  # (B,2)
        prev_idx = ti[B_range, *prev_ij.unbind(1)]  # (B,)
        curr_idx = ti[B_range, *curr_ij.unbind(1)]  # (B,)
        jump_fail = (curr_idx - prev_idx).abs() > 1
        return old_fail | jump_fail

    def _compute_step_limits(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
    ) -> torch.IntTensor:
        # Identical to TrunkCrossing
        dists = torch.linalg.norm(goal_pos - start_pos, dim=-1)  # distances from starts to goals
        fastest_traversal = dists / (self.robot_model.v_max * self.env.effective_dt)  # time to reach the furthest goal
        steps = (fastest_traversal * self.iteration_limit_factor).ceil()
        return steps.int()

    def start_goal_to_simview(self, start: PhysicsState, goal: PhysicsState):
        # Identical to TrunkCrossing
        try:
            from simview import SimViewStaticObject, BodyShapeType
        except ImportError:
            logging.warning("SimView is not installed. Cannot visualize start/goal positions.")
            return []

        # start
        pos = start.x
        start_object = SimViewStaticObject.create_batched(
            name="Start",
            shape_type=BodyShapeType.POINTCLOUD,
            shapes_kwargs=[
                {
                    "points": pos[i, None],
                    "color": "#ff0000",
                }
                for i in range(pos.shape[0])
            ],
        )
        # goal
        pos = goal.x
        goal_object = SimViewStaticObject.create_batched(
            name="Goal",
            shape_type=BodyShapeType.POINTCLOUD,
            shapes_kwargs=[
                {
                    "points": pos[i, None],
                    "color": "#0000ff",
                }
                for i in range(pos.shape[0])
            ],
        )
        return [start_object, goal_object]
