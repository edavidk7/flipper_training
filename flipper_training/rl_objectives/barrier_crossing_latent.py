import torch
import logging
from tqdm import trange
from typing import Literal, override
from dataclasses import dataclass

from flipper_training.engine.engine_state import PhysicsState
from flipper_training.rl_objectives import BaseObjective
from flipper_training.utils.geometry import euler_to_quaternion, quaternion_to_euler
from flipper_training.heightmaps.barrier import BarrierZones


@dataclass
class BarrierCrossingWithLatentControl(BaseObjective):
    min_dist_to_goal: float
    max_dist_to_goal: float
    goal_reached_threshold: float
    start_z_offset: float
    goal_z_offset: float
    iteration_limit_factor: float
    max_feasible_pitch: float
    max_feasible_roll: float
    enforce_path_through_barrier: bool
    start_position_orientation: Literal["random", "towards_goal"]
    init_joint_angles: torch.Tensor | Literal["max", "min", "random"]
    cache_size: int
    resample_random_joint_angles_on_reset: bool = False
    _cache_cursor: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.terrain_config.grid_extras is None or "suitable_mask" not in self.terrain_config.grid_extras:
            raise ValueError("World configuration must contain the barrier suitable_mask in grid_extras.")
        self._init_cache()

    def state_dict(self):
        return {"cache_cursor": self._cache_cursor}

    def load_state_dict(self, state_dict):
        self._cache_cursor = state_dict["cache_cursor"]

    def _init_cache(self) -> None:
        B = self.physics_config.num_robots
        total = self.cache_size
        start_pos = torch.empty((total, B, 3), dtype=torch.float32)
        goal_pos = torch.empty((total, B, 3), dtype=torch.float32)

        for b in trange(B, desc="Init barrier start/goal cache"):
            # import matplotlib.pyplot as plt

            # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            # ax[0].set_title("Heightmap")
            # ax[0].contourf(
            #     self.terrain_config.x_grid[b].cpu(),
            #     self.terrain_config.y_grid[b].cpu(),
            #     self.terrain_config.z_grid[b].cpu(),
            #     levels=100,
            #     cmap="viridis",
            # )
            # ax[1].set_title("Barrier mask")
            # ax[1].contourf(
            #     self.terrain_config.x_grid[b].cpu(),
            #     self.terrain_config.y_grid[b].cpu(),
            #     self.terrain_config.grid_extras["suitable_mask"][b].cpu(),
            #     levels=[-0.5, 0.5, 1.5, 2.5],
            #     cmap="tab10",
            # )
            # plt.show()
            mask = self.terrain_config.grid_extras["suitable_mask"][b].cpu()
            left_idx = torch.nonzero(mask == BarrierZones.LEFT, as_tuple=False).cpu()
            right_idx = torch.nonzero(mask == BarrierZones.RIGHT, as_tuple=False).cpu()
            collected = 0
            factor = 1
            while collected < total:
                rem = total - collected
                n = min(rem * factor, left_idx.shape[0], right_idx.shape[0])
                idxL = torch.randperm(left_idx.shape[0], generator=self.rng)[:n]
                idxR = torch.randperm(right_idx.shape[0], generator=self.rng)[:n]
                L = left_idx[idxL]
                R = right_idx[idxR]
                pick_left = torch.bernoulli(torch.full((n,), 0.5), generator=self.rng).bool().unsqueeze(-1)
                start_ij = torch.where(pick_left, L, R)
                goal_ij = torch.where(pick_left, R, L)
                if (mask[*start_ij.unbind(-1)] == mask[*goal_ij.unbind(-1)]).any():
                    raise ValueError("Start and goal positions are not on different sides of the barrier.")

                sp = torch.stack(
                    [g[b, *start_ij.unbind(-1)] for g in [self.terrain_config.x_grid, self.terrain_config.y_grid, self.terrain_config.z_grid]], dim=-1
                ).to("cpu")
                gp = torch.stack(
                    [g[b, *goal_ij.unbind(-1)] for g in [self.terrain_config.x_grid, self.terrain_config.y_grid, self.terrain_config.z_grid]], dim=-1
                ).to("cpu")

                valid = self._is_start_goal_xyz_valid(sp, gp)
                if self.enforce_path_through_barrier:
                    midpoints = torch.floor((start_ij + goal_ij) / 2).long()
                    valid &= mask[*midpoints.unbind(-1)] == BarrierZones.BARRIER
                vs, vg = sp[valid], gp[valid]
                take = min(vs.shape[0], rem)
                start_pos[collected : collected + take, b] = vs[:take]
                goal_pos[collected : collected + take, b] = vg[:take]
                collected += take
                factor *= 2

        self.cache = {
            "start": start_pos,
            "goal": goal_pos,
            "ori": self._get_initial_orientation_quat(start_pos, goal_pos),
            "step_limits": self._compute_step_limits(start_pos, goal_pos),
            "joint_angles": torch.stack([self._get_initial_joint_angles() for _ in range(self.cache_size)], dim=0),
            "latent_params": torch.rand(self.cache_size, self.physics_config.num_robots, 1) * 2 - 1,
        }
        self._cache_cursor = 0

    def _get_initial_joint_angles(self) -> torch.Tensor:
        high = self.robot_model.joint_limits[None, 1].cpu()
        low = self.robot_model.joint_limits[None, 0].cpu()
        match self.init_joint_angles:
            case "max":
                return high.repeat(self.physics_config.num_robots, 1)
            case "min":
                return low.repeat(self.physics_config.num_robots, 1)
            case "random":
                ang = (
                    torch.rand(
                        (self.physics_config.num_robots, self.robot_model.num_driving_parts),
                        generator=self.rng,
                    )
                    * (high - low)
                    + low
                )
                return ang.clamp(min=low, max=high)
            case torch.Tensor():
                if len(self.init_joint_angles) != self.robot_model.num_driving_parts:
                    raise ValueError(
                        f"Invalid shape for init_joint_angles: {self.init_joint_angles.shape}. Expected {self.robot_model.num_driving_parts}."
                    )
                ang = self.init_joint_angles.repeat(self.physics_config.num_robots, 1)
                return ang.clamp(min=low, max=high)
            case _:
                raise ValueError("Invalid value for init_joint_angles.")

    def _construct_full_start_goal_states(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        oris: torch.Tensor,
        thetas: torch.Tensor,
    ) -> tuple[PhysicsState, PhysicsState]:
        start_state = PhysicsState.dummy(
            robot_model=self.robot_model,
            batch_size=start_pos.shape[0],
            device=self.device,
            x=start_pos.to(self.device),
            q=oris.to(self.device),
            thetas=thetas.to(self.device),
        )
        goal_state = PhysicsState.dummy(
            robot_model=self.robot_model,
            batch_size=goal_pos.shape[0],
            device=self.device,
            x=goal_pos.to(self.device),
            q=oris.to(self.device),
        )
        goal_state.x[..., 2] += self.goal_z_offset
        start_state.x[..., 2] += self.start_z_offset
        return start_state, goal_state

    def _is_start_goal_xyz_valid(self, start_xyz: torch.Tensor, goal_xyz: torch.Tensor) -> torch.BoolTensor:
        dist = torch.linalg.norm(start_xyz - goal_xyz, dim=-1)
        start_xy_to_edge = self.terrain_config.max_coord - start_xyz[..., :2].abs()
        goal_xy_to_edge = self.terrain_config.max_coord - goal_xyz[..., :2].abs()
        return (
            (dist >= self.min_dist_to_goal)
            & (dist <= self.max_dist_to_goal)
            & (start_xy_to_edge.min(dim=-1).values > 2 * self.robot_model.radius)
            & (goal_xy_to_edge.min(dim=-1).values > 2 * self.robot_model.radius)
        )

    @override
    def reset(self, reset_mask, training):
        self.last_reset_mask = reset_mask

    def _reset_latent_params(self, latent_params: torch.Tensor):
        current_latent_params = getattr(self.env, "latent_control_params", None)
        if current_latent_params is None:
            self.env.latent_control_params = self.cache["latent_params"][self._cache_cursor].to(self.env.device)
        else:
            current_latent_params[self.last_reset_mask] = self.cache["latent_params"][self._cache_cursor].to(self.env.device)[self.last_reset_mask]
            self.env.latent_control_params = current_latent_params

    @override
    def generate_start_goal_states(self) -> tuple[PhysicsState, PhysicsState, torch.IntTensor]:
        if self._cache_cursor < self.cache_size:
            start_state, goal_state = self._construct_full_start_goal_states(
                self.cache["start"][self._cache_cursor],
                self.cache["goal"][self._cache_cursor],
                self.cache["ori"][self._cache_cursor],
                self.cache["joint_angles"][self._cache_cursor],
            )
            step_limits = self.cache["step_limits"][self._cache_cursor].to(self.device)
            self._reset_latent_params(self.cache["latent_params"][self._cache_cursor])
            self._cache_cursor += 1
            return start_state, goal_state, step_limits
        else:
            logging.warning("Start/goal cache exhausted, doubling size and generating new start/goal positions")
            self.cache_size *= 2
            self._init_cache()
            return self.generate_start_goal_states()

    def _get_initial_orientation_quat(self, starts_x: torch.Tensor, goals_x: torch.Tensor) -> torch.Tensor:
        match self.start_position_orientation:
            case "towards_goal":
                diff_vecs = goals_x[..., :2] - starts_x[..., :2]
                ori = torch.atan2(diff_vecs[..., 1], diff_vecs[..., 0])
            case "random":
                ori = torch.rand(starts_x.shape[0], device=starts_x.device, generator=self.rng) * 2 * torch.pi
            case _:
                raise ValueError(f"Invalid start_position_orientation: {self.start_position_orientation}")
        return euler_to_quaternion(torch.zeros_like(ori), torch.zeros_like(ori), ori)

    def check_reached_goal(self, prev_state: PhysicsState, state: PhysicsState, goal: PhysicsState) -> torch.BoolTensor:
        return torch.linalg.norm(state.x - goal.x, dim=-1) <= self.goal_reached_threshold

    def check_terminated_wrong(self, prev_state: PhysicsState, state: PhysicsState, goal: PhysicsState) -> torch.BoolTensor:
        rolls, pitches, _ = quaternion_to_euler(state.q)
        return (
            (pitches.abs() > self.max_feasible_pitch)
            | (rolls.abs() > self.max_feasible_roll)
            | (state.x.abs() > self.terrain_config.max_coord).any(dim=-1)
        )

    def _compute_step_limits(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
    ) -> torch.IntTensor:
        dists = torch.linalg.norm(goal_pos - start_pos, dim=-1)
        fastest_traversal = dists / (self.robot_model.v_max * self.env.effective_dt)
        steps = (fastest_traversal * self.iteration_limit_factor).ceil()
        return steps.int()

    def start_goal_to_simview(self, start: PhysicsState, goal: PhysicsState):
        try:
            from simview import SimViewStaticObject, BodyShapeType
        except ImportError:
            logging.warning("SimView is not installed. Cannot visualize start/goal positions.")
            return []

        pos = start.x
        start_object = SimViewStaticObject.create_batched(
            name="Start",
            shape_type=BodyShapeType.POINTCLOUD,
            shapes_kwargs=[{"points": pos[i, None], "color": "#ff0000"} for i in range(pos.shape[0])],
        )
        pos = goal.x
        goal_object = SimViewStaticObject.create_batched(
            name="Goal",
            shape_type=BodyShapeType.POINTCLOUD,
            shapes_kwargs=[{"points": pos[i, None], "color": "#0000ff"} for i in range(pos.shape[0])],
        )
        return [start_object, goal_object]
