from dataclasses import dataclass

import torch
from torchrl.data import Unbounded
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.utils.geometry import (
    inverse_quaternion,
    rotate_vector_by_quaternion,
    quaternion_to_euler,
)
from . import Observation
from .robot_state import LocalStateVectorEncoder


@dataclass
class LocalStateVectorWithAction(Observation):
    """
    Generates the observation vector for the robot state from kinematics and dynamics.
    """

    supports_vecnorm = True

    def __post_init__(self):
        if self.apply_noise:
            if not isinstance(self.noise_scale, (float, torch.Tensor)):
                raise ValueError("Noise scale must be specified if apply_noise is True and must be a float or tensor.")
            if isinstance(self.noise_scale, float):
                self.noise_scale = torch.tensor([self.noise_scale], dtype=self.env.out_dtype, device=self.env.device)
            if self.noise_scale.shape[0] not in (1, self.dim):
                raise ValueError(f"Noise scale tensor must have shape (1,) or ({self.dim},) but got {self.noise_scale.shape}.")
        self.max_dist = self.env.terrain_cfg.max_coord * 2**1.5
        self.theta_range = self.env.robot_cfg.joint_limits[1] - self.env.robot_cfg.joint_limits[0]

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
    ) -> torch.Tensor:
        goal_vecs = self.env.goal.x - curr_state.x  # (n_robots, 3)
        inv_q = inverse_quaternion(curr_state.q)  # (n_robots, 4)
        goal_vecs_local = rotate_vector_by_quaternion(goal_vecs.unsqueeze(1), inv_q).squeeze(1)  # (n_robots, 3)
        goal_vecs_local /= self.max_dist
        xd_local = rotate_vector_by_quaternion(curr_state.xd.unsqueeze(1), inv_q).squeeze(1)
        xd_local /= self.max_dist
        omega_local = rotate_vector_by_quaternion(curr_state.omega.unsqueeze(1), inv_q).squeeze(1) / torch.pi
        thetas = (curr_state.thetas - self.env.robot_cfg.joint_limits[None, 0]) / self.theta_range.unsqueeze(0)  # (n_robots, num_driving_parts)
        rolls, pitches, _ = quaternion_to_euler(curr_state.q)
        rolls.div_(torch.pi)
        pitches.div_(torch.pi)
        obs = torch.cat(
            [
                rolls.unsqueeze(1),
                pitches.unsqueeze(1),
                xd_local,
                omega_local,
                thetas,
                goal_vecs_local,
                action,
            ],
            dim=1,
        ).to(self.env.out_dtype)
        if self.apply_noise:
            noise = torch.randn_like(obs) * self.noise_scale.view(1, -1)
            obs.add_(noise)
        return obs

    @property
    def dim(self) -> int:
        """
        The dimension of the observation vector.
        """
        dim = 3  # velocity vector
        dim += 2  # roll and pitch angles
        dim += 3  # angular velocity vector
        dim += self.env.robot_cfg.num_driving_parts * 3  # joint angles and action
        dim += 3  # goal vector
        return dim

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.n_robots, self.dim),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )

    def get_encoder(self) -> LocalStateVectorEncoder:
        return LocalStateVectorEncoder(
            input_dim=self.dim,
            **self.encoder_opts,
        )
