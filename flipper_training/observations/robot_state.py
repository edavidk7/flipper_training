from dataclasses import dataclass

import torch
from torchrl.data import Unbounded

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.utils.geometry import (
    inverse_quaternion,
    quaternion_to_euler,
    rotate_vector_by_quaternion,
)

from . import Observation


@dataclass
class LocalStateVector(Observation):
    """
    Generates the observation vector for the robot state from kinematics and dynamics.
    """

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
        rolls, pitches, _ = quaternion_to_euler(curr_state.q)
        xd_local = rotate_vector_by_quaternion(curr_state.xd.unsqueeze(1), inv_q).squeeze(1)
        omega_local = rotate_vector_by_quaternion(curr_state.omega.unsqueeze(1), inv_q).squeeze(1)
        return torch.cat(
            [
                xd_local,
                omega_local,
                curr_state.thetas,
                rolls.unsqueeze(-1),
                pitches.unsqueeze(-1),
                goal_vecs_local,
            ],
            dim=1,
        ).to(self.env.out_dtype)

    def get_spec(self) -> Unbounded:
        dim = 3  # velocity vector
        dim += 3  # angular velocity vector
        dim += self.env.robot_cfg.num_driving_parts  # joint angles
        dim += 1  # roll
        dim += 1  # pitch
        dim += 3  # goal vector
        return Unbounded(
            shape=(self.env.n_robots, dim),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )
