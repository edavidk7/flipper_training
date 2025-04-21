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
from flipper_training.policies import make_mlp_layer_module


class LocalStateVectorEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_hidden: int, hidden_dim: int, ln: bool):
        super(LocalStateVectorEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden = num_hidden
        self.hidden_dim = hidden_dim
        self.ln = ln
        self.encoder = torch.nn.Sequential(
            make_mlp_layer_module(input_dim, hidden_dim, ln=ln),  # input dimension
            *[make_mlp_layer_module(hidden_dim, ln=ln) for _ in range(num_hidden)],  # hidden layers
            make_mlp_layer_module(hidden_dim, output_dim, ln=ln),  # output dimension
        )

    def forward(self, x):
        return self.encoder(x)


@dataclass
class LocalStateVector(Observation):
    """
    Generates the observation vector for the robot state from kinematics and dynamics.
    """

    supports_vecnorm = True

    def __post_init__(self):
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
        return torch.cat(
            [
                rolls.unsqueeze(1),
                pitches.unsqueeze(1),
                xd_local,
                omega_local,
                thetas,
                goal_vecs_local,
            ],
            dim=1,
        ).to(self.env.out_dtype)

    def get_spec(self) -> Unbounded:
        dim = 3  # velocity vector
        dim += 2  # roll and pitch angles
        dim += 3  # angular velocity vector
        dim += self.env.robot_cfg.num_driving_parts  # joint angles
        dim += 3  # goal vector
        return Unbounded(
            shape=(self.env.n_robots, dim),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )

    def get_encoder(self, output_dim, num_hidden: int, hidden_dim: int, ln: bool) -> LocalStateVectorEncoder:
        in_dim = self.get_spec().shape[-1]
        return LocalStateVectorEncoder(
            input_dim=in_dim,
            output_dim=output_dim,
            num_hidden=num_hidden,
            hidden_dim=hidden_dim,
            ln=ln,
        )
