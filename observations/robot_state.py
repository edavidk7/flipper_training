import torch
from flipper_training.engine.engine_state import PhysicsState, AuxEngineInfo, PhysicsStateDer
from flipper_training.utils.geometry import (
    quaternion_to_euler,
    inverse_quaternion,
    rotate_vector_by_quaternion,
)
from .obs import Observation
from dataclasses import dataclass
from torchrl.data import Unbounded


@dataclass
class RobotStateVector(Observation):
    """
    Generates the observation vector for the robot state from kinematics and dynamics.
    """

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        state_der: PhysicsStateDer,
        curr_state: PhysicsState,
        aux_info: AuxEngineInfo,
    ) -> torch.Tensor:
        goal_vecs = self.env.goal.x - curr_state.x  # (n_robots, 3)
        goal_vecs_local = rotate_vector_by_quaternion(
            goal_vecs.unsqueeze(1), inverse_quaternion(curr_state.q)
        ).squeeze(1)  # (n_robots, 3)
        rolls, pitches, _ = quaternion_to_euler(curr_state.q)
        return torch.cat(
            [
                curr_state.xd,
                curr_state.omega,
                curr_state.thetas,
                rolls.unsqueeze(0),
                pitches.unsqueeze(0),
                goal_vecs_local,
            ],
            dim=1,
        )

    def get_spec(self) -> Unbounded:
        dim = 3  # velocity vector
        dim += 3  # angular velocity vector
        dim += self.env.robot_cfg.num_joints  # joint angles
        dim += 1  # roll
        dim += 1  # pitch
        dim += 3  # goal vector
        return Unbounded(
            shape=(self.env.n_robots, dim),
            dtype=torch.float32,
            device=self.env.device,
        )
