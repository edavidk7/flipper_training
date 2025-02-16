import torch
from flipper_training.engine.engine_state import PhysicsState, AuxEngineInfo, PhysicsStateDer
from .obs import Observation, PhysicsState
from dataclasses import dataclass
from torchrl.data import Unbounded
from flipper_training.utils.geometry import roll_from_R, pitch_from_R


@dataclass
class RobotStateVector(Observation):
    """
    Generates the observation vector for the robot state from kinematics and dynamics.
    """

    def __call__(self, prev_state: PhysicsState,
                 action: torch.Tensor,
                 state_der: PhysicsStateDer,
                 curr_state: PhysicsState,
                 aux_info: AuxEngineInfo) -> torch.Tensor:
        goal_vecs = torch.bmm((self.env.goal.x - curr_state.x).unsqueeze(1), curr_state.R).squeeze(dim=1)
        rolls = roll_from_R(curr_state.R).reshape(-1, 1)
        pitches = pitch_from_R(curr_state.R).reshape(-1, 1)
        return torch.cat([curr_state.xd, curr_state.omega, curr_state.thetas, rolls, pitches, goal_vecs], dim=1)

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
