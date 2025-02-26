import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING
from flipper_training.engine.engine_state import PhysicsState, AuxEngineInfo, PhysicsStateDer
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from flipper_training.environment.env import Env
    
__all__ = ["Reward", "RollPitchGoal"]


@dataclass
class Reward(ABC):
    """
    Reward function for the environment.
    """

    @abstractmethod
    def __call__(self,
                 prev_state: PhysicsState,
                 action: torch.Tensor,
                 state_der: PhysicsStateDer,
                 curr_state: PhysicsState,
                 aux_info: AuxEngineInfo,
                 success: torch.BoolTensor,
                 fail: torch.BoolTensor,
                 env: "Env"
                 ) -> torch.Tensor:
        """
        Calculate the reward for the environment.

        Args:
            prev_state (PhysicsState): The previous state of the environment.
            action (torch.Tensor): The action taken in the environment.
            state_der (PhysicsStateDer): The state derivative of the environment.
            curr_state (PhysicsState): The current state of the environment.
            aux_info (AuxEngineInfo): The auxiliary information from the engine.
            success (torch.BoolTensor): The success tensor.
            fail (torch.BoolTensor): The fail tensor.
            env (Env): The environment.

        Returns:
            The reward tensor of shape (batch_size,1).
        """

        raise NotImplementedError


@dataclass
class RollPitchGoal(Reward):
    """
    Reward function for the environment. 
    Robot is rewarded for minimizing the roll and pitch angles and for moving towards the goal position, penalized for moving away from the goal position.
    """

    goal_reached_reward: float
    failed_reward: float
    omega_weight: float
    goal_weight: float

    def __call__(self,
                 prev_state: PhysicsState,
                 action: torch.Tensor,
                 state_der: PhysicsStateDer,
                 curr_state: PhysicsState,
                 aux_info: AuxEngineInfo,
                 success: torch.BoolTensor,
                 fail: torch.BoolTensor,
                 env: "Env"
                 ) -> torch.Tensor:
        roll_pitch_rates_sq = curr_state.omega[..., :2].pow(2)  # shape (batch_size, 2)
        goal_diff_curr = (env.goal.x - curr_state.x).norm(dim=-1, keepdim=True)
        goal_diff_prev = (env.goal.x - prev_state.x).norm(dim=-1, keepdim=True)
        diff_delta = goal_diff_curr - goal_diff_prev
        reward = -self.omega_weight * roll_pitch_rates_sq.sum(dim=-1, keepdim=True) + self.goal_weight * diff_delta
        reward[success] = self.goal_reached_reward
        reward[fail] = self.failed_reward
        return reward
