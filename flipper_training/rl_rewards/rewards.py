from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.rl_rewards import Reward

if TYPE_CHECKING:
    from flipper_training.environment.env import Env

__all__ = ["RollPitchGoal", "Goal"]


@dataclass
class RollPitchGoal(Reward):
    """
    Reward function for the environment.
    Robot is rewarded for minimizing the roll and pitch angles and for moving towards the goal position,
    penalized for moving away from the goal position.
    """

    goal_reached_reward: float
    failed_reward: float
    omega_weight: float
    goal_weight: float

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
        success: torch.BoolTensor,
        fail: torch.BoolTensor,
        env: "Env",
    ) -> torch.Tensor:
        roll_pitch_rates_sq = curr_state.omega[..., :2].pow(2)  # shape (batch_size, 2)
        goal_diff_curr = (env.goal.x - curr_state.x).norm(dim=-1, keepdim=True)
        goal_diff_prev = (env.goal.x - prev_state.x).norm(dim=-1, keepdim=True)
        diff_delta = goal_diff_curr - goal_diff_prev
        reward = -self.omega_weight * roll_pitch_rates_sq.sum(dim=-1, keepdim=True) + self.goal_weight * diff_delta
        reward[success] = self.goal_reached_reward
        reward[fail] = self.failed_reward
        return reward.to(env.out_dtype)


@dataclass
class Goal(Reward):
    goal_reached_reward: float
    failed_reward: float
    weight: float
    exp: float | int = 2

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
        success: torch.BoolTensor,
        fail: torch.BoolTensor,
        env: "Env",
    ) -> torch.Tensor:
        goal_diff = (env.goal.x - curr_state.x).norm(dim=-1, keepdim=True)
        reward = -self.weight * goal_diff.pow(self.exp)
        reward[success] += self.goal_reached_reward
        reward[fail] += self.failed_reward
        return reward.to(env.out_dtype)
