from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.rl_rewards import Reward

__all__ = [
    "RollPitchGoal",
    "PotentialGoal",
    "PotentialGoalWithVelocityBonus",
    "PotentialGoalWithConditionalVelocityBonus",
    "PotentialGoalWithConditionalVelocityBonusAndJointCommandBonus",
    "GoalDistance",
]


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
        start_state: PhysicsState,
        goal_state: PhysicsState,
    ) -> torch.Tensor:
        roll_pitch_rates_sq = curr_state.omega[..., :2].pow(2)  # shape (batch_size, 2)
        goal_diff_curr = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        goal_diff_prev = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        diff_delta = goal_diff_curr - goal_diff_prev
        reward = self.omega_weight * roll_pitch_rates_sq.sum(dim=-1, keepdim=True) - self.goal_weight * diff_delta
        reward[success] = self.goal_reached_reward
        reward[fail] = self.failed_reward
        return reward.to(self.env.out_dtype)


@dataclass
class GoalDistance(Reward):
    goal_reached_reward: float
    failed_reward: float
    weight: float
    exp: float | int = 1

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
        success: torch.BoolTensor,
        fail: torch.BoolTensor,
        start_state: PhysicsState,
        goal_state: PhysicsState,
    ) -> torch.Tensor:
        goal_diff = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True) / (self.env.terrain_cfg.max_coord * 2**1.5)
        reward = -self.weight * goal_diff.pow(self.exp)
        reward[success] += self.goal_reached_reward
        reward[fail] += self.failed_reward
        return reward.to(self.env.out_dtype)


@dataclass
class PotentialGoal(Reward):
    goal_reached_reward: float
    failed_reward: float
    gamma: float
    step_penalty: float
    potential_coef: float

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
        success: torch.BoolTensor,
        fail: torch.BoolTensor,
        start_state: PhysicsState,
        goal_state: PhysicsState,
    ) -> torch.Tensor:
        curr_dist = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        prev_dist = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        neg_goal_dist_curr = -curr_dist  # phi(s')
        neg_goal_dist_prev = -prev_dist  # phi(s)
        reward = self.potential_coef * (self.gamma * neg_goal_dist_curr - neg_goal_dist_prev) + self.step_penalty
        reward[success] += self.goal_reached_reward
        reward[fail] += self.failed_reward
        return reward.to(self.env.out_dtype)


@dataclass
class PotentialGoalWithVelocityBonus(Reward):
    goal_reached_reward: float
    failed_reward: float
    gamma: float
    step_penalty: float
    potential_coef: float
    velocity_bonus_coef: float

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
        success: torch.BoolTensor,
        fail: torch.BoolTensor,
        start_state: PhysicsState,
        goal_state: PhysicsState,
    ) -> torch.Tensor:
        curr_dist = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        prev_dist = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        # Normalized potential to bound rewards
        neg_goal_dist_curr = -curr_dist  # phi(s')
        neg_goal_dist_prev = -prev_dist  # phi(s)
        reward = (
            self.potential_coef * (self.gamma * neg_goal_dist_curr - neg_goal_dist_prev)
            + self.step_penalty
            + self.velocity_bonus_coef * curr_state.xd.norm(dim=-1, keepdim=True)
        )
        reward[success] += self.goal_reached_reward
        reward[fail] += self.failed_reward
        return reward.to(self.env.out_dtype)


@dataclass
class PotentialGoalWithConditionalVelocityBonus(Reward):
    goal_reached_reward: float
    failed_reward: float
    gamma: float
    step_penalty: float
    potential_coef: float
    velocity_bonus_coef: float

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
        success: torch.BoolTensor,
        fail: torch.BoolTensor,
        start_state: PhysicsState,
        goal_state: PhysicsState,
    ) -> torch.Tensor:
        curr_dist = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        prev_dist = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        # Normalized potential to bound rewards
        neg_goal_dist_curr = -curr_dist  # phi(s')
        neg_goal_dist_prev = -prev_dist  # phi(s)
        reward = (
            self.potential_coef * (self.gamma * neg_goal_dist_curr - neg_goal_dist_prev)
            + self.step_penalty
            + self.velocity_bonus_coef
            * curr_state.xd.norm(dim=-1, keepdim=True)
            * (curr_dist <= prev_dist).float()  # award only if the robot is getting closer to the goal
        )
        reward[success] += self.goal_reached_reward
        reward[fail] += self.failed_reward
        return reward.to(self.env.out_dtype)


@dataclass
class PotentialGoalWithConditionalVelocityBonusAndJointCommandBonus(Reward):
    goal_reached_reward: float
    failed_reward: float
    gamma: float
    step_penalty: float
    potential_coef: float
    velocity_bonus_coef: float
    joint_command_bonus_coef: float

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
        success: torch.BoolTensor,
        fail: torch.BoolTensor,
        start_state: PhysicsState,
        goal_state: PhysicsState,
    ) -> torch.Tensor:
        curr_dist = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        prev_dist = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        # Normalized potential to bound rewards
        neg_goal_dist_curr = -curr_dist  # phi(s')
        neg_goal_dist_prev = -prev_dist  # phi(s)
        reward = (
            self.potential_coef * (self.gamma * neg_goal_dist_curr - neg_goal_dist_prev)
            + self.step_penalty
            + self.velocity_bonus_coef
            * curr_state.xd.norm(dim=-1, keepdim=True)
            * (curr_dist <= prev_dist).float()  # award only if the robot is getting closer to the goal
            + action[..., action.shape[1] // 2 :].pow(2).sum(dim=-1, keepdim=True) * self.joint_command_bonus_coef
        )
        reward[success] += self.goal_reached_reward
        reward[fail] += self.failed_reward
        return reward.to(self.env.out_dtype)


@dataclass
class PotentialGoalWithJointVelVariancePenalty(Reward):
    goal_reached_reward: float
    failed_reward: float
    gamma: float
    step_penalty: float
    potential_coef: float
    joint_vel_variance_coef: float

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
        success: torch.BoolTensor,
        fail: torch.BoolTensor,
        start_state: PhysicsState,
        goal_state: PhysicsState,
    ) -> torch.Tensor:
        curr_dist = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        prev_dist = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        neg_goal_dist_curr = -curr_dist  # phi(s')
        neg_goal_dist_prev = -prev_dist  # phi(s)
        reward = self.potential_coef * (self.gamma * neg_goal_dist_curr - neg_goal_dist_prev) + self.step_penalty
        joint_vel_variances = action[..., action.shape[1] // 2 :].abs().var(dim=-1, keepdim=True)
        reward -= self.joint_vel_variance_coef * joint_vel_variances
        reward[success] += self.goal_reached_reward
        reward[fail] += self.failed_reward
        return reward.to(self.env.out_dtype)


@dataclass
class PotentialGoalWithFinishVelocityPenalty(Reward):
    goal_reached_reward: float
    failed_reward: float
    gamma: float
    step_penalty: float
    potential_coef: float
    finish_velocity_coef: float

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
        success: torch.BoolTensor,
        fail: torch.BoolTensor,
        start_state: PhysicsState,
        goal_state: PhysicsState,
    ) -> torch.Tensor:
        curr_dist = (goal_state.x - curr_state.x).norm(dim=-1, keepdim=True)
        prev_dist = (goal_state.x - prev_state.x).norm(dim=-1, keepdim=True)
        neg_goal_dist_curr = -curr_dist  # phi(s')
        neg_goal_dist_prev = -prev_dist  # phi(s)
        reward = self.potential_coef * (self.gamma * neg_goal_dist_curr - neg_goal_dist_prev) + self.step_penalty
        reward[success] += self.goal_reached_reward - self.finish_velocity_coef * curr_state.xd[success].norm(dim=-1, keepdim=True)
        reward[fail] += self.failed_reward
        return reward.to(self.env.out_dtype)
