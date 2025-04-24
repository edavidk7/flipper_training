# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn


class SimpleMPPIPlanner(nn.Module):
    """A concise MPPI Planner that uses the environment's batch size as the number of candidates.

    Args:
        env: The environment to perform planning on (batch-locked).
        temperature: Softness parameter for trajectory weighting.
        planning_horizon: Length of the simulated trajectories.
        optim_steps: Number of optimization steps.
        top_k: Number of top trajectories to update the action distribution.
        gamma: Discount factor for computing the discounted reward sum as advantage.
        reward_key: Key in the TensorDict to retrieve the reward (default: ("next", "reward")).
    """

    def __init__(
        self,
        env,
        temperature: float,
        planning_horizon: int,
        optim_steps: int,
        top_k: int,
        gamma: float = 0.99,
        reward_key: tuple[str, str] = ("next", "reward"),
    ):
        super().__init__()
        self.env = env
        self.temperature = torch.as_tensor(temperature, device=env.device)
        self.planning_horizon = planning_horizon
        self.optim_steps = optim_steps
        self.top_k = top_k
        self.gamma = gamma
        self.reward_key = reward_key
        # Use the environment's batch size as the number of candidates
        self.num_candidates = env.batch_size[0] if env.batch_size else 1
        if self.top_k > self.num_candidates:
            raise ValueError(f"top_k ({self.top_k}) must be less than or equal to num_candidates ({self.num_candidates})")

    def forward(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Perform MPPI planning and return the optimal action for the current state.

        Args:
            tensordict: Input TensorDict containing the initial state.

        Returns:
            torch.Tensor: The planned action for the current step.
        """
        action_shape = (self.num_candidates, self.planning_horizon, *self.env.action_spec.shape)
        action_stats_shape = (1, self.planning_horizon, *self.env.action_spec.shape)
        expanded_tensordict = tensordict
        # Initialize action distribution
        action_means = torch.zeros(
            *action_stats_shape,
            device=tensordict.device,
            dtype=self.env.action_spec.dtype,
        )
        action_stds = torch.ones_like(action_means)
        # MPPI optimization loop
        for _ in range(self.optim_steps):
            # Sample candidate actions
            actions = action_means + action_stds * torch.randn(
                *action_shape,
                device=action_means.device,
                dtype=action_means.dtype,
            )
            actions = self.env.action_spec.project(actions)

            # Roll out trajectories
            rollout_tensordict = expanded_tensordict.clone()
            policy = _PrecomputedActionsSequentialSetter(actions)
            rollout_tensordict = self.env.rollout(
                max_steps=self.planning_horizon,
                policy=policy,
                auto_reset=False,
                tensordict=rollout_tensordict,
            )

            # Compute advantage as discounted reward sum
            rewards = rollout_tensordict.get(self.reward_key)  # Shape: [..., planning_horizon, 1]
            timesteps = torch.arange(rewards.shape[-2], device=rewards.device)
            discounts = self.gamma**timesteps
            discounted_rewards = rewards * discounts.view(1, -1, 1)
            advantage = discounted_rewards.sum(dim=-2, keepdim=True)  # Shape: [..., 1, 1]

            # Select top-k trajectories
            K_DIM = -3  # Dimension of num_candidates
            _, top_k = advantage.topk(self.top_k, dim=K_DIM)
            top_k_expanded = top_k.expand(self.top_k, self.planning_horizon, *self.env.action_spec.shape)
            vals = advantage.gather(K_DIM, top_k)  # Shape: [..., top_k, 1, 1]
            omegas = (self.temperature * vals).exp()

            # Update action distribution
            best_actions = actions.gather(K_DIM, top_k_expanded)
            action_means = (omegas * best_actions).sum(dim=K_DIM, keepdim=True) / omegas.sum(K_DIM, keepdim=True)
            action_stds = ((omegas * (best_actions - action_means).pow(2)).sum(dim=K_DIM, keepdim=True) / omegas.sum(K_DIM, keepdim=True)).sqrt()

        # Return the action for the current step
        return action_means[..., 0, 0, :].expand(self.env.action_spec.shape)


class _PrecomputedActionsSequentialSetter:
    def __init__(self, actions):
        self.actions = actions
        self.cmpt = 0

    def __call__(self, tensordict):
        if self.cmpt >= self.actions.shape[-2]:
            raise ValueError("Precomputed actions sequence is too short")
        tensordict = tensordict.set("action", self.actions[..., self.cmpt, :])
        self.cmpt += 1
        return tensordict
