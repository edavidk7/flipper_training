from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer

if TYPE_CHECKING:
    from flipper_training.environment.env import Env


@dataclass
class Reward(ABC):
    """
    Reward function for the environment.
    """

    @abstractmethod
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
        """
        Calculate the reward for the environment.

        Args:
            prev_state (PhysicsState): The previous state of the environment.
            action (torch.Tensor): The action taken in the environment.
            prev_state_der (PhysicsStateDer): The state derivative of the environment.
            curr_state (PhysicsState): The current state of the environment.
            success (torch.BoolTensor): The success tensor.
            fail (torch.BoolTensor): The fail tensor.
            env (Env): The environment.

        Returns:
            The reward tensor of shape (batch_size,1).
        """

        raise NotImplementedError
