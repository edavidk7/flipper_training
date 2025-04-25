from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
from functools import wraps

import torch
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer

if TYPE_CHECKING:
    from flipper_training.environment.env import Env


@dataclass
class Reward(ABC):
    """
    Reward function for the environment.
    """

    env: "Env"

    @abstractmethod
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
        """
        Calculate the reward for the environment.

        Args:
            prev_state (PhysicsState): The previous state of the environment.
            action (torch.Tensor): The action taken in the environment.
            prev_state_der (PhysicsStateDer): The state derivative of the environment.
            curr_state (PhysicsState): The current state of the environment.
            success (torch.BoolTensor): The success tensor.
            fail (torch.BoolTensor): The fail tensor.
            start_state (PhysicsState): The start state for the current episode.
            goal_state (PhysicsState): The goal state for the current episode.

        Returns:
            The reward tensor of shape (batch_size,1).
        """

        raise NotImplementedError

    @property
    def name(self) -> str:
        """
        Name of the reward function.
        """
        return self.__class__.__name__

    @classmethod
    def make_factory(cls, **opts):
        """
        Factory method to create a reward function with the given options.
        """

        @wraps(cls)
        def factory(env: "Env"):
            return cls(env=env, **opts)

        return factory


RewardFactory = Callable[["Env"], Reward]
