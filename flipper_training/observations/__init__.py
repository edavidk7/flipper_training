from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer

if TYPE_CHECKING:
    from flipper_training.environment.env import Env


@dataclass
class Observation(ABC):
    """
    Abstract class for observation generators.

    Args:
        env (Env): The environment.
    """

    env: "Env"
    supports_vecnorm: ClassVar[bool] = NotImplemented

    @abstractmethod
    def __call__(
        self, prev_state: PhysicsState, action: torch.Tensor, prev_state_der: PhysicsStateDer, curr_state: PhysicsState
    ) -> torch.Tensor | TensorDict:
        """
        Generate observations from the current state of the environment.

        Args:
            prev_state (PhysicsState): The previous state of the environment.
            action (torch.Tensor): The action taken in previous state.
            prev_state_der (PhysicsStateDer): The derivative of the previous state.
            curr_state (PhysicsState): The current state of the environment.

        Returns:
            The observation tensor.
        """
        pass

    @abstractmethod
    def get_spec(self) -> Bounded | Unbounded | Composite:
        """
        Get the observation spec.

        Returns:
            The observation spec.
        """
        pass

    @abstractmethod
    def get_encoder(self, output_dim: int, *args, **kwargs) -> torch.nn.Module:
        """
        Get the encoder for the observation.

        Returns:
            The observation encoder.
        """
        pass
