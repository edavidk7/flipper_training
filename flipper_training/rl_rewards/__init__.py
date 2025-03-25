import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING
from flipper_training.engine.engine_state import PhysicsState, AuxEngineInfo, PhysicsStateDer
from abc import ABC, abstractmethod

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
        state_der: PhysicsStateDer,
        curr_state: PhysicsState,
        aux_info: AuxEngineInfo,
        success: torch.BoolTensor,
        fail: torch.BoolTensor,
        env: "Env",
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
