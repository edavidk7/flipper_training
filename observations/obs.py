import torch
from dataclasses import dataclass
from typing import Type, Any, TYPE_CHECKING
from flipper_training.engine.engine_state import PhysicsState, AuxEngineInfo, PhysicsStateDer
from torchrl.data import Composite, Unbounded, Bounded
from abc import ABC, abstractmethod

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

    @abstractmethod
    def __call__(self, prev_state: PhysicsState,
                 action: torch.Tensor,
                 state_der: PhysicsStateDer,
                 curr_state: PhysicsState,
                 aux_info: AuxEngineInfo) -> torch.Tensor:
        """
        Generate observations from the current state of the environment.

        Args:
            prev_state (PhysicsState): The previous state of the environment.
            action (torch.Tensor): The action taken in the environment.
            state_der (PhysicsStateDer): The state derivative of the environment.
            curr_state (PhysicsState): The current state of the environment.
            aux_info (AuxEngineInfo): The auxiliary information from the engine.

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
