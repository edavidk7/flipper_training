from abc import ABC
import torch
from tensordict import TensorDict


class BaseConfig(ABC):
    """
    Base configuration class. This class is used to store the configuration of the simulation.
    """

    def to(self, device: torch.device | str):
        """
        Moves all tensors to the specified device.

        Args:
            device (torch.device): device to move the tensors to.

        Returns:
            None
        """
        for attr, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))
            elif isinstance(val, TensorDict):
                val.to(device)
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    if isinstance(item, torch.Tensor):
                        val[i] = item.to(device)
                    elif isinstance(item, TensorDict):
                        item.to(device)
            elif isinstance(val, dict):
                for key, item in val.items():
                    if isinstance(item, torch.Tensor):
                        val[key] = item.to(device)
                    elif isinstance(item, TensorDict):
                        item.to(device)

        return self
