from abc import ABC
from dataclasses import dataclass
from torch import device, Tensor


@dataclass
class BaseConfig(ABC):
    """
    Base configuration class. This class is used to store the configuration of the simulation.
    """
    pass

    def move_all_tensors_to_device(self, device: device | str) -> None:
        """
        Moves all tensors to the specified device.

        Args:
            device (torch.device): device to move the tensors to.

        Returns:
            None
        """
        for attr, val in self.__dict__.items():
            if isinstance(val, Tensor):
                setattr(self, attr, val.to(device))
