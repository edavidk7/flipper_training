from flipper_training.configs.base_config import BaseConfig
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EnvConfig(BaseConfig):
    """
    Basic environment configuration

    Attributes:
        percep_shape: tuple[int] = (256, 256) - The shape of the perception grid
        percep_extent: tuple[float] = (1.0, 1.0, -1.0, -1.0) - The extent of the perception grid (x_min, y_min, x_max, y_max) in the robot's local frame
        percep_type: Literal['heightmap', 'pointcloud'] = 'heightmap' - The type of perception data to use, either a heightmap or a pointcloud representation
    """
    percep_shape: tuple[int, int] = (256, 256)
    percep_extent: tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0)
    percep_type: Literal['heightmap', 'pointcloud'] = 'heightmap'
