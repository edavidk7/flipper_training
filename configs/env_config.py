from flipper_training.configs.base_config import BaseConfig
from dataclasses import dataclass
from typing import Literal


@dataclass
class EnvConfig(BaseConfig):
    """
    Basic environment configuration

    Attributes:
        percep_dim: int = 256 - The dimension of the perception grid (percep_dim x percep_dim)
        percep_coord: float = 1.0 - The metric coordinate of the perception grid furthest from the origin at the center (so for a value of 1.0 the grid spans from -1 to 1)
        percep_type: Literal['heightmap', 'pointcloud'] = 'heightmap' - The type of perception data to use, either a heightmap or a pointcloud representation
    """
    percep_dim: int = 256
    percep_coord: float = 1.0
    percep_type: Literal['heightmap', 'pointcloud'] = 'heightmap'
