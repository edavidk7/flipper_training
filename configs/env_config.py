from flipper_training.configs.base_config import BaseConfig
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EnvConfig(BaseConfig):
    """
    Basic environment configuration

    Attributes:
        control_type: Literal["per-track", "lon_ang"] = "lon_ang" - control type for the environment, either per-track or lon_ang
        differentiable: bool = False - whether the environment is differentiable, if False, the simulation step is performed in a no_grad context
    """
    control_type: Literal["per-track", "lon_ang"] = "lon_ang"
    differentiable: bool = False
