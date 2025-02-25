from flipper_training.configs.base_config import BaseConfig
from dataclasses import dataclass


@dataclass
class PhysicsEngineConfig(BaseConfig):
    """
    Physics Engine configuration. Contains the physical constants of the simulation, such as gravity, time step, etc.

    Attributes:
        num_robots (int): number of robots in the simulation.
        dt (float): time step for numerical integration. Default is 0.01. (100 Hz)
        gravity (float): acceleration due to gravity. Default is 9.81 m/s^2.
        torque_limit (float): torque limit that can be generated on CoG. The physics engine clips it to this value. Default is 500.0 Nm.
        damping_alpha (float): damping coefficient modifier, should be between 0 and 2. Default is 1.0 (damping is critical damping)
    """

    num_robots: int
    dt: float = 0.01
    gravity: float = 9.81
    torque_limit: float = 200.0
    damping_alpha: float = 1.0
