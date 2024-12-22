from typing import Literal
import torch
from dataclasses import dataclass
from flipper_training.utils.numerical import IntegrationMode
from flipper_training.utils.environment import compute_heightmap_gradients
from flipper_training.engine.robot_model import RobotModelConfig


@dataclass
class PhysicsEngineConfig:
    """
    Physics Engine configuration. Contains the physical constants of the world, coordinates of the world frame etc.
    Model of the robot, its mass and geometry, and the parameters of the height map.

    **Note**:
        1) the robot is assumed to be symmetric with respect to the x-axis.
        2) the convention for grids is torch's "xy" indexing of the meshgrid. This means that the first
        dimension of the grid corresponds to the y-coordinate and the second dimension corresponds to the x-coordinate. The coordinate increases with increasing index, i.e. the Y down and X right. Generate them with torch.meshgrid(y, x, indexing="xy").
        3) origin [0,0] is at the center of the grid.

    Attributes:
        device (str): device to run the simulation on. Default is "cpu".
        num_robots (int): number of robots in the simulation.
        x_grid (torch.Tensor):  x-coordinates of the grid. 
        y_grid (torch.Tensor): y-coordinates of the grid.
        z_grid (torch.Tensor): z-coordinates of the grid.
        grid_res (float): resolution of the grid in meters. Represents the metric distance between 2 centers of adjacent grid cells.
        max_coord (float): maximum coordinate of the grid.
        robot_model (RobotModelConfig): configuration of the robot model.
        integration_mode (IntegrationMode): integration mode for the physics engine. Default is "rk4".
        dt (float): time step for numerical integration. Default is 0.01. (100 Hz)
        gravity (float): acceleration due to gravity. Default is 9.81 m/s^2.
        k_stiffness (float or torch.Tensor): stiffness of the terrain. Default is 20_000.  
        k_damping (float or torch.Tensor or None): damping of the terrain. Default is None, it is calculated based on the critical damping, i.e. sqrt(4 * robot_mass * k_stiffness).
        k_friction (float or torch.Tensor): friction of the terrain. Default is 1.0.    
        torque_limit (float): torque limit that can be generated on CoG. The physics engine clips it to this value. Default is 1000.0 Nm.
        contact_threshold (float): distance threshold for contact detection. Positive = above the terrain, negative = below the terrain. Default is 0.01 m.
        gravity_force (torch.Tensor): gravity force acting on the robot. It is calculated as [0, 0, -robot_mass * gravity].
    """
    device: str
    x_grid: torch.Tensor
    y_grid: torch.Tensor
    z_grid: torch.Tensor
    grid_res: float
    max_coord: float
    robot_model: RobotModelConfig
    integration_mode: IntegrationMode = "rk4"
    dt: float = 0.01
    gravity: float = 9.81
    k_stiffness: float | torch.Tensor = 20_000.
    k_friction: float | torch.Tensor = 1.0
    torque_limit: float = 1000.0
    contact_threshold: float = 0.01

    def __post_init__(self):
        self.k_damping = (4 * self.robot_model.mass * self.k_stiffness)**0.5
        self.z_grid_grad = compute_heightmap_gradients(self.z_grid, self.grid_res)
        self.F_gravity = torch.tensor([0., 0., -self.robot_model.mass * self.gravity], device=self.device)
        # Move all tensors to the device
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(self.device))
        self.robot_model.move_all_tensors_to_device(self.device)
