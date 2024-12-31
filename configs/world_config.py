import torch
from dataclasses import dataclass
from flipper_training.configs.base_config import BaseConfig
from flipper_training.utils.environment import compute_heightmap_gradients


@dataclass
class WorldConfig(BaseConfig):
    """
    World configuration. Contains the physical constants of the world, coordinates of the world frame etc.

    **Note**:
        1) the convention for grids is torch's "xy" indexing of the meshgrid. This means that the first
            dimension of the grid corresponds to the y-coordinate and the second dimension corresponds to the x-coordinate. The coordinate increases with increasing index, i.e. the Y down and X right. Generate them with torch.meshgrid(y, x, indexing="xy").
        2) origin [0,0] is at the center of the grid.

    x_grid (torch.Tensor):  x-coordinates of the grid. 
    y_grid (torch.Tensor): y-coordinates of the grid.
    z_grid (torch.Tensor): z-coordinates of the grid.
    grid_res (float): resolution of the grid in meters. Represents the metric distance between 2 centers of adjacent grid cells.
    max_coord (float): maximum coordinate of the grid.
    z_grid_grad (torch.Tensor): gradients of the heightmap. Shape (2, grid_dim, grid_dim). The first dimension corresponds to the x and y gradients respectively.
    k_stiffness (float or torch.Tensor): stiffness of the terrain. Default is 20_000.
    k_friction (float or torch.Tensor): friction of the terrain. Default is 1.0.    
    suitable_mask (torch.Tensor | None): mask of suitable terrain. Shape (grid_dim, grid_dim). 1 if suitable, 0 if not. Default is None.
    """

    x_grid: torch.Tensor
    y_grid: torch.Tensor
    z_grid: torch.Tensor
    grid_res: float
    max_coord: float
    k_stiffness: float | torch.Tensor = 20_000.
    k_friction: float | torch.Tensor = 1.0
    suitable_mask: torch.Tensor | None = None

    def __post_init__(self):
        self.z_grid_grad = compute_heightmap_gradients(self.z_grid, self.grid_res)
