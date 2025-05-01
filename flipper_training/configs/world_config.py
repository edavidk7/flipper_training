from dataclasses import dataclass

import torch

from flipper_training.configs.base_config import BaseConfig
from flipper_training.utils.environment import compute_heightmap_gradients


@dataclass
class TerrainConfig(BaseConfig):
    """
    Terrain configuration. Contains the physical constants of the world, coordinates of the world frame etc.

    **Note**:
        1) the convention for grids is torch's "xy" indexing of the meshgrid. This means that the first
            dimension of the grid corresponds to the y-coordinate and the second dimension corresponds to the x-coordinate.
            The coordinate increases with increasing index, i.e. the Y down and X right. Generate them with torch.meshgrid(y, x, indexing="xy").
        2) origin [0,0] is at the center of the grid.

    x_grid (torch.Tensor):  x-coordinates of the grid, shape (B, grid_dim, grid)
    y_grid (torch.Tensor): y-coordinates of the grid, shape (B, grid_dim, grid)
    z_grid (torch.Tensor): z-coordinates of the grid, shape (B, grid_dim, grid)
    z_grid_grad (torch.Tensor): gradients of the z-coordinates of the grid. Shape (B, 2, grid_dim, grid_dim).
    normals (torch.Tensor): normals of the terrain. Shape (B, 3, grid_dim, grid_dim).
    grid_res (float): resolution of the grid in meters. Represents the metric distance between 2 centers of adjacent grid cells.
    max_coord (float): maximum coordinate of the grid.
    k_stiffness (float or torch.Tensor): stiffness of the terrain. Default is 20_000.
    k_friction (float or torch.Tensor): friction of the terrain. Default is 1.0.
    grid_extras (dict | None): extra information about the grid. Default is None.
    """

    x_grid: torch.Tensor
    y_grid: torch.Tensor
    z_grid: torch.Tensor
    grid_res: float
    max_coord: float
    k_stiffness: float | torch.Tensor = 20_000.0
    k_friction_lon: float | torch.Tensor = 0.5
    k_friction_lat: float | torch.Tensor = 0.2
    grid_extras: dict[str, torch.Tensor] | None = None

    def __post_init__(self):
        self.z_grid_grad = compute_heightmap_gradients(self.z_grid, self.grid_res)  # (B, 2, D, D)
        ones = torch.ones_like(self.z_grid_grad[:, 0]).unsqueeze(1)
        self.normals = torch.cat((-self.z_grid_grad, ones), dim=1)
        self.normals /= torch.linalg.norm(self.normals, dim=1, keepdim=True)

    @property
    def grid_size(self) -> int:
        """
        Returns the size of the grid.

        Returns:
            int: size of the grid.
        """
        return self.z_grid.shape[-1]

    def xy2ij(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Converts x-y coordinates to i-j coordinates.

        Args:
            xy (torch.Tensor): x-y coordinates of shape (B, 2).

        Returns:
            torch.Tensor: i-j coordinates of shape (B, 2).
        """
        return ((xy + self.max_coord) / self.grid_res).long().clamp(0, self.grid_size - 1)

    def ij2xy(self, ij: torch.Tensor) -> torch.Tensor:
        """
        Converts i-j coordinates to x-y coordinates.

        Args:
            ij (torch.Tensor): i-j coordinates of shape (B, 2).

        Returns:
            torch.Tensor: x-y coordinates of shape (B, 2).
        """
        return ((ij * self.grid_res) - self.max_coord).clamp(-self.max_coord, self.max_coord)
