from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from flipper_training.utils.environment import make_x_y_grids


@dataclass
class BaseHeightmapGenerator(ABC):
    """
    Base class for heightmap generators.

    Attributes:
    - add_random_noise: bool - whether to add random noise to the heightmap
    - noise_std: float - standard deviation of the Gaussian noise in meters
    - noise_mu: float - mean of the Gaussian noise in meters
    """

    add_random_noise: bool = False
    noise_std: float = 0.01  # Standard deviation of the Gaussian noise in meters
    noise_mu: float = 0.0  # Mean of the Gaussian noise in meters

    def __call__(
        self, grid_res: float, max_coord: float, num_robots: int, rng: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Generates a heightmap.

        Args:
        - grid_res: Resolution of the grid in meters.
        - max_coord: Maximum coordinate in meters.
        - num_robots: Number of robots.
        - rng: Random number generator.

        Returns:
        - x: Tensor of x coordinates. Shape is (B, D, D).
        - y: Tensor of y coordinates. Shape is (B, D, D).
        - z: Heightmap tensor of shape (B, D, D).
        - mask: Suitability mask tensor of shape (B, D, D).
        """
        x, y = make_x_y_grids(max_coord, grid_res, num_robots)
        z, extras = self._generate_heightmap(x, y, max_coord, rng)
        if self.add_random_noise:
            z = self._add_noise_to_heightmap(z, rng)
        return x, y, z, extras

    @abstractmethod
    def _generate_heightmap(
        self, x: torch.Tensor, y: torch.Tensor, max_coord: float, rng: torch.Generator | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Generates a heightmap.

        Args:
        - x: Tensor of x coordinates. Shape is (B, D, D).
        - y: Tensor of y coordinates. Shape is (B, D, D).
        - max_coord: Maximum coordinate in meters.
        - rng: Random number generator.

        Returns:
        - Heightmap tensor of shape (B, D, D) and an extras dictionary.
        """
        raise NotImplementedError

    def _add_noise_to_heightmap(self, z: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
        """
        Adds Gaussian noise to a heightmap.

        Args:
        - z: Heightmap tensor of shape (B, D, D).
        - rng: Random number generator.

        Returns:
        - Heightmap tensor of shape (B, D, D) with added noise.
        """
        noise = torch.normal(self.noise_mu, self.noise_std, size=z.shape, generator=rng, device=z.device)
        return z + noise
