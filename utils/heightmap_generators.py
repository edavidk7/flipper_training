import torch
from dataclasses import dataclass
from typing import ClassVar
from abc import ABC, abstractmethod


@dataclass
class BaseHeightmapGenerator(ABC):
    """
    Base class for heightmap generators.

    Attributes:
    - returns_suitability_mask: Whether the heightmap generator also returns a suitability mask. A suitability mask is a binary mask that indicates which parts of the heightmap are suitable for start/goal positions.
    """
    returns_suitability_mask: ClassVar[bool] = False
    add_random_noise: bool = False
    noise_std: float = 0.01  # Standard deviation of the Gaussian noise in meters
    noise_mu: float = 0.0  # Mean of the Gaussian noise in meters

    def __call__(self, x: torch.Tensor, y: torch.Tensor, max_coord: float, rng: torch.Generator | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a heightmap.

        Args:
        - x: Tensor of x coordinates. Shape is (D, D).
        - y: Tensor of y coordinates. Shape is (D, D).
        - max_coord: Maximum coordinate in meters.
        - rng: Random number generator.

        Returns:
        - Heightmap tensor of shape (D, D) or a tuple of heightmap tensor and suitability mask tensor.
        """
        r = self._generate_heightmap(x, y, max_coord, rng)
        if isinstance(r, tuple):
            z, mask = r
        elif isinstance(r, torch.Tensor):
            z = r
            mask = None
        else:
            raise ValueError(f"Invalid return type from generate_heightmap: {type(r)}")
        if self.add_random_noise:
            z = self._add_noise_to_heightmap(z, rng)
        if mask is not None:
            return z, mask
        return z

    @abstractmethod
    def _generate_heightmap(self, x: torch.Tensor, y: torch.Tensor, max_coord: float, rng: torch.Generator | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a heightmap.

        Args:
        - x: Tensor of x coordinates. Shape is (D, D).
        - y: Tensor of y coordinates. Shape is (D, D).
        - max_coord: Maximum coordinate in meters.
        - rng: Random number generator.

        Returns:
        - Heightmap tensor of shape (D, D) or a tuple of heightmap tensor and suitability mask tensor.
        """
        raise NotImplementedError

    def _add_noise_to_heightmap(self, z: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
        """
        Adds Gaussian noise to a heightmap.

        Args:
        - z: Heightmap tensor of shape (D, D).
        - rng: Random number generator.

        Returns:
        - Heightmap tensor of shape (D, D) with added noise.
        """
        noise = torch.normal(self.noise_mu, self.noise_std, size=z.shape, generator=rng, device=z.device)
        return z + noise


@ dataclass
class MultiGaussianHeightmapGenerator(BaseHeightmapGenerator):
    """
    Generates a heightmap using multiple gaussians.
    """
    returns_suitability_mask: ClassVar[bool] = False
    min_gaussians: int = 5
    max_gaussians: int = 10
    max_height_fraction: float = 0.1
    min_std_fraction: float = 0.2
    max_std_fraction: float = 0.5
    min_sigma_ratio: float = 0.5

    def _generate_heightmap(self, x: torch.Tensor,
                            y: torch.Tensor,
                            max_coord: float,
                            rng: torch.Generator | None = None) -> torch.Tensor:
        D, D = x.shape
        z = torch.zeros((D, D), device=x.device)
        # Generate random number of gaussians
        num_gaussians = int(torch.randint(self.min_gaussians, self.max_gaussians + 1, (1,), generator=rng).item())
        # Generate means from -max_coord to max_coord
        mus = torch.rand((num_gaussians, 2), device=x.device) * 2 * max_coord - max_coord
        # Generate standard deviations from min_std_fraction * max_coord to max_std_fraction * max_coord
        sigmas = torch.rand((num_gaussians, 1), device=x.device) * (self.max_std_fraction - self.min_std_fraction) * max_coord + self.min_std_fraction * max_coord
        ratios = torch.rand((num_gaussians,), device=x.device) * (1 - self.min_sigma_ratio) + self.min_sigma_ratio  # ratio of the standard deviations of the x and y components in range [min_sigma_ratio, 1]
        higher_indices = torch.randint(0, 2, (num_gaussians,), device=x.device)  # whether the x or y component has the  higher standard deviation
        sigmas = sigmas.repeat(1, 2)
        sigmas[torch.arange(num_gaussians), higher_indices] *= ratios
        heights = torch.rand((num_gaussians,), device=x.device) * self.max_height_fraction * max_coord  # from 0 to max_height_fraction * max_coord
        for i in range(num_gaussians):
            z += heights[i] * torch.exp(-((x - mus[i, 0])**2 / (2 * sigmas[i, 0]**2) + (y - mus[i, 1])**2 / (2 * sigmas[i, 1]**2)))
        return z
