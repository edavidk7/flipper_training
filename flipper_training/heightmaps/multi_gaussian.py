from dataclasses import dataclass

import torch

from . import BaseHeightmapGenerator


@dataclass
class MultiGaussianHeightmapGenerator(BaseHeightmapGenerator):
    """
    Generates a heightmap using multiple gaussians.
    """

    min_gaussians: int = 30
    max_gaussians: int = 50
    min_height_fraction: float = 0.05
    max_height_fraction: float = 0.1
    min_std_fraction: float = 0.05
    max_std_fraction: float = 0.3
    min_sigma_ratio: float = 0.3
    top_height_percentile_cutoff: float | None = None  # Heights above this percentile are exluded from the suitability mask

    def _generate_heightmap(self, x, y, max_coord, rng):
        B = x.shape[0]
        z = torch.zeros_like(x)
        suitable_mask = torch.ones_like(x, dtype=torch.bool, device=x.device)
        for i in range(B):
            # Generate random number of gaussians
            num_gaussians = int(torch.randint(self.min_gaussians, self.max_gaussians + 1, (1,), generator=rng).item())
            # Generate means from -max_coord to max_coord
            mus = torch.rand((2, num_gaussians), device=x.device, generator=rng) * 2 * max_coord - max_coord
            # Generate standard deviations from min_std_fraction * max_coord to max_std_fraction * max_coord
            sigmas = (
                torch.rand((num_gaussians, 1), generator=rng).to(x.device) * (self.max_std_fraction - self.min_std_fraction) * max_coord
                + self.min_std_fraction * max_coord
            )
            ratios = (
                torch.rand((num_gaussians,), generator=rng).to(x.device) * (1 - self.min_sigma_ratio) + self.min_sigma_ratio
            )  # ratio of the standard deviations of the x and y components in range [min_sigma_ratio, 1]
            higher_indices = torch.randint(0, 2, (num_gaussians,), generator=rng).to(
                x.device
            )  # whether the x or y component has the  higher standard deviation
            sigmas = sigmas.repeat(1, 2)
            sigmas[torch.arange(num_gaussians), higher_indices] *= ratios
            heights = (
                torch.rand((num_gaussians,), generator=rng).to(x.device) * (self.max_height_fraction - self.min_height_fraction)
                + self.min_height_fraction
            )
            z_i = torch.sum(
                heights.view(-1, 1, 1)
                * torch.exp(
                    -(
                        (x[None, i] - mus[0].view(-1, 1, 1)) ** 2 / (2 * sigmas[..., 0].view(-1, 1, 1) ** 2)
                        + (y[None, i] - mus[1].view(-1, 1, 1)) ** 2 / (2 * sigmas[..., 1].view(-1, 1, 1) ** 2)
                    )
                ),
                dim=0,
            )
            if self.top_height_percentile_cutoff is not None:
                # Set the suitable mask to False where the height is above the percentile
                # Get the top height percentile cutoff
                percentile_height = torch.quantile(z_i, self.top_height_percentile_cutoff)
                violation_mask = z_i > percentile_height
                suitable_mask[i] = ~violation_mask
            z[i] = z_i
        return z, {"suitable_mask": suitable_mask}
