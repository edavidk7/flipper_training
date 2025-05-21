from dataclasses import dataclass

import torch

from . import BaseHeightmapGenerator


@dataclass
class FromFileHeightmap(BaseHeightmapGenerator):
    """
    Generates a heightmap using multiple gaussians.
    """

    file_path: str
    top_height_percentile_cutoff: float | None = None  # Heights above this percentile are exluded from the suitability mask

    def _generate_heightmap(self, x, y, max_coord, rng):
        z = torch.zeros_like(x)
        suitable_mask = torch.ones_like(x, dtype=torch.bool, device=x.device)
        z = torch.load(self.file_path, map_location=x.device)
        if z.ndim == 2:
            z = z.expand_as(x)
        elif z.ndim == 3:
            assert z.shape[0] == x.shape[0], "Batch dimension mismatch"
            assert z.shape[1:] == x.shape[1:], "Heightmap shape mismatch"
        if self.top_height_percentile_cutoff is not None:
            # Set the suitable mask to False where the height is above the percentile
            # Get the top height percentile cutoff
            percentile_height = torch.quantile(z, self.top_height_percentile_cutoff)
            violation_mask = z > percentile_height
            suitable_mask = ~violation_mask
        return z, {"suitable_mask": suitable_mask}
