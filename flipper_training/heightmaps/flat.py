from dataclasses import dataclass

import torch
from . import BaseHeightmapGenerator


@dataclass
class FlatHeightmapGenerator(BaseHeightmapGenerator):
    """
    Generates a flat heightmap.
    """

    def _generate_heightmap(self, x, y, max_coord, rng=None):
        return torch.zeros_like(x), {}
