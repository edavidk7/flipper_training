import math
from dataclasses import dataclass
from enum import IntEnum

import torch
from flipper_training.heightmaps import BaseHeightmapGenerator

class BarrierZones(IntEnum):
    """
    Enum for the different zones of the barrier.
    """

    DEAD = 0
    LEFT = 1
    RIGHT = 2
    BARRIER = 3


@dataclass
class MixedHeightmapGenerator(BaseHeightmapGenerator):
    """
    Instantiates a list of heightmap generators and randomly selects one of them for each robot.
    The sampling  weights can be optionally specified.
    """

    classes: list[type[BaseHeightmapGenerator]]
    opts: list[dict]
    weights: list[float] | None = None
    
    def __post_init__(self):
        self.generators = [g(**o) for g, o in zip(self.classes, self.opts)]
        if self.weights is None:
            self.weights = [1.0 / len(self.generators)] * len(self.generators)
        self.weights_tensor = torch.tensor(self.weights)

    def _generate_heightmap(self, x, y, max_coord, rng=None):
        B, D, _ = x.shape
        z = torch.zeros_like(x)
        extras = {}
        typelist = [] # store the used generator types
        
            
        return z, {"suitable_mask": suitable_mask}


@dataclass
class FixedBarrierHeightmapGenerator(BaseHeightmapGenerator):
    """
    Fixed-version of the barrier: orientation and offset are constants.
    """

    angle: float = 0.0
    offset: float = 0.0
    length: float = 1.0
    thickness: float = 0.2
    deadzone: float = 0.1
    height: float = 0.4
    exp: float | int | None = None

    def _generate_heightmap(self, x, y, max_coord, rng=None):
        normal = torch.tensor([math.cos(self.angle), math.sin(self.angle)], device=x.device)
        tangent = torch.tensor([-normal[1], normal[0]], device=x.device)
        dist = x * normal[0] + y * normal[1] - self.offset
        par = x * tangent[0] + y * tangent[1]
        hl, ht = self.length / 2, self.thickness / 2

        # define barrier region
        barrier = (par.abs() <= hl) & (dist.abs() <= ht)

        if self.exp is None:
            z = torch.zeros_like(x)
            z[barrier] = self.height
        else:
            mask_len = par.abs() <= hl
            prof = (1 - (dist.abs() ** self.exp) / (ht**self.exp)) * self.height
            prof.clamp_min_(0)
            z = torch.zeros_like(x)
            z[mask_len] = prof[mask_len]

        # assign mask value 3 for barrier cells
        suitable_mask = torch.zeros_like(x, dtype=torch.long)
        suitable_mask[barrier] = BarrierZones.BARRIER

        dz = self.deadzone
        # mask sides outside deadzone along the longer obstacle axis
        if self.length <= self.thickness:
            side1 = par < -(hl + dz)
            side2 = par > (hl + dz)
        else:
            side1 = dist < -(ht + dz)
            side2 = dist > (ht + dz)
        suitable_mask[side1] = BarrierZones.LEFT
        suitable_mask[side2] = BarrierZones.RIGHT
        return z, {"suitable_mask": suitable_mask}


if __name__ == "__main__":
    import torch
    import math
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import matplotlib

    
