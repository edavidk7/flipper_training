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
class BarrierHeightmapGenerator(BaseHeightmapGenerator):
    """
    Generates a flat barrier segment of given length and thickness, offset randomly.
    Produces z=height inside barrier, 0 elsewhere, and a barrier_mask: 1 on one side, 2 on the other.
    """

    min_length: float = 0.8
    max_length: float = 1.2
    min_thickness: float = 0.15
    max_thickness: float = 0.25
    max_dist_from_origin: float = 0.5
    deadzone: float = 0.1
    min_height: float = 0.3
    max_height: float = 0.5
    exp: float | int | None = None

    def _generate_heightmap(self, x, y, max_coord, rng=None):
        B, D, _ = x.shape
        z = torch.zeros_like(x)
        barrier_mask = torch.zeros_like(x, dtype=torch.long)
        for i in range(B):
            # random obstacle parameters
            length = torch.rand((), generator=rng, device=x.device) * (self.max_length - self.min_length) + self.min_length
            thickness = torch.rand((), generator=rng, device=x.device) * (self.max_thickness - self.min_thickness) + self.min_thickness
            height = torch.rand((), generator=rng, device=x.device) * (self.max_height - self.min_height) + self.min_height

            # random orientation & offset
            angle = torch.rand((), generator=rng, device=x.device) * 2 * math.pi
            offset = (torch.rand((), generator=rng, device=x.device) * 2 - 1) * self.max_dist_from_origin
            normal = torch.stack([torch.cos(angle), torch.sin(angle)])
            tangent = torch.stack([-normal[1], normal[0]])
            # signed distance and parallel coord
            dist = x[i] * normal[0] + y[i] * normal[1] - offset
            par = x[i] * tangent[0] + y[i] * tangent[1]
            hl, ht = length / 2, thickness / 2

            # define barrier region
            barrier = (par.abs() <= hl) & (dist.abs() <= ht)

            if self.exp is None:
                z[i][barrier] = height
            else:
                mask_len = par.abs() <= hl
                prof = (1 - (dist.abs() ** self.exp) / (ht**self.exp)) * height
                prof.clamp_min_(0)
                z[i][mask_len] = prof[mask_len]

            # assign mask value 3 for barrier cells
            barrier_mask[i][barrier] = BarrierZones.BARRIER

            # deadzone
            dz = self.deadzone
            # mask sides outside deadzone along the longer obstacle axis
            if length <= thickness:
                side1 = par < -(hl + dz)
                side2 = par > (hl + dz)
            else:
                side1 = dist < -(ht + dz)
                side2 = dist > (ht + dz)
            barrier_mask[i][side1] = BarrierZones.LEFT
            barrier_mask[i][side2] = BarrierZones.RIGHT
        return z, {"barrier_mask": barrier_mask}


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
        barrier_mask = torch.zeros_like(x, dtype=torch.long)
        barrier_mask[barrier] = BarrierZones.BARRIER

        dz = self.deadzone
        # mask sides outside deadzone along the longer obstacle axis
        if self.length <= self.thickness:
            side1 = par < -(hl + dz)
            side2 = par > (hl + dz)
        else:
            side1 = dist < -(ht + dz)
            side2 = dist > (ht + dz)
        barrier_mask[side1] = BarrierZones.LEFT
        barrier_mask[side2] = BarrierZones.RIGHT
        return z, {"barrier_mask": barrier_mask}


if __name__ == "__main__":
    import torch
    import math
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import matplotlib

    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt

    # config
    resolution = 0.05
    max_coord = 3.2
    batch = 1
    rng = torch.manual_seed(42)

    # random barrier
    rand_gen = BarrierHeightmapGenerator(
        min_length=0.5,
        max_length=1.5,
        min_thickness=0.3,
        max_thickness=0.8,
        max_dist_from_origin=0.5,
        deadzone=0.5,
        min_height=0.4,
        max_height=0.5,
        exp=4,
    )
    x_r, y_r, z_r, e_r = rand_gen(grid_res=resolution, max_coord=max_coord, num_robots=batch, rng=rng)
    mask_r = e_r["barrier_mask"]

    # fixed barrier
    fix_gen = FixedBarrierHeightmapGenerator(angle=math.pi / 4, offset=0.0, length=1.5, thickness=0.3, deadzone=0.1, height=0.5)
    x_f, y_f, z_f, e_f = fix_gen(grid_res=resolution, max_coord=max_coord, num_robots=batch, rng=rng)
    mask_f = e_f["barrier_mask"]

    # Plotly surfaces
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "surface"}]], subplot_titles=("Random Barrier", "Fixed Barrier"))
    fig.add_trace(go.Surface(x=x_r[0].cpu(), y=y_r[0].cpu(), z=z_r[0].cpu(), showscale=False), row=1, col=1)
    fig.add_trace(go.Surface(x=x_f[0].cpu(), y=y_f[0].cpu(), z=z_f[0].cpu(), showscale=False), row=1, col=2)
    fig.update_layout(title_text="Barrier Heightmaps", margin=dict(l=10, r=10, t=40, b=10))

    # --- new: enforce equal axis scaling ---
    max_z = max(z_r.max(), z_f.max()).item()
    aspect = dict(x=1, y=1, z=max(0.1, max_z / (2 * max_coord)))
    fig.update_scenes(aspectmode="manual", aspectratio=aspect)

    fig.show()

    # Matplotlib masks
    fig2, axs = plt.subplots(1, 2, figsize=(12, 4))
    levels = [-0.5, 0.5, 1.5, 2.5]
    cmap = "tab10"

    cont0 = axs[0].contourf(x_r[0].cpu(), y_r[0].cpu(), mask_r[0].cpu(), levels=levels, cmap=cmap)
    axs[0].set_title("Random Suitability Mask")
    axs[0].axis("equal")  # <--- equal scaling
    fig2.colorbar(cont0, ax=axs[0], ticks=[0, 1, 2], label="Suitability Mask")

    cont1 = axs[1].contourf(x_f[0].cpu(), y_f[0].cpu(), mask_f[0].cpu(), levels=levels, cmap=cmap)
    axs[1].set_title("Fixed Suitability Mask")
    axs[1].axis("equal")  # <--- equal scaling
    fig2.colorbar(cont1, ax=axs[1], ticks=[0, 1, 2], label="Suitability Mask")

    plt.tight_layout()
    plt.show()
