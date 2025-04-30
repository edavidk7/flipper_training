from dataclasses import dataclass
import math

import torch
from flipper_training.heightmaps import BaseHeightmapGenerator


def make_stairs(
    angle: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    max_coord: float,
    n_steps: torch.Tensor,
    step_height: float,
    exponent: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a heightmap with stairs facing a given angle.
    """
    cos_a = torch.cos(angle).item()
    sin_a = torch.sin(angle).item()
    normal = torch.tensor([cos_a, sin_a], device=x.device)
    abs_cos_a = abs(cos_a)
    abs_sin_a = abs(sin_a)
    offset = -max_coord * (abs_cos_a + abs_sin_a)
    total_extent = 2 * max_coord * (abs_cos_a + abs_sin_a)
    step_width = total_extent / n_steps
    dist = x * normal[0] + y * normal[1] - offset
    step_index_raw = torch.floor(dist / (step_width + 1e-9))
    step_index = (n_steps - 1 - step_index_raw).clamp_(min=0)
    if exponent is None:
        z = step_index_raw * step_height
    else:
        base_height = step_index_raw * step_height
        dist_from_step_start = dist - step_index_raw * step_width
        normalized_dist = (dist_from_step_start / (step_width + 1e-9)).clamp_(0.0, 1.0)
        height_increase = step_height * (1 - (1 - normalized_dist) ** exponent)
        z = base_height + height_increase
    max_h = (n_steps - 1) * step_height
    z.clamp_(min=0.0, max=max_h)
    return z, step_index_raw, step_index


@dataclass
class StairsHeightmapGenerator(BaseHeightmapGenerator):
    """
    Generates a heightmap with stairs facing a random direction, spanning the grid.
    Step width is determined implicitly by the grid size and number of steps.
    Can optionally create soft transitions between steps using an exponent.
    Returns a `step_indices` mask indicating the step number for each cell
    (highest index at top step, decreasing downwards, -1 outside stairs).
    """

    min_steps: int = 5
    max_steps: int | None = None
    min_step_height: float = 0.05
    max_step_height: float | None = None
    exponent: float | None = None

    def __post_init__(self):
        if self.min_steps <= 0 or (self.max_steps is not None and self.max_steps < self.min_steps):
            raise ValueError("Invalid step count configuration.")
        if self.min_step_height <= 0 or (self.max_step_height is not None and self.max_step_height < self.min_step_height):
            raise ValueError("Invalid step height configuration.")
        if self.exponent is not None and self.exponent <= 0:
            raise ValueError("Exponent must be positive if specified.")

    def _generate_heightmap(self, x, y, max_coord, rng=None):
        B = x.shape[0]
        z = torch.zeros_like(x)
        step_indices = torch.full_like(x, -1, dtype=torch.long)
        suitable_mask = torch.ones_like(x, dtype=torch.bool)

        for i in range(B):
            angle = torch.rand((1,), generator=rng, device=x.device) * 2 * math.pi
            n_steps = (
                torch.randint(self.min_steps, self.max_steps + 1, (1,), generator=rng, device=x.device).item() if self.max_steps else self.min_steps
            )
            step_height = (
                (torch.rand((1,), generator=rng, device=x.device) * (self.max_step_height - self.min_step_height) + self.min_step_height).item()
                if self.max_step_height
                else self.min_step_height
            )
            z_i, step_index_raw, step_index = make_stairs(angle, x[i], y[i], max_coord, n_steps, step_height, self.exponent)
            z[i] = z_i
            valid_step_mask = (step_index_raw >= 0) & (step_index_raw < n_steps)
            step_indices[i][valid_step_mask] = step_index[valid_step_mask].long()

        return z, {"suitable_mask": suitable_mask, "step_indices": step_indices}


@dataclass
class FixedStairsHeightmapGenerator(BaseHeightmapGenerator):
    """
    Generates a heightmap with stairs facing a fixed direction, spanning the grid.
    Step width is determined implicitly by the grid size and number of steps.
    Can optionally create soft transitions between steps using an exponent.
    Returns a `step_indices` mask indicating the step number for each cell
    (highest index at top step, decreasing downwards, -1 outside stairs).
    """

    normal_angle: float = 0.0
    n_steps: int = 10
    step_height: float = 0.1
    exponent: float | None = None

    def __post_init__(self):
        if self.n_steps <= 0:
            raise ValueError("Number of steps must be positive.")
        if self.step_height <= 0:
            raise ValueError("Step height must be positive.")
        if self.exponent is not None and self.exponent <= 0:
            raise ValueError("Exponent must be positive if specified.")

    def _generate_heightmap(self, x, y, max_coord, rng=None):
        step_indices = torch.full_like(x, -1, dtype=torch.long)
        z, step_index_raw, step_index = make_stairs(
            torch.tensor(self.normal_angle), x, y, max_coord, torch.tensor(self.n_steps), self.step_height, self.exponent
        )
        valid_step_mask = (step_index_raw >= 0) & (step_index_raw < self.n_steps)
        step_indices[valid_step_mask] = step_index[valid_step_mask].long()
        suitable_mask = torch.ones_like(x, dtype=torch.bool)
        return z, {"suitable_mask": suitable_mask, "step_indices": step_indices}


@dataclass
class BidirectionalStairsHeightmapGenerator(BaseHeightmapGenerator):
    """
    Generates a heightmap with stairs facing a random direction, spanning the grid.
    Step width is determined implicitly by the grid size and number of steps.
    Can optionally create soft transitions between steps using an exponent.
    Returns a `step_indices` mask indicating the step number for each cell
    (highest index at top step, decreasing downwards, -1 outside stairs).
    """

    min_steps: int = 5
    max_steps: int | None = None
    min_step_height: float = 0.05
    max_step_height: float | None = None
    exponent: float | None = None

    def __post_init__(self):
        if self.min_steps <= 0 or (self.max_steps is not None and self.max_steps < self.min_steps):
            raise ValueError("Invalid step count configuration.")
        if self.min_step_height <= 0 or (self.max_step_height is not None and self.max_step_height < self.min_step_height):
            raise ValueError("Invalid step height configuration.")
        if self.exponent is not None and self.exponent <= 0:
            raise ValueError("Exponent must be positive if specified.")

    def _generate_heightmap(self, x, y, max_coord, rng=None):
        B = x.shape[0]
        z = torch.zeros_like(x)
        step_indices = torch.full_like(x, -1, dtype=torch.long)
        suitable_mask = torch.ones_like(x, dtype=torch.bool)
        for i in range(B):
            angle = torch.rand((1,), generator=rng, device=x.device) * 2 * math.pi
            n_steps = (
                torch.randint(self.min_steps, self.max_steps + 1, (1,), generator=rng, device=x.device).item() if self.max_steps else self.min_steps
            )
            step_height = (
                (torch.rand((1,), generator=rng, device=x.device) * (self.max_step_height - self.min_step_height) + self.min_step_height).item()
                if self.max_step_height
                else self.min_step_height
            )
            # Generate stairs in one direction
            z1, step_index_raw1, step_index1 = make_stairs(angle, x[i], y[i], max_coord, n_steps, step_height, self.exponent)
            z2, step_index_raw2, step_index2 = make_stairs(angle + math.pi, x[i], y[i], max_coord, n_steps, step_height, self.exponent)
            # Create the line coefficients and determine left/right sides
            left_mask = (x[i] * math.sin(angle) - y[i] * math.cos(angle)) < 0
            right_mask = ~left_mask
            left_highest_point = z1[left_mask].max()
            z[i][left_mask] = z1[left_mask]
            z[i][right_mask] = z2[right_mask] + left_highest_point - z2[right_mask].min()  # this creates the landing
            # Set the indices for the left side
            valid_step_mask1 = (step_index_raw1 >= 0) & (step_index_raw1 < n_steps)
            step_indices[i][valid_step_mask1 & left_mask] = step_index1[valid_step_mask1 & left_mask].long()
            # Right side indices are increased by n_steps
            valid_step_mask2 = (step_index_raw2 >= 0) & (step_index_raw2 < n_steps)
            step_indices[i][right_mask & valid_step_mask2] = step_index2[right_mask & valid_step_mask2].long() + n_steps - 1

        return z, {"suitable_mask": suitable_mask, "step_indices": step_indices}


@dataclass
class FixedBidirectionalStairsHeightmapGenerator(BaseHeightmapGenerator):
    """
    Generates a heightmap with stairs facing a fixed direction, spanning the grid.
    Step width is determined implicitly by the grid size and number of steps.
    Can optionally create soft transitions between steps using an exponent.
    Returns a `step_indices` mask indicating the step number for each cell
    (highest index at top step, decreasing downwards, -1 outside stairs).
    """

    normal_angle: float = 0.0
    n_steps: int = 10
    step_height: float = 0.1
    exponent: float | None = None

    def __post_init__(self):
        if self.n_steps <= 0:
            raise ValueError("Number of steps must be positive.")
        if self.step_height <= 0:
            raise ValueError("Step height must be positive.")
        if self.exponent is not None and self.exponent <= 0:
            raise ValueError("Exponent must be positive if specified.")

    def _generate_heightmap(self, x, y, max_coord, rng=None):
        step_indices = torch.full_like(x, -1, dtype=torch.long)
        z1, step_index_raw1, step_index1 = make_stairs(
            torch.tensor(self.normal_angle), x, y, max_coord, torch.tensor(self.n_steps), self.step_height, self.exponent
        )
        z2, step_index_raw2, step_index2 = make_stairs(
            torch.tensor(self.normal_angle + math.pi), x, y, max_coord, torch.tensor(self.n_steps), self.step_height, self.exponent
        )
        # Create the line coefficients and determine left/right sides
        left_mask = (x * math.sin(self.normal_angle) - y * math.cos(self.normal_angle)) < 0
        right_mask = ~left_mask
        left_highest_point = z1[left_mask].max()
        z = torch.zeros_like(x)
        z[left_mask] = z1[left_mask]
        z[right_mask] = z2[right_mask] + left_highest_point - z2[right_mask].min()  # this creates the landing
        # Set the indices for the left side
        valid_step_mask1 = (step_index_raw1 >= 0) & (step_index_raw1 < self.n_steps)
        step_indices[left_mask & valid_step_mask1] = step_index1[valid_step_mask1 & left_mask].long()
        # Right side indices are increased by n_steps
        valid_step_mask2 = (step_index_raw2 >= 0) & (step_index_raw2 < self.n_steps)
        step_indices[right_mask & valid_step_mask2] = step_index2[valid_step_mask2 & right_mask].long() + self.n_steps - 1
        # Create the suitable mask
        suitable_mask = torch.ones_like(x, dtype=torch.bool)
        return z, {"suitable_mask": suitable_mask, "step_indices": step_indices}


if __name__ == "__main__":
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import matplotlib

    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt

    # Configuration
    resolution = 0.05
    max_coord = 3.2
    batch_size = 1
    device = "cpu"
    seed = 47
    rng = torch.manual_seed(seed)
    soft_exponent = 100

    # Standard Stairs Generation
    print("Generating standard stairs...")
    rand_hard = StairsHeightmapGenerator(min_steps=3, max_steps=5, min_step_height=0.1, max_step_height=0.2, exponent=None)
    x_rh, y_rh, z_rh, e_rh = rand_hard(grid_res=resolution, max_coord=max_coord, num_robots=batch_size, rng=rng)
    idx_rh = e_rh["step_indices"]
    rand_soft = StairsHeightmapGenerator(min_steps=3, max_steps=5, min_step_height=0.1, max_step_height=0.2, exponent=soft_exponent)
    x_rs, y_rs, z_rs, e_rs = rand_soft(grid_res=resolution, max_coord=max_coord, num_robots=batch_size, rng=rng)
    idx_rs = e_rs["step_indices"]
    fixed_hard = FixedStairsHeightmapGenerator(normal_angle=0, n_steps=5, step_height=0.25, exponent=None)
    x_fh, y_fh, z_fh, e_fh = fixed_hard(grid_res=resolution, max_coord=max_coord, num_robots=batch_size, rng=rng)
    idx_fh = e_fh["step_indices"]
    fixed_soft = FixedStairsHeightmapGenerator(normal_angle=0, n_steps=5, step_height=0.25, exponent=soft_exponent)
    x_fs, y_fs, z_fs, e_fs = fixed_soft(grid_res=resolution, max_coord=max_coord, num_robots=batch_size, rng=rng)
    idx_fs = e_fs["step_indices"]

    # Bidirectional Stairs Generation (Opposite Directions)
    print("Generating bidirectional stairs with opposite directions...")
    step_height = 0.2
    bi_rand_hard = BidirectionalStairsHeightmapGenerator(
        min_steps=3,
        max_steps=5,
        min_step_height=step_height,
        max_step_height=step_height,
        exponent=None,
    )
    x_brh, y_brh, z_brh, e_brh = bi_rand_hard(grid_res=resolution, max_coord=max_coord, num_robots=batch_size, rng=rng)
    idx_brh = e_brh["step_indices"]
    bi_rand_soft = BidirectionalStairsHeightmapGenerator(
        min_steps=3,
        max_steps=3,
        min_step_height=step_height,
        max_step_height=step_height,
        exponent=soft_exponent,
    )
    x_brs, y_brs, z_brs, e_brs = bi_rand_soft(grid_res=resolution, max_coord=max_coord, num_robots=batch_size, rng=rng)
    idx_brs = e_brs["step_indices"]

    # Plotly Visualization (Standard Stairs)
    print("Visualizing standard stairs...")
    fig_std = make_subplots(
        rows=2, cols=2, specs=[[{"type": "surface"}] * 2] * 2, subplot_titles=("Random Hard", "Random Soft", "Fixed Hard", "Fixed Soft")
    )
    fig_std.add_trace(go.Surface(x=x_rh[0].cpu(), y=y_rh[0].cpu(), z=z_rh[0].cpu(), showscale=False), row=1, col=1)
    fig_std.add_trace(go.Surface(x=x_rs[0].cpu(), y=y_rs[0].cpu(), z=z_rs[0].cpu(), showscale=False), row=1, col=2)
    fig_std.add_trace(go.Surface(x=x_fh[0].cpu(), y=y_fh[0].cpu(), z=z_fh[0].cpu(), showscale=False), row=2, col=1)
    fig_std.add_trace(go.Surface(x=x_fs[0].cpu(), y=y_fs[0].cpu(), z=z_fs[0].cpu(), showscale=False), row=2, col=2)
    fig_std.update_layout(title_text="Standard Stairs Variations")
    max_z_std = max(z_rh.max(), z_rs.max(), z_fh.max(), z_fs.max()).item()
    aspect_std = dict(x=1, y=1, z=max(0.1, max_z_std / (2 * max_coord)))
    fig_std.update_scenes(aspectmode="manual", aspectratio=aspect_std)
    fig_std.show()

    # Plotly Visualization (Bidirectional Stairs)
    print("Visualizing bidirectional stairs with opposite directions...")
    fig_bi = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}] * 2], subplot_titles=("Hard Opposite Directions", "Soft Opposite Directions"))
    fig_bi.add_trace(go.Surface(x=x_brh[0].cpu(), y=y_brh[0].cpu(), z=z_brh[0].cpu(), showscale=False), row=1, col=1)
    fig_bi.add_trace(go.Surface(x=x_brs[0].cpu(), y=y_brs[0].cpu(), z=z_brs[0].cpu(), showscale=False), row=1, col=2)
    fig_bi.update_layout(title_text="Bidirectional Stairs (Opposite Directions)")
    max_z_bi = max(z_brh.max(), z_brs.max()).item()
    aspect_bi = dict(x=1, y=1, z=max(0.1, max_z_bi / (2 * max_coord)))
    fig_bi.update_scenes(aspectmode="manual", aspectratio=aspect_bi)
    fig_bi.show()

    # Matplotlib Index Maps (Standard Stairs)
    print("Displaying standard index maps...")
    fig_idx_std, axs_std = plt.subplots(2, 2, figsize=(10, 9))
    fig_idx_std.suptitle("Standard Stairs - Step Indices (Top=Highest)")

    def plot_index_map(fig, ax, x, y, indices, title):
        max_idx = indices.max().item()
        min_idx = indices.min().item()
        levels = max_idx - min_idx + 2 if max_idx >= min_idx else 1
        cont = ax.contourf(x.cpu(), y.cpu(), indices.cpu(), levels=levels)
        fig.colorbar(cont, ax=ax, label="Step Index")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")

    plot_index_map(fig_idx_std, axs_std[0, 0], x_rh[0], y_rh[0], idx_rh[0], "Random Hard")
    plot_index_map(fig_idx_std, axs_std[0, 1], x_rs[0], y_rs[0], idx_rs[0], f"Random Soft (Exp={soft_exponent})")
    plot_index_map(fig_idx_std, axs_std[1, 0], x_fh[0], y_fh[0], idx_fh[0], "Fixed Hard")
    plot_index_map(fig_idx_std, axs_std[1, 1], x_fs[0], y_fs[0], idx_fs[0], f"Fixed Soft (Exp={soft_exponent})")
    fig_idx_std.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Matplotlib Index Maps (Bidirectional Stairs)
    print("Displaying bidirectional index maps...")
    fig_idx_bi, axs_bi = plt.subplots(1, 2, figsize=(10, 5))
    fig_idx_bi.suptitle("Bidirectional Stairs (Opposite Directions) - Step Indices (Flight1: 0..N-1, Middle: -2, Flight2: N..M)")

    plot_index_map(fig_idx_bi, axs_bi[0], x_brh[0], y_brh[0], idx_brh[0], "Hard Opposite Directions")
    plot_index_map(fig_idx_bi, axs_bi[1], x_brs[0], y_brs[0], idx_brs[0], f"Soft Opposite Directions (Exp={soft_exponent})")
    fig_idx_bi.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()

    print("Done.")
