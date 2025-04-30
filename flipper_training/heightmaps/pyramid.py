from dataclasses import dataclass
import math

import torch

from flipper_training.heightmaps import BaseHeightmapGenerator


@dataclass
class PyramidHeightmapGenerator(BaseHeightmapGenerator):
    """
    Generates a heightmap with a 4-sided pyramid shape centered at the origin,
    potentially rotated by a random angle. Spans the grid up to max_coord.
    Level width is determined implicitly by max_coord and number of levels.
    Can optionally create soft transitions between levels using an exponent.
    Returns a `level_indices` mask indicating the level number for each cell
    (highest index at peak, decreasing outwards, -1 outside pyramid).
    """

    min_levels: int = 5  # Number of levels from peak to edge
    max_levels: int | None = None
    min_level_height: float = 0.05  # Height difference between adjacent levels
    max_level_height: float | None = None
    exponent: float | None = None  # Exponent for soft transitions (None for hard steps)

    def __post_init__(self):
        if self.min_levels <= 0 or (self.max_levels is not None and self.max_levels < self.min_levels):
            raise ValueError("Invalid level count configuration.")
        if self.min_level_height <= 0 or (self.max_level_height is not None and self.max_level_height < self.min_level_height):
            raise ValueError("Invalid level height configuration.")
        if self.exponent is not None and self.exponent <= 0:
            raise ValueError("Exponent must be positive if specified.")

    def _generate_heightmap(self, x, y, max_coord, rng=None):
        B = x.shape[0]
        z = torch.zeros_like(x)
        level_indices = torch.full_like(x, -1, dtype=torch.long)  # Initialize level index mask
        suitable_mask = torch.ones_like(x, dtype=torch.bool)  # Placeholder

        for i in range(B):
            # Sample parameters
            angle = torch.rand((1,), generator=rng, device=x.device) * 2 * math.pi
            n_levels = (
                torch.randint(self.min_levels, self.max_levels + 1, (1,), generator=rng, device=x.device).item()
                if self.max_levels
                else self.min_levels
            )
            level_height = (
                (torch.rand((1,), generator=rng, device=x.device) * (self.max_level_height - self.min_level_height) + self.min_level_height).item()
                if self.max_level_height
                else self.min_level_height
            )

            # Calculate rotation and implicit level width
            cos_a = torch.cos(-angle).item()  # Use negative angle for coordinate rotation
            sin_a = torch.sin(-angle).item()
            # Level width based on max_coord (distance from center to edge in rotated frame)
            level_width = max_coord / n_levels

            # Rotate coordinates and calculate L-infinity distance from origin
            x_rot = x[i] * cos_a - y[i] * sin_a
            y_rot = x[i] * sin_a + y[i] * cos_a
            dist = torch.max(x_rot.abs(), y_rot.abs())

            # Calculate level index (raw: 0 near peak, increasing outwards)
            level_index_raw = torch.floor(dist / (level_width + 1e-9))
            # Flipped level index (highest index near peak)
            level_index = (n_levels - 1 - level_index_raw).clamp_(min=0)

            # Calculate max height at the peak
            max_height = (n_levels - 1) * level_height

            # Calculate height
            if self.exponent is None:  # Hard steps
                z_i = level_index * level_height
            else:  # Soft steps
                h_start = level_index * level_height
                dist_from_inner_edge = dist - level_index_raw * level_width
                normalized_dist = (dist_from_inner_edge / (level_width + 1e-9)).clamp_(0.0, 1.0)
                # Apply exponent for soft transition profile (drops by level_height across the level)
                height_decrease = level_height * (normalized_dist ** float(self.exponent))
                z_i = h_start - height_decrease

            # Clamp overall height
            z_i.clamp_(min=0.0, max=max_height)
            z[i] = z_i

            # Create level index mask (using flipped index)
            valid_level_mask = (level_index_raw >= 0) & (level_index_raw < n_levels)
            level_indices[i][valid_level_mask] = level_index[valid_level_mask].long()

        return z, {"suitable_mask": suitable_mask, "level_indices": level_indices}


@dataclass
class FixedPyramidHeightmapGenerator(BaseHeightmapGenerator):
    """
    Generates a heightmap with a 4-sided pyramid shape centered at the origin,
    rotated by a fixed angle. Spans the grid up to max_coord.
    Level width is determined implicitly by max_coord and number of levels.
    Can optionally create soft transitions between levels using an exponent.
    Returns a `level_indices` mask indicating the level number for each cell
    (highest index at peak, decreasing outwards, -1 outside pyramid).
    """

    rotation_angle: float = 0.0  # Angle defining the pyramid's orientation
    n_levels: int = 10  # Number of levels from peak to edge
    level_height: float = 0.1  # Height difference between adjacent levels
    exponent: float | None = None  # Exponent for soft transitions (None for hard steps)

    def __post_init__(self):
        if self.n_levels <= 0:
            raise ValueError("Number of levels must be positive.")
        if self.level_height <= 0:
            raise ValueError("Level height must be positive.")
        if self.exponent is not None and self.exponent <= 0:
            raise ValueError("Exponent must be positive if specified.")

    def _generate_heightmap(self, x, y, max_coord, rng=None):
        # Calculate rotation and implicit level width
        cos_a = math.cos(-self.rotation_angle)  # Use negative angle for coordinate rotation
        sin_a = math.sin(-self.rotation_angle)
        # Level width based on max_coord
        level_width = max_coord / self.n_levels

        # Rotate coordinates and calculate L-infinity distance from origin
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        dist = torch.max(x_rot.abs(), y_rot.abs())

        # Calculate level index (raw: 0 near peak, increasing outwards)
        level_index_raw = torch.floor(dist / (level_width + 1e-9))
        # Flipped level index (highest index near peak)
        level_index = (self.n_levels - 1 - level_index_raw).clamp_(min=0)

        # Calculate max height at the peak
        max_height = (self.n_levels - 1) * self.level_height

        # Calculate height
        if self.exponent is None:  # Hard steps
            z = level_index * self.level_height
        else:  # Soft steps
            h_start = level_index * self.level_height
            dist_from_inner_edge = dist - level_index_raw * level_width
            normalized_dist = (dist_from_inner_edge / (level_width + 1e-9)).clamp_(0.0, 1.0)
            # Apply exponent for soft transition profile (drops by level_height across the level)
            height_decrease = self.level_height * (normalized_dist ** float(self.exponent))
            z = h_start - height_decrease

        # Clamp overall height
        z.clamp_(min=0.0, max=max_height)

        # Create level index mask (using flipped index)
        level_indices = torch.full_like(x, -1, dtype=torch.long)
        valid_level_mask = (level_index_raw >= 0) & (level_index_raw < self.n_levels)
        level_indices[valid_level_mask] = level_index[valid_level_mask].long()

        suitable_mask = torch.ones_like(x, dtype=torch.bool)  # Placeholder

        return z, {"suitable_mask": suitable_mask, "level_indices": level_indices}


if __name__ == "__main__":
    # Ensure flipper_training is in path or installed
    try:
        from flipper_training.vis.static_vis import plot_heightmap_3d
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError:
        print("Error: Could not import flipper_training.vis.static_vis.")
        print("Ensure the flipper_training package is installed or accessible in the Python path.")
        exit()

    import matplotlib

    try:
        matplotlib.use("QtAgg")  # Or another suitable backend like 'TkAgg'
    except ImportError:
        print("Warning: Could not set Matplotlib backend 'QtAgg'. Using default.")
    import matplotlib.pyplot as plt

    # --- Configuration ---
    resolution = 0.05
    max_coord = 2.5  # Slightly larger grid
    batch_size = 1  # Use batch size 1 for simplicity
    device = "cpu"
    seed = 45
    rng = torch.manual_seed(seed)
    soft_exponent = 50.0  # Make exponent smaller for softer look

    # --- Random Pyramid Generation (Hard) ---
    print("Generating random 4-sided pyramid (Hard)...")
    random_gen_hard = PyramidHeightmapGenerator(min_levels=3, max_levels=5, min_level_height=0.18, max_level_height=0.3, exponent=None)
    x_random_hard, y_random_hard, z_random_hard, extras_hard_rand = random_gen_hard(
        grid_res=resolution, max_coord=max_coord, num_robots=batch_size, rng=rng
    )
    indices_hard_rand = extras_hard_rand["level_indices"]

    # --- Random Pyramid Generation (Soft) ---
    print("Generating random 4-sided pyramid (Soft)...")
    random_gen_soft = PyramidHeightmapGenerator(min_levels=3, max_levels=5, min_level_height=0.18, max_level_height=0.3, exponent=soft_exponent)
    x_random_soft, y_random_soft, z_random_soft, extras_soft_rand = random_gen_soft(
        grid_res=resolution, max_coord=max_coord, num_robots=batch_size, rng=rng
    )
    indices_soft_rand = extras_soft_rand["level_indices"]

    # --- Fixed Pyramid Generation (Hard) ---
    print("Generating fixed 4-sided pyramid (Hard)...")
    fixed_gen_hard = FixedPyramidHeightmapGenerator(
        rotation_angle=0,  # Rotate 30 deg
        n_levels=5,
        level_height=0.2,
        exponent=None,
    )
    x_fixed_hard, y_fixed_hard, z_fixed_hard, extras_hard_fixed = fixed_gen_hard(
        grid_res=resolution, max_coord=max_coord, num_robots=batch_size, rng=rng
    )
    indices_hard_fixed = extras_hard_fixed["level_indices"]

    # --- Fixed Pyramid Generation (Soft) ---
    print("Generating fixed 4-sided pyramid (Soft)...")
    fixed_gen_soft = FixedPyramidHeightmapGenerator(
        rotation_angle=0,  # Rotate 30 deg
        n_levels=5,
        level_height=0.2,
        exponent=soft_exponent,
    )
    x_fixed_soft, y_fixed_soft, z_fixed_soft, extras_soft_fixed = fixed_gen_soft(
        grid_res=resolution, max_coord=max_coord, num_robots=batch_size, rng=rng
    )
    indices_soft_fixed = extras_soft_fixed["level_indices"]

    # --- Plotly Visualization (Combined) ---
    print("Visualizing 4-sided pyramids...")
    fig = make_subplots(
        rows=2, cols=2, specs=[[{"type": "surface"}] * 2] * 2, subplot_titles=("Random Hard", "Random Soft", "Fixed Hard", "Fixed Soft")
    )

    fig.add_trace(go.Surface(x=x_random_hard[0].cpu(), y=y_random_hard[0].cpu(), z=z_random_hard[0].cpu(), showscale=False), row=1, col=1)
    fig.add_trace(go.Surface(x=x_random_soft[0].cpu(), y=y_random_soft[0].cpu(), z=z_random_soft[0].cpu(), showscale=False), row=1, col=2)
    fig.add_trace(go.Surface(x=x_fixed_hard[0].cpu(), y=y_fixed_hard[0].cpu(), z=z_fixed_hard[0].cpu(), showscale=False), row=2, col=1)
    fig.add_trace(go.Surface(x=x_fixed_soft[0].cpu(), y=y_fixed_soft[0].cpu(), z=z_fixed_soft[0].cpu(), showscale=False), row=2, col=2)

    fig.update_layout(title_text="4-Sided Pyramid Variations")
    # Update scene aspect ratio for all subplots
    max_z = max(z_random_hard.max(), z_random_soft.max(), z_fixed_hard.max(), z_fixed_soft.max()).item()
    aspect = dict(x=1, y=1, z=max(0.1, max_z / (2 * max_coord)))  # Ensure z aspect is reasonable
    fig.update_scenes(aspectmode="manual", aspectratio=aspect)
    # Minimize figure margins to bring subplots closer
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))  # Adjust left, right, top, bottom margins
    fig.show()

    # --- Matplotlib Index Maps (Combined) ---
    print("Displaying index maps...")
    fig_idx, axs = plt.subplots(2, 2, figsize=(10, 9))
    fig_idx.suptitle("4-Sided Pyramid - Level Indices (Peak=Highest)")

    def plot_index_map(ax, x, y, indices, title):
        max_idx = indices.max().item()
        min_idx = indices.min().item()
        levels = max_idx - min_idx + 2 if max_idx >= min_idx else 1
        cont = ax.contourf(x.cpu(), y.cpu(), indices.cpu(), levels=levels)
        fig_idx.colorbar(cont, ax=ax, label="Level Index")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")

    plot_index_map(axs[0, 0], x_random_hard[0], y_random_hard[0], indices_hard_rand[0], "Random Hard")
    plot_index_map(axs[0, 1], x_random_soft[0], y_random_soft[0], indices_soft_rand[0], f"Random Soft (Exp={soft_exponent})")
    plot_index_map(axs[1, 0], x_fixed_hard[0], y_fixed_hard[0], indices_hard_fixed[0], "Fixed Hard")
    plot_index_map(axs[1, 1], x_fixed_soft[0], y_fixed_soft[0], indices_soft_fixed[0], f"Fixed Soft (Exp={soft_exponent})")

    fig_idx.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle
    plt.show()

    print("Done.")
