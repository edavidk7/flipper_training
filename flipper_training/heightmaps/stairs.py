from dataclasses import dataclass
import math

import torch
from flipper_training.heightmaps import BaseHeightmapGenerator


@dataclass
class StairsHeightmapGenerator(BaseHeightmapGenerator):
    """
    Generates a heightmap with stairs facing a random direction, spanning the grid.
    Step width is determined implicitly by the grid size and number of steps.
    Can optionally create soft transitions between steps using an exponent.
    Returns a `step_indices` mask indicating the step number for each cell (-1 if not on a step).
    """

    min_steps: int = 5
    max_steps: int | None = None
    min_step_height: float = 0.05
    max_step_height: float | None = None
    exponent: float | None = None  # Exponent for soft transitions (None for hard steps)
    # Removed step width parameters
    # Removed max_dist_from_origin

    def __post_init__(self):
        if self.min_steps <= 0 or (self.max_steps is not None and self.max_steps < self.min_steps):
            raise ValueError("Invalid step count configuration.")
        if self.min_step_height <= 0 or (self.max_step_height is not None and self.max_step_height < self.min_step_height):
            raise ValueError("Invalid step height configuration.")
        if self.exponent is not None and self.exponent <= 0:
            raise ValueError("Exponent must be positive if specified.")
        # Removed step width checks

    def _generate_heightmap(self, x, y, max_coord, rng=None):
        B = x.shape[0]
        z = torch.zeros_like(x)
        step_indices = torch.full_like(x, -1, dtype=torch.long)  # Initialize step index mask
        suitable_mask = torch.ones_like(x, dtype=torch.bool)  # Placeholder, adjust if needed

        for i in range(B):
            # Sample parameters
            angle = torch.rand((1,), generator=rng, device=x.device) * 2 * math.pi
            n_steps = (
                torch.randint(self.min_steps, self.max_steps + 1, (1,), generator=rng, device=x.device).item() if self.max_steps else self.min_steps
            )
            step_height = (
                (torch.rand((1,), generator=rng, device=x.device) * (self.max_step_height - self.min_step_height) + self.min_step_height).item()
                if self.max_step_height
                else self.min_step_height
            )

            # Calculate normal, offset, and implicit step width
            cos_a = torch.cos(angle).item()
            sin_a = torch.sin(angle).item()
            normal = torch.tensor([cos_a, sin_a], device=x.device)
            abs_cos_a = abs(cos_a)
            abs_sin_a = abs(sin_a)
            offset = -max_coord * (abs_cos_a + abs_sin_a)
            total_extent = 2 * max_coord * (abs_cos_a + abs_sin_a)
            step_width = total_extent / n_steps

            # Calculate distance from the line defining the first step riser
            dist = x[i] * normal[0] + y[i] * normal[1] - offset

            # Calculate step index
            step_index = torch.floor(dist / (step_width + 1e-9))

            # Calculate height
            if self.exponent is None:  # Hard steps
                z_i = step_index * step_height
            else:  # Soft steps
                base_height = step_index * step_height
                dist_from_step_start = dist - step_index * step_width
                # Normalize distance within the step width, clamp to avoid issues at edges
                normalized_dist = (dist_from_step_start / (step_width + 1e-9)).clamp_(0.0, 1.0)
                # Apply exponent for soft transition profile (rises from 0 to step_height)
                height_increase = step_height * (1 - (1 - normalized_dist) ** self.exponent)
                z_i = base_height + height_increase

            # Clamp overall height
            z_i.clamp_(min=0.0, max=n_steps * step_height)
            z[i] = z_i

            # Create step index mask (based on hard step boundaries)
            valid_step_mask = (step_index >= 0) & (step_index < n_steps)
            step_indices[i][valid_step_mask] = step_index[valid_step_mask].long()

        return z, {"suitable_mask": suitable_mask, "step_indices": step_indices}


@dataclass
class FixedStairsHeightmapGenerator(BaseHeightmapGenerator):
    """
    Generates a heightmap with stairs facing a fixed direction, spanning the grid.
    Step width is determined implicitly by the grid size and number of steps.
    Can optionally create soft transitions between steps using an exponent.
    Returns a `step_indices` mask indicating the step number for each cell (-1 if not on a step).
    """

    normal_angle: float = 0.0
    n_steps: int = 10
    step_height: float = 0.1
    exponent: float | None = None  # Exponent for soft transitions (None for hard steps)
    # Removed step_width parameter
    # Removed dist_from_origin

    def __post_init__(self):
        if self.n_steps <= 0:
            raise ValueError("Number of steps must be positive.")
        if self.step_height <= 0:
            raise ValueError("Step height must be positive.")
        if self.exponent is not None and self.exponent <= 0:
            raise ValueError("Exponent must be positive if specified.")
        # Removed step width check

    def _generate_heightmap(self, x, y, max_coord, rng=None):
        # Calculate normal, offset, and implicit step width
        cos_a = math.cos(self.normal_angle)
        sin_a = math.sin(self.normal_angle)
        normal = torch.tensor([cos_a, sin_a], device=x.device)
        abs_cos_a = abs(cos_a)
        abs_sin_a = abs(sin_a)
        offset = -max_coord * (abs_cos_a + abs_sin_a)
        total_extent = 2 * max_coord * (abs_cos_a + abs_sin_a)
        step_width = total_extent / self.n_steps

        # Calculate distance from the line defining the first step riser
        dist = x * normal[0] + y * normal[1] - offset

        # Calculate step index
        step_index = torch.floor(dist / (step_width + 1e-9))

        # Calculate height
        if self.exponent is None:  # Hard steps
            z = step_index * self.step_height
        else:  # Soft steps
            base_height = step_index * self.step_height
            dist_from_step_start = dist - step_index * step_width
            # Normalize distance within the step width, clamp to avoid issues at edges
            normalized_dist = (dist_from_step_start / (step_width + 1e-9)).clamp_(0.0, 1.0)
            # Apply exponent for soft transition profile (rises from 0 to step_height)
            height_increase = self.step_height * (1 - (1 - normalized_dist) ** self.exponent)
            z = base_height + height_increase

        # Clamp overall height
        z.clamp_(min=0.0, max=self.n_steps * self.step_height)

        # Create step index mask (based on hard step boundaries)
        step_indices = torch.full_like(x, -1, dtype=torch.long)
        valid_step_mask = (step_index >= 0) & (step_index < self.n_steps)
        step_indices[valid_step_mask] = step_index[valid_step_mask].long()

        suitable_mask = torch.ones_like(x, dtype=torch.bool)  # Placeholder

        return z, {"suitable_mask": suitable_mask, "step_indices": step_indices}


if __name__ == "__main__":
    from flipper_training.vis.static_vis import plot_heightmap_3d
    import matplotlib

    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt

    # --- Configuration ---
    resolution = 0.05
    max_coord = 2.0
    batch_size = 2
    device = "cpu"
    seed = 44  # Slightly different seed
    rng = torch.Generator(device=device).manual_seed(seed)
    soft_exponent = 10.0

    # --- Random Stairs Generation (Hard) ---
    print("Generating random stairs (Hard)...")
    random_stairs_gen_hard = StairsHeightmapGenerator(min_steps=8, max_steps=12, min_step_height=0.1, max_step_height=0.2, exponent=None)
    x_random_hard, y_random_hard, z_random_hard, _ = random_stairs_gen_hard(grid_res=resolution, max_coord=max_coord, num_robots=batch_size, rng=rng)

    # --- Random Stairs Generation (Soft) ---
    print("Generating random stairs (Soft)...")
    # Use same parameters but with exponent
    random_stairs_gen_soft = StairsHeightmapGenerator(min_steps=8, max_steps=12, min_step_height=0.1, max_step_height=0.2, exponent=soft_exponent)
    # Need a new RNG state or reseed if we want the *exact* same steps/heights but soft
    rng.manual_seed(seed)  # Reset RNG for comparable generation
    x_random_soft, y_random_soft, z_random_soft, _ = random_stairs_gen_soft(grid_res=resolution, max_coord=max_coord, num_robots=batch_size, rng=rng)

    # --- Fixed Stairs Generation (Hard) ---
    print("Generating fixed stairs (Hard)...")
    fixed_stairs_gen_hard = FixedStairsHeightmapGenerator(normal_angle=0, n_steps=10, step_height=0.25, exponent=None)
    x_fixed_hard, y_fixed_hard, z_fixed_hard, _ = fixed_stairs_gen_hard(grid_res=resolution, max_coord=max_coord, num_robots=batch_size, rng=rng)

    # --- Fixed Stairs Generation (Soft) ---
    print("Generating fixed stairs (Soft)...")
    fixed_stairs_gen_soft = FixedStairsHeightmapGenerator(normal_angle=0, n_steps=10, step_height=0.25, exponent=soft_exponent)
    x_fixed_soft, y_fixed_soft, z_fixed_soft, _ = fixed_stairs_gen_soft(grid_res=resolution, max_coord=max_coord, num_robots=batch_size, rng=rng)

    # --- Visualization ---
    print("Visualizing random stairs (Batch 0 - Hard vs Soft)...")
    fig_random_hard = plot_heightmap_3d(x_random_hard[0], y_random_hard[0], z_random_hard[0])
    fig_random_hard.update_layout(title=f"Random Stairs (Hard, Batch 0)")
    fig_random_hard.show()

    fig_random_soft = plot_heightmap_3d(x_random_soft[0], y_random_soft[0], z_random_soft[0])
    fig_random_soft.update_layout(title=f"Random Stairs (Soft Exp={soft_exponent}, Batch 0)")
    fig_random_soft.show()

    print("Visualizing fixed stairs (Batch 0 - Hard vs Soft)...")
    fig_fixed_hard = plot_heightmap_3d(x_fixed_hard[0], y_fixed_hard[0], z_fixed_hard[0])
    fig_fixed_hard.update_layout(title=f"Fixed Stairs (Hard, Batch 0)")
    fig_fixed_hard.show()

    fig_fixed_soft = plot_heightmap_3d(x_fixed_soft[0], y_fixed_soft[0], z_fixed_soft[0])
    fig_fixed_soft.update_layout(title=f"Fixed Stairs (Soft Exp={soft_exponent}, Batch 0)")
    fig_fixed_soft.show()

    print("Done.")
