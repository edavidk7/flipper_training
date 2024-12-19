import torch
from .geometry import normalized


def compute_heightmap_gradients(z_grid: torch.Tensor, grid_res: float) -> torch.Tensor:
    """
    Computes the gradients of a heightmap along the x and y axes using torch.gradient.

    **Note**: The input grid is assumed to use torch's "xy" indexing convention, i.e., the first dimension is the y-axis and the second dimension is the x-axis.

    Parameters:
    - z_grid: Heightmap tensor of shape (B, H, W).
    - grid_res: Resolution of the grid in meters.
    """
    return torch.stack(torch.gradient(z_grid, spacing=grid_res, dim=(2, 1), edge_order=2), dim=1)


@torch.compile
def surface_normals(z_grid_grads: torch.Tensor, query: torch.Tensor, max_coord: float) -> torch.Tensor:
    """
    Computes the surface normals and tangents at the queried coordinates.

    Parameters:
    - z_grid_grads: torch.Tensor of gradients of the heightmap along the x and y axes (3D array), (B, 2, H, W).
    - query: Tensor of desired point coordinates for interpolation (3D array), (B, N, 2).

    Returns:
    - Surface normals at the queried coordinates.
    """
    norm_query = query / max_coord  # Normalize to [-1, 1]
    # Query coordinates of shape (B, N, 1, 2)
    B, N = query.shape[:2]
    grid_coords = norm_query.unsqueeze(2)
    # Interpolate the grid values into shape (B, 2, N, 1)
    grad_query = torch.nn.functional.grid_sample(z_grid_grads, grid_coords, align_corners=True, mode="bilinear", padding_mode="border").squeeze(-1).transpose(1, 2)  # (B, N, 2)
    # Compute the surface normals
    n = torch.dstack([-grad_query, torch.ones((B, N, 1))])  # n = [-dz/dx, -dz/dy, 1]
    n = normalized(n)
    return n


@torch.compile
def interpolate_grid(grid: torch.Tensor, query: torch.Tensor, max_coord: float | torch.Tensor) -> torch.Tensor:
    """
    Interpolates the height at the desired (query[0], query[1]]) coordinates.

    Parameters:
    - grid: Tensor of grid values corresponding to the x and y coordinates (3D array), (B, D, D). Top-left corner is (-max_coord, -max_coord). The indexing order follows the "xy" convention, meaning the first dimension is the y-axis and the second dimension is the x-axis.
    - query: Tensor of desired point coordinates for interpolation (3D array), (B, N, 2). Range is from -max_coord to max_coord.
    Returns:
    - Interpolated grid values at the queried coordinates in shape (B, N, 1).
    """
    norm_query = query / max_coord  # Normalize to [-1, 1]
    # Query coordinates of shape (B, N, 1, 2)
    grid_coords = norm_query.unsqueeze(2)
    # Grid of shape (B, 1, H, W)
    grid_w_c = grid.unsqueeze(1)
    # Interpolate the grid values into shape (B, 1, N, 1)
    z_query = torch.nn.functional.grid_sample(grid_w_c, grid_coords, align_corners=True, mode="bilinear", padding_mode="border")
    return z_query.squeeze(1)
