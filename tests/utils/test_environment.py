import pytest
import torch
from flipper_training.utils.environment import compute_heightmap_gradients, surface_normals, interpolate_grid


def test_heightmap_gradients() -> None:
    z_grid = torch.tensor([[[1.0, 4.0, 7.0],
                            [2.0, 5.0, 8.0],
                            [3.0, 6.0, 9.0]]])
    grid_res = 0.1
    gradients = compute_heightmap_gradients(z_grid, grid_res)[0]
    expected_dz_dx = torch.tensor([[[0.5, 0.5, 0.5],
                                    [1.0, 1.0, 1.0],
                                    [0.5, 0.5, 0.5]]]) / grid_res
    expected_dz_dy = torch.tensor([[[1.5, 3.0, 1.5],
                                    [1.5, 3.0, 1.5],
                                    [1.5, 3.0, 1.5]]]) / grid_res
    assert torch.allclose(gradients[0], expected_dz_dx) and torch.allclose(gradients[1], expected_dz_dy), f"Expected gradients: {expected_dz_dx}, {expected_dz_dy}, but got {gradients[0]}, {gradients[1]}"


def test_surface_normals() -> None:
    grid_res = 0.1  # 0.1 m per cell
    max_coord = 0.1  # origin is in the middle of the grid
    z_grid_grads = torch.tensor([[[0.5, 0.5, 0.5],
                                  [1.0, 1.0, 1.0],
                                  [0.5, 0.5, 0.5]],
                                 [[1.5, 3.0, 1.5],
                                     [1.5, 3.0, 1.5],
                                     [1.5, 3.0, 1.5]]]) / grid_res
