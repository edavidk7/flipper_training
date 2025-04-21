from dataclasses import dataclass

import torch
import torch.nn as nn
from torchrl.data import Unbounded

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.utils.environment import interpolate_grid
from flipper_training.utils.geometry import planar_rot_from_q

from . import Observation


class HeightmapEncoder(torch.nn.Module):
    def __init__(self, img_shape: tuple[int, int], output_dim: int):
        super(HeightmapEncoder, self).__init__()
        self.img_shape = img_shape
        self.output_dim = output_dim
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),  # instead of BatchNorm2d(16)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (img_shape[0] // 16) * (img_shape[1] // 16), output_dim),
        )

    def forward(self, hm):
        if hm.ndim > 4:
            B, T = hm.shape[:2]
            hm = hm.flatten(0, 1)
            y_ter = self.encoder(hm)
            y_ter = y_ter.reshape((B, T, -1))
        else:
            y_ter = self.encoder(hm)
        return y_ter


@dataclass
class Heightmap(Observation):
    """
    Generates heightmap observation from the environment using 2D transformations.
    """

    percep_shape: tuple[int, int]
    percep_extent: tuple[float, float, float, float]
    supports_vecnorm = False

    def __post_init__(self):
        self._initialize_perception_grid()

    def _initialize_perception_grid(self) -> None:
        """
        Initialize the 2D perception grid points.
        """
        x_space = torch.linspace(self.percep_extent[0], self.percep_extent[2], self.percep_shape[0])
        y_space = torch.linspace(self.percep_extent[1], self.percep_extent[3], self.percep_shape[1])
        px, py = torch.meshgrid(x_space, y_space, indexing="ij")
        # Store as 2D points (N, 2)
        percep_grid_points_2d = torch.dstack([px, py]).reshape(-1, 2)
        # Repeat for batch size (B, N, 2)
        self.percep_grid_points_2d = percep_grid_points_2d.unsqueeze(0).repeat(self.env.n_robots, 1, 1).to(self.env.device)

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
    ) -> torch.Tensor:
        B = curr_state.x.shape[0]
        # Get 2D rotation matrix from quaternion (B, 2, 2)
        R_yaw_2d = planar_rot_from_q(curr_state.q)

        # Rotate local 2D points (B, N, 2)
        rotated_points_2d = torch.bmm(self.percep_grid_points_2d, R_yaw_2d.transpose(1, 2))

        # Translate points by robot's XY position (B, N, 2)
        global_percep_points_2d = rotated_points_2d + curr_state.x[..., :2].unsqueeze(1)

        # Interpolate height at global 2D points (B, N)
        z_coords = interpolate_grid(self.env.terrain_cfg.z_grid, global_percep_points_2d, self.env.terrain_cfg.max_coord)

        # Reshape and make height relative to robot's Z coordinate
        hm = z_coords.reshape(B, 1, self.percep_shape[0], self.percep_shape[1]) - curr_state.x[..., 2].reshape(-1, 1, 1, 1)
        return hm.to(self.env.out_dtype)

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.n_robots, 1, self.percep_shape[0], self.percep_shape[1]),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )

    def get_encoder(self, output_dim) -> HeightmapEncoder:
        return HeightmapEncoder(self.percep_shape, output_dim)
