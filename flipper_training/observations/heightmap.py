from dataclasses import dataclass

import torch
import torch.nn as nn
from torchrl.data import Unbounded

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.utils.environment import interpolate_grid
from flipper_training.utils.geometry import planar_rot_from_q

from . import Observation, ObservationEncoder


class ResBlock(nn.Module):
    """
    Residual Block with GroupNorm. Handles downsampling and channel changes.
    """

    def __init__(self, in_channels, out_channels, stride=1, groups_in_norm=8):
        super().__init__()
        # Ensure groups_in_norm divides channels, adjust if necessary
        num_groups = max(1, out_channels // groups_in_norm)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

        # Shortcut connection (for identity or projection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Need projection to match dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.GroupNorm(num_groups, out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += identity  # Add the shortcut connection
        out = self.relu(out)
        return out


class HeightmapEncoder(ObservationEncoder):
    def __init__(self, img_shape: tuple[int, int], output_dim: int):
        super(HeightmapEncoder, self).__init__(output_dim)
        self.img_shape = img_shape  # Keep for reference if needed, but not used in layer defs anymore
        groups_in_norm = 8  # Approx group size for GroupNorm

        # Define the sequential convolutional layers
        # Each block roughly corresponds to a downsampling stage in the original
        self.encoder = nn.Sequential(
            # Layer 1: Similar to the original stem but using 3x3 kernel
            # Input: (B, 1, H, W)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(max(1, 32 // groups_in_norm), 32),
            nn.ReLU(inplace=True),
            # Output: (B, 32, H/2, W/2)
            # Layer 2: Downsample, increase channels
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(max(1, 64 // groups_in_norm), 64),
            nn.ReLU(inplace=True),
            # Output: (B, 64, H/4, W/4)
            # Layer 3: Same channels, same spatial size
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(max(1, 64 // groups_in_norm), 64),
            nn.ReLU(inplace=True),
            # Layer 4: Downsample, increase channels
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(max(1, 128 // groups_in_norm), 128),
            nn.ReLU(inplace=True),
            # Output: (B, 128, H/8, W/8)
            # Layer 5: Downsample, increase channels (Optional, could stop earlier)
            # If the input image is small, H/16 might become too small.
            # Let's keep it for now to match the original depth approximately.
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(max(1, 128 // groups_in_norm), 128),
            nn.ReLU(inplace=True),
            # Output: (B, 128, H/16, W/16)
            # Final pooling and linear layer (same as before)
            nn.AdaptiveAvgPool2d((1, 1)),  # Pool to 1x1 spatial dimensions
            nn.Flatten(),  # Flatten features -> (B, 128)
            nn.Linear(128, output_dim),  # Linear layer -> (B, output_dim)
        )

    def forward(self, hm):
        # Handle potential time dimension (same as before)
        if hm.ndim > 4:
            B, T = hm.shape[:2]
            # Input shape expected: (B, T, C, H, W)
            C, H, W = hm.shape[2:]
            hm = hm.view(B * T, C, H, W)  # Use view for efficiency
            y_ter = self.encoder(hm)
            # Output shape expected: (B, T, output_dim)
            y_ter = y_ter.view(B, T, -1)
        else:
            # Input shape expected: (B, C, H, W)
            y_ter = self.encoder(hm)
            # Output shape expected: (B, output_dim)
        return y_ter


@dataclass
class Heightmap(Observation):
    """
    Generates heightmap observation from the environment using 2D transformations.
    """

    percep_shape: tuple[int, int]
    percep_extent: tuple[float, float, float, float]
    interval: tuple[float, float]
    normalize_to_interval: bool = False
    supports_vecnorm = False
    name = "heightmap"

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
        hm.clamp_(self.interval[0], self.interval[1])
        # Normalize to interval if specified
        if self.normalize_to_interval:
            hm.div_(self.interval[1] - self.interval[0])
        return hm.to(self.env.out_dtype)

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.n_robots, 1, self.percep_shape[0], self.percep_shape[1]),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )

    def get_encoder(self) -> HeightmapEncoder:
        return HeightmapEncoder(self.percep_shape, **self.encoder_opts)
