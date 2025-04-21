from dataclasses import dataclass

import torch
from torchrl.data import Unbounded

from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.utils.environment import interpolate_grid
from flipper_training.utils.geometry import global_to_local_q, local_to_global_q

from . import Observation, ObservationEncoder


class PointcloudEncoder(ObservationEncoder):
    def __init__(self, img_shape: tuple[int, int], output_dim: int):
        super(PointcloudEncoder, self).__init__(output_dim)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(8, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(16, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * (img_shape[0] // 16) * (img_shape[1] // 16), output_dim),
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
class Pointcloud(Observation):
    """
    Generates grid of points observation from the environment.
    """

    percep_shape: tuple[int, int]
    percep_extent: tuple[float, float, float, float]
    supports_vecnorm = False
    name = "pointcloud"

    def __post_init__(self):
        self._initialize_perception_grid()

    def _initialize_perception_grid(self) -> None:
        """
        Initialize the perception grid points.
        """
        x_space = torch.linspace(self.percep_extent[0], self.percep_extent[2], self.percep_shape[0])
        y_space = torch.linspace(self.percep_extent[1], self.percep_extent[3], self.percep_shape[1])
        px, py = torch.meshgrid(
            x_space, y_space, indexing="ij"
        )  # TODO check this, but we want the first coordinate to be the vertical one on the grid.
        percep_grid_points = torch.dstack([px, py, torch.zeros_like(px)]).reshape(-1, 3)  # add the z coordinate (0)
        self.percep_grid_points = percep_grid_points.unsqueeze(0).repeat(self.env.n_robots, 1, 1).to(self.env.device)

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
    ) -> torch.Tensor:
        global_percep_points = local_to_global_q(curr_state.x, curr_state.q, self.percep_grid_points)
        z = interpolate_grid(self.env.world_cfg.z_grid, global_percep_points[..., :2], self.env.world_cfg.max_coord)
        global_percep_points[..., 2] = z.squeeze(-1)
        local_percep_points = global_to_local_q(curr_state.x, curr_state.q, global_percep_points)  # shape (B, N, 3)
        return local_percep_points.permute(0, 2, 1).reshape(-1, 3, self.percep_shape[0], self.percep_shape[1]).to(self.env.out_dtype)

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.n_robots, 3, self.percep_shape[0], self.percep_shape[1]),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )

    def get_encoder(self) -> PointcloudEncoder:
        return PointcloudEncoder(self.percep_shape, **self.encoder_opts)
