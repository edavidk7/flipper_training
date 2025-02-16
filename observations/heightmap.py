import torch
from flipper_training.engine.engine_state import PhysicsState, AuxEngineInfo, PhysicsStateDer
from .obs import Observation, PhysicsState
from dataclasses import dataclass
from flipper_training.utils.geometry import local_to_global
from flipper_training.utils.environment import interpolate_grid
from torchrl.data import Unbounded


@dataclass
class Heightmap(Observation):
    """
    Generates heightmap observation from the environment.
    """

    percep_shape: tuple[int, int]
    percep_extent: tuple[float, float, float, float]

    def __post_init__(self):
        self._initialize_perception_grid()

    def _initialize_perception_grid(self) -> None:
        """
        Initialize the perception grid points.
        """
        x_space = torch.linspace(self.percep_extent[0], self.percep_extent[2], self.percep_shape[0])
        y_space = torch.linspace(self.percep_extent[1], self.percep_extent[3], self.percep_shape[1])
        px, py = torch.meshgrid(x_space, y_space, indexing="ij")  # TODO check this, but we want the first coordinate to be the vertical one on the grid.
        percep_grid_points = torch.dstack([px, py, torch.zeros_like(px)]).reshape(-1, 3)  # add the z coordinate (0)
        self.percep_grid_points = percep_grid_points.unsqueeze(0).repeat(self.env.n_robots, 1, 1).to(self.env.device)

    def __call__(self, prev_state: PhysicsState,
                 action: torch.Tensor,
                 state_der: PhysicsStateDer,
                 curr_state: PhysicsState,
                 aux_info: AuxEngineInfo) -> torch.Tensor:
        global_percep_points = local_to_global(curr_state.x, curr_state.R, self.percep_grid_points)
        z_coords = interpolate_grid(self.env.world_cfg.z_grid, global_percep_points[..., :2], self.env.world_cfg.max_coord)
        return z_coords.reshape(-1, 1, self.percep_shape[0], self.percep_shape[1]) - curr_state.x[..., 2].reshape(-1, 1, 1, 1)

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.n_robots, 1, self.percep_shape[0], self.percep_shape[1]),
            dtype=torch.float32,
            device=self.env.device,
        )
