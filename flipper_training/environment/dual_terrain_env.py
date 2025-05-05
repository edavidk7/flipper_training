"""
Dual Terrain environment, there is the main terrain config and the physics terrain config, they can differ
The physics terrain config is used to generate the physics engine, while the main terrain config is used to generate the heightmap observations
"""

from typing import Literal

import torch
from flipper_training.configs.terrain_config import TerrainConfig
from flipper_training.vis.static_vis import plot_heightmap_3d
from flipper_training.environment.env import Env


class DualTerrainEnv(Env):
    def __init__(
        self,
        terrain_config_obs: TerrainConfig,
        terrain_config_phys: TerrainConfig,
        **kwargs,
    ):
        self.terrain_cfg_phys = terrain_config_phys
        super().__init__(**kwargs, terrain_config=terrain_config_obs)

    def visualize(self, mode: Literal["phys", "obs"] = "obs"):
        for i in range(self.n_robots):
            plot_heightmap_3d(
                self.terrain_cfg_phys.x_grid[i] if mode == "phys" else self.terrain_cfg.x_grid[i],
                self.terrain_cfg_phys.y_grid[i] if mode == "phys" else self.terrain_cfg.y_grid[i],
                self.terrain_cfg_phys.z_grid[i] if mode == "phys" else self.terrain_cfg.z_grid[i],
                start=self.start.x[i],
                end=self.goal.x[i],
            ).show()

    def _step_engine(self, prev_state, action, terrain_cfg_phys: TerrainConfig | None = None):
        curr_state = prev_state
        first_prev_state_der = None
        if terrain_cfg_phys is None:
            terrain_cfg_phys = self.terrain_cfg_phys
        with torch.inference_mode(not self.differentiable):
            for _ in range(self.engine_iters_per_step):
                curr_state, prev_state_der = self.engine(curr_state, action, terrain_cfg_phys)
                if first_prev_state_der is None:
                    first_prev_state_der = prev_state_der.clone()
                curr_state = curr_state.clone()
        return first_prev_state_der, curr_state
