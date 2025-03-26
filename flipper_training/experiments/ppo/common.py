import logging
from typing import TYPE_CHECKING, Tuple

import torch
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer
from torchrl.envs import (
    Compose,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)

from flipper_training.configs import PhysicsEngineConfig, RobotModelConfig, WorldConfig
from flipper_training.utils.torch_utils import set_device

if TYPE_CHECKING:
    from config import PPOExperimentConfig
    from torchrl.modules import SafeSequential

    from flipper_training.environment.env import Env

logging.getLogger().setLevel(logging.INFO)


def prepare_configs(rng: torch.Generator, cfg: "PPOExperimentConfig") -> Tuple[WorldConfig, PhysicsEngineConfig, RobotModelConfig, torch.device]:
    heightmap_gen = cfg.heightmap_gen(**cfg.heightmap_gen_opts)
    x, y, z, extras = heightmap_gen(cfg.grid_res, cfg.max_coord, cfg.num_robots, rng)
    world_config = WorldConfig(
        x_grid=x,
        y_grid=y,
        z_grid=z,
        grid_extras=extras,
        **cfg.world_opts,
        grid_res=cfg.grid_res,
        max_coord=cfg.max_coord,
    )
    robot_model = RobotModelConfig(**cfg.robot_model_opts)
    physics_config = PhysicsEngineConfig(num_robots=cfg.num_robots, **cfg.engine_opts)
    device = set_device(cfg.device)
    robot_model.to(device)
    world_config.to(device)
    physics_config.to(device)
    return world_config, physics_config, robot_model, device


def make_transformed_env(env: "Env", cfg: "PPOExperimentConfig"):
    tf_env = TransformedEnv(
        env,
        Compose(
            *[ObservationNorm(in_keys=[k], standard_normal=True) for k in cfg.observations],
            StepCounter(),
        ),
    )
    for t in tf_env.transform:
        if isinstance(t, ObservationNorm):
            t.init_stats(cfg.norm_init_iters, reduce_dim=(0, 1), cat_dim=1)
    tf_env.reset(reset_all=True)
    return tf_env


def prepare_data_collection(env: "Env", policy: "SafeSequential", cfg: "PPOExperimentConfig") -> Tuple[SyncDataCollector, TensorDictReplayBuffer]:
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=cfg.frames_per_batch * cfg.num_robots,
        total_frames=cfg.total_frames,
        **cfg.data_collector_opts,
        device=env.device,
    )
    # Replay Buffer
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            max_size=cfg.frames_per_batch * cfg.num_robots,
            ndim=1,  # This storage will mix the batch and time dimensions
            device=env.device,
            compilable=True,
        ),
        sampler=SamplerWithoutReplacement(drop_last=True),
        batch_size=cfg.frames_per_sub_batch * cfg.num_robots,  # sample flattened and mixed data
        dim_extend=0,
        compilable=True,
    )
    return collector, replay_buffer
