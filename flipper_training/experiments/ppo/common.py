from typing import TYPE_CHECKING, Literal, Tuple

import torch
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer
from torchrl.envs.utils import check_env_specs

from flipper_training.configs import PhysicsEngineConfig, RobotModelConfig, TerrainConfig
from flipper_training.configs.experiment_config import make_partial_observations
from flipper_training.environment.env import Env
from flipper_training.policies import make_actor_value_policy
from flipper_training.utils.torch_utils import seed_all, set_device

if TYPE_CHECKING:
    from config import PPOExperimentConfig
    from torchrl.modules import SafeSequential


def prepare_configs(rng: torch.Generator, cfg: "PPOExperimentConfig") -> Tuple[TerrainConfig, PhysicsEngineConfig, RobotModelConfig, torch.device]:
    heightmap_gen = cfg.heightmap_gen(**cfg.heightmap_gen_opts)
    x, y, z, extras = heightmap_gen(cfg.grid_res, cfg.max_coord, cfg.num_robots, rng)
    world_config = TerrainConfig(
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


def prepare_env(train_config: "PPOExperimentConfig", mode: Literal["train", "eval"]) -> tuple["Env", torch.device, torch.Generator]:
    # Init configs and RL-related objects
    rng = seed_all(train_config.seed)
    world_config, physics_config, robot_model, device = prepare_configs(rng, train_config)
    training_objective = train_config.objective(
        device=device,
        physics_config=physics_config,
        robot_model=robot_model,
        world_config=world_config,
        rng=rng,
        **train_config.objective_opts,
    )
    # Create environment
    base_env = Env(
        objective=training_objective,
        reward=train_config.reward(**train_config.reward_opts),
        observations=make_partial_observations(train_config.observations),
        terrain_config=world_config,
        physics_config=physics_config,
        robot_model_config=robot_model,
        device=device,
        batch_size=[train_config.num_robots],
        differentiable=False,
        engine_compile_opts=train_config.engine_compile_opts,
        out_dtype=train_config.training_dtype,
        return_derivative=mode == "eval",  # needed for evaluation
    )
    check_env_specs(base_env)
    return base_env, device, rng


def make_policy(env: "Env", cfg: "PPOExperimentConfig", device: torch.device, weights_path: str | None = None) -> "SafeSequential":
    actor_value_policy = make_actor_value_policy(env, **cfg.policy_opts)
    actor_value_policy.to(device)
    if weights_path is not None:
        actor_value_policy.load_state_dict(torch.load(weights_path, map_location=device))
    return actor_value_policy
