from typing import TYPE_CHECKING, Literal, Tuple

import torch
from collections import defaultdict
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer
from torchrl.envs.utils import check_env_specs

from flipper_training.configs import PhysicsEngineConfig, RobotModelConfig, TerrainConfig
from flipper_training.configs.experiment_config import make_partial_observations
from flipper_training.environment.env import Env
from flipper_training.utils.torch_utils import seed_all, set_device
from argparse import ArgumentParser
from pathlib import Path
from config import PPOExperimentConfig
from omegaconf import DictConfig, OmegaConf
from flipper_training.utils.logutils import LocalRunReader, WandbRunReader

from torchrl.envs import (
    Compose,
    VecNorm,
    StepCounter,
    TransformedEnv,
)

if TYPE_CHECKING:
    from config import PPOExperimentConfig
    from torchrl.modules import SafeSequential
    from tensordict import TensorDict

EVAL_LOG_OPT = {
    "precision": {
        "step_reward": 4,
        "step_count": 0,
        "succeeded": 3,
        "failed": 3,
        "truncated": 3,
    },
    "groups": [
        ["step_reward"],
        ["step_count"],
        ["succeeded", "failed", "truncated"],
    ],
    "group_labels": {
        0: "Eval per-step reward",
        1: "Eval step count",
        2: "Eval state statistics",
    },
}


def prepare_configs(rng: torch.Generator, cfg: "PPOExperimentConfig") -> Tuple[TerrainConfig, PhysicsEngineConfig, RobotModelConfig, torch.device]:
    heightmap_gen = cfg.heightmap_gen(**cfg.heightmap_gen_opts if cfg.heightmap_gen_opts else {})
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


def make_formatted_str_lines(eval_log: dict[str, int | float], format_dict: dict, init_eval_log: dict[str, int | float] | None = None) -> list[str]:
    var_dict = defaultdict(list)
    for key, value in eval_log.items():
        statistic = key.split("/")[1]  # remove the prefix specifying the type of value
        var_name = statistic.split("_", maxsplit=1)[-1]  # remove the prefix specifying the type of value
        if var_name in format_dict["precision"]:
            prec = format_dict["precision"].get(var_name, 4)
            s = f"{key}: {value:.{prec}f}"
            if init_eval_log is not None and key in init_eval_log:
                s += f" (init {init_eval_log[key]:.{prec}f})"
            var_dict[var_name].append(s)

    lines = []
    for i, group in enumerate(format_dict["groups"]):
        s = f"{format_dict['group_labels'][i]}: "
        for var_name in group:
            for entry in var_dict[var_name]:
                s += f"{entry} "
        lines.append(s)
    return lines


def log_from_eval_rollout(eval_rollout: "TensorDict") -> dict[str, int | float]:
    """
    Computes the statistics from the evaluation rollout and returns them as a dictionary.
    """
    last_step_count = eval_rollout["step_count"][:, -1].float()
    last_succeeded_mean = eval_rollout["next", "succeeded"][:, -1].float().mean().item()
    last_failed_mean = eval_rollout["next", "failed"][:, -1].float().mean().item()
    return {
        "eval/mean_step_reward": eval_rollout["next", "reward"].mean().item(),
        "eval/max_step_reward": eval_rollout["next", "reward"].max().item(),
        "eval/min_step_reward": eval_rollout["next", "reward"].min().item(),
        "eval/mean_step_count": last_step_count.mean().item(),
        "eval/max_step_count": last_step_count.max().item(),
        "eval/min_step_count": last_step_count.min().item(),
        "eval/pct_succeeded": last_succeeded_mean,
        "eval/pct_failed": last_failed_mean,
        "eval/pct_truncated": 1 - last_succeeded_mean - last_failed_mean,  # eithered none of the states at the end of the rollout
    }


def make_normed_env(
    env: "Env", train_config: "PPOExperimentConfig", vecnorm_weights_path: str | Path | None = None, freeze_vecnorm: bool = False
) -> TransformedEnv:
    vecnorm_keys = [o.name for o in env.observations if o.supports_vecnorm]
    if train_config.vecnorm_on_reward:
        vecnorm_keys.append("reward")
    transforms = Compose(
        StepCounter(),
        VecNorm(in_keys=vecnorm_keys, **train_config.vecnorm_opts),
    )
    if vecnorm_weights_path is not None:
        transforms[-1].load_state_dict(torch.load(vecnorm_weights_path, map_location=env.device))
    if freeze_vecnorm:
        transforms[-1].eval()
    return TransformedEnv(env, transforms)


def download_config_and_paths(reader: WandbRunReader | LocalRunReader, weight_step: int | None) -> tuple[DictConfig, Path | None, Path | None]:
    run_omegaconf = reader.load_config()
    if not isinstance(run_omegaconf, DictConfig):
        raise ValueError("Config must be a DictConfig")
    if run_omegaconf["type"].__name__ != PPOExperimentConfig.__name__:
        raise ValueError("Config must be of type PPOExperimentConfig")
    if weight_step is not None:
        policy_weights_path = reader.get_weights_path(f"policy_step_{weight_step}")
        vecnorm_weights_path = reader.get_weights_path(f"vecnorm_step_{weight_step}")
    else:
        policy_weights_path = vecnorm_weights_path = None
    return run_omegaconf, policy_weights_path, vecnorm_weights_path


def parse_and_load_config() -> dict:
    parser = ArgumentParser()
    parser.add_argument("--local", type=Path, required=False, default=None, help="Path to the local run directory")
    parser.add_argument("--wandb", type=Path, required=False, default=None, help="Name of the run to evaluate")
    parser.add_argument("--weight_step", type=int, required=False, help="Step from which to load the weights", default=None)
    args, unknown = parser.parse_known_args()
    if args.local is None and args.wandb is None:
        raise ValueError("Either --local or --wandb must be provided")
    if args.local is not None and args.wandb is not None:
        raise ValueError("Only one of --local or --wandb must be provided")
    if args.local is not None and "yaml" in args.local.name:
        parsed_omegaconf = OmegaConf.load(args.local)
        policy_weights_path = vecnorm_weights_path = None
    else:
        run_reader = WandbRunReader(args.wandb, category="ppo") if args.wandb else LocalRunReader(Path("runs/ppo") / args.local)
        parsed_omegaconf, policy_weights_path, vecnorm_weights_path = download_config_and_paths(run_reader, args.weight_step)
    cli_omegaconf = OmegaConf.from_dotlist(unknown)
    merged_omegaconf = OmegaConf.merge(parsed_omegaconf, cli_omegaconf)
    return {
        "config": merged_omegaconf,
        "policy_weights_path": policy_weights_path,
        "vecnorm_weights_path": vecnorm_weights_path,
    }
