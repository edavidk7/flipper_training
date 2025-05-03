import gc
from typing import TYPE_CHECKING, Tuple

import torch
from pathlib import Path
from simview import SimView
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.environment.env import Env
from flipper_training.vis.simview import physics_state_to_simview_body_states, simview_bodies_from_robot_config, simview_terrain_from_config
from collections import defaultdict
from torchrl.envs.utils import check_env_specs
from simple_mppi import SimpleMPPIPlanner
from flipper_training.configs import PhysicsEngineConfig, RobotModelConfig, TerrainConfig
from flipper_training.utils.torch_utils import seed_all, set_device
from argparse import ArgumentParser
from omegaconf import DictConfig, OmegaConf
from config import MPPIExperimentConfig, hash_omegaconf

if TYPE_CHECKING:
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


def prepare_configs(rng: torch.Generator, cfg: "MPPIExperimentConfig") -> Tuple[TerrainConfig, PhysicsEngineConfig, RobotModelConfig, torch.device]:
    heightmap_gen = cfg.heightmap_gen(**cfg.heightmap_gen_opts if cfg.heightmap_gen_opts else {})
    x, y, z, extras = heightmap_gen(cfg.grid_res, cfg.max_coord, cfg.num_candidates, rng)
    terrain_config = TerrainConfig(
        x_grid=x,
        y_grid=y,
        z_grid=z,
        grid_extras=extras,
        **cfg.world_opts,
        grid_res=cfg.grid_res,
        max_coord=cfg.max_coord,
    )
    robot_model = RobotModelConfig(**cfg.robot_model_opts)
    physics_config = PhysicsEngineConfig(num_robots=cfg.num_candidates, **cfg.engine_opts)
    device = set_device(cfg.device)
    robot_model.to(device)
    terrain_config.to(device)
    physics_config.to(device)
    return terrain_config, physics_config, robot_model, device


def prepare_env(train_config: "MPPIExperimentConfig") -> tuple["Env", torch.device, torch.Generator]:
    # Init configs and RL-related objects
    rng = seed_all(train_config.seed)
    terrain_config, physics_config, robot_model, device = prepare_configs(rng, train_config)
    training_objective = train_config.objective(
        device=device,
        physics_config=physics_config,
        robot_model=robot_model,
        terrain_config=terrain_config,
        rng=rng,
        **train_config.objective_opts,
    )
    # Create environment
    base_env = Env(
        objective_factory=training_objective,
        reward_factory=train_config.reward(**train_config.reward_opts),
        observation_factories=[],
        terrain_config=terrain_config,
        physics_config=physics_config,
        robot_model_config=robot_model,
        device=device,
        batch_size=[train_config.num_candidates],
        differentiable=False,
        engine_compile_opts=train_config.engine_compile_opts,
        return_derivative=True,  # needed for evaluation
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


def get_eval_rollout(train_config: MPPIExperimentConfig) -> tuple["Env", "TensorDict"]:
    env, device, rng = prepare_env(train_config)
    env.eval()
    planner = SimpleMPPIPlanner(
        env=env,
        gamma=train_config.gamma,
        temperature=train_config.temperature,
        planning_horizon=train_config.planning_horizon,
        optim_steps=train_config.optim_steps,
        top_k=train_config.top_k,
    )
    planner.to(device)
    eval_rollout = env.rollout(train_config.max_eval_steps, planner, auto_reset=True, break_when_all_done=True)
    return env, eval_rollout


def eval_mppi(config: "DictConfig"):
    train_config = MPPIExperimentConfig(**config)
    config_hash = hash_omegaconf(config)
    simview = SimView(
        run_name=config_hash,
        batch_size=train_config.num_candidates,
        scalar_names=["cumulative reward", "reward", "terminated", "truncated"],
        dt=train_config.engine_opts["dt"],
        collapse=False,  # display all batches independently
        use_cache=False,
    )
    if simview.is_ready:
        simview.visualize()
    else:
        # Compose the simview model
        env, rollout = get_eval_rollout(train_config)
        str_lines = make_formatted_str_lines(
            log_from_eval_rollout(rollout),
            EVAL_LOG_OPT,
        )
        print(*str_lines, sep="\n")
        simview.model.add_terrain(simview_terrain_from_config(env.terrain_cfg))
        for body in simview_bodies_from_robot_config(env.robot_cfg):
            simview.model.add_body(body)
        for static_object in env.objective.start_goal_to_simview(env.start, env.goal):
            simview.model.add_static_object(static_object)
        # Correct the rewards for robots that have finished the episode
        dones = torch.roll(rollout["next", "done"].float(), 1, 1)  # shifted by one timestep forward
        dones[:, 0] = 0  # first timestep is not done
        reward_masked = rollout["next", "reward"] * (1 - dones)
        reward_masked = reward_masked.squeeze()
        cum_reward = torch.cumsum(reward_masked, dim=1)
        for i in range(rollout.shape[1]):
            s = PhysicsState.from_tensordict(rollout[Env.STATE_KEY][:, i])
            # Derivative of current state is returned with the next state
            ds = (
                PhysicsStateDer.from_tensordict(rollout[Env.PREV_STATE_DER_KEY][:, i + 1])
                if i < rollout.shape[1] - 1
                else PhysicsStateDer.from_tensordict(rollout[Env.PREV_STATE_DER_KEY][:, i])
            )
            control = rollout["action"][:, i]
            body_states = physics_state_to_simview_body_states(
                env.robot_cfg,
                s,
                ds,
                control,
            )
            simview.add_state(
                env.phys_cfg.dt * i,
                body_states=body_states,
                scalar_values={
                    "cumulative reward": cum_reward[:, i].squeeze().tolist(),
                    "reward": reward_masked[:, i].squeeze().tolist(),
                    "terminated": rollout["next", "terminated"][:, i].int().squeeze().tolist(),
                    "truncated": rollout["next", "truncated"][:, i].int().squeeze().tolist(),
                },
            )
        if not simview.is_ready:
            raise RuntimeError("SimView is not ready. Check if the model is loaded correctly.")
        del env
        del rollout
        del train_config
        gc.collect()
        simview.visualize()


if __name__ == "__main__":
    parser = ArgumentParser(description="MPPI evaluation script")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    config_path = Path(args.config)
    eval_mppi(OmegaConf.load(config_path))
