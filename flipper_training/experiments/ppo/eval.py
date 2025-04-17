import gc
from typing import TYPE_CHECKING

import torch
from common import make_policy, prepare_env, make_formatted_str_lines, log_from_eval_rollout, EVAL_LOG_OPT
from config import PPOExperimentConfig
from simview import SimView
from torchrl.envs import (
    Compose,
    StepCounter,
    TransformedEnv,
    VecNorm,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type

from flipper_training.configs.experiment_config import hash_omegaconf
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.environment.env import Env
from flipper_training.utils.logging import LocalRunReader, WandbRunReader
from flipper_training.vis.simview import physics_state_to_simview_body_states, simview_bodies_from_robot_config, simview_terrain_from_config

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from tensordict import TensorDictBase


def frozen_normed_env(env: "Env", train_config: "PPOExperimentConfig", vecnorm_weights_path: str) -> TransformedEnv:
    vecnorm_keys = list(train_config.observations.keys())
    if train_config.vecnorm_on_reward:
        vecnorm_keys.append("reward")
    norm = VecNorm(
        in_keys=vecnorm_keys,
        **train_config.vecnorm_opts,
    )
    norm.eval()
    norm.load_state_dict(torch.load(vecnorm_weights_path, map_location=env.device))
    norm = norm.to_observation_norm()
    transform = Compose(
        StepCounter(),
        norm,
    )
    return TransformedEnv(env, transform)


def get_eval_rollout(train_config: PPOExperimentConfig, weights_path: str, vecnorm_weights_path: str) -> tuple["TransformedEnv", "TensorDictBase"]:
    env, device, rng = prepare_env(train_config, mode="eval")
    env = frozen_normed_env(env, train_config, vecnorm_weights_path)
    actor_value_policy = make_policy(env, train_config, device, weights_path=weights_path)
    actor_operator = actor_value_policy.get_policy_operator()
    actor_operator.eval()
    env.reset()
    env.eval()
    with (
        set_exploration_type(ExplorationType.DETERMINISTIC),
        torch.inference_mode(),
    ):
        eval_rollout = env.rollout(train_config.max_eval_steps, actor_operator, auto_reset=True, break_when_all_done=True)
    return env, eval_rollout


def main(train_omegaconf: "DictConfig", weights_path: str, vecnorm_weights_path: str):
    train_config = PPOExperimentConfig(**train_omegaconf)
    config_hash = hash_omegaconf(train_omegaconf)
    simview = SimView(
        run_name=config_hash,
        batch_size=train_config.num_robots,
        scalar_names=["cumulative reward", "reward", "terminated", "truncated"],
        dt=train_config.engine_opts["dt"],
        collapse=False,  # display all batches independently
        use_cache=False,
    )
    if simview.is_ready:
        simview.visualize()
    else:
        # Compose the simview model
        env, rollout = get_eval_rollout(train_config, weights_path, vecnorm_weights_path)
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
    from argparse import ArgumentParser
    from pathlib import Path

    from omegaconf import DictConfig, OmegaConf

    parser = ArgumentParser()
    parser.add_argument("--local", type=Path, required=False, default=None, help="Path to the local run directory")
    parser.add_argument("--wandb", type=Path, required=False, default=None, help="Name of the run to evaluate")
    parser.add_argument("--weights", type=str, required=True, help="Name of the weights to evaluate")
    args, unknown = parser.parse_known_args()
    if args.local is None and args.wandb is None:
        raise ValueError("Either --local or --wandb must be provided")
    run_reader = WandbRunReader(args.wandb, category="ppo") if args.wandb else LocalRunReader(Path("runs/ppo") / args.local)
    run_omegaconf = run_reader.load_config()
    cli_omegaconf = OmegaConf.from_dotlist(unknown)
    train_omegaconf = OmegaConf.merge(run_omegaconf, cli_omegaconf)
    weights_path = run_reader.get_weights_path(args.weights)
    vecnorm_weights_path = run_reader.get_weights_path(f"vecnorm_{args.weights}")
    if not isinstance(train_omegaconf, DictConfig):
        raise ValueError("Config must be a DictConfig")
    if train_omegaconf["type"].__name__ != PPOExperimentConfig.__name__:
        raise ValueError("Config must be of type PPOExperimentConfig")
    main(train_omegaconf, weights_path, vecnorm_weights_path)
