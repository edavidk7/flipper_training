from typing import TYPE_CHECKING

import torch
from common import prepare_env, make_policy
from config import PPOExperimentConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type
from flipper_training.utils.logging import get_run_reader
from flipper_training.configs.experiment_config import hash_omegaconf
from simview import SimView
from flipper_training.vis.simview import simview_terrain_from_config, simview_bodies_from_robot_config, physics_state_to_simview_body_states
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from tensordict import TensorDictBase
    from flipper_training.environment.env import Env


def get_eval_rollout(train_config: PPOExperimentConfig) -> tuple["Env", "TensorDictBase"]:
    env, device, rng = prepare_env(train_config, mode="eval")
    actor_value_policy = make_policy(env, train_config, device)
    actor_operator = actor_value_policy.get_policy_operator()
    actor_operator.eval()
    env.reset()
    env.set_truncation(False)  # disable for eval
    with (
        set_exploration_type(ExplorationType.DETERMINISTIC),
        torch.no_grad(),
    ):
        eval_rollout = env.rollout(train_config.max_eval_steps, actor_operator, auto_reset=True, break_when_all_done=True)
    return env, eval_rollout


def main(train_omegaconf: "DictConfig"):
    train_config = PPOExperimentConfig(**train_omegaconf)
    config_hash = hash_omegaconf(train_omegaconf)
    simview = SimView(
        run_name=config_hash,
        batch_size=train_config.num_robots,
        scalar_names=["cumulative reward", "reward"],
        dt=train_config.engine_opts["dt"],
        collapse=False,  # display all batches independently
        use_cache=False,
    )
    if simview.is_ready:
        simview.visualize()
    else:
        env, rollout = get_eval_rollout(train_config)
        simview.model.add_terrain(simview_terrain_from_config(env.terrain_cfg))
        for body in simview_bodies_from_robot_config(env.robot_cfg):
            simview.model.add_body(body.name, body)
        rollout["cumulative reward"] = rollout["next", "reward"].cumsum(dim=1).squeeze()
        for i in range(rollout.shape[1] - 1):
            time = env.phys_cfg.dt * i
            s = PhysicsState(**rollout["physics_state"][:, i])
            ds = PhysicsStateDer(**rollout["physics_state_der"][:, i + 1])  # Derivative of current state is returned with the next state
            control = rollout["action"][:, i]
            body_states = physics_state_to_simview_body_states(
                env.robot_cfg,
                s,
                ds,
                control,
            )
            simview.add_state(
                time,
                body_states=body_states,
                scalar_values={
                    "cumulative reward": rollout["cumulative reward"][:, i].squeeze().tolist(),
                    "reward": rollout["next", "reward"][:, i].squeeze().tolist(),
                },
            )
        assert simview.is_ready, "SimView is not ready. Something went wrong."
        simview.visualize()


if __name__ == "__main__":
    from argparse import ArgumentParser
    from omegaconf import DictConfig, OmegaConf

    parser = ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="Path to the local run directory or to the wandb run name prefixed with 'wandb:'")
    parser.add_argument("--weights", type=str, required=True, help="Name of the weights to evaluate")
    args, unknown = parser.parse_known_args()
    run_reader = get_run_reader(args.run, "ppo")
    run_omegaconf = run_reader.load_config()
    cli_omegaconf = OmegaConf.from_dotlist(unknown)
    train_omegaconf = OmegaConf.merge(run_omegaconf, cli_omegaconf)
    weights_path = run_reader.get_weights_path(args.weights)
    train_omegaconf["policy_opts"]["weights_path"] = weights_path
    if not isinstance(train_omegaconf, DictConfig):
        raise ValueError("Config must be a DictConfig")
    if train_omegaconf["type"].__name__ != PPOExperimentConfig.__name__:
        raise ValueError("Config must be of type PPOExperimentConfig")
    main(train_omegaconf)
