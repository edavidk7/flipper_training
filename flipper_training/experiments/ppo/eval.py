import gc
from typing import TYPE_CHECKING

import torch
from common import prepare_env, make_formatted_str_lines, log_from_eval_rollout, EVAL_LOG_OPT, make_transformed_env, parse_and_load_config
from config import PPOExperimentConfig, hash_omegaconf, OmegaConf
from pathlib import Path
from simview import SimView
from torchrl.envs.utils import ExplorationType, set_exploration_type
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.environment.env import Env
from flipper_training.vis.simview import physics_state_to_simview_body_states, simview_bodies_from_robot_config, simview_terrain_from_config

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torchrl.envs import TransformedEnv
    from tensordict import TensorDictBase


def get_eval_rollout(
    train_config: PPOExperimentConfig, policy_weights_path: str | Path, vecnorm_weights_path: str | Path
) -> tuple["TransformedEnv", "TensorDictBase"]:
    env, device, rng = prepare_env(train_config, mode="eval")
    policy_config = train_config.policy_config(**train_config.policy_opts)
    actor_value_wrapper, optim_groups, policy_transforms = policy_config.create(
        env=env,
        weights_path=policy_weights_path,
        device=device,
    )
    actor_operator = actor_value_wrapper.get_policy_operator()
    env, vecnorm = make_transformed_env(env, train_config, policy_transforms)
    if vecnorm_weights_path is not None:
        vecnorm.load_state_dict(torch.load(vecnorm_weights_path, map_location=device))
    actor_operator.eval()
    env.eval()
    with (
        set_exploration_type(ExplorationType.DETERMINISTIC),
        torch.inference_mode(),
    ):
        eval_rollout = env.rollout(train_config.max_eval_steps, actor_operator, auto_reset=True, break_when_all_done=True)
    return env, eval_rollout


def eval_ppo(config: "DictConfig"):
    print(OmegaConf.to_yaml(config, sort_keys=True))
    train_config = PPOExperimentConfig(**config)
    if train_config.vecnorm_weights_path is None or train_config.policy_weights_path is None:
        raise ValueError("Policy and VecNorm weights paths must be provided for evaluation.")
    # Compose the simview model
    env, rollout = get_eval_rollout(train_config, train_config.policy_weights_path, train_config.vecnorm_weights_path)
    str_lines = make_formatted_str_lines(
        log_from_eval_rollout(rollout),
        EVAL_LOG_OPT,
    )
    print(*str_lines, sep="\n")
    simview = SimView(
        run_name=hash_omegaconf(config),
        batch_size=train_config.num_robots,
        scalar_names=["cumulative reward", "reward", "terminated", "truncated"],
        dt=env.effective_dt,
        collapse=False,  # display all batches independently
        use_cache=False,
    )
    simview.model.add_terrain(simview_terrain_from_config(env.terrain_cfg))
    for body in simview_bodies_from_robot_config(env.robot_cfg):
        simview.model.add_body(body)
    for static_object in env.objective.start_goal_to_simview(env.start, env.goal):
        simview.model.add_static_object(static_object)
    # Correct the rewards for robots that have finished the episode
    dones = torch.roll(rollout["next", "done"].float(), 1, 1)  # shifted by one timestep forward
    dones[:, 0] = 0  # first timestep is not done
    reward_masked = rollout["next", "raw_reward"] * (1 - dones)
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
            env.effective_dt * i,
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
    eval_ppo(parse_and_load_config())
