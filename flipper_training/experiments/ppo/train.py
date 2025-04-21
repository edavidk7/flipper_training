from typing import TYPE_CHECKING
from pathlib import Path
import torch
from common import (
    prepare_data_collection,
    prepare_env,
    make_formatted_str_lines,
    log_from_eval_rollout,
    EVAL_LOG_OPT,
    parse_and_load_config,
    make_normed_env,
)
from config import PPOExperimentConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from flipper_training.environment.env import Env
from flipper_training.utils.logging import RunLogger, print_sticky_tqdm
from flipper_training.policies.basic_policy import make_policy, make_value_function

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from tensordict import TensorDict

TRAIN_LOG_OPT = {
    "precision": {
        "action_sample_log_prob": 4,
        "critic_loss": 4,
        "objective_loss": 4,
        "entropy_loss": 4,
        "entropy": 4,
        "kl_approx": 4,
        "clip_fraction": 4,
        "lr": 4,
        "advantage": 4,
        "grad_norm": 4,
    },
    "groups": [
        ["action_sample_log_prob", "entropy", "kl_approx"],
        ["critic_loss", "objective_loss", "entropy_loss"],
        ["advantage"],
        ["clip_fraction"],
        ["grad_norm"],
        ["lr"],
    ],
    "group_labels": {
        0: "Train policy statistics",
        1: "Train PPO loss statistics",
        2: "Train advantage statistics",
        3: "Train PPO clipped fraction",
        4: "Train gradient norm",
        5: "Train learning rate",
    },
}


def train_tensordicts_to_log(rollout_td: "TensorDict", loss_td: "TensorDict", grad_norm: torch.Tensor) -> dict[str, float]:
    """
    Extract important data from the training tensordicts to log.
    """
    return {
        "train/mean_action_sample_log_prob": rollout_td["sample_log_prob"].mean().item(),
        "train/mean_critic_loss": loss_td["loss_critic"].mean().item(),
        "train/mean_objective_loss": loss_td["loss_objective"].mean().item(),
        "train/mean_entropy_loss": loss_td["loss_entropy"].mean().item(),
        "train/mean_entropy": loss_td["entropy"].mean().item(),
        "train/mean_kl_approx": loss_td["kl_approx"].mean().item(),
        "train/mean_clip_fraction": loss_td["clip_fraction"].mean().item(),
        "train/mean_advantage": rollout_td["advantage"].mean().item(),
        "train/std_advantage": rollout_td["advantage"].std().item(),
        "train/total_grad_norm": grad_norm.item(),
    }


def train_ppo(
    config: "DictConfig",
    policy_weights_path: str | Path | None = None,
    value_weights_path: str | Path | None = None,
    vecnorm_weights_path: str | Path | None = None,
):
    train_config = PPOExperimentConfig(**config)
    env, device, rng = prepare_env(train_config, mode="train")
    env = make_normed_env(env, train_config, vecnorm_weights_path)
    actor_operator = make_policy(
        env,
        policy_opts=train_config.policy_opts,
        encoders_opts=train_config.observation_encoders_opts,
        device=device,
        weights_path=policy_weights_path,
    )
    value_operator = make_value_function(
        env,
        value_opts=train_config.value_function_opts,
        encoders_opts=train_config.observation_encoders_opts,
        device=device,
        weights_path=value_weights_path,
    )
    # Collector
    collector, replay_buffer = prepare_data_collection(env, actor_operator, train_config)
    # PPO setup
    advantage_module = GAE(
        **train_config.gae_opts, value_network=value_operator, time_dim=1, device=device, differentiable=False
    )  # here we still expect (B, T, ...) collected data
    advantage_module = advantage_module.to(train_config.training_dtype)
    loss_module = ClipPPOLoss(actor_operator, value_operator, **train_config.ppo_opts)
    loss_module = loss_module.to(train_config.training_dtype)
    # Compile
    if train_config.gae_compile_opts:
        advantage_module.compile(**train_config.gae_compile_opts)
    if train_config.ppo_compile_opts:
        loss_module.compile(**train_config.ppo_compile_opts)
    # Optim
    optim = train_config.optimizer(
        [
            {"params": actor_operator.parameters(), **train_config.optimizer_opts["actor"]},
            {"params": value_operator.parameters(), **train_config.optimizer_opts["critic"]},
        ],
    )
    scheduler = train_config.scheduler(
        optim,
        **train_config.scheduler_opts,
    )
    if train_config.frames_per_batch // train_config.frames_per_sub_batch == 0:
        raise ValueError("frames_per_batch must be divisible by frames_per_sub_batch")
    # Training loop
    logger = RunLogger(
        train_config=config,
        category="ppo",
        use_wandb=train_config.use_wandb,
        step_metric_name="collected_frames",
    )
    pbar = tqdm(total=train_config.total_frames, desc="Training", unit="frames")
    init_train_log = {}
    init_eval_log = {}
    eval_lines = []
    for i, tensordict_data in enumerate(collector):
        # collected (B, T, *specs) where B is the batch size and T the number of steps
        tensordict_data.pop(Env.STATE_KEY)  # we don't need this
        tensordict_data.pop(("next", Env.STATE_KEY))  # we don't need this
        actor_operator.train()
        value_operator.train()
        env.train()
        for _ in range(train_config.epochs_per_batch):
            advantage_module(tensordict_data)
            replay_buffer.extend(tensordict_data.reshape(-1))  # we can now safely flatten the data
            for _ in range(train_config.frames_per_batch // train_config.frames_per_sub_batch):
                sub_batch = replay_buffer.sample()
                loss_vals = loss_module(sub_batch)
                loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                loss_value.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), train_config.max_grad_norm)
                optim.step()
                optim.zero_grad()

        total_collected_frames = (i + 1) * train_config.frames_per_batch * train_config.num_robots

        log = train_tensordicts_to_log(tensordict_data, loss_vals, grad_norm)
        log |= {"train/lr": optim.param_groups[0]["lr"]}
        if not init_train_log:
            init_train_log |= log
        train_lines = make_formatted_str_lines(log, TRAIN_LOG_OPT, init_train_log)
        if i % train_config.eval_and_save_every == 0:
            actor_operator.eval()
            value_operator.eval()
            env.eval()
            with (
                set_exploration_type(ExplorationType.DETERMINISTIC),
                torch.inference_mode(),
            ):
                eval_rollout = env.rollout(train_config.max_eval_steps, actor_operator, break_when_all_done=True, auto_reset=True)
                eval_log = log_from_eval_rollout(eval_rollout)
                if not init_eval_log:
                    init_eval_log |= eval_log
                eval_lines = make_formatted_str_lines(eval_log, EVAL_LOG_OPT, init_eval_log)
                log.update(eval_log)
                logger.save_weights(actor_operator.state_dict(), f"policy_step_{total_collected_frames}")
                logger.save_weights(value_operator.state_dict(), f"value_step_{total_collected_frames}")
                logger.save_weights(env.transform[-1].state_dict(), f"vecnorm_step_{total_collected_frames}")
                del eval_rollout

        all_lines = train_lines + eval_lines
        print_sticky_tqdm(all_lines)
        logger.log_data(log, total_collected_frames)
        pbar.update(train_config.frames_per_batch * train_config.num_robots)
        scheduler.step()

    pbar.close()
    logger.close()


if __name__ == "__main__":
    train_ppo(**parse_and_load_config())
