from typing import TYPE_CHECKING

import torch
from common import prepare_data_collection, prepare_env, make_policy
from config import PPOExperimentConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from flipper_training.utils.logging import RunLogger

if TYPE_CHECKING:
    from omegaconf import DictConfig


def main(train_omegaconf: "DictConfig"):
    train_config = PPOExperimentConfig(**train_omegaconf)
    env, device, rng = prepare_env(train_config, mode="train")
    actor_value_policy = make_policy(env, train_config, device)
    actor_value_policy.train()
    actor_operator = actor_value_policy.get_policy_operator()
    value_operator = actor_value_policy.get_value_operator()
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
    optim = train_config.optimizer(actor_value_policy.parameters(), **train_config.optimizer_opts)
    scheduler = train_config.scheduler(
        optim,
        **train_config.scheduler_opts,
    )
    if train_config.frames_per_batch // train_config.frames_per_sub_batch == 0:
        raise ValueError("frames_per_batch must be divisible by frames_per_sub_batch")
    # Training loop
    init_log = {}
    eval_str = ""
    best_reward = -float("inf")
    logger = RunLogger(
        train_config=train_omegaconf,
        category="ppo",
        use_wandb=train_config.use_wandb,
        step_metric_name="collected_frames",
    )
    pbar = tqdm(total=train_config.total_frames)
    for i, tensordict_data in enumerate(collector):
        # collected (B, T, *specs) where B is the batch size and T the number of steps
        tensordict_data.pop("physics_state")  # we don't need this
        tensordict_data.pop(("next", "physics_state"))
        for _ in range(train_config.epochs_per_batch):
            advantage_module(tensordict_data)
            replay_buffer.extend(tensordict_data.reshape(-1))  # we can now safely flatten the data
            for _ in range(train_config.frames_per_batch // train_config.frames_per_sub_batch):
                sub_batch = replay_buffer.sample()
                loss_vals = loss_module(sub_batch)
                loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), train_config.max_grad_norm)
                optim.step()
                optim.zero_grad()

        total_collected_frames = (i + 1) * train_config.frames_per_batch * train_config.num_robots

        log = {
            "train/reward": tensordict_data["next", "reward"].mean().item(),
            "train/lr": optim.param_groups[0]["lr"],
        }

        if not init_log:
            init_log |= log

        train_str = f"train/reward: {log['train/reward']:.4f} (init: {init_log['train/reward']: .4f}), train/lr: {log['train/lr']:0.6f}"

        if i % train_config.evaluate_every == 0:
            actor_value_policy.eval()
            env.set_truncation(False)
            with (
                set_exploration_type(ExplorationType.DETERMINISTIC),
                torch.no_grad(),
            ):
                eval_rollout = env.rollout(train_config.max_eval_steps, actor_operator, break_when_all_done=True, auto_reset=True)

                eval_log = {
                    "eval/reward": eval_rollout["next", "reward"].mean().item(),
                    "eval/step_count": eval_rollout["step_count"].max().item(),
                }

                if eval_str == "":
                    init_log |= eval_log

                if eval_log["eval/reward"] > best_reward:
                    best_reward = eval_log["eval/reward"]
                    logger.save_weights(actor_value_policy.state_dict(), "best")

                eval_str = (
                    f"eval/reward: {eval_log['eval/reward']:.4f} "
                    f"(init: {init_log['eval/reward']:.4f}), "
                    f"eval/step_count: {eval_log['eval/step_count']} "
                    f"(init: {init_log['eval/step_count']})"
                )

                log.update(eval_log)

                del eval_rollout
            env.set_truncation(True)
            actor_value_policy.train()

        logger.log_data(log, total_collected_frames)

        if i % train_config.save_weights_every == 0:
            logger.save_weights(actor_value_policy.state_dict(), f"weights_{total_collected_frames}")

        pbar.set_description(", ".join([eval_str, train_str]))
        pbar.update(train_config.frames_per_batch * train_config.num_robots)
        scheduler.step()

    pbar.close()
    logger.close()


if __name__ == "__main__":
    from argparse import ArgumentParser

    from omegaconf import DictConfig, OmegaConf

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, unknown = parser.parse_known_args()
    file_omegaconf = OmegaConf.load(args.config)
    cli_omegaconf = OmegaConf.from_dotlist(unknown)
    train_omegaconf = OmegaConf.merge(file_omegaconf, cli_omegaconf)
    if not isinstance(train_omegaconf, DictConfig):
        raise ValueError("Config must be a DictConfig")
    if train_omegaconf["type"].__name__ != PPOExperimentConfig.__name__:
        raise ValueError("Config must be of type PPOExperimentConfig")
    try:
        main(train_omegaconf)
    except Exception as e:
        import wandb

        wandb.finish()
        raise e
