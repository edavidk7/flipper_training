from typing import TYPE_CHECKING, Tuple
import torch
import traceback
from common import (
    prepare_env,
    log_from_eval_rollout,
    parse_and_load_config,
    make_transformed_env,
    # EVAL_LOG_OPT,
    # make_formatted_str_lines,
)
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer
from config import PPOExperimentConfig, OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from flipper_training.environment.env import Env
from flipper_training.utils.logutils import RunLogger, get_terminal_logger

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from tensordict import TensorDict
    from torchrl.modules import SafeSequential

TERM_LOGGER = get_terminal_logger("ppo_train")
RUN_LOGGER = None


def train_step_to_log(rollout_td: "TensorDict", loss_td: "TensorDict", grad_norm: torch.Tensor, optim: torch.optim.Optimizer) -> dict[str, float]:
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
        **{f"train/{g['name']}_lr": g["lr"] for g in optim.param_groups},
    }


def prepare_data_collection(
    env: "Env", policy: "SafeSequential", cfg: "PPOExperimentConfig", rng: torch.Generator
) -> Tuple[SyncDataCollector, TensorDictReplayBuffer]:
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=cfg.time_steps_per_batch * cfg.num_robots,  # iteration size
        total_frames=cfg.total_frames,
        **cfg.data_collector_opts,
        device=env.device,
    )
    # Replay Buffer
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            max_size=cfg.time_steps_per_batch * cfg.num_robots,  # iteration size
            ndim=1,  # This storage will mix the batch and time dimensions
            device=env.device,
            compilable=True,
        ),
        sampler=SamplerWithoutReplacement(drop_last=True, shuffle=True),
        batch_size=cfg.frames_per_sub_batch,  # sample flattened and mixed data
        dim_extend=0,
        compilable=True,
    )
    return collector, replay_buffer


def train_ppo(
    config: "DictConfig",
):
    global RUN_LOGGER
    train_config = PPOExperimentConfig(**config)
    # Logging
    RUN_LOGGER = RunLogger(
        train_config=config,
        category="ppo",
        use_wandb=train_config.use_wandb,
        use_tensorboard=train_config.use_tensorboard,
        step_metric_name="collected_frames",
    )
    print(OmegaConf.to_yaml(config, sort_keys=True))
    iteration_size = train_config.time_steps_per_batch * train_config.num_robots  # total number of frames per iteration (stable baselines convention)
    if not iteration_size % train_config.frames_per_sub_batch == 0:
        raise ValueError(f"The iteration size must be divisible by the frames per sub-batch: {iteration_size % train_config.frames_per_sub_batch=}")
    TERM_LOGGER.info(
        f"Iteration size: {iteration_size}, frames per sub-batch: {train_config.frames_per_sub_batch}, optimization steps: {iteration_size // train_config.frames_per_sub_batch}"
    )
    # RL
    env, device, rng = prepare_env(train_config, mode="train")
    policy_config = train_config.policy_config(**train_config.policy_opts)
    actor_value_wrapper, optim_groups, policy_transforms = policy_config.create(
        env=env,
        weights_path=train_config.policy_weights_path,
        device=device,
    )
    actor_operator = actor_value_wrapper.get_policy_operator()
    value_operator = actor_value_wrapper.get_value_operator()
    env, vecnorm = make_transformed_env(env, train_config, policy_transforms)
    if train_config.vecnorm_weights_path is not None:
        vecnorm.load_state_dict(torch.load(train_config.vecnorm_weights_path, map_location=device), strict=False)
    # Collector
    collector, replay_buffer = prepare_data_collection(env, actor_operator, train_config, rng)
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
        optim_groups,
        **(train_config.optimizer_opts or {}),
    )
    scheduler = train_config.scheduler(
        optim,
        **train_config.scheduler_opts,
    )
    # Loop
    pbar = tqdm(total=train_config.total_frames, desc="Training", unit="frames")
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
            for _ in range(iteration_size // train_config.frames_per_sub_batch):
                sub_batch = replay_buffer.sample()
                loss_vals = loss_module(sub_batch)
                loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                loss_value.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), train_config.max_grad_norm, error_if_nonfinite=True)
                optim.step()
                optim.zero_grad()

        log = train_step_to_log(tensordict_data, loss_vals, grad_norm, optim)
        total_collected_frames = (i + 1) * iteration_size

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
                log.update(eval_log)
                RUN_LOGGER.save_weights(actor_value_wrapper.state_dict(), f"policy_step_{total_collected_frames}")
                RUN_LOGGER.save_weights(vecnorm.state_dict(), f"vecnorm_step_{total_collected_frames}")
                del eval_rollout

        RUN_LOGGER.log_data(log, total_collected_frames)
        pbar.update(iteration_size)
        scheduler.step()

    # env.eval()
    # actor_operator.eval()
    # value_operator.eval()
    # with (
    #     set_exploration_type(ExplorationType.DETERMINISTIC),
    #     torch.inference_mode(),
    # ):
    #     eval_rollout = env.rollout(train_config.max_eval_steps, actor_operator, break_when_all_done=True, auto_reset=True)
    #     for _ in range(train_config.eval_repeats_after_training - 1):
    #         eval_rollout += env.rollout(train_config.max_eval_steps, actor_operator, break_when_all_done=True, auto_reset=True)
    #     avg_eval_rollout = eval_rollout / train_config.eval_repeats_after_training
    #     avg_eval_log = log_from_eval_rollout(avg_eval_rollout)
    #     log.update(avg_eval_log)
    #     RUN_LOGGER.save_weights(actor_value_wrapper.state_dict(), "policy_final")
    #     RUN_LOGGER.save_weights(vecnorm.state_dict(), "vecnorm_final")
    #     del eval_rollout
    #     del avg_eval_rollout
    # RUN_LOGGER.log_data(log, train_config.total_frames)
    # print(make_formatted_str_lines(avg_eval_log, EVAL_LOG_OPT))
    # return avg_eval_log


if __name__ == "__main__":
    try:
        train_ppo(parse_and_load_config())
    except KeyboardInterrupt:
        TERM_LOGGER.info("Training interrupted by user.")
    except Exception as e:
        TERM_LOGGER.error(f"Training failed with error: {e}")
        traceback.print_exception(e)
    if RUN_LOGGER is not None:
        RUN_LOGGER.close()
