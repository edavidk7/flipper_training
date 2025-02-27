import os

os.environ["TORCH_LOGS"] = "dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

from collections import defaultdict
import deepcopy
import matplotlib.pyplot as plt
import torch
import torchrl
import torchrl.data as data
import torchrl.envs as envs
import torchrl.collectors
import torchrl.objectives
import torchrl.objectives.value
from flipper_training.configs import PhysicsEngineConfig, RobotModelConfig, WorldConfig
from flipper_training.environment.env import Env
from flipper_training.policies import make_actor_value_policy
from flipper_training.utils.environment import generate_heightmaps, make_x_y_grids
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from tqdm import tqdm


def prepare_configs(train_config):
    pass


def main(train_config):
    train_config = deepcopy.deepcopy(train_config)
    x_grid, y_grid = make_x_y_grids(train_config["max_coord"], train_config["grid_res"], train_config["num_robots"])
    heightmap_gen = train_config["heightmap_gen"](**train_config["heightmap_gen_opts"])
    robot_model = RobotModelConfig(**train_config["robot_model_opts"])
    physics_config = PhysicsEngineConfig(num_robots=train_config["num_robots"])
    z_grid, suit_mask = generate_heightmaps(x_grid, y_grid, heightmap_gen)
    world_config = WorldConfig(
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        suitable_mask=suit_mask,
        **train_config["world_opts"],
        grid_res=train_config["grid_res"],
        max_coord=train_config["max_coord"],
    )
    device = train_config["device"]
    robot_model.to(device)
    world_config.to(device)
    physics_config.to(device)
    training_objective = train_config["training_objective"](**train_config["objective_opts"])
    reward = train_config["reward"](**train_config["reward_opts"])
    base_env = Env(
        objective=training_objective,
        reward=reward,
        observations=train_config["observations"],
        world_config=world_config,
        physics_config=physics_config,
        robot_model_config=robot_model,
        device=device,
        batch_size=[train_config["num_robots"]],
        differentiable=False,
        engine_compile_opts=train_config["engine_compile_opts"] if "engine_compile_opts" in train_config else None,
    )
    torchrl.envs.utils.check_env_specs(base_env)
    env = TransformedEnv(
        base_env,
        Compose(
            ObservationNorm(in_keys=["observation"], standard_normal=True),
            ObservationNorm(in_keys=["perception"], standard_normal=True),
            StepCounter(),
        ),
    )
    for t in env.transform:
        if isinstance(t, ObservationNorm):
            t.init_stats(1000, (0, 1), 1)
    env.reset(reset_all=True)
    actor_value_policy = make_actor_value_policy(env, **train_config["policy_opts"])
    actor_value_policy.to(device)
    actor_value_policy.train()
    actor_operator = actor_value_policy.get_policy_operator()
    value_operator = actor_value_policy.get_value_operator()
    replay_buffer = data.ReplayBuffer(
        storage=data.replay_buffers.LazyTensorStorage(max_size=train_config["data_collector_opts"]["frames_per_batch"], device=device),
        sampler=data.replay_buffers.SamplerWithoutReplacement(),
        **train_config["replay_buffer_opts"],
    )
    collector = torchrl.collectors.SyncDataCollector(env, 
                                                     actor_value_policy.get_policy_operator(),
                                                     frames_per_batch=train_config["data_collector_opts"].pop("frames_per_batch") * train_config["num_robots"],
                                                     **train_config["data_collector_opts"], 
                                                     device=device)
    advantage_module = torchrl.objectives.value.GAE(**train_config["gae_opts"], value_network=value_operator, time_dim=1, device=device)
    loss_module = torchrl.objectives.ClipPPOLoss(actor_operator, value_operator, **train_config["ppo_opts"])
    optim = train_config["optimizer"](actor_value_policy.parameters(), lr=train_config["learning_rate"])
    scheduler = train_config["scheduler"](
        optim,
        T_max=train_config["data_collector_opts"]["total_frames"] // train_config["data_collector_opts"]["frames_per_batch"],
    )

    logs = defaultdict(list)
    pbar = tqdm(total=train_config["data_collector_opts"]["total_frames"])
    eval_str = ""

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector): # collected (B, T, *specs) where B is the batch size and T the number of steps
        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(train_config["epochs"]):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(tensordict_data)
            replay_buffer.extend(tensordict_data)
            for _ in range(train_config["data_collector_opts"]["frames_per_batch"] // train_config["sub_batch_size"]):
                subdata = replay_buffer.sample(train_config["sub_batch_size"])
                loss_vals = loss_module(subdata)
                loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), train_config["max_grad_norm"])
                optim.step()
                optim.zero_grad()

            logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            pbar.update(tensordict_data.numel())
            cum_reward_str = f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
            logs["step_count"].append(tensordict_data["step_count"].max().item())
            stepcount_str = f"step count (max): {logs['step_count'][-1]}"
            logs["lr"].append(optim.param_groups[0]["lr"])
            lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
            if i % 10 == 0:
                # We evaluate the policy once every 10 batches of data.
                # Evaluation is rather simple: execute the policy without exploration
                # (take the expected value of the action distribution) for a given
                # number of steps (1000, which is our ``env`` horizon).
                # The ``rollout`` method of the ``env`` can take a policy as argument:
                # it will then execute this policy at each step.
                with (
                    envs.utils.set_exploration_type(envs.utils.ExplorationType.DETERMINISTIC),
                    torch.no_grad(),
                ):
                    # execute a rollout with the trained policy
                    eval_rollout = env.rollout(1000, actor_value_policy.get_policy_operator())
                    logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                    logs["eval reward (sum)"].append(eval_rollout["next", "reward"].sum().item())
                    logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                    eval_str = (
                        f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                        f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                        f"eval step-count: {logs['eval step_count'][-1]}"
                    )
                    del eval_rollout
            pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

            # We're also using a learning rate scheduler. Like the gradient clipping,
            # this is a nice-to-have but nothing necessary for PPO to work.
            scheduler.step()

        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(logs["reward"])
        plt.title("training rewards (average)")
        plt.subplot(2, 2, 2)
        plt.plot(logs["step_count"])
        plt.title("Max step count (training)")
        plt.subplot(2, 2, 3)
        plt.plot(logs["eval reward (sum)"])
        plt.title("Return (test)")
        plt.subplot(2, 2, 4)
        plt.plot(logs["eval step_count"])
        plt.title("Max step count (test)")
        plt.show()


if __name__ == "__main__":
    from flipper_training.train_config import train_config

    main(train_config)
