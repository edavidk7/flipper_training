from functools import partial

from torch.optim import Adam, lr_scheduler

from flipper_training.configs import *
from flipper_training.observations import *
from flipper_training.rl_objectives import *
from flipper_training.rl_rewards.rewards import RollPitchGoal
from flipper_training.utils.heightmap_generators import *

train_config = {
    "device": "cuda",
    "num_robots": 32,  # represents the number of robots in the environment simulated in parallel
    "grid_res": 0.05,  # cm per cell
    "max_coord": 3.2,  # meters, the grid stretches from -max_coord to max_coord in x and y
    "refresh_heightmap_every": 100,  # number of steps after which the heightmap is regenerated
    "learning_rate": 1e-4,
    "optimizer": Adam,
    "scheduler": lr_scheduler.CosineAnnealingLR,
    "epochs": 10000,
    "sub_batch_size": 64,
    "max_grad_norm": 1e3,
    "engine_compile_opts": {"max-autotune": True, "triton.cudagraphs": True, "coordinate_descent_tuning": True},
    "gae_opts": {
        "gamma": 0.99,
        "lmbda": 0.95,
        "average_gae": True,
    },
    "ppo_opts": {
        "clip_epsilon": 0.2,
        "entropy_bonus": True,
        "entropy_coef": 0.01,
        "critic_coef": 1.0,
        "loss_critic_type": "smooth_l1",
    },
    "data_collector_opts": {
        "frames_per_batch": 128, # true time steps, not divided by the number of robots
        "total_frames": 1_048_576,
        "split_trajs": False,
    },
    "replay_buffer_opts": {
        "dim_extend":1,
    },
    "heightmap_gen": MultiGaussianHeightmapGenerator,
    "heightmap_gen_opts": {
        "min_gaussians": 400,
        "max_gaussians": 600,
        "min_height_fraction": 0.03,
        "max_height_fraction": 0.12,
        "min_std_fraction": 0.03,
        "max_std_fraction": 0.08,
        "min_sigma_ratio": 0.6,
    },
    "robot_model_opts": {
        "robot_type": "marv",
        "mesh_voxel_size": 0.01,
        "points_per_driving_part": 192,
        "points_per_body": 256,
        "wheel_assignment_margin": 0.02,
        "linear_track_assignment_margin": 0.05,
    },
    "world_opts": {
        "k_stiffness": 20_000,
        "k_friction": 1.0,
    },
    "training_objective": SimpleStabilizationObjective,
    "objective_opts": {
        "higher_allowed": 0.5,
        "min_dist_to_goal": 0.5,
        "max_dist_to_goal": 0.8,
        "start_drop": 0.1,
        "iteration_limit_factor": 10,
        "start_position_orientation": "towards_goal",
    },
    "reward": RollPitchGoal,
    "reward_opts": {
        "goal_reached_reward": 1000.0,
        "failed_reward": -100.0,
        "omega_weight": 1.0,
        "goal_weight": 1.0,
    },
    "observations": {
        "perception": partial(Heightmap, percep_shape=(128, 128), percep_extent=(1.0, 1.0, -1.0, -1.0)),
        "observation": partial(RobotStateVector),
    },
    "policy_opts": {
        "hidden_dim": 64,
        "value_mlp_layers": 2,
        "actor_mlp_layers": 2,
    },
}


if __name__ == "__main__":
    import pprint

    from flipper_training.utils.config_to_json import load_config, save_config

    pprint.pprint(train_config)
    save_config(train_config, "/tmp/train_config.json")
    loaded_config = load_config("/tmp/train_config.json")
    pprint.pprint(loaded_config)
