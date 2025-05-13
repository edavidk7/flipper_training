from flipper_training import ROOT
from pathlib import Path
from omegaconf import OmegaConf
from flipper_training.experiments.ppo.eval import get_eval_rollout, log_from_eval_rollout, PPOExperimentConfig
from flipper_training.experiments.ppo.common import parse_and_load_config
from flipper_training.utils.logutils import get_terminal_logger
import contextlib
from tqdm import tqdm
import pickle
import os
import sys
import torch

CONFIGS_DIR = ROOT / "cross_eval_configs"
RESULTS_DIR = ROOT / "cross_eval_results"
with open(CONFIGS_DIR / "cross_eval_seeds.txt", "r") as f:
    SEEDS = [int(line.strip()) for line in f.readlines()]

NUM_ENVS_PER_EVAL = 16  # robots in parallel
MAX_EVAL_STEPS = 1000
LOGGER = get_terminal_logger("cross_eval")


def cross_eval(dict_config) -> None:
    train_config = PPOExperimentConfig(**dict_config)
    assert train_config.policy_weights_path is not None, "Policy weights path must be specified in the config"
    assert train_config.vecnorm_weights_path is not None, "Vecnorm weights path must be specified in the config"
    if not isinstance(train_config.policy_weights_path, Path):
        train_config.policy_weights_path = Path(train_config.policy_weights_path)
    if not isinstance(train_config.vecnorm_weights_path, Path):
        train_config.vecnorm_weights_path = Path(train_config.vecnorm_weights_path)
    train_config.num_robots = NUM_ENVS_PER_EVAL
    train_config.max_eval_steps = MAX_EVAL_STEPS
    train_config.objective_opts["cache_size"] = 10
    devnull_handle = open(os.devnull, "w")
    FULL_RESULT_DIR = RESULTS_DIR / f"{train_config.name}_{train_config.policy_weights_path.stem}"
    FULL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if not (FULL_RESULT_DIR / "training.pkl").exists():
        tqdm.write("Training environment")
        results = []
        for seed in tqdm(SEEDS, desc="Training environment"):
            tqdm.write(f"Evaluating seed {seed}")
            train_config.seed = seed
            with contextlib.redirect_stdout(devnull_handle):
                with contextlib.redirect_stderr(devnull_handle):
                    env, eval_rollout = get_eval_rollout(train_config)

            log = log_from_eval_rollout(eval_rollout)
            del env, eval_rollout
            results.append(log)
        with open(FULL_RESULT_DIR / "training.pkl", "wb") as f:
            pickle.dump(results, f)

    for test_config_path in CONFIGS_DIR.glob("*.yaml"):
        if (FULL_RESULT_DIR / test_config_path.stem).exists():
            print(f"Skipping {test_config_path.stem} as it already exists")
            continue
        print(f"Evaluating {test_config_path.stem}")
        test_config = OmegaConf.load(test_config_path)
        train_config.objective_opts = test_config["objective_opts"]
        train_config.heightmap_gen_opts = test_config["heightmap_gen_opts"]
        train_config.objective = test_config["objective"]
        train_config.heightmap_gen = test_config["heightmap_gen"]
        train_config.reward = test_config["reward"]
        train_config.reward_opts = test_config["reward_opts"]
        results = []
        for seed in tqdm(SEEDS, desc=f"Evaluating {test_config_path.stem}"):
            train_config.seed = seed
            with contextlib.redirect_stdout(devnull_handle):
                with contextlib.redirect_stderr(devnull_handle):
                    env, eval_rollout = get_eval_rollout(train_config)
            log = log_from_eval_rollout(eval_rollout)
            del env, eval_rollout
            results.append(log)

        with open(FULL_RESULT_DIR / f"{test_config_path.stem}.pkl", "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    cross_eval(parse_and_load_config())
