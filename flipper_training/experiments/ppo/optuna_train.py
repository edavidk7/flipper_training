import optuna
from typing import Any
from dataclasses import dataclass
from optuna.storages import RDBStorage
from optuna.study import MaxTrialsCallback
import multiprocessing
import optunahub
from copy import deepcopy
from omegaconf import OmegaConf
import argparse
import time
import os
from flipper_training import ROOT

DB_SECRET = OmegaConf.load(ROOT / "optuna_db.yaml")


@dataclass
class OptunaConfig:
    study_name: str
    directions: list[str]
    metrics_to_optimize: list[str]
    num_trials: int
    workers_per_gpu: int
    use_gpus: list[str]
    delay: int
    train_config_overrides: dict[str, Any]
    optuna_keys: list[str]
    optuna_types: list[str]
    optuna_values: list[str]


def define_search_space(trial, keys, types, values):
    params = {}
    for key, typ, val in zip(keys, types, values):
        if typ == "float":
            params[key] = trial.suggest_float(key, val[0], val[1])
        elif typ == "int":
            params[key] = trial.suggest_int(key, val[0], val[1])
        elif typ == "categorical":
            params[key] = trial.suggest_categorical(key, val)
        elif typ == "bool":
            params[key] = trial.suggest_categorical(key, [True, False])
    return params


def objective(trial, base_config, keys, types, values, metrics_to_optimize):
    from train import train_ppo

    params = define_search_space(trial, keys, types, values)
    dotlist = [f"{k}={v}" for k, v in params.items()]
    updated_config = OmegaConf.merge(base_config, OmegaConf.from_dotlist(dotlist))
    metrics = train_ppo(updated_config)  # Returns a dict like {"eval/mean_reward": value}
    return tuple(metrics[metric] for metric in metrics_to_optimize)


def run_trial(gpu, base_config, keys, types, values, study_name, storage, total_trials, metrics_to_optimize):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    cfg = deepcopy(base_config)
    cfg["device"] = "cuda:0"
    study = optuna.load_study(study_name=study_name, storage=storage)
    callback = MaxTrialsCallback(total_trials)
    study.optimize(lambda trial: objective(trial, cfg, keys, types, values, metrics_to_optimize), callbacks=[callback])


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Optuna optimization for PPO training")
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--optuna_config", type=str, required=True)
    args = parser.parse_args()

    optuna_config = OptunaConfig(**OmegaConf.load(args.optuna_config))
    train_config = OmegaConf.load(args.train_config)
    train_config = OmegaConf.merge(train_config, optuna_config.train_config_overrides)

    # Validate argument lengths
    if len(optuna_config.optuna_keys) != len(optuna_config.optuna_types) or len(optuna_config.optuna_keys) != len(optuna_config.optuna_values):
        raise ValueError("optuna_keys, optuna_types, and optuna_values must have the same length")

    # Set up Optuna study
    storage = RDBStorage(
        f"postgresql+psycopg2://{DB_SECRET['db_user']}:{DB_SECRET['db_password']}@{DB_SECRET['db_host']}:{DB_SECRET['db_port']}/{DB_SECRET['db_name']}?sslmode=require"
    )
    study = optuna.create_study(
        study_name=optuna_config.study_name,
        storage=storage,
        directions=optuna_config.directions,
        load_if_exists=True,
        sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(),  # Automatically selects an algorithm internally
    )

    # Assign GPUs to workers
    gpu_assignments = [gpu for gpu in optuna_config.use_gpus for _ in range(optuna_config.workers_per_gpu)]

    multiprocessing.set_start_method("spawn", force=True)

    # Start worker processes
    processes = []
    for gpu in gpu_assignments:
        p = multiprocessing.Process(
            target=run_trial,
            args=(
                gpu,
                train_config,
                optuna_config.optuna_keys,
                optuna_config.optuna_types,
                optuna_config.optuna_values,
                optuna_config.study_name,
                storage,
                optuna_config.num_trials,
                optuna_config.metrics_to_optimize,
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)
        time.sleep(optuna_config.delay)

    # Wait for completion
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
