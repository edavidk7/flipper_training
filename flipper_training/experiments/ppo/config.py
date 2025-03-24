from dataclasses import dataclass
from typing import Any
from flipper_training.configs.experiment_config import BaseExperimentConfig


@dataclass
class PPOExperimentConfig(BaseExperimentConfig):
    evaluate_every: int
    epochs_per_batch: int
    frames_per_sub_batch: int
    norm_init_iters: int
    gae_compile_opts: dict[str, Any] | None
    ppo_compile_opts: dict[str, Any] | None
    gae_opts: dict[str, Any]
    ppo_opts: dict[str, Any]
    data_collector_opts: dict[str, Any]
    policy_opts: dict[str, Any]
