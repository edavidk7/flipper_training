from dataclasses import dataclass
from typing import Any, TypedDict, Literal
from flipper_training.configs.experiment_config import BaseExperimentConfig


class PolicyConfig(TypedDict):
    encoders_mode: Literal["shared", "separate"]
    actor_opts: dict[str, Any]
    value_opts: dict[str, Any]
    observation_encoders_opts: dict[str, Any]
    actor_lr: float
    value_lr: float


@dataclass
class PPOExperimentConfig(BaseExperimentConfig):
    epochs_per_batch: int
    frames_per_sub_batch: int
    gae_opts: dict[str, Any]
    ppo_opts: dict[str, Any]
    data_collector_opts: dict[str, Any]
    policy_config: PolicyConfig
    vecnorm_opts: dict[str, Any]
    vecnorm_on_reward: bool
    gae_compile_opts: dict[str, Any] | None = None
    ppo_compile_opts: dict[str, Any] | None = None
