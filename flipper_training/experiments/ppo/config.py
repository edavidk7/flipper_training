from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from flipper_training.policies import PolicyConfig

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from omegaconf import DictConfig
    from flipper_training.observations import Observation
    from flipper_training.rl_objectives import BaseObjective
    from flipper_training.rl_rewards.rewards import Reward
    from flipper_training.heightmaps import BaseHeightmapGenerator

import hashlib
from functools import partial
from typing import Type, TypedDict, List
from omegaconf import OmegaConf

import torch


class ObservationConfig(TypedDict):
    cls: "Type[Observation]"
    opts: dict[str, Any] | None


def make_partial_observations(observations: List[ObservationConfig]) -> List[partial]:
    return [partial(o["cls"], **(o["opts"] or {})) for o in observations]


def hash_omegaconf(omegaconf: "DictConfig") -> str:
    """Hashes the omegaconf config to a string."""
    s = OmegaConf.to_yaml(omegaconf, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()


@dataclass
class PPOExperimentConfig:
    name: str
    comment: str
    training_dtype: torch.dtype
    use_wandb: bool
    use_tensorboard: bool
    seed: int
    device: str
    num_robots: int
    grid_res: float
    max_coord: float
    robot_model_opts: dict[str, Any]
    optimizer: "Type[Optimizer]"
    scheduler: "Type[LRScheduler]"
    scheduler_opts: dict[str, Any]
    max_grad_norm: float
    total_frames: int
    frames_per_batch: int
    heightmap_gen: "Type[BaseHeightmapGenerator]"
    world_opts: dict[str, float]
    engine_opts: dict[str, Any]
    observations: List[ObservationConfig]
    objective: "Type[BaseObjective]"
    objective_opts: dict[str, Any]
    reward: "Type[Reward]"
    reward_opts: dict[str, Any]
    eval_and_save_every: int
    max_eval_steps: int
    epochs_per_batch: int
    frames_per_sub_batch: int
    gae_opts: dict[str, Any]
    ppo_opts: dict[str, Any]
    data_collector_opts: dict[str, Any]
    policy_config: type[PolicyConfig]
    policy_opts: dict[str, Any]
    vecnorm_opts: dict[str, Any]
    vecnorm_on_reward: bool
    optimizer_opts: dict[str, Any] = field(default_factory=dict)
    heightmap_gen_opts: dict[str, Any] = field(default_factory=dict)
    engine_compile_opts: dict[str, Any] = field(default_factory=dict)
    gae_compile_opts: dict[str, Any] = field(default_factory=dict)
    ppo_compile_opts: dict[str, Any] = field(default_factory=dict)
    # Compatibility with old configs
    type: Any = None
