import math
import hashlib
from ast import literal_eval
from abc import ABC
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict, Type, TypedDict

import torch
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from omegaconf import DictConfig
    from flipper_training.observations import Observation
    from flipper_training.rl_objectives import BaseObjective
    from flipper_training.rl_rewards.rewards import Reward
    from flipper_training.heightmaps import BaseHeightmapGenerator


def resolve_class(typename: str) -> Type:
    module, class_name = typename.rsplit(".", 1)
    return getattr(import_module(module), class_name)


OmegaConf.register_new_resolver("add", lambda *args: sum(args))
OmegaConf.register_new_resolver("mul", lambda *args: math.prod(args))
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("intdiv", lambda a, b: a // b)
OmegaConf.register_new_resolver("cls", resolve_class)
OmegaConf.register_new_resolver("lmbda", lambda s: literal_eval(s))  # evaluate a lambda string
OmegaConf.register_new_resolver("dtype", lambda s: getattr(torch, s))  # get a torch dtype
OmegaConf.register_new_resolver("tensor", lambda s: torch.tensor(s))


class ObservationConfig(TypedDict):
    observation: "Type[Observation]"
    opts: dict[str, Any]


def make_partial_observations(observations: Dict[str, ObservationConfig]):
    return {k: partial(v["observation"], **v.get("opts", {})) for k, v in observations.items()}


def hash_omegaconf(omegaconf: "DictConfig") -> str:
    """Hashes the omegaconf config to a string."""
    s = OmegaConf.to_yaml(omegaconf, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()


@dataclass(kw_only=True)
class BaseExperimentConfig(ABC):
    type: "Type[BaseExperimentConfig]"
    name: str
    comment: str
    training_dtype: torch.dtype
    use_wandb: bool
    seed: int
    device: str
    num_robots: int
    grid_res: float
    max_coord: float
    robot_model_opts: dict[str, Any]
    optimizer: "Type[Optimizer]"
    optimizer_opts: dict[str, Any]
    scheduler: "Type[LRScheduler]"
    scheduler_opts: dict[str, Any]
    max_grad_norm: float
    total_frames: int
    frames_per_batch: int
    heightmap_gen: "Type[BaseHeightmapGenerator]"
    heightmap_gen_opts: dict[str, Any] | None = None
    world_opts: dict[str, float]
    engine_compile_opts: dict[str, Any] | None = None
    engine_opts: dict[str, Any]
    observations: Dict[str, ObservationConfig]
    objective: "Type[BaseObjective]"
    objective_opts: dict[str, Any]
    reward: "Type[Reward]"
    reward_opts: dict[str, Any]
    eval_and_save_every: int
    max_eval_steps: int
