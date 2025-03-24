import math
from dataclasses import dataclass
from importlib import import_module
from functools import partial
from typing import TYPE_CHECKING, Any, Type, TypedDict, Dict

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from flipper_training.observations import Observation
    from flipper_training.utils.heightmap_generators import BaseHeightmapGenerator
    from flipper_training.rl_objectives import BaseObjective
    from flipper_training.rl_rewards.rewards import Reward


def resolve_class(type: str) -> Type:
    module, class_name = type.rsplit(".", 1)
    return getattr(import_module(module), class_name)


OmegaConf.register_new_resolver("add", lambda *args: sum(args))
OmegaConf.register_new_resolver("mul", lambda *args: math.prod(args))
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("intdiv", lambda a, b: a // b)
OmegaConf.register_new_resolver("cls", resolve_class)


class ObservationConfig(TypedDict):
    observation: "Type[Observation]"
    opts: dict[str, Any]


def make_partial_observations(observations: Dict[str, ObservationConfig]):
    return {k: partial(v["observation"], **v["opts"]) for k, v in observations.items()}


@dataclass
class BaseExperimentConfig:
    type: "Type[BaseExperimentConfig]"
    name: str
    use_wandb: bool
    seed: int
    device: str
    num_robots: int
    grid_res: float
    max_coord: float
    learning_rate: float
    robot_model_opts: dict[str, Any]
    optimizer: "Type[Optimizer]"
    optimizer_opts: dict[str, Any]
    scheduler: "Type[LRScheduler]"
    scheduler_opts: dict[str, Any]
    max_grad_norm: float
    total_frames: int
    frames_per_batch: int
    heightmap_gen: "Type[BaseHeightmapGenerator]"
    heightmap_gen_opts: dict[str, Any]
    world_opts: dict[str, float]
    engine_compile_opts: dict[str, Any] | None
    engine_opts: dict[str, Any]
    observations: Dict[str, ObservationConfig]
    objective: "Type[BaseObjective]"
    objective_opts: dict[str, Any]
    reward: "Type[Reward]"
    reward_opts: dict[str, Any]
    save_weights_every: int
    max_eval_steps: int
