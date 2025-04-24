from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, Type
import hashlib
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from flipper_training.rl_objectives import BaseObjective
    from flipper_training.rl_rewards.rewards import Reward
    from flipper_training.heightmaps import BaseHeightmapGenerator


def hash_omegaconf(omegaconf: "DictConfig") -> str:
    """Hashes the omegaconf config to a string."""
    s = OmegaConf.to_yaml(omegaconf, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()


@dataclass
class MPPIExperimentConfig:
    name: str
    comment: str
    seed: int
    device: str
    grid_res: float
    max_coord: float
    robot_model_opts: dict[str, Any]
    heightmap_gen: "Type[BaseHeightmapGenerator]"
    world_opts: dict[str, float]
    engine_opts: dict[str, Any]
    objective: "Type[BaseObjective]"
    objective_opts: dict[str, Any]
    reward: "Type[Reward]"
    reward_opts: dict[str, Any]
    max_eval_steps: int
    gamma: float
    temperature: float
    planning_horizon: int
    optim_steps: int
    num_candidates: int
    top_k: int
    heightmap_gen_opts: dict[str, Any] = field(default_factory=dict)
    engine_compile_opts: dict[str, Any] = field(default_factory=dict)
