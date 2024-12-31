import torch
from dataclasses import dataclass
from typing import Literal, Callable, Optional
from torchrl.envs import EnvBase
from torchrl.data import Composite, Bounded, Unbounded
from tensordict import TensorDict
from flipper_training.environment.base_env import BaseDPhysicsEnv
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.configs import *


@dataclass
class TorchRLEnvConfig(EnvConfig):
    """
    Configuration for the TorchRL environment, derived from the base environment configuration

    Extra attributes:
    reward_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] - reward function to use, defaults to L2 norm of angular velocity in radians (roll, pitch)
    control_type: Literal["per-track","lon_ang"] = "lon_ang" - control type for the environment, either per-track or lon_ang
    dist_to_goal: float = 0.1 - distance to goal for termination condition in meters
    """
    reward_func: Callable[[PhysicsState, PhysicsStateDer, torch.Tensor, PhysicsState], torch.Tensor] = lambda x, xd, u, xn: x.omega[..., :2].norm(dim=-1)
    control_type: Literal["per-track", "lon_ang"] = "lon_ang"
    min_dist_to_goal: float = 1  # minimum distance from start to goal in meters
    dist_to_goal: float = 0.1  # distance to goal for termination condition in meters
    max_iters: int = 1000  # maximum number of iterations for the environment before termination


class TorchRLEnv(EnvBase):
    _batch_locked = True

    def __init__(self, env_config: TorchRLEnvConfig,
                 physics_config: PhysicsEngineConfig,
                 robot_model_config: RobotModelConfig,
                 device: torch.device | str):
        super().__init__(device=device)
        # action spec
        self.n_robots = physics_config.num_robots
        self.action_spec = self._make_action_spec(n_robots, robot_model_config, env_config.control_type)
        # observation spec
        dummy_state = PhysicsState.dummy(n_robots, robot_model_config.robot_points)
        self.observation_spec = self._make_observation_spec(n_robots, dummy_state, env_config)
        # reward
        self.reward_spec = self._make_reward_spec(n_robots)
        # done
        self.done_spec = self._make_done_spec(n_robots)
        self._env = BaseDPhysicsEnv(env_config, physics_config, robot_model_config, device)

    def _make_action_spec(self, n_robots: int, robot_model: RobotModelConfig, control_type: Literal["per-track", "lon_ang"]) -> Bounded:
        match control_type:
            case "per-track":
                track_low = torch.full((n_robots, robot_model.num_joints), -robot_model.vel_max).repeat(n_robots, 1)
                track_high = -track_low
            case "lon_ang":
                track_low = torch.tensor([-robot_model.vel_max, -robot_model.omega_max]).repeat(n_robots, 1)
                track_high = -track_low
            case _:
                raise ValueError(f"Invalid control type: {control_type}")
        joint_low = robot_model.joint_limits[0].repeat(n_robots, 1)
        joint_high = robot_model.joint_limits[1].repeat(n_robots, 1)
        return Bounded(
            low=torch.cat([track_low, joint_low], dim=-1),
            high=torch.cat([track_high, joint_high], dim=-1),
            shape=(n_robots, track_low.shape[1] + robot_model.num_joints),
            device=self.device,
            dtype=torch.float32,
        )

    def _make_observation_spec(self, num_robots: int, dummy_state: PhysicsState, env_config: TorchRLEnvConfig) -> Composite:
        match env_config.percep_type:
            case "heightmap":
                percep_shape = (num_robots, 1, env_config.percep_dim, env_config.percep_dim)
            case "pointcloud":
                percep_shape = (num_robots, env_config.percep_dim**2, 3)
            case _:
                raise ValueError(f"Invalid perception type: {env_config.percep_type}")
        return Composite({
            "perception": Unbounded(  # perception data
                shape=percep_shape,
                dtype=torch.float32,
                device=self.device
            ),
            "velocity": Unbounded(  # velocity of the robot
                shape=dummy_state.xd.shape,
                dtype=torch.float32,
                device=self.device
            ),
            "rotation": Unbounded(  # rotation matrix of the robot
                shape=dummy_state.R.shape,
                dtype=torch.float32,
                device=self.device
            ),
            "omega": Unbounded(  # angular velocity of the robot
                shape=dummy_state.omega.shape,
                dtype=torch.float32,
                device=self.device
            ),
            "thetas": Unbounded(  # joint angles of the robot
                shape=dummy_state.thetas.shape,
                dtype=torch.float32,
                device=self.device,
            ),
            "direction": Unbounded(  # direction to the goal
                shape=(num_robots, 3),
                dtype=torch.float32,
                device=self.device,
            )
        })

    def _make_reward_spec(self, num_robots: int) -> Unbounded:
        return Unbounded(
            shape=(num_robots,),
            dtype=torch.float32,
            device=self.device
        )

    def _make_done_spec(self, num_robots: int) -> Bounded:
        return Bounded(
            low=0,
            high=1,
            shape=(num_robots,),
            dtype=torch.bool,
            device=self.device
        )
    
    def _reset_and_generate_goals(self):
        raise NotImplementedError

    def init(self, world_config: WorldConfig, seed: Optional[int] = None):
        """
        Initialize the environment with the given seed and world configuration
        """
        self._set_seed(seed)
        self._env.init(**kwargs)

    def compile(self, **kwargs):
        """
        Compile the environment with the given configuration
        """
        super().compile(**kwargs)
        self._env.compile(**kwargs)

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _reset(self, tensordict=None, **kwargs):
        state, percep_data = self._env.reset(**kwargs)
        self.goals = torch.zeros((self.n_robots, 3), device=device)
        self.done = torch.zeros((self.n_robots,), device=device, dtype=torch.bool)
        self.terminated = torch.zeros((self.n_robots,), device=device, dtype=torch.bool)
        self.step_count = torch.zeros((self.n_robots,), device=device, dtype=torch.int32)
        tensordict = TensorDict({
            "perception": percep_data.to(self.device),
            "reward": torch.zeros(self.reward_spec.shape, device=self.device),
            "done": torch.zeros(self.done_spec.shape, device=self.device)
        }, batch_size=[self._env.phys_cfg.num_robots])
        
        return tensordict

    def _step(self, tensordict):
        action = tensordict.get("action").to(self.device)
        next_state, state_der, aux_info, percep_data = self._env.step(action, sample_percep=True)
        reward = self.compute_reward(next_state, action)
        done = self.check_done(next_state)
        next_tensordict = TensorDict({
            "perception": percep_data.to(self.device),
            "reward": reward,
            "done": done
        }, batch_size=[self._env.phys_cfg.num_robots])
        return next_tensordict

    def compute_reward(self, state, action):
        # implement your reward function
        return torch.zeros(self.reward_spec.shape, device=self.device)

    def check_done(self, state):
        # implement your termination condition
        return torch.zeros(self.done_spec.shape, device=self.device, dtype=torch.bool)
