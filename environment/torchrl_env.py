import torch
from dataclasses import dataclass, field
from typing import Literal, Optional
from torchrl.envs import EnvBase
from torchrl.data import Composite, Unbounded, Bounded
from tensordict import TensorDict
from flipper_training.environment.base_env import BaseDPhysicsEnv
from flipper_training.engine.engine_state import PhysicsState
from flipper_training.configs import *
from flipper_training.rl_objectives import *
from flipper_training.utils.heightmap_generators import *
from flipper_training.vis.static_vis import plot_heightmap_3d

DEFAULT_SEED = 0


@dataclass
class TorchRLEnvConfig(EnvConfig):
    """
    Configuration for the TorchRL environment, derived from the base environment configuration

    Extra attributes:
        objective: BaseObjective = SimpleDrivingObjective() - the objective to use for the environment
        control_type: Literal["per-track","lon_ang"] = "lon_ang" - control type for the environment, either per-track or lon_ang
    """
    control_type: Literal["per-track", "lon_ang"] = "lon_ang"


class TorchRLEnv(EnvBase):
    _batch_locked = True

    def __init__(self,
                 objective,
                 env_config: TorchRLEnvConfig,
                 world_config: WorldConfig,
                 physics_config: PhysicsEngineConfig,
                 robot_model_config: RobotModelConfig,
                 device: torch.device | str = "cpu",
                 **kwargs):
        super().__init__(device=device, **kwargs)
        self._env = BaseDPhysicsEnv(env_config, physics_config, robot_model_config, device)
        # action spec
        self.n_robots = physics_config.num_robots
        self.action_spec = self._make_action_spec(robot_model_config, env_config.control_type)
        # observation spec
        self.observation_spec = self._make_observation_spec(env_config)
        # reward
        self.reward_spec = self._make_reward_spec()
        # done
        self.done_spec = self._make_done_spec()
        self.config = env_config
        self.objective = objective
        self.world_config = world_config
        self.rng = torch.manual_seed(DEFAULT_SEED)

    @property
    def world_config(self) -> WorldConfig:
        return self._world_config

    @world_config.setter
    def world_config(self, world_config: WorldConfig):
        assert world_config.suitable_mask is not None, "Suitable mask is required for the world configuration"
        self._world_config = world_config

    def _make_action_spec(self, robot_model: RobotModelConfig, control_type: Literal["per-track", "lon_ang"]) -> Composite:
        match control_type:
            case "per-track":
                track_low = torch.full((self.n_robots, robot_model.num_joints), -robot_model.vel_max).repeat(self.n_robots, 1)
                track_high = -track_low
            case "lon_ang":
                track_low = torch.tensor([-robot_model.vel_max, -robot_model.omega_max]).repeat(self.n_robots, 1)
                track_high = -track_low
            case _:
                raise ValueError(f"Invalid control type: {control_type}")
        joint_low = robot_model.joint_limits[0].repeat(self.n_robots, 1)
        joint_high = robot_model.joint_limits[1].repeat(self.n_robots, 1)
        return Bounded(
            low=torch.cat([track_low, joint_low], dim=-1),
            high=torch.cat([track_high, joint_high], dim=-1),
            shape=(self.n_robots, robot_model.num_joints + track_low.shape[1]),
            device=self.device,
            dtype=torch.float32,
        )

    def _make_observation_spec(self, env_config: TorchRLEnvConfig) -> Composite:
        dummy_state = PhysicsState.dummy(batch_size=self.n_robots, robot_model=self._env.robot_cfg)
        match env_config.percep_type:
            case "heightmap":
                percep_shape = (self.n_robots, 1, env_config.percep_dim, env_config.percep_dim)
            case "pointcloud":
                percep_shape = (self.n_robots, env_config.percep_dim**2, 3)
            case _:
                raise ValueError(f"Invalid perception type: {env_config.percep_type}")
        return Composite({
            "perception": Unbounded(  # perception data
                shape=percep_shape,
                dtype=torch.float32,
                device=self.device,
            ),
            "velocity": Unbounded(  # velocity of the robot
                shape=dummy_state.xd.shape,
                dtype=torch.float32,
                device=self.device,
            ),
            "rotation": Unbounded(  # rotation matrix of the robot
                shape=dummy_state.R.shape,
                dtype=torch.float32,
                device=self.device,
            ),
            "omega": Unbounded(  # angular velocity of the robot
                shape=dummy_state.omega.shape,
                dtype=torch.float32,
                device=self.device,
            ),
            "thetas": Unbounded(  # joint angles of the robot
                shape=dummy_state.thetas.shape,
                dtype=torch.float32,
                device=self.device,
            ),
            "goal_vec": Unbounded(  # goal vector in the robot's frame
                shape=(self.n_robots, 3),
                dtype=torch.float32,
                device=self.device,
            ),

        }, shape=(self.n_robots,))

    def _make_reward_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.n_robots,),
            dtype=torch.float32,
            device=self.device,
        )

    def _make_done_spec(self) -> Bounded:
        return Bounded(
            low=0,
            high=1,
            shape=(self.n_robots,),
            dtype=torch.bool,
            device=self.device,
        )

    def compile(self, **kwargs):
        """
        Compile the environment with the given configuration
        """
        super().compile(**kwargs)
        self._env.compile(**kwargs)

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def visualize_curr_state(self):
        """
        Visualize the current state of the environment
        """
        if self._env.last_step_misc:
            aux = self._env.last_step_misc["aux_info"]
        else:
            aux = None
        for i in range(self.n_robots):
            kwargs = {
                "start": self.start.x[i],
                "end": self.goal.x[i],
            }
            if aux is not None:
                kwargs["robot_points"] = aux.global_robot_points[i]
            plot_heightmap_3d(self.world_config.x_grid[i], self.world_config.y_grid[i], self.world_config.z_grid[i], **kwargs).show()

    def _state_ret_to_obs_tensordict(self, state: PhysicsState, percep_data: torch.Tensor) -> TensorDict:
        goal_vecs = torch.bmm((self.goal.x - state.x).unsqueeze(1), state.R).squeeze(dim=1)  # transposed matmul with rotation means transforming the goal vector to the robot's frame
        return TensorDict({
            "perception": percep_data,
            "velocity": state.xd,
            "rotation": state.R,
            "omega": state.omega,
            "thetas": state.thetas,
            "goal_vec": goal_vecs
        }, batch_size=[self.n_robots])

    def _reset(self, tensordict=None, **kwargs):
        self.world_config = kwargs.get("world_config", None) or self.world_config
        # Generate start and goal states, iteration limits
        self.start, self.goal = self.objective.generate_start_goal_states(self.world_config, self._env.robot_cfg, self.rng)
        self.iteration_limits = self.objective.compute_iteration_limits(self.start, self.goal, self._env.robot_cfg, self._env.phys_cfg.dt)
        # Reset indicators
        self.done_or_terminated = torch.zeros((self.n_robots,), device=self.device, dtype=torch.bool)
        self.step_count = torch.zeros((self.n_robots,), device=self.device, dtype=torch.int32)
        # Reset the environment
        state, percep_data = self._env.reset(self.world_config, state=self.start)
        obs_td = self._state_ret_to_obs_tensordict(state, percep_data)
        obs_td["action"] = self.action_spec.ones()
        return obs_td

    def _step(self, tensordict):
        self.step_count += 1
        action = tensordict.get("action").to(self.device)
        prev_state = self._env.state
        next_state, state_der, aux_info, percep_data = self._env.step(action, sample_percep=True)
        reward = self.objective.compute_reward(prev_state, action, state_der, next_state, self.goal)
        self.done_or_terminated |= self.objective.check_reached_goal_or_terminate(next_state, self.goal)
        self.done_or_terminated |= self.step_count >= self.iteration_limits
        obs_td = self._state_ret_to_obs_tensordict(next_state, percep_data)
        obs_td["reward"] = reward
        obs_td["done"] = self.done_or_terminated
        return obs_td
