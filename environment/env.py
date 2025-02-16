import torch
from typing import Any, TYPE_CHECKING, Callable
from torchrl.envs import EnvBase
from torchrl.data import Composite, Unbounded, Bounded
from tensordict import TensorDict
from flipper_training.engine.engine_state import PhysicsState
from flipper_training.configs import *
from flipper_training.rl_objectives import *
from flipper_training.rl_rewards.rewards import *
from flipper_training.utils.heightmap_generators import *
from flipper_training.vis.static_vis import plot_heightmap_3d
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer, AuxEngineInfo
from flipper_training.engine.engine import DPhysicsEngine
from flipper_training.configs import PhysicsEngineConfig, WorldConfig, EnvConfig, RobotModelConfig

if TYPE_CHECKING:
    from flipper_training.observations import Observation

DEFAULT_SEED = 0


class Env(EnvBase):
    _batch_locked = True

    def __init__(self,
                 objective: BaseObjective,
                 reward: Reward,
                 observations: dict[str, Callable[["Env"], "Observation"],],
                 env_config: EnvConfig,
                 world_config: WorldConfig,
                 physics_config: PhysicsEngineConfig,
                 robot_model_config: RobotModelConfig,
                 device: torch.device | str = "cpu",
                 **kwargs):
        super().__init__(device=device, **kwargs)
        # Misc
        self._set_seed(kwargs.get("seed", None))
        self.n_robots = self.batch_size[0]
        # Physics configs
        self.phys_cfg = physics_config.to(device)
        self.robot_cfg = robot_model_config.to(device)
        self.env_cfg = env_config.to(device)
        self.world_cfg = world_config.to(device)
        # Engine
        self.engine = DPhysicsEngine(physics_config, robot_model_config, device)
        # Engine state variables
        self.state: PhysicsState | None = None
        self.last_step_aux_info: AuxEngineInfo | None = None
        self.last_step_der: PhysicsStateDer | None = None
        # RL components
        self.observations = {k: o(self) for k, o in observations.items()}
        self.objective = objective
        self.reward = reward
        # RL State variables
        self.done = torch.ones((self.n_robots,), device=self.device, dtype=torch.bool)
        self.step_count = torch.zeros((self.n_robots,), device=self.device, dtype=torch.int32)
        self.iteration_limits = torch.zeros((self.n_robots,), device=self.device, dtype=torch.int32)
        self.start = PhysicsState.dummy(batch_size=self.n_robots, robot_model=robot_model_config, device=self.device)
        self.goal = PhysicsState.dummy(batch_size=self.n_robots, robot_model=robot_model_config, device=self.device)
        # Specs
        self.action_spec = self._make_action_spec()
        self.observation_spec = self._make_observation_spec()
        self.reward_spec = self._make_reward_spec()
        self.done_spec = self._make_done_spec()
        # Reset the environment
        self._reset(reset_all=True)

    @property
    def world_config(self) -> WorldConfig:
        return self._world_config

    @world_config.setter
    def world_config(self, world_config: WorldConfig):
        assert world_config.suitable_mask is not None, "Suitable mask is required for the world configuration"
        self._world_config = world_config

    def _set_seed(self, seed: int | None):
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed or DEFAULT_SEED)
        self.rng = rng

    def _make_action_spec(self) -> Composite:
        match self.env_cfg.control_type:
            case "per-track":
                track_low = torch.full((self.n_robots, self.robot_cfg.num_joints), -self.robot_cfg.vel_max, device=self.device)
                track_high = -track_low
            case "lon_ang":
                track_low = torch.tensor([-self.robot_cfg.vel_max, -self.robot_cfg.omega_max], device=self.device).repeat(1, self.n_robots)
                track_high = -track_low
            case _:
                raise ValueError(f"Invalid control type: {self.env_cfg.control_type}")
        joint_low = self.robot_cfg.joint_limits[0].repeat(self.n_robots, 1)
        joint_high = self.robot_cfg.joint_limits[1].repeat(self.n_robots, 1)
        return Bounded(
            low=torch.cat([track_low, joint_low], dim=1),
            high=torch.cat([track_high, joint_high], dim=1),
            shape=(self.n_robots, self.robot_cfg.num_joints + track_low.shape[1]),
            device=self.device,
            dtype=torch.float32,
        )

    def _make_observation_spec(self) -> Composite:
        return Composite({
            k: obs.get_spec() for k, obs in self.observations.items()
        }, device=self.device, shape=(self.n_robots,))

    def _make_reward_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.n_robots, 1),
            dtype=torch.float32,
            device=self.device,
        )

    def _make_done_spec(self) -> Bounded:
        bool_spec = Bounded(
            low=0,
            high=1,
            shape=(self.n_robots, 1),
            dtype=torch.bool,
            device=self.device,
        )
        return Composite({
            "done": bool_spec,
            "truncated": bool_spec,
            "terminated": bool_spec,
        }, device=self.device, shape=(self.n_robots,))

    def _get_observations(self, prev_state: PhysicsState, action: torch.Tensor, state_der: PhysicsStateDer, curr_state: PhysicsState, aux_info: AuxEngineInfo) -> TensorDict:
        return TensorDict({
            k: obs(prev_state, action, state_der, curr_state, aux_info) for k, obs in self.observations.items()
        }, device=self.device, batch_size=[self.n_robots])

    def visualize_curr_state(self):
        """
        Visualize the current state of the environment
        """
        for i in range(self.n_robots):
            plot_heightmap_3d(
                self.world_cfg.x_grid[i],
                self.world_cfg.y_grid[i],
                self.world_cfg.z_grid[i],
                start=self.start.x[i],
                end=self.goal.x[i],
                robot_points=self.last_step_aux_info.global_robot_points[i]
            ).show()

    def _compute_full_controls(self, controls: torch.Tensor) -> torch.Tensor:
        """
        Compute the full controls from the input controls.
        """
        if controls.shape[1] == 2 * self.robot_cfg.num_joints:
            full_controls = controls.to(self.device)
        elif controls.shape[1] == 2 + self.robot_cfg.num_joints:
            full_controls = torch.zeros((self.phys_cfg.num_robots, 2 * self.robot_cfg.num_joints), device=self.device)
            full_controls[..., :self.robot_cfg.num_joints] = self.robot_cfg.get_controls(controls[:, :2]).unsqueeze(0)
            full_controls[..., self.robot_cfg.num_joints:] = controls[:, 2:]
        else:
            raise ValueError(f"Invalid shape for controls: {controls.shape}. Expected {(self.phys_cfg.num_robots, 2 + self.robot_cfg.num_joints)} or {(self.phys_cfg.num_robots, 2 * self.robot_cfg.num_joints)}.")
        return full_controls

    def _step_engine(self, action: torch.Tensor) -> tuple[PhysicsState, PhysicsStateDer, AuxEngineInfo, PhysicsState]:
        prev_state = self.state.clone()
        full_controls = self._compute_full_controls(action)
        if self.env_cfg.differentiable:
            next_state, state_der, aux_info = self.engine(self.state, full_controls, self.world_cfg)
        else:
            with torch.no_grad():
                next_state, state_der, aux_info = self.engine(self.state, full_controls, self.world_cfg)
        return prev_state, state_der, aux_info, next_state

    def _reset(self, tensordict=None, **kwargs) -> TensorDict:
        self.world_cfg = kwargs.get("world_config", None) or self.world_cfg
        reset_all = kwargs.get("reset_all", False)
        # Reset only the robots that are not done or terminated
        reset_mask = self.done | reset_all
        skip_mask = ~reset_mask
        # Generate start and goal states, iteration limits for done/terminated robots
        new_start, new_goal = self.objective.generate_start_goal_states(self.world_cfg, self.robot_cfg, self.rng, skip_mask=skip_mask.unsqueeze(-1))
        new_iteration_limits = self.objective.compute_iteration_limits(self.start, self.goal, self.robot_cfg, self.phys_cfg.dt)
        # Update the state variables
        self.start[reset_mask] = new_start[reset_mask]
        self.goal[reset_mask] = new_goal[reset_mask]
        self.iteration_limits[reset_mask] = new_iteration_limits[reset_mask]
        self.done[reset_mask] = False
        self.step_count[reset_mask] = 0
        # Reset the environment
        if self.state is None:
            self.state = self.start.clone()
        else:
            self.state[reset_mask] = self.start[reset_mask].clone()
        # Step the engine to get the first observation
        zeros_action = self.action_spec.zeros()
        prev_state, state_der, aux_info, next_state = self._step_engine(zeros_action)
        # Set the state variables
        self.state = next_state
        self.last_step_aux_info = aux_info
        self.last_step_der = state_der
        # Output tensordict
        obs_td = self._get_observations(prev_state, zeros_action, state_der, next_state, aux_info)
        obs_td["action"] = zeros_action
        return obs_td

    def _step(self, tensordict) -> TensorDict:
        action = tensordict.get("action").to(self.device)
        # Step the engine
        prev_state, state_der, aux_info, next_state = self._step_engine(action)
        self.state = next_state
        self.last_step_aux_info = aux_info
        self.last_step_der = state_der
        # Check if the robots have reached the goal or terminated
        reached_goal = self.objective.check_reached_goal(self.state, self.goal)
        failed = self.objective.check_terminated_wrong(self.state, self.goal)
        truncated = self.step_count >= self.iteration_limits
        # Output tensordict
        obs_td = self._get_observations(prev_state, action, state_der, next_state, aux_info)
        obs_td["terminated"] = failed | reached_goal
        obs_td["truncated"] = truncated
        obs_td["done"] = failed | reached_goal | truncated
        obs_td["reward"] = self.reward(prev_state,
                                       action,
                                       state_der,
                                       next_state,
                                       aux_info,
                                       reached_goal,
                                       failed,
                                       self)
        self.step_count += 1
        return obs_td
