import torch
from typing import TYPE_CHECKING, Callable
from torchrl.envs import EnvBase
from torchrl.data import Composite, Unbounded, Bounded
from tensordict import TensorDict
from flipper_training.configs import PhysicsEngineConfig, WorldConfig, RobotModelConfig
from flipper_training.rl_objectives import BaseObjective
from flipper_training.rl_rewards.rewards import Reward
from flipper_training.vis.static_vis import plot_heightmap_3d
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer, AuxEngineInfo
from flipper_training.engine.engine import DPhysicsEngine

if TYPE_CHECKING:
    from flipper_training.observations import Observation

DEFAULT_SEED = 0


class Env(EnvBase):
    _batch_locked = True

    def __init__(
        self,
        objective: BaseObjective,
        reward: Reward,
        observations: dict[
            str,
            Callable[["Env"], "Observation"],
        ],
        world_config: WorldConfig,
        physics_config: PhysicsEngineConfig,
        robot_model_config: RobotModelConfig,
        device: torch.device | str = "cpu",
        differentiable: bool = False,
        engine_compile_opts: dict | None = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        # Misc
        self._set_seed(kwargs.get("seed", None))
        self.n_robots = self.batch_size[0]
        self.differentiable = differentiable
        # Physics configs
        self.phys_cfg = physics_config.to(device)
        self.robot_cfg = robot_model_config.to(device)
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
        if engine_compile_opts is not None:
            self._compile_engine(engine_compile_opts)
        else:
            self._needs_engine_io_buffer = False
        self._reset(reset_all=True)

    @property
    def world_config(self) -> WorldConfig:
        return self._world_config

    @world_config.setter
    def world_config(self, world_config: WorldConfig):
        assert world_config.suitable_mask is not None, "Suitable mask is required for the world configuration"
        self._world_config = world_config

    def _compile_engine(self, compile_opts: dict) -> None:
        print(f"Environment: Compiling engine with options {compile_opts}")
        self._needs_engine_io_buffer = "triton.cudagraphs" in compile_opts and self.device == torch.device(
            "cuda"
        )  # If cudagraphs is used, we need to allocate the engine's input and output memory buffers
        if self._needs_engine_io_buffer:
            print("Environment: Engine compilation requires static input and output memory buffers")
        act = self.action_spec.zeros()  # Dummy action
        state = self.start.clone()  # Dummy state
        comp_engine = torch.compile(self.engine, options=compile_opts)
        try:
            out_state, out_state_der, out_aux_info = self.engine(state, act, self.world_cfg)  # Dummy forward pass to compile the engine, record the return tensors
        except Exception as e:
            print(f"Engine compilation failed: {e}, falling back to non-compiled engine")
            return
        self.engine = comp_engine  # Replace the engine with the compiled one
        if self._needs_engine_io_buffer:
            self._engine_io_buffer = {  # Record the input and output tensors for the engine, whose memory locations are fixed in the cuda graph
                "state": state,
                "action": act,
                "out_state": out_state,
                "out_state_der": out_state_der,
                "out_aux_info": out_aux_info,
            }

    def _set_seed(self, seed: int | None):
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed or DEFAULT_SEED)
        self.rng = rng

    def _make_action_spec(self) -> Composite:
        track_low = torch.full(
            (self.n_robots, self.robot_cfg.num_driving_parts),
            -self.robot_cfg.v_max,
            device=self.device,
        )
        track_high = -track_low
        joint_low = self.robot_cfg.joint_limits[0].repeat(self.n_robots, 1)
        joint_high = self.robot_cfg.joint_limits[1].repeat(self.n_robots, 1)
        return Bounded(
            low=torch.cat([track_low, joint_low], dim=1),
            high=torch.cat([track_high, joint_high], dim=1),
            shape=(self.n_robots, 2 * self.robot_cfg.num_driving_parts),
            device=self.device,
            dtype=torch.float32,
        )

    def _make_observation_spec(self) -> Composite:
        return Composite(
            {k: obs.get_spec() for k, obs in self.observations.items()},
            device=self.device,
            shape=(self.n_robots,),
        )

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
        return Composite(
            {
                "done": bool_spec,
                "truncated": bool_spec,
                "terminated": bool_spec,
            },
            device=self.device,
            shape=(self.n_robots,),
        )

    def _get_observations(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        state_der: PhysicsStateDer,
        curr_state: PhysicsState,
        aux_info: AuxEngineInfo,
    ) -> TensorDict:
        return TensorDict(
            {k: obs(prev_state, action, state_der, curr_state, aux_info) for k, obs in self.observations.items()},
            device=self.device,
            batch_size=[self.n_robots],
        )

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
                robot_points=self.last_step_aux_info.global_robot_points[i],
            ).show()

    def _maybe_put_to_buffer(self, prev_state: PhysicsState, action: torch.Tensor) -> tuple[PhysicsState, torch.Tensor]:
        if self._needs_engine_io_buffer:
            self._engine_io_buffer["state"].copy_(prev_state)
            self._engine_io_buffer["action"].copy_(action)
            return self._engine_io_buffer["state"], self._engine_io_buffer["action"]
        return prev_state, action

    def _maybe_get_from_buffer(
        self, next_state: PhysicsState, state_der: PhysicsStateDer, aux_info: AuxEngineInfo
    ) -> tuple[PhysicsState, PhysicsStateDer, AuxEngineInfo]:
        if self._needs_engine_io_buffer:
            return self._engine_io_buffer["out_state"].clone(), self._engine_io_buffer["out_state_der"].clone(), self._engine_io_buffer["out_aux_info"].clone()
        return next_state, state_der, aux_info

    def _step_engine(self, action: torch.Tensor) -> tuple[PhysicsState, PhysicsStateDer, AuxEngineInfo, PhysicsState]:
        prev_state = self.state.clone()
        action = action.to(self.device)
        s, a = self._maybe_put_to_buffer(prev_state, action)
        if self.differentiable:
            next_state, state_der, aux_info = self.engine(s, a, self.world_cfg)
        else:
            with torch.no_grad():
                next_state, state_der, aux_info = self.engine(s, a, self.world_cfg)
        next_state, state_der, aux_info = self._maybe_get_from_buffer(next_state, state_der, aux_info)
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
        obs_td["reward"] = self.reward(prev_state, action, state_der, next_state, aux_info, reached_goal, failed, self)
        self.step_count += 1
        return obs_td
