import time
from typing import TYPE_CHECKING, Callable

import torch
import logging
from flipper_training.configs import PhysicsEngineConfig, RobotModelConfig, WorldConfig
from flipper_training.engine.engine import DPhysicsEngine
from flipper_training.engine.engine_state import AuxEngineInfo, PhysicsState, PhysicsStateDer
from flipper_training.rl_objectives import BaseObjective
from flipper_training.rl_rewards.rewards import Reward
from flipper_training.vis.static_vis import plot_heightmap_3d
from tensordict import TensorDict, assert_allclose_td
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

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
        out_dtype: torch.dtype = torch.float32,
        differentiable: bool = False,
        engine_compile_opts: dict | None = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        # Misc
        self._set_seed(kwargs.get("seed", None))
        self.n_robots = self.batch_size[0]
        self.differentiable = differentiable
        self.out_dtype = out_dtype
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
            self._compile_engine(**engine_compile_opts)
        self.reset(reset_all=True)

    def _compile_engine(self, correctness_iters: int = 100, benchmark_iters: int = 1000, atol: float = 1e-3, rtol: float = 1e-3, **kwargs) -> None:
        logging.info(f"Environment: Compiling engine with options {kwargs}")
        act = self.action_spec.rand()
        state = self.start.clone()  # Dummy state
        # Capture the return tensors from the engine for correctness
        states, state_ders, aux_infos = [], [], []
        for _ in range(correctness_iters):
            state, state_der, aux_info = self.engine(state, act, self.world_cfg)
            states.append(state)
            state_ders.append(state_der)
            aux_infos.append(aux_info)
        # Compile the engine
        self.engine = torch.compile(self.engine, options=kwargs)
        self.engine(state, act, self.world_cfg)  # Dummy forward pass to compile the engine, record the return tensors
        state = self.start.clone()  # Reset the state
        logging.info(f"Engine compiled successfully, testing correctness with {atol=}, {rtol=}")
        for _ in range(correctness_iters):
            next_state, state_der, aux_info = self.engine(state, act, self.world_cfg)
            # Check correctness
            assert_allclose_td(next_state, states.pop(0), atol=atol, rtol=rtol, msg="compiled engine produced incorrect state")
            assert_allclose_td(state_der, state_ders.pop(0), atol=atol, rtol=rtol, msg="compiled engine produced incorrect state der")
            assert_allclose_td(aux_info, aux_infos.pop(0), atol=atol, rtol=rtol, msg="compiled engine produced incorrect aux info")
            state = next_state.clone()
        logging.info("Compiled engine passed correctness test")
        # Benchmark the compiled engine
        start_time = time.perf_counter_ns()
        for _ in range(benchmark_iters):
            _ = self.engine(state, act, self.world_cfg)
        end_time = time.perf_counter_ns()
        logging.info(f"Compiled engine takes {((end_time - start_time) / benchmark_iters) / 1e6} ms per step")

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
            dtype=self.out_dtype,
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

    def _step_engine(self, action: torch.Tensor) -> tuple[PhysicsState, PhysicsStateDer, AuxEngineInfo, PhysicsState]:
        prev_state = self.state.clone()
        action = action.to(self.device)
        if self.differentiable:
            next_state, state_der, aux_info = self.engine(prev_state, action, self.world_cfg)
        else:
            with torch.no_grad():
                next_state, state_der, aux_info = self.engine(prev_state, action, self.world_cfg)
        return prev_state, state_der.clone(), aux_info.clone(), next_state.clone()

    def _reset(self, tensordict=None, **kwargs) -> TensorDict:
        # Generate start and goal states, iteration limits for done/terminated robots
        new_start, new_goal, new_iteration_limits = self.objective.generate_start_goal_states()
        # Update the state variables for the done robots
        reset_mask = self.done | kwargs.get("reset_all", False)
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
        # Set the state variables for reset robots
        self.state[reset_mask] = next_state[reset_mask].clone()  # Update the state for the reset robots
        # Update the last step aux info and state der for the reset robots
        if self.last_step_aux_info is None:
            self.last_step_aux_info = aux_info
        else:
            self.last_step_aux_info[reset_mask] = aux_info[reset_mask]
        if self.last_step_der is None:
            self.last_step_der = state_der
        else:
            self.last_step_der[reset_mask] = state_der[reset_mask]
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
        self.done |= obs_td["done"]
        return obs_td
