import time
from typing import TYPE_CHECKING, Callable

import torch
from tensordict import TensorDict, assert_allclose_td
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase, make_composite_from_td

from flipper_training.configs import PhysicsEngineConfig, RobotModelConfig, TerrainConfig
from flipper_training.engine.engine import DPhysicsEngine
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.rl_objectives import BaseObjective
from flipper_training.rl_rewards.rewards import Reward
from flipper_training.utils.logging import get_terminal_logger

if TYPE_CHECKING:
    from flipper_training.observations import Observation


class Env(EnvBase):
    _batch_locked = True
    STATE_KEY = "curr_state"
    PREV_STATE_DER_KEY = "prev_state_der"

    def __init__(
        self,
        objective: BaseObjective,
        reward: Reward,
        observations: dict[
            str,
            Callable[["Env"], "Observation"],
        ],
        terrain_config: TerrainConfig,
        physics_config: PhysicsEngineConfig,
        robot_model_config: RobotModelConfig,
        device: torch.device | str = "cpu",
        out_dtype: torch.dtype = torch.float32,
        differentiable: bool = False,
        engine_compile_opts: dict | None = None,
        return_derivative: bool = False,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        # Misc
        self.n_robots = self.batch_size[0]
        self.differentiable = differentiable
        self.out_dtype = out_dtype
        self.logger = get_terminal_logger("environment")
        self.return_derivative = return_derivative
        # Physics configs
        self.phys_cfg = physics_config.to(device)
        self.robot_cfg = robot_model_config.to(device)
        self.terrain_cfg = terrain_config.to(device)
        # Engine
        self.engine = DPhysicsEngine(physics_config, robot_model_config, device)
        # RL components
        self.observations = {k: o(self) for k, o in observations.items()}
        self.objective = objective
        self.reward = reward
        # RL State variables
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
        self.reset()

    def _compile_engine(
        self,
        correctness_iters: int = 100,
        benchmark_iters: int = 1000,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        **kwargs,
    ) -> None:
        self.logger.info(f"Environment: Compiling engine with options {kwargs}")
        act = self.action_spec.rand()
        state = self.start.clone()  # Dummy state
        # Capture the return tensors from the engine for correctness
        states, prev_state_ders = [], []
        for _ in range(correctness_iters):
            state, prev_state_der = self.engine(state, act, self.terrain_cfg)
            states.append(state)
            prev_state_ders.append(prev_state_der)
        # Compile the engine
        self.engine = torch.compile(self.engine, options=kwargs)
        self.engine(state, act, self.terrain_cfg)  # Dummy forward pass to compile the engine, record the return tensors
        state = self.start.clone()  # Reset the state
        self.logger.info(f"Engine compiled successfully, testing correctness with {atol=}, {rtol=}")
        for _ in range(correctness_iters):
            curr_state, prev_state_der = self.engine(state, act, self.terrain_cfg)
            # Check correctness
            assert_allclose_td(curr_state, states.pop(0), atol=atol, rtol=rtol, msg="compiled engine produced incorrect state")
            assert_allclose_td(prev_state_der, prev_state_ders.pop(0), atol=atol, rtol=rtol, msg="compiled engine produced incorrect state der")
            state = curr_state.clone()
        self.logger.info("Compiled engine passed correctness test")
        # Benchmark the compiled engine
        start_time = time.perf_counter_ns()
        for _ in range(benchmark_iters):
            _ = self.engine(state, act, self.terrain_cfg)
        end_time = time.perf_counter_ns()
        self.logger.info(f"Compiled engine takes {((end_time - start_time) / benchmark_iters) / 1e6} ms per step")

    def _set_seed(self, seed: int | None):
        self.logger.warning(f"This environment is not seedable, ignoring seed {seed}, please set the default generator seed")

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
        obs_specs = {k: obs.get_spec() for k, obs in self.observations.items()}
        state_spec = {Env.STATE_KEY: make_composite_from_td(self.start)}
        if self.return_derivative:
            der_spec = {
                Env.PREV_STATE_DER_KEY: make_composite_from_td(PhysicsStateDer.dummy(self.robot_cfg, device=self.device, batch_size=self.n_robots))
            }
        else:
            der_spec = {}
        return Composite(
            obs_specs | state_spec | der_spec,  # Include the physics state in the observation spec
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
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
    ) -> TensorDict:
        obs_td = TensorDict(
            {
                k: obs(prev_state=prev_state, action=action, prev_state_der=prev_state_der, curr_state=curr_state)
                for k, obs in self.observations.items()
            },
            device=self.device,
            batch_size=[self.n_robots],
        )
        obs_td[Env.STATE_KEY] = curr_state.to_tensordict()
        if self.return_derivative:
            obs_td[Env.PREV_STATE_DER_KEY] = prev_state_der.to_tensordict()
        return obs_td

    def _step_engine(self, prev_state: PhysicsState, action: torch.Tensor) -> tuple[PhysicsStateDer, PhysicsState]:
        if self.differentiable:
            curr_state, prev_state_der = self.engine(prev_state, action, self.terrain_cfg)
        else:
            with torch.no_grad():
                curr_state, prev_state_der = self.engine(prev_state, action, self.terrain_cfg)
        return prev_state_der.clone(), curr_state.clone()

    def _reset(self, tensordict=None, **kwargs) -> TensorDict:
        # Generate start and goal states, iteration limits for done/terminated robots
        new_start, new_goal, new_iteration_limits = self.objective.generate_start_goal_states()
        # Update the state variables for the done robots
        if tensordict is not None and "_reset" in tensordict:
            reset_mask = tensordict["_reset"].squeeze(-1)
        else:
            reset_mask = torch.full((self.n_robots,), True, device=self.device, dtype=torch.bool)
        self.start[reset_mask] = new_start[reset_mask]
        self.goal[reset_mask] = new_goal[reset_mask]
        self.iteration_limits[reset_mask] = new_iteration_limits[reset_mask]
        self.step_count[reset_mask] = 0
        # Take a dummy step to get the first observation
        if tensordict is not None and Env.STATE_KEY in tensordict:
            prev_state = PhysicsState(**tensordict.get(Env.STATE_KEY))
        else:
            prev_state = self.start.clone()
        zeros_action = self.action_spec.zeros()
        prev_state_der, curr_state = self._step_engine(prev_state, zeros_action)
        # Output tensordict
        obs_td = self._get_observations(prev_state=prev_state, action=zeros_action, prev_state_der=prev_state_der, curr_state=curr_state)
        return obs_td

    def _step(self, tensordict) -> TensorDict:
        action = tensordict.get("action").to(self.device)
        prev_state = PhysicsState.from_tensordict(tensordict.get(Env.STATE_KEY))
        # Step the engine
        prev_state_der, curr_state = self._step_engine(prev_state=prev_state, action=action)
        # Check if the robots have reached the goal or terminated
        reached_goal = self.objective.check_reached_goal(state=curr_state, goal=self.goal)
        failed = self.objective.check_terminated_wrong(state=curr_state, goal=self.goal)
        truncated = self.step_count >= self.iteration_limits
        # Output tensordict
        obs_td = self._get_observations(prev_state=prev_state, action=action, prev_state_der=prev_state_der, curr_state=curr_state)
        obs_td["terminated"] = failed | reached_goal
        obs_td["truncated"] = truncated
        obs_td["reward"] = self.reward(
            prev_state=prev_state, action=action, prev_state_der=prev_state_der, curr_state=curr_state, success=reached_goal, fail=failed, env=self
        )
        self.step_count += 1
        return obs_td
