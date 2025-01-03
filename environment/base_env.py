from numpy import full
import torch
from copy import deepcopy
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer, AuxEngineInfo
from flipper_training.engine.engine import DPhysicsEngine
from flipper_training.configs import PhysicsEngineConfig, WorldConfig, EnvConfig, RobotModelConfig
from flipper_training.utils.geometry import planar_rot_from_R3, local_to_global, global_to_local
from flipper_training.utils.environment import interpolate_grid


class BaseDPhysicsEnv():

    def __init__(self, env_config: EnvConfig,
                 physics_config: PhysicsEngineConfig,
                 robot_model_config: RobotModelConfig,
                 device: torch.device | str):
        self.phys_cfg = physics_config
        self.robot_cfg = robot_model_config
        self.env_cfg = env_config
        self.device = device
        for cfg in (physics_config, robot_model_config, env_config):
            cfg.to(device)
        self.engine = DPhysicsEngine(physics_config, robot_model_config, device)
        self.initialize_perception_grid()
        self.state: PhysicsState | None = None
        self.last_reset_data: dict[str, torch.Tensor | WorldConfig] = {}

    def initialize_perception_grid(self) -> None:
        """
        Initialize the perception grid points.
        """
        space = torch.linspace(self.env_cfg.percep_coord, -self.env_cfg.percep_coord, self.env_cfg.percep_dim)
        px, py = torch.meshgrid(space, space, indexing="ij")  # TODO check this, but we want the first coordinate to be the vertical one on the grid. The robot's (1,1) should be the top left corner.
        percep_grid_points = torch.dstack([px, py, torch.zeros_like(px)]).reshape(-1, 3)  # add the z coordinate (0)
        self.percep_grid_points = percep_grid_points.unsqueeze(0).repeat(self.phys_cfg.num_robots, 1, 1).to(self.device)

    def compile(self, **compile_kwargs) -> None:
        """
        Compile the physics engine. The method will execute compile on all methods and the engine. It will take some time to finish.
        """
        assert self.state is not None, "State is not initialized! Call reset with proper arguments first."
        # Call compile on all methods
        self.engine.compile(**compile_kwargs)
        self._rotate_percep_grid = torch.compile(self._rotate_percep_grid, **compile_kwargs)
        self._sample_heightmap = torch.compile(self._sample_heightmap, **compile_kwargs)
        self._sample_pointcloud = torch.compile(self._sample_pointcloud, **compile_kwargs)
        # Run the forward passes to trigger the compilation
        _ = self.engine(PhysicsState.dummy_like(self.state), torch.zeros(self.phys_cfg.num_robots, 2 * self.robot_cfg.num_joints, device=self.device), self.last_reset_data["world_cfg"])
        _ = self._rotate_percep_grid(self.state.R)
        _ = self._sample_heightmap(self.state)
        _ = self._sample_pointcloud(self.state)

    def reset(self, world_cfg: WorldConfig, **kwargs) -> tuple[PhysicsState, torch.Tensor]:
        """
        Reset/initialize the state used by the physics engine. This should be called before starting the simulation.

        Args:
            world_cfg (WorldConfig): The world configuration.
            **kwargs: Additional keyword arguments containing initial state information.

        """
        world_cfg.to(self.device)
        self.state = PhysicsState.dummy(**kwargs, batch_size=self.phys_cfg.num_robots, device=self.device, robot_model=self.robot_cfg)
        self.last_reset_data = {"state": self.state.clone(), "world_cfg": world_cfg}
        return self.state, self._sample_percep_data()

    def _sample_heightmap(self, state: PhysicsState) -> torch.Tensor:
        """
        Sample the heightmap data from the perception grid.

        Rotate robot's local perception grid to the world frame and interpolate the z coordinates. 
        This represents a "bird's eye view" of the environment around the robot.
        The result is a tensor of shape (n_robots, percep_dim, percep_dim).
        """
        world_cfg: WorldConfig = self.last_reset_data["world_cfg"]
        global_percep_points = local_to_global(state.x, state.R, self.percep_grid_points)
        z_coords = interpolate_grid(world_cfg.z_grid, global_percep_points[..., :2], world_cfg.max_coord)
        return z_coords.reshape(-1, 1, self.env_cfg.percep_dim, self.env_cfg.percep_dim)

    def _sample_pointcloud(self, state: PhysicsState) -> torch.Tensor:
        """
        Sample the point cloud data from the perception grid.

        Rotate robot's local perception grid to the world frame and interpolate the z coordinates, then convert it back to the local frame.
        This represents a "point cloud" of the environment around the robot.
        """
        world_cfg: WorldConfig = self.last_reset_data["world_cfg"]
        global_percep_points = local_to_global(state.x, state.R, self.percep_grid_points)
        global_percep_points[..., 2] = interpolate_grid(world_cfg.z_grid, global_percep_points[..., :2], world_cfg.max_coord)
        local_percep_points = global_to_local(global_percep_points, state.R, state.x)
        return local_percep_points

    def _sample_percep_data(self) -> torch.Tensor:
        """
        Sample the perception data from the environment.
        """
        assert self.state is not None, "State is not initialized! Call reset first."
        match self.env_cfg.percep_type:
            case "heightmap":
                return self._sample_heightmap(self.state)
            case "pointcloud":
                return self._sample_pointcloud(self.state)
            case _:
                raise ValueError(f"Invalid perception type: {self.env_cfg.percep_type}")

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

    def step(self, controls: torch.Tensor, sample_percep: bool = False) -> tuple[PhysicsState, PhysicsStateDer, AuxEngineInfo, torch.Tensor | None]:
        """
        Perform one step of the environment.

        Args:
            controls (torch.Tensor): The control inputs for the robots. There are 2 options:

                1) Shape (n_robots, 2 + n_joints) for the longitudinal, rotational and joint angular velocities.

                2) Shape (n_robots, 2*n_joints) for the per-joint longitudinal and rotational velocities.

            All inpus are limited by the robot's physical limits.
            sample_percep (bool): Whether to sample the perception data from the environment.

        Returns:
            tuple[PhysicsState, PhysicsStateDer, AuxEngineInfo, torch.Tensor | None, torch.Tensor | None]: The next state of the environment, the state derivative, the auxiliary engine information, the perception data and the reward.
        """
        assert self.state is not None, "State is not initialized! Call reset with proper arguments first."
        # Recompute controls if needed
        full_controls = self._compute_full_controls(controls)
        # Engine step
        next_state, state_der, aux_info = self.engine(self.state, full_controls, self.last_reset_data["world_cfg"])
        # Perception data
        percep_data = self._sample_percep_data() if sample_percep else None
        self.state = next_state
        return next_state, state_der, aux_info, percep_data
