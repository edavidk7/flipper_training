import torch
from copy import deepcopy
from flipper_training.configs.world_config import WorldConfig
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer, AuxEngineInfo
from flipper_training.engine.engine import DPhysicsEngine
from configs.engine_config import PhysicsEngineConfig
from configs.robot_config import RobotModelConfig
from flipper_training.utils.geometry import planar_rot_from_R3, local_to_global
from flipper_training.utils.environment import interpolate_grid
from configs.env_config import EnvConfig


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
            cfg.move_all_tensors_to_device(device)
        self.engine = DPhysicsEngine(physics_config, robot_model_config, device)
        self.initialize_perception_grid()
        self.state: PhysicsState | None = None
        self.world_cfg: WorldConfig | None = None
        self.init_data: dict[str, torch.Tensor | WorldConfig] = {}

    def initialize_perception_grid(self) -> None:
        """
        Initialize the perception grid points.
        """
        space = torch.linspace(self.env_cfg.percep_coord, -self.env_cfg.percep_coord, self.env_cfg.percep_dim)
        grid = torch.meshgrid(space, space, indexing="ij")  # TODO check this, but we want the first coordinate to be the vertical one on the grid. The robot's (1,1) should be the top left corner.
        percep_grid_points = torch.dstack(grid).reshape(-1, 2)
        self.percep_grid_points = percep_grid_points.unsqueeze(0).repeat(self.phys_cfg.num_robots, 1, 1).to(self.device)

    def reset(self, **kwargs) -> None:
        """
        Reset the environment to its initial state.

        Args:
            The arguments to update the initial state of the environment, they should be the same as the ones passed to the init method.
        """
        assert self.init_data, "Environment is not initialized! Call init with proper arguments first."
        new_init = deepcopy(self.init_data) | kwargs
        self.init(**new_init)

    def init(self, positions: torch.Tensor, rotations: torch.Tensor, thetas: torch.Tensor, world_cfg: WorldConfig) -> None:
        """
        Initialize the state used by the physics engine. This should be called before starting the simulation.

        Args:
            positions (torch.Tensor): The initial positions of the robots. Shape (n_robots, 3).
            rotations (torch.Tensor): The initial rotations of the robots. Shape (n_robots, 3).
            thetas (torch.Tensor): The initial joint angles of the robots. Shape (n_robots, n_joints).

        """
        conf_n_robots = self.phys_cfg.num_robots
        assert positions.shape == (conf_n_robots, 3), f"Invalid shape for positions: {positions.shape}. Expected {(conf_n_robots, 3)}."
        assert rotations.shape == (conf_n_robots, 3), f"Invalid shape for rotations: {rotations.shape}. Expected {(conf_n_robots, 3)}."
        assert thetas.shape == (conf_n_robots, self.robot_cfg.num_joints), f"Invalid shape for thetas: {thetas.shape}. Expected {(conf_n_robots, self.robot_cfg.num_joints)}."
        x0 = positions.to(self.device)
        R0 = rotations.to(self.device)
        thetas0 = thetas.to(self.device)
        xd0 = torch.zeros_like(positions)
        omega0 = torch.zeros_like(positions)  # this is the angular velocity as a vector
        # Create the local robot points and rotate the joints as needed
        local_robot_points0 = self.robot_cfg.robot_points.unsqueeze(0).repeat(conf_n_robots, 1, 1).to(self.device)
        local_robot_points0 = self.engine.rotate_joints(local_robot_points0, thetas0)
        # Initialize the state
        world_cfg.move_all_tensors_to_device(self.device)
        self.state = PhysicsState(x=x0, xd=xd0, R=R0, omega=omega0, thetas=thetas0, local_robot_points=local_robot_points0)
        self.world_cfg = world_cfg
        self.init_data = {"positions": x0, "rotations": R0, "thetas": thetas0, "world_cfg": deepcopy(world_cfg)}

    def _rotate_percep_grid(self, R: torch.Tensor) -> torch.Tensor:
        """
        Rotate the perception grid points by yaw angle extracted from the rotation matrix.
        """
        planar_rot = planar_rot_from_R3(R)
        return local_to_global(torch.zeros(2, device=self.device), planar_rot, self.percep_grid_points)

    def _sample_heightmap(self, state: PhysicsState) -> torch.Tensor:
        """
        Sample the heightmap data from the perception grid.
        """
        rotated_percep_points = self._rotate_percep_grid(state.R)
        z_coords = interpolate_grid(self.world_cfg.z_grid, rotated_percep_points, self.world_cfg.max_coord)
        return z_coords.reshape(-1, self.env_cfg.percep_dim, self.env_cfg.percep_dim)

    def _sample_pointcloud(self, state: PhysicsState) -> torch.Tensor:
        """
        Sample the point cloud data from the perception grid.
        """
        rotated_percep_points = self._rotate_percep_grid(state.R)
        z_coords = interpolate_grid(self.world_cfg.z_grid, rotated_percep_points, self.world_cfg.max_coord)
        return torch.cat((self.percep_grid_points, z_coords.unsqueeze(-1)), dim=-1)  # local coordinates + z coordinates

    def _sample_percep_data(self) -> torch.Tensor:
        """
        Sample the perception data from the environment.
        """
        assert self.state is not None, "State is not initialized! Call init with proper arguments first."
        assert self.world_cfg is not None, "World configuration is not set! Call init with proper arguments first."
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
            vw = controls[:, :2]
            joint_v = self.robot_cfg.get_controls(vw)
            join_w = controls[:, 2:]
            full_controls = torch.cat((joint_v, join_w), dim=1).to(self.device)
        else:
            raise ValueError(f"Invalid shape for controls: {controls.shape}. Expected {(self.phys_cfg.num_robots, 2 + self.robot_cfg.num_joints)} or {(self.phys_cfg.num_robots, 2*self.robot_cfg.num_joints)}.")
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
        assert self.state is not None, "State is not initialized! Call init with proper arguments first."
        assert self.world_cfg is not None, "World configuration is not set! Call init with proper arguments first."
        # Recompute controls if needed
        full_controls = self._compute_full_controls(controls)
        # Engine step
        next_state, state_der, aux_info = self.engine(self.state, full_controls, self.world_cfg)
        # Perception data
        percep_data = self._sample_percep_data() if sample_percep else None
        self.state = next_state
        return next_state, state_der, aux_info, percep_data
