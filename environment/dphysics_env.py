import torch
from torch.nn import functional as F
from engine.engine_state import PhysicsState
from utils.pid import PID
from environment.control_cfg import ControlConfig
from engine.engine import DPhysicsEngine
from engine.engine_config import DPhysConfig
from utils.geometry import planar_rot_from_R3, local_to_global
from utils.environment import compute_heightmap_gradients


class DPhysicsEnv():
    PERCEP_DIM = 256
    PERCEP_GRID_POINTS = torch.dstack(torch.meshgrid(torch.linspace(1, -1, PERCEP_DIM), torch.linspace(1, -1, PERCEP_DIM))).reshape(-1, 2)

    def __init__(self, dphys_cfg: DPhysConfig, control_cfg: ControlConfig, device: str = 'cpu'):
        self.dphys_cfg = dphys_cfg
        self.physics = DPhysicsEngine(dphys_cfg, device=device)
        self.device = device
        self.flipper_controllers = PID(
            kp=control_cfg.flipper_ki,
            ki=control_cfg.flipper_ki,
            kd=control_cfg.flipper_kd,
            max_output=control_cfg.flipper_max_rot_vel,
            min_output=-control_cfg.flipper_max_rot_vel)

    def init_state(self, num_robots: int) -> PhysicsState:
        x = torch.tensor([0.0, 0.0, 0.0]).to(self.device).repeat(num_robots, 1)
        xd = torch.zeros_like(x)
        R = torch.eye(3).to(self.device).repeat(num_robots, 1, 1)
        omega = torch.zeros_like(x)
        robot_points = torch.as_tensor(self.dphys_cfg.robot_points, device=self.device)
        robot_points = robot_points.repeat(num_robots, 1, 1)
        if self.dphys_cfg.has_movable_flippers:
            flipper_angles = torch.tensor(list(self.dphys_cfg.joint_angles.values()), device=self.device).repeat(num_robots, 1)
            self.flipper_controllers.reset(torch.zeros_like(flipper_angles), torch.zeros_like(flipper_angles))
        else:
            flipper_angles = None
        return PhysicsState(x, xd, R, omega, robot_points, flipper_angles)

    def initialize_world(self, x_grid: torch.Tensor, y_grid: torch.Tensor, z_grid: torch.Tensor, n_robots: int):
        self.state = self.init_state(n_robots)
        self.dstate = None
        self.z_grid = z_grid
        self.z_grad = compute_heightmap_gradients(z_grid, self.dphys_cfg.grid_res)
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.percep_grids = torch.repeat_interleave(self.PERCEP_GRID_POINTS.unsqueeze(0), n_robots, dim=0)
        self.num_robots = n_robots

    def sample_percep_data(self):
        planar_points = self.state.x[:, :2].clone()
        planar_rotations = planar_rot_from_R3(self.state.R)
        percep_points = local_to_global(planar_points, planar_rotations, self.percep_grids)  # (num_robots, PERCEP_DIM**2, 2)
        percep_Z = self.physics.interpolate_grid(self.z_grid, percep_points)
        return percep_Z.reshape(self.num_robots, self.PERCEP_DIM, self.PERCEP_DIM), percep_points

    def get_flipper_controls(self, flipper_setpoints: torch.Tensor):
        """Update the angles of the flippers based on the setpoints."""
        return self.flipper_controllers.step(flipper_setpoints, self.state.joint_angles, self.dphys_cfg.dt)

    def step(self, controls: torch.Tensor):
        """
        Step function for the environment. Updates the state of the robots based on the control inputs.
        Args:
            controls (torch.Tensor): Control inputs for the robots (num_robots, 2xnumber of driving parts).
        """
        if self.dphys_cfg.has_movable_flippers:
            controls[:, -4:] = self.get_flipper_controls(controls[:, -4:])
        else:
            controls = controls[:, :-4]
        self.dstate, forces = self.physics.forward_kinematics(state=self.state, z_grid=self.z_grid,
                                                              z_grid_grad=self.z_grad,
                                                              stiffness=self.dphys_cfg.k_stiffness, damping=self.dphys_cfg.k_damping, friction=self.dphys_cfg.k_friction,
                                                              driving_parts=self.dphys_cfg.driving_parts,
                                                              controls=controls)
        self.state = self.physics.update_state(self.state, self.dstate, self.dphys_cfg.dt)
        return self.state, self.dstate, forces
