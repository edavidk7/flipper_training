import torch
from flipper_training.engine.engine_config import PhysicsEngineConfig
from typing import List, Tuple, Dict, Literal
from flipper_training.utils.geometry import normalized, local_to_global, rot_Y
from flipper_training.utils.dynamics import inertia_tensor
from flipper_training.engine.engine_state import *
from flipper_training.utils.numerical import get_integrator_fn, integrate_rotation
from flipper_training.utils.environment import surface_normals, interpolate_grid


class DPhysicsEngine(torch.nn.Module):
    def __init__(self, config: PhysicsEngineConfig, device: str | torch.device = 'cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self.integrator_fn = get_integrator_fn(self.config.integration_mode)

    @torch.compile
    def forward(self, state: PhysicsState, controls: torch.Tensor) -> tuple[PhysicsState, PhysicsStateDer, AuxEngineInfo]:
        """
        Forward pass of the physics engine.
        """
        state_der, aux_info = self.forward_kinematics(state, controls)
        return self.update_state(state, state_der), state_der, aux_info

    @torch.compile
    def forward_kinematics(self, state: PhysicsState,
                           controls: torch.Tensor) -> Tuple[PhysicsStateDer,
                                                            AuxEngineInfo]:
        # unpack state
        x = state.x
        xd = state.xd
        R = state.R
        omega = state.omega
        thetas = state.thetas
        global_robot_points, I = self.construct_global_robot_points(state)

        B, n_pts, D = global_robot_points.shape

        # compute the terrain properties at the robot points
        if not isinstance(self.config.k_stiffness, (int, float)):
            stiffness_points = interpolate_grid(self.config.k_stiffness, global_robot_points[..., :2], self.config.max_coord)
            assert stiffness_points.shape == (B, n_pts, 1)
        else:
            stiffness_points = self.config.k_stiffness
        if not isinstance(self.config.k_damping, (int, float)):
            damping_points = interpolate_grid(self.config.k_damping, global_robot_points[..., :2], self.config.max_coord)
            assert damping_points.shape == (B, n_pts, 1)
        else:
            damping_points = self.config.k_damping
        if not isinstance(self.config.k_friction, (int, float)):
            friction_points = interpolate_grid(self.config.k_friction, global_robot_points[..., :2], self.config.max_coord)
            assert friction_points.shape == (B, n_pts, 1)
        else:
            friction_points = self.config.k_friction

        # find the contact points
        in_contact, dh_points = self.find_contact_points(global_robot_points)
        assert in_contact.shape == (B, n_pts, 1), in_contact.shape

        # compute surface normals at the contact points
        n = surface_normals(self.config.z_grid_grad, global_robot_points[..., :2], self.config.max_coord)
        # Compute current point velocities based on CoG motion and rotation
        xd_points = xd.unsqueeze(1) + torch.cross(omega.unsqueeze(1), global_robot_points - x.unsqueeze(1), dim=-1)
        # reaction at the contact points as spring-damper forces
        xd_points_n = (xd_points * n).sum(dim=-1, keepdims=True)  # normal velocity
        assert xd_points_n.shape == (B, n_pts, 1)
        F_spring = -torch.mul((stiffness_points * dh_points + damping_points * xd_points_n), n)  # F_s = -k * dh - b * v_n
        num_contacts = in_contact.sum(dim=1, keepdims=True)
        F_spring = torch.mul(F_spring, in_contact) / torch.clamp(num_contacts, min=1)  # apply forces only at the contact points
        assert F_spring.shape == (B, n_pts, 3)

        # friction forces: shttps://en.wikipedia.org/wiki/Friction
        N = torch.norm(F_spring, dim=2)  # normal force magnitude at the contact points
        m, g = self.config.robot_model.mass, self.config.gravity
        F_friction = torch.zeros_like(F_spring)  # initialize friction forces
        driving_parts = self.config.robot_model.driving_parts
        thrust_dir = normalized(R[..., 0])  # thrust direction in the global frame
        for i, mask in enumerate(driving_parts):
            # thrust_dir_local = rot_Y(thetas[:, i])[..., 0, None]  # thrust direction in the local frame
            # thrust_dir = torch.bmm(R, thrust_dir_local).squeeze(-1)  # thrust direction in the global frame
            u = controls[:, i].unsqueeze(1)  # control input
            v_cmd = u * thrust_dir  # commanded velocity
            # F_fr = -mu * N * tanh(v_cmd - xd_points)  # tracks friction forces
            dv = v_cmd.unsqueeze(1) - xd_points
            dv_n = (dv * n).sum(dim=-1, keepdims=True)  # normal component of the relative velocity
            dv_tau = dv - dv_n * n  # tangential component of the relative velocity
            F_friction[:, mask] = (friction_points * N.unsqueeze(2) * torch.tanh(dv_tau))[:, mask]
        assert F_friction.shape == (B, n_pts, 3)

        # rigid body rotation: M = sum(r_i x F_i)
        torque = torch.sum(torch.cross(global_robot_points - x.unsqueeze(1), F_spring + F_friction, dim=-1), dim=1)
        # Note that the intertial tensor is a symmetric matrix, so we don't care about the transposition
        omega_d = torch.linalg.solve_ex(I, torque)[0]  # omega_d = I^-1 M÷
        # motion of the cog
        F_grav = torch.tensor([[0.0, 0.0, -m * g]], device=self.device)  # F_grav = [0, 0, -m * g]
        F_cog = F_grav + F_spring.sum(dim=1) + F_friction.sum(dim=1)  # ma = sum(F_i)
        xdd = F_cog / m  # a = F / m
        assert xdd.shape == (B, 3)

        # joint rotational velocities
        thetas_d = controls[:, len(driving_parts):]
        # new state derivative
        new_state_der = PhysicsStateDer(xd=xd, xdd=xdd, omega_d=omega_d, thetas_d=thetas_d)
        # auxiliary information (e.g. for visualization)
        aux_info = AuxEngineInfo(F_spring=F_spring, F_friction=F_friction, in_contact=in_contact, normals=n, global_robot_points=global_robot_points)
        return new_state_der, aux_info

    @ torch.compile
    def find_contact_points(self, robot_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the contact points on the robot.
        """
        z_points = interpolate_grid(self.config.z_grid, robot_points[..., :2], self.config.max_coord)
        dh_points = robot_points[..., 2:3] - z_points
        on_grid = (robot_points[..., 0:1] >= -self.config.max_coord) & (robot_points[..., 0:1] <= self.config.max_coord) & \
            (robot_points[..., 1:2] >= -self.config.max_coord) & (robot_points[..., 1:2] <= self.config.max_coord)
        in_contact = ((dh_points <= 0.0) & on_grid).float()
        return in_contact, dh_points

    @ torch.compile
    def update_state(self, state: PhysicsState, dstate: PhysicsStateDer) -> PhysicsState:
        """
        Integrates the states of the rigid body for the next time step.
        """
        x, xd, R, local_robot_points, omega, thetas = state
        _, xdd, omega_d, thetas_d = dstate
        dt = self.config.dt
        xd = self.integrator_fn(xd, xdd, dt)
        x = self.integrator_fn(x, xd, dt)
        delta_thetas = self.integrator_fn(0., thetas_d, dt)
        local_robot_points = self.rotate_joints(local_robot_points, delta_thetas)
        thetas += delta_thetas
        omega = self.integrator_fn(omega, omega_d, dt)
        R = integrate_rotation(R, omega, dt, integrator=self.integrator_fn)
        return PhysicsState(x, xd, R, local_robot_points, omega, thetas)

    @ torch.compile
    def construct_global_robot_points(self, state: PhysicsState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct the global robot points from the local robot points in the state.
        """
        x = state.x
        R = state.R
        robot_points = state.local_robot_points
        I = inertia_tensor(self.config.robot_model.pointwise_mass, robot_points)
        robot_points = local_to_global(x, R, robot_points)
        return robot_points, I

    @ torch.compile
    def rotate_joints(self, robot_points: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        """
        Rotate points on the robot pointcloud corresponding to the joints based on the joint angles.

        Args:
            robot_points: The robot points in LOCAL coordinates.
            thetas: The joint angles to rotate by.
        """
        # Update each driving part simultaneously on all robots
        for i, (pmask, ppos) in enumerate(zip(self.config.robot_model.driving_parts, self.config.robot_model.joint_positions)):
            rot_Ys = rot_Y(thetas[:, i])
            flippter_pts = robot_points[:, pmask] - ppos.to(device=self.device)
            robot_points[:, pmask] = torch.bmm(flippter_pts, rot_Ys) + ppos
        return robot_points
