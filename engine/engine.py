import torch
from engine.engine_config import DPhysConfig
from typing import List, Tuple, Dict
from utils.geometry import normalized, local_to_global, rot_Y
from utils.dynamics import inertia_tensor
from engine.engine_state import PhysicsState, PhysicsStateDer
from utils.numerical import get_integrator_fn, integrate_rotation
from utils.environment import surface_normals, interpolate_grid


class DPhysicsEngine(torch.nn.Module):
    def __init__(self, config: DPhysConfig, device: str | torch.device = 'cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self.integrator_fn = get_integrator_fn(self.config.integration_mode)

    def forward_kinematics(self, state: PhysicsState,
                           z_grid: torch.Tensor,
                           z_grid_grad: torch.Tensor,
                           stiffness: torch.Tensor | float,
                           damping: torch.Tensor | float,
                           friction: torch.Tensor | float,
                           driving_parts: List[torch.Tensor],
                           controls: torch.Tensor) -> Tuple[PhysicsStateDer, tuple[torch.Tensor, torch.Tensor]]:

        # unpack state
        x = state.x
        xd = state.xd
        R = state.R
        I = state.I
        robot_points = state.robot_points
        omega = state.omega
        x_points = local_to_global(x, R, robot_points)  # Convert points to global coordinates
        I_inv = torch.inverse(I)  # Inverse of the inertia tensor

        B, n_pts, D = x_points.shape

        # compute the terrain properties at the robot points
        z_points = interpolate_grid(z_grid, x_points[..., :2], self.config.d_max)
        assert z_points.shape == (B, n_pts, 1)
        if not isinstance(stiffness, (int, float)):
            stiffness_points = interpolate_grid(stiffness, x_points[..., :2], self.config.d_max)
            assert stiffness_points.shape == (B, n_pts, 1)
        else:
            stiffness_points = stiffness
        if not isinstance(damping, (int, float)):
            damping_points = interpolate_grid(damping, x_points[..., :2], self.config.d_max)
            assert damping_points.shape == (B, n_pts, 1)
        else:
            damping_points = damping
        if not isinstance(friction, (int, float)):
            friction_points = interpolate_grid(friction, x_points[..., :2], self.config.d_max)
            assert friction_points.shape == (B, n_pts, 1)
        else:
            friction_points = friction

        # check if the rigid body is in contact with the terrain
        dh_points = x_points[..., 2:3] - z_points
        on_grid = (x_points[..., 0:1] >= -self.config.d_max) & (x_points[..., 0:1] <= self.config.d_max) & \
            (x_points[..., 1:2] >= -self.config.d_max) & (x_points[..., 1:2] <= self.config.d_max)
        in_contact = ((dh_points <= 0.0) & on_grid).float()
        assert in_contact.shape == (B, n_pts, 1)

        # compute surface normals at the contact points
        n = surface_normals(z_grid_grad, x_points[..., :2], self.config.d_max)
        assert n.shape == (B, n_pts, 3)

        # Compute current point velocities based on CoG motion and rotation
        xd_points = xd.unsqueeze(1) + torch.cross(omega.unsqueeze(1), x_points - x.unsqueeze(1))

        # reaction at the contact points as spring-damper forces
        xd_points_n = (xd_points * n).sum(dim=-1, keepdims=True)  # normal velocity
        assert xd_points_n.shape == (B, n_pts, 1)
        F_spring = -torch.mul((stiffness_points * dh_points + damping_points * xd_points_n), n)  # F_s = -k * dh - b * v_n
        F_spring = torch.mul(F_spring, in_contact) / n_pts  # apply forces only at the contact points
        assert F_spring.shape == (B, n_pts, 3)

        # friction forces: https://en.wikipedia.org/wiki/Friction
        # thrust_vec_local = torch.tensor(B * [[1.0, 0.0, 0.0]], device=self.device)
        # thrust_dir = normalized(rotate_vector_by_quaternion(thrust_vec_local, q))  # direction of the thrust
        thrust_dir = normalized(R[..., 0])  # direction of the thrust
        N = torch.norm(F_spring, dim=2)  # normal force magnitude at the contact points
        m, g = self.config.robot_mass, self.config.gravity
        F_friction = torch.zeros_like(F_spring)  # initialize friction forces
        for i in range(len(driving_parts)):
            u = controls[:, i].unsqueeze(1)  # control input
            v_cmd = u * thrust_dir  # commanded velocity
            mask = driving_parts[i]
            # F_fr = -mu * N * tanh(v_cmd - xd_points)  # tracks friction forces
            dv = v_cmd.unsqueeze(1) - xd_points
            dv_n = (dv * n).sum(dim=-1, keepdims=True)  # normal component of the relative velocity
            dv_tau = dv - dv_n * n  # tangential component of the relative velocity
            F_friction[:, mask] = (friction_points * N.unsqueeze(2) * torch.tanh(dv_tau))[:, mask]
        assert F_friction.shape == (B, n_pts, 3)

        # rigid body rotation: M = sum(r_i x F_i)
        torque = torch.sum(torch.cross(x_points - x.unsqueeze(1), F_spring + F_friction), dim=1)
        # Note that the intertial tensor is a symmetric matrix, so we don't care about the transposition
        # The equation would be transposed, as by linear algebra we have I^-1 @ M = omega_d, where M should be (3,B), here it is (B,3)
        omega_d = torch.bmm(I_inv, torque.unsqueeze(2)).squeeze(2)  # omega_d = I^-1 M

        # motion of the cog
        F_grav = torch.tensor([[0.0, 0.0, -m * g]], device=self.device)  # F_grav = [0, 0, -m * g]
        F_cog = F_grav + F_spring.sum(dim=1) + F_friction.sum(dim=1)  # ma = sum(F_i)
        xdd = F_cog / m  # a = F / m
        assert xdd.shape == (B, 3)

        # joint rotational velocities
        joint_omegas = controls[:, len(driving_parts):] if len(driving_parts) < controls.shape[1] else None
        new_state_der = PhysicsStateDer(xd=xd, xdd=xdd, omega_d=omega_d, joint_omegas=joint_omegas)
        return new_state_der, (F_spring, F_friction)

    def update_state(self, state: PhysicsState, dstate: PhysicsStateDer, dt: float) -> PhysicsState:
        """
        Integrates the states of the rigid body for the next time step.
        """
        x, xd, R, omega, robot_points, I, joint_angles = state
        _, xdd, omega_d, joint_omegas = dstate
        xd = self.integrator_fn(xd, xdd, dt)
        x = self.integrator_fn(x, xd, dt)
        # if joint_angles is not None:
        #     robot_points, joint_angles, I = self.update_joints(robot_points, joint_angles, joint_omegas, dt)
        omega = self.integrator_fn(omega, omega_d, dt)
        R = integrate_rotation(R, omega, dt, integrator=self.integrator_fn)
        return PhysicsState(x, xd, R, omega, robot_points, I, joint_angles)

    @torch.compile
    def update_joints(self, robot_points: torch.Tensor, joint_angles: torch.Tensor, joint_omegas: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Updates the robot joint angles based on their angular velocities.
        """
        # Angular differences
        joint_angle_deltas = self.integrator_fn(torch.zeros_like(joint_omegas), joint_omegas, dt)
        # Update each driving part on all robots simultaneously
        for i, (pmask, ppos) in enumerate(zip(self.config.driving_parts, self.config.joint_positions.values())):
            ppos = torch.tensor(ppos, device=self.device)
            deltas = joint_angle_deltas[:, i]
            rot_Ys = rot_Y(-deltas).squeeze(0)
            flippter_pts = robot_points[:, pmask] - ppos
            torch.bmm(flippter_pts, rot_Ys, out=flippter_pts)
            robot_points[:, pmask] = flippter_pts + ppos
        return robot_points, joint_angles + joint_angle_deltas, inertia_tensor(self.config.robot_mass, robot_points)
