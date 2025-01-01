import torch
from typing import Tuple
from flipper_training.configs.engine_config import PhysicsEngineConfig
from flipper_training.configs.robot_config import RobotModelConfig
from flipper_training.configs.world_config import WorldConfig
from flipper_training.utils.geometry import normalized, local_to_global, rot_Y
from flipper_training.utils.dynamics import inertia_tensor, cog
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer, AuxEngineInfo
from flipper_training.utils.numerical import get_integrator_fn, integrate_rotation
from flipper_training.utils.environment import surface_normals, interpolate_grid


class DPhysicsEngine(torch.nn.Module):
    def __init__(self, config: PhysicsEngineConfig, robot_model: RobotModelConfig, device: torch.device | str):
        super().__init__()
        self.config = config
        self.device = device
        self.robot_model = robot_model
        self.integrator_fn = get_integrator_fn(self.config.integration_mode)

    def forward(self, state: PhysicsState,
                controls: torch.Tensor,
                world_config: WorldConfig) -> tuple[PhysicsState, PhysicsStateDer, AuxEngineInfo]:
        """
        Forward pass of the physics engine.
        """
        state_der, aux_info = self.forward_kinematics(state, controls, world_config)
        return self.update_state(state, state_der), state_der, aux_info

    def forward_kinematics(self,
                           state: PhysicsState,
                           controls: torch.Tensor,
                           world_config: WorldConfig) -> Tuple[PhysicsStateDer,
                                                               AuxEngineInfo]:

        global_robot_points, global_I, cog_corrected_points, global_cogs = self.construct_global_robot_points(state)

        # find the contact points
        in_contact, dh_points = self.find_contact_points(global_robot_points, world_config)

        # compute surface normals at the contact points
        n = surface_normals(world_config.z_grid_grad, global_robot_points[..., :2], world_config.max_coord)

        # Compute current point velocities based on CoG motion and rotation
        xd_points = state.xd.unsqueeze(1) + torch.cross(state.omega.unsqueeze(1), cog_corrected_points, dim=-1)

        # normal velocity computed as v_n = v . n
        xd_points_n = (xd_points * n).sum(dim=-1, keepdims=True)

        # Reaction at the contact points as spring-damper forces
        k_damping = (4 * self.robot_model.mass * world_config.k_stiffness)**0.5 * self.config.damping_alpha
        # F_s = -k * dh - b * v_n, multiply by -n to get the force vector
        F_spring = (world_config.k_stiffness * dh_points + k_damping * xd_points_n) * (-n)
        F_spring = F_spring * in_contact / torch.clamp(torch.sum(in_contact, dim=1, keepdims=True), min=1)

        # friction forces
        k_friction = world_config.k_friction
        F_friction = self.calculate_friction(F_spring, state.R, xd_points, controls, n, k_friction)

        # rigid body rotation: M = sum(r_i x F_i)
        act_force = F_spring + F_friction  # total force acting on the robot's points
        torque, omega_d = self.calculate_torque_omega_d(act_force, cog_corrected_points, global_I)

        # motion of the cog
        F_cog = torch.tensor([0., 0., -self.robot_model.mass * self.config.gravity], device=self.device) + act_force.sum(dim=1)  # F = F_spring + F_friction + F_grav
        xdd = F_cog / self.robot_model.mass  # a = F / m, very funny xdd

        # joint rotational velocities, shape (B, n_joints)
        thetas_d = self.compute_joint_angular_velocities(controls)

        # new state derivative
        new_state_der = PhysicsStateDer(xd=state.xd, xdd=xdd, omega_d=omega_d, thetas_d=thetas_d)

        # auxiliary information (e.g. for visualization)
        aux_info = AuxEngineInfo(F_spring=F_spring,
                                 F_friction=F_friction,
                                 in_contact=in_contact,
                                 normals=n,
                                 global_robot_points=global_robot_points,
                                 torque=torque,
                                 global_cog_coords=global_cogs,
                                 cog_corrected_points=cog_corrected_points,
                                 I_global=global_I)

        return new_state_der, aux_info

    def compute_joint_angular_velocities(self, controls: torch.Tensor) -> torch.Tensor:
        """
        Compute the joint angle velocities based on the control inputs and the current joint angles.

        Args:
            controls: The control inputs.
            thetas: The current joint angles.

        Returns:
            thetas_d: The joint angle velocities.
        """
        thetas_d = controls[:, self.robot_model.num_joints:]
        thetas_d = thetas_d.clamp(-self.robot_model.joint_vel_limits, self.robot_model.joint_vel_limits)
        return thetas_d

    def calculate_torque_omega_d(self, act_force: torch.Tensor, cog_corrected_points: torch.Tensor, global_I: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the angular acceleration of the robot.

        Args:
            act_force: The total force acting on the robot's points.
            cog_corrected_points: The CoG corrected robot points in global coordinates.
            global_I: The inertia tensor in the global frame.

        Returns:
            omega_d: The angular acceleration of the robot.
        """
        torque = torch.sum(torch.cross(cog_corrected_points, act_force, dim=-1), dim=1)
        torque = torch.clamp(torque, -self.config.torque_limit, self.config.torque_limit)
        omega_d = torch.linalg.solve_ex(global_I, torque)[0]
        return torque, omega_d

    def calculate_friction(self, F_normal: torch.Tensor, R: torch.Tensor, xd_points: torch.Tensor, controls: torch.Tensor, n: torch.Tensor, k_friction: float | torch.Tensor) -> torch.Tensor:
        """
        Calculate the friction force acting on the robot points.

        This function replaces naive boolean masking with multiplication and addition, which is faster and doesn't break threads on a GPU.

        Args:
            F_normal: The normal force acting on the robot points.
            R: The rotation matrix of the robot.
            xd_points: The velocity of the robot points.
            controls: The control inputs.
            n: The surface normals of robot points.

        Returns:
            F_friction: The friction force acting on the robot points.
        """
        # friction forces: shttps://en.wikipedia.org/wiki/Friction
        N = torch.norm(F_normal, dim=2)  # normal force magnitude at the contact points, guaranteed to be zero if not in contact because of the spring force being zero
        F_friction = torch.zeros_like(F_normal)  # initialize friction forces
        thrust_dir = normalized(R[..., 0])  # thrust direction in the global frame
        for i in range(self.robot_model.num_joints):
            u = controls[:, i].unsqueeze(1)  # control input
            v_cmd = u * thrust_dir  # commanded velocity
            dv = v_cmd.unsqueeze(1) - xd_points  # velocity difference between the commanded and the actual velocity of the robot points
            dv_n = (dv * n).sum(dim=-1, keepdims=True)  # normal component of the relative velocity computed as dv_n = dv . n
            dv_tau = dv - dv_n * n  # tangential component of the relative velocity
            dv_tau_sat = torch.tanh(dv_tau)  # saturation of the tangential velocity using tanh
            mask = self.robot_model.driving_part_masks[i].unsqueeze(-1)  # Shape (n_pts,1)
            F_friction += k_friction * N.unsqueeze(2) * dv_tau_sat * mask
        return F_friction

    def find_contact_points(self, robot_points: torch.Tensor, world_config: WorldConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the contact points on the robot.

        We use a "dampened contact" model which starts considering points even above the terrain as in contact if they are close enough to the terrain. Their penetration is then calculated with the penetration model function.

        Args:
            robot_points: The robot points in GLOBAL coordinates.

        Returns:
            in_contact: A boolean tensor of shape (B, n_pts, 1) indicating whether the points are in contact with the terrain.
            dh_points: The penetration depth of the points. Shape (B, n_pts, 1). It is computed using the penetration model.
        """
        z_points = interpolate_grid(world_config.z_grid, robot_points[..., :2], world_config.max_coord)
        dh_points = robot_points[..., 2:3] - z_points
        on_grid = (robot_points[..., 0:1] >= -world_config.max_coord) & (robot_points[..., 0:1] <= world_config.max_coord) & \
            (robot_points[..., 1:2] >= -world_config.max_coord) & (robot_points[..., 1:2] <= world_config.max_coord)
        in_contact = ((dh_points <= 0.0) & on_grid).float()
        return in_contact, dh_points * in_contact

    def update_state(self, state: PhysicsState, dstate: PhysicsStateDer) -> PhysicsState:
        """
        Integrates the states of the rigid body for the next time step.
        """
        x, xd, R, local_robot_points, omega, thetas = state
        _, xdd, omega_d, thetas_d = dstate
        xd = xd + self.integrator_fn(xdd, self.config.dt)
        x = x + self.integrator_fn(xd, self.config.dt)
        delta_thetas = self.integrator_fn(thetas_d, self.config.dt)
        local_robot_points = self.rotate_joints(local_robot_points, delta_thetas)
        thetas = thetas + delta_thetas
        thetas = thetas.clamp(self.robot_model.joint_limits[0], self.robot_model.joint_limits[1])
        omega = omega + self.integrator_fn(omega_d, self.config.dt)
        R = integrate_rotation(R, omega, self.config.dt, integrator=self.integrator_fn)
        return PhysicsState(x, xd, R, local_robot_points, omega, thetas)

    def construct_global_robot_points(self, state: PhysicsState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct the global robot points from the default local robot points.
        Also compute the global, CoG corrected robot points. Use them to compute the inertia tensor in the global frame.
        """
        robot_points = local_to_global(state.x, state.R, state.local_robot_points)  # global robot points
        global_cogs = cog(self.robot_model.pointwise_mass, robot_points)  # center of gravity in global coordinates
        cog_corrected_points = robot_points - global_cogs.unsqueeze(1)  # CoG corrected robot points in global coordinates (CoG at origin, same rotation as global frame)
        I_global = inertia_tensor(self.robot_model.pointwise_mass, cog_corrected_points)
        return robot_points, I_global, cog_corrected_points, global_cogs

    def rotate_joints(self, robot_points: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        """
        Rotate points on the robot pointcloud corresponding to the joints based on the joint angles. The rotation is done in LOCAL coordinates, inplace.

        This function replaces naive boolean masking with multiplication and addition, which is faster and doesn't break threads on a GPU.

        Args:
            robot_points: The robot points in LOCAL coordinates.
            thetas: The joint angles to rotate by.
        """
        new_robot_points = torch.zeros_like(robot_points)
        for i in range(self.robot_model.num_joints):
            rot_Ys = rot_Y(thetas[:, i])
            joint_pos = self.robot_model.joint_positions[i]
            flippter_coord_system_pts = robot_points - joint_pos
            rotated_pts = torch.bmm(flippter_coord_system_pts, rot_Ys) + joint_pos
            part_mask = self.robot_model.driving_part_masks[i].unsqueeze(-1)  # Shape (n_pts,1)
            new_robot_points += part_mask * rotated_pts
        new_robot_points += self.robot_model.body_mask.unsqueeze(-1) * robot_points
        return new_robot_points
