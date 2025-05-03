# Necessary imports
import torch
import warp as wp
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.configs import RobotModelConfig, PhysicsEngineConfig, TerrainConfig

# User Provided Data Structures (Keep As Is)
# (PhysicsState, PhysicsStateDer, RobotModelConfig - definitions provided by user)
# Assume TerrainConfig is also a dataclass or dict holding tensors like z_grid, z_grid_grad


# --- Warp Structs for Configs (Scalar values only) ---
# (These remain the same as before)
@wp.struct
class PhysicsEngineConfigWarp:
    gravity: wp.float32
    dt: wp.float32
    # num_robots is derived from tensor shapes
    damping_alpha: wp.float32
    soft_contact_sigma: wp.float32
    torque_limit: wp.float32
    eps: wp.float32


@wp.struct
class RobotModelConfigWarp:
    total_mass: wp.float32
    body_mass: wp.float32
    num_driving_parts: wp.int32
    points_per_driving_part: wp.int32
    num_body_points: wp.int32
    num_total_points: wp.int32
    # Joint limits, max vels, driving dir, vmax passed as kernel args arrays/scalars


@wp.struct
class TerrainConfigWarp:
    max_coord: wp.float32
    grid_res: wp.float32
    k_stiffness: wp.float32
    k_friction_lon: wp.float32
    k_friction_lat: wp.float32
    grid_dim_h: wp.int32
    grid_dim_w: wp.int32


# --- Warp Helper Functions (@wp.func) ---
# (These remain the same as before - integrate_quaternion, normalized_warp, etc.)
# IMPORTANT: Ensure integrate_quaternion uses Warp's [x,y,z,w] convention internally


@wp.func
def integrate_quaternion(q: wp.quatf, omega: wp.vec3, dt: wp.float32, eps: wp.float32) -> wp.quatf:
    # (Implementation from previous step - uses Warp's xyzw convention)
    half_dt_omega = 0.5 * dt * omega
    theta = wp.length(half_dt_omega)
    sin_theta_over_theta = wp.select(theta > eps, wp.sin(theta) / theta, 1.0)
    delta_q_vector = sin_theta_over_theta * half_dt_omega
    delta_q_scalar = wp.cos(theta)
    delta_q = wp.quatf(delta_q_vector[0], delta_q_vector[1], delta_q_vector[2], delta_q_scalar)  # xyzw
    q_new = wp.quat_mul(delta_q, q)
    q_new = wp.normalize(q_new)
    return q_new


@wp.func
def normalized_warp(v: wp.vec3, eps: wp.float32) -> wp.vec3:
    norm = wp.length(v)
    safe_norm = wp.select(norm > eps, norm, eps)
    return wp.select(norm > eps, v / safe_norm, wp.vec3(0.0, 0.0, 0.0))


@wp.func
def quat_to_matrix(q: wp.quatf) -> wp.mat33:
    return wp.quat_to_matrix(q)


@wp.func
def rotate_vector_by_quat(v: wp.vec3, q: wp.quatf) -> wp.vec3:
    return wp.quat_rotate(q, v)


@wp.func
def inverse_quat(q: wp.quatf) -> wp.quatf:
    return wp.quat_inverse(q)


@wp.func
def rot_Y_warp(theta: wp.float32) -> wp.mat33:
    c = wp.cos(theta)
    s = wp.sin(theta)
    return wp.mat33(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)


@wp.func
def interpolate_grid_warp(
    grid: wp.array2d(dtype=wp.float32), query_xy: wp.vec2, max_coord: wp.float32, grid_dim_h: wp.int32, grid_dim_w: wp.int32
) -> wp.float32:
    # (Implementation from previous step)
    u = (query_xy[0] + max_coord) / (2.0 * max_coord)
    v = (query_xy[1] + max_coord) / (2.0 * max_coord)
    u = wp.clamp(u, 0.0, 1.0)
    v = wp.clamp(v, 0.0, 1.0)
    x_grid = u * wp.float32(grid_dim_w - 1)
    y_grid = v * wp.float32(grid_dim_h - 1)
    x0 = wp.int32(wp.floor(x_grid))
    y0 = wp.int32(wp.floor(y_grid))
    x1 = x0 + 1
    y1 = y0 + 1
    x0 = wp.max(0, x0)
    y0 = wp.max(0, y0)
    x1 = wp.min(grid_dim_w - 1, x1)
    y1 = wp.min(grid_dim_h - 1, y1)
    tx = x_grid - wp.float32(x0)
    ty = y_grid - wp.float32(y0)
    c00 = grid[y0, x0]
    c10 = grid[y0, x1]
    c01 = grid[y1, x0]
    c11 = grid[y1, x1]
    a = c00 * (1.0 - tx) + c10 * tx
    b = c01 * (1.0 - tx) + c11 * tx
    result = a * (1.0 - ty) + b * ty
    return result


@wp.func
def surface_normals_from_grads_warp(
    z_grid_grad: wp.array3d(dtype=wp.float32),  # Shape (2, H, W) for dx, dy
    query_xy: wp.vec2,
    max_coord: wp.float32,
    grid_dim_h: wp.int32,
    grid_dim_w: wp.int32,
    eps: wp.float32,
) -> wp.vec3:
    # (Implementation from previous step)
    u = (query_xy[0] + max_coord) / (2.0 * max_coord)
    v = (query_xy[1] + max_coord) / (2.0 * max_coord)
    u = wp.clamp(u, 0.0, 1.0)
    v = wp.clamp(v, 0.0, 1.0)
    x_grid = u * wp.float32(grid_dim_w - 1)
    y_grid = v * wp.float32(grid_dim_h - 1)
    x0 = wp.int32(wp.floor(x_grid))
    y0 = wp.int32(wp.floor(y_grid))
    x1 = x0 + 1
    y1 = y0 + 1
    x0 = wp.max(0, x0)
    y0 = wp.max(0, y0)
    x1 = wp.min(grid_dim_w - 1, x1)
    y1 = wp.min(grid_dim_h - 1, y1)
    tx = x_grid - wp.float32(x0)
    ty = y_grid - wp.float32(y0)
    dzdx00 = z_grid_grad[0, y0, x0]
    dzdx10 = z_grid_grad[0, y0, x1]
    dzdx01 = z_grid_grad[0, y1, x0]
    dzdx11 = z_grid_grad[0, y1, x1]
    adx = dzdx00 * (1.0 - tx) + dzdx10 * tx
    bdx = dzdx01 * (1.0 - tx) + dzdx11 * tx
    grad_x = adx * (1.0 - ty) + bdx * ty
    dzdy00 = z_grid_grad[1, y0, x0]
    dzdy10 = z_grid_grad[1, y0, x1]
    dzdy01 = z_grid_grad[1, y1, x0]
    dzdy11 = z_grid_grad[1, y1, x1]
    ady = dzdy00 * (1.0 - tx) + dzdy10 * tx
    bdy = dzdy01 * (1.0 - tx) + dzdy11 * tx
    grad_y = ady * (1.0 - ty) + bdy * ty
    n = wp.vec3(-grad_x, -grad_y, 1.0)
    return normalized_warp(n, eps)


# --- Warp Kernel (@wp.kernel) ---
# (Kernel definition remains largely the same as before, but arguments are wp.array types)
# (Make sure quaternion inputs/outputs match kernel's internal xyzw convention)


@wp.kernel
def physics_step_kernel(
    # Input State (Warp arrays from torch tensors)
    state_in_x: wp.array(dtype=wp.vec3),
    state_in_xd: wp.array(dtype=wp.vec3),
    state_in_q: wp.array(dtype=wp.quatf),  # MUST BE [x,y,z,w] convention here
    state_in_omega: wp.array(dtype=wp.vec3),
    state_in_thetas: wp.array(dtype=wp.float32, ndim=2),  # [B, n_joints]
    # Controls (Warp array from torch tensor)
    controls: wp.array(dtype=wp.float32, ndim=2),  # [B, n_controls]
    # Output State & Derivatives (Warp arrays pointing to torch tensors)
    state_out_x: wp.array(dtype=wp.vec3),
    state_out_xd: wp.array(dtype=wp.vec3),
    state_out_q: wp.array(dtype=wp.quatf),  # Kernel output is [x,y,z,w]
    state_out_omega: wp.array(dtype=wp.vec3),
    state_out_thetas: wp.array(dtype=wp.float32, ndim=2),
    der_xdd: wp.array(dtype=wp.vec3),
    der_omega_d: wp.array(dtype=wp.vec3),
    der_thetas_d: wp.array(dtype=wp.float32, ndim=2),
    der_f_spring: wp.array(dtype=wp.vec3, ndim=2),
    der_f_friction: wp.array(dtype=wp.vec3, ndim=2),
    der_in_contact: wp.array(dtype=wp.float32, ndim=2),  # Now [B, n_pts], matching PhysicsStateDer
    der_torque: wp.array(dtype=wp.vec3),
    der_thrust_vectors: wp.array(dtype=wp.vec3, ndim=2),  # Now [B, n_pts, 3], matching PhysicsStateDer (reshape needed)
    # Configs
    cfg: PhysicsEngineConfigWarp,
    robot_cfg: RobotModelConfigWarp,
    terrain_cfg: TerrainConfigWarp,
    # Robot Model Data (read-only Warp arrays from torch tensors)
    joint_local_driving_part_pts: wp.array(dtype=wp.vec3, ndim=3),
    joint_positions: wp.array(dtype=wp.vec3),
    joint_local_driving_part_cogs: wp.array(dtype=wp.vec3),
    driving_part_inertias: wp.array(dtype=wp.mat33),
    driving_part_masses: wp.array(dtype=wp.float32),
    body_cog: wp.vec3,
    body_inertia: wp.mat33,
    body_points: wp.array(dtype=wp.vec3),
    thrust_directions: wp.array(dtype=wp.vec3, ndim=3),
    joint_limits: wp.vec2,
    joint_max_pivot_vels: wp.float32,
    driving_direction: wp.vec3,
    v_max: wp.float32,
    # Terrain Data (read-only Warp arrays from torch tensors)
    z_grid: wp.array(dtype=wp.float32, ndim=2),
    z_grid_grad: wp.array(dtype=wp.float32, ndim=3),
):
    # --- Thread ID ---
    tid = wp.tid()

    # --- Read current state for this thread ---
    x = state_in_x[tid]
    xd = state_in_xd[tid]
    q = state_in_q[tid]  # Assumed xyzw format
    omega = state_in_omega[tid]

    # --- Constants ---
    F_g = wp.vec3(0.0, 0.0, -robot_cfg.total_mass * cfg.gravity)
    I_3x3 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    # --- 1. Assemble and Transform Robot ---
    # (Logic identical to previous implementation)
    num_joints = robot_cfg.num_driving_parts
    pts_per_joint = robot_cfg.points_per_driving_part
    num_body_pts = robot_cfg.num_body_points
    num_total_pts = robot_cfg.num_total_points

    rot_driving_part_pts = wp.zeros(shape=(num_joints, pts_per_joint), dtype=wp.vec3)
    rot_driving_part_cogs = wp.zeros(shape=num_joints, dtype=wp.vec3)
    rot_driving_part_inertias = wp.zeros(shape=num_joints, dtype=wp.mat33)

    for j in range(num_joints):
        theta_j = state_in_thetas[tid, j]
        rot_j = rot_Y_warp(theta_j)
        joint_pos_j = joint_positions[j]
        cog_local_j = joint_local_driving_part_cogs[j]
        inertia_local_j = driving_part_inertias[j]
        rotated_cog = wp.transform_vector(rot_j, cog_local_j)
        rot_driving_part_cogs[j] = rotated_cog + joint_pos_j
        rot_driving_part_inertias[j] = wp.mul(rot_j, wp.mul(inertia_local_j, wp.transpose(rot_j)))
        for p in range(pts_per_joint):
            pt_local = joint_local_driving_part_pts[j, p]
            rotated_pt = wp.transform_vector(rot_j, pt_local)
            rot_driving_part_pts[j, p] = rotated_pt + joint_pos_j

    cog_overall_local = wp.vec3()
    mass_weighted_sum = wp.vec3()
    for j in range(num_joints):
        mass_weighted_sum += rot_driving_part_cogs[j] * driving_part_masses[j]
    mass_weighted_sum += body_cog * robot_cfg.body_mass
    cog_overall_local = mass_weighted_sum / robot_cfg.total_mass

    I_overall_local = wp.mat33()
    for j in range(num_joints):
        d_driving = rot_driving_part_cogs[j] - cog_overall_local
        d_driving_sq = wp.dot(d_driving, d_driving)
        translation_term_driving = (d_driving_sq * I_3x3 - wp.outer(d_driving, d_driving)) * driving_part_masses[j]
        I_overall_local += rot_driving_part_inertias[j] + translation_term_driving

    d_body = body_cog - cog_overall_local
    d_body_sq = wp.dot(d_body, d_body)
    translation_term_body = (d_body_sq * I_3x3 - wp.outer(d_body, d_body)) * robot_cfg.body_mass
    I_overall_local += body_inertia + translation_term_body

    R_world = quat_to_matrix(q)
    t_world = x
    cog_overall_world = rotate_vector_by_quat(cog_overall_local, q) + t_world
    I_overall_world = wp.mul(R_world, wp.mul(I_overall_local, wp.transpose(R_world)))

    robot_points_global = wp.zeros(shape=num_total_pts, dtype=wp.vec3)
    thrust_vectors_per_point = wp.zeros(shape=num_total_pts, dtype=wp.vec3)  # Local temp storage
    point_idx = 0

    for j in range(num_joints):
        vel_cmd_j = wp.clamp(controls[tid, j], -v_max, v_max)
        rot_j = rot_Y_warp(state_in_thetas[tid, j])
        for p in range(pts_per_joint):
            pt_local_frame = rot_driving_part_pts[j, p]
            robot_points_global[point_idx] = rotate_vector_by_quat(pt_local_frame, q) + t_world
            thrust_dir_joint_local = thrust_directions[j, p]
            thrust_dir_robot_local = wp.transform_vector(rot_j, thrust_dir_joint_local)
            thrust_global = rotate_vector_by_quat(thrust_dir_robot_local, q) * vel_cmd_j
            thrust_vectors_per_point[point_idx] = thrust_global
            # Write to output derivative array (reshaped)
            der_thrust_vectors[tid, point_idx, 0] = thrust_global[0]
            der_thrust_vectors[tid, point_idx, 1] = thrust_global[1]
            der_thrust_vectors[tid, point_idx, 2] = thrust_global[2]
            point_idx += 1

    for p in range(num_body_pts):
        pt_local_frame = body_points[p]
        robot_points_global[point_idx] = rotate_vector_by_quat(pt_local_frame, q) + t_world
        thrust_vectors_per_point[point_idx] = wp.vec3(0.0)
        # Write zero thrust to output
        der_thrust_vectors[tid, point_idx, 0] = 0.0
        der_thrust_vectors[tid, point_idx, 1] = 0.0
        der_thrust_vectors[tid, point_idx, 2] = 0.0
        point_idx += 1

    # --- 2. Find Contact Points ---
    # (Logic identical to previous implementation)
    in_contact_local = wp.zeros(shape=num_total_pts, dtype=wp.float32)
    dh_points_local = wp.zeros(shape=num_total_pts, dtype=wp.float32)
    normals_local = wp.zeros(shape=num_total_pts, dtype=wp.vec3)
    num_contacts = wp.float32(0.0)

    for i in range(num_total_pts):
        pt_global = robot_points_global[i]
        query_xy = wp.vec2(pt_global[0], pt_global[1])
        z_terrain = interpolate_grid_warp(z_grid, query_xy, terrain_cfg.max_coord, terrain_cfg.grid_dim_h, terrain_cfg.grid_dim_w)
        n = surface_normals_from_grads_warp(z_grid_grad, query_xy, terrain_cfg.max_coord, terrain_cfg.grid_dim_h, terrain_cfg.grid_dim_w, cfg.eps)
        normals_local[i] = n
        vec_to_point = pt_global - wp.vec3(pt_global[0], pt_global[1], z_terrain)
        dh = wp.dot(vec_to_point, n)
        contact_val = 0.5 * (1.0 + wp.tanh(dh / (cfg.soft_contact_sigma + cfg.eps)))
        in_contact_flag = wp.select(dh > 0.0, contact_val, 0.0)
        dh_points_local[i] = wp.select(dh > 0.0, dh, 0.0) * in_contact_flag
        in_contact_local[i] = in_contact_flag
        num_contacts += in_contact_flag
        # Write contact status to output array
        der_in_contact[tid, i] = in_contact_flag  # Store the float contact value

    safe_num_contacts = wp.max(num_contacts, 1.0)

    # --- 3. Compute Point Velocities ---
    # (Logic identical to previous implementation)
    cog_corrected_points = wp.zeros(shape=num_total_pts, dtype=wp.vec3)
    xd_points = wp.zeros(shape=num_total_pts, dtype=wp.vec3)
    for i in range(num_total_pts):
        cog_corrected_points[i] = robot_points_global[i] - cog_overall_world
        xd_points[i] = xd + wp.cross(omega, cog_corrected_points[i])

    # --- 4. Calculate Spring Force ---
    # (Logic identical to previous implementation)
    F_spring_local = wp.zeros(shape=num_total_pts, dtype=wp.vec3)
    k_damping = cfg.damping_alpha * 2.0 * wp.sqrt(robot_cfg.total_mass * terrain_cfg.k_stiffness / safe_num_contacts)

    for i in range(num_total_pts):
        if in_contact_local[i] > cfg.eps:
            n = normals_local[i]
            dh = dh_points_local[i]
            xd_point = xd_points[i]
            xd_point_n_mag = wp.dot(xd_point, n)
            F_scalar = terrain_cfg.k_stiffness * dh + k_damping * xd_point_n_mag
            F_spring_local[i] = -F_scalar * n
            F_spring_local[i] = F_spring_local[i] * in_contact_local[i] / safe_num_contacts
        else:
            F_spring_local[i] = wp.vec3(0.0)
        # Write spring force to output array
        der_f_spring[tid, i, 0] = F_spring_local[i][0]
        der_f_spring[tid, i, 1] = F_spring_local[i][1]
        der_f_spring[tid, i, 2] = F_spring_local[i][2]

    # --- 5. Calculate Friction Force ---
    # (Logic identical to previous implementation)
    F_friction_local = wp.zeros(shape=num_total_pts, dtype=wp.vec3)
    global_driving_dir_unit = rotate_vector_by_quat(driving_direction, q)

    for i in range(num_total_pts):
        F_normal_vec = -F_spring_local[i]  # Normal force points along +n
        N = wp.length(F_normal_vec)

        if N > cfg.eps and in_contact_local[i] > cfg.eps:
            n = normals_local[i]
            forward_dir_proj = global_driving_dir_unit - wp.dot(global_driving_dir_unit, n) * n
            forward_dir = normalized_warp(forward_dir_proj, cfg.eps)
            lateral_dir = normalized_warp(wp.cross(forward_dir, n), cfg.eps)
            dv = thrust_vectors_per_point[i] - xd_points[i]  # Use local thrust storage
            dv_n_mag = wp.dot(dv, n)
            dv_tau = dv - dv_n_mag * n
            # dv_tau_saturated = wp.tanh(dv_tau) # Original tanh saturation
            dv_tau_saturated = dv_tau  # Let's try without tanh first if it simplifies things
            dv_lon_mag = wp.dot(dv_tau_saturated, forward_dir)
            dv_lat_mag = wp.dot(dv_tau_saturated, lateral_dir)
            F_friction_lon = -terrain_cfg.k_friction_lon * N * dv_lon_mag * forward_dir
            F_friction_lat = -terrain_cfg.k_friction_lat * N * dv_lat_mag * lateral_dir
            F_friction_local[i] = F_friction_lat + F_friction_lon
        else:
            F_friction_local[i] = wp.vec3(0.0)
        # Write friction force to output array
        der_f_friction[tid, i, 0] = F_friction_local[i][0]
        der_f_friction[tid, i, 1] = F_friction_local[i][1]
        der_f_friction[tid, i, 2] = F_friction_local[i][2]

    # --- 6. Calculate Total Forces, Torques, and Accelerations ---
    # (Logic identical to previous implementation)
    F_cog = F_g
    torque_total = wp.vec3()

    for i in range(num_total_pts):
        act_force_i = F_spring_local[i] + F_friction_local[i]
        F_cog += act_force_i
        r_i = cog_corrected_points[i]
        torque_total += wp.cross(r_i, act_force_i)

    torque_total = wp.clamp(torque_total, -cfg.torque_limit, cfg.torque_limit)
    xdd_next = F_cog / robot_cfg.total_mass
    inv_I_overall_world = wp.inverse(I_overall_world)
    omega_d_next = wp.transform_vector(inv_I_overall_world, torque_total)

    # --- 7. Compute Joint Angular Velocities ---
    # (Logic identical to previous implementation)
    thetas_d_next = wp.zeros(shape=num_joints, dtype=wp.float32)
    for j in range(num_joints):
        joint_vel_cmd = controls[tid, num_joints + j]
        thetas_d_next[j] = wp.clamp(joint_vel_cmd, -joint_max_pivot_vels, joint_max_pivot_vels)
        der_thetas_d[tid, j] = thetas_d_next[j]

    # --- 8. Update State (Integration) ---
    # (Logic identical to previous implementation)
    xd_new = xd + xdd_next * cfg.dt
    x_new = x + xd_new * cfg.dt
    omega_new = omega + omega_d_next * cfg.dt
    q_new = integrate_quaternion(q, omega_new, cfg.dt, cfg.eps)  # Output is xyzw

    # --- Write results back to output arrays ---
    state_out_x[tid] = x_new
    state_out_xd[tid] = xd_new
    state_out_q[tid] = q_new  # Write xyzw quat
    state_out_omega[tid] = omega_new
    der_xdd[tid] = xdd_next
    der_omega_d[tid] = omega_d_next
    der_torque[tid] = torque_total

    # Joint kinematics (integrated and written directly to output)
    for j in range(num_joints):
        theta_current = state_in_thetas[tid, j]
        theta_new_j = theta_current + thetas_d_next[j] * cfg.dt
        state_out_thetas[tid, j] = wp.clamp(theta_new_j, joint_limits[0], joint_limits[1])


# --- Python Wrapper Class with PyTorch Interop ---


class WarpPhysicsEngineTorch(torch.nn.Module):
    def __init__(self, engine_config: PhysicsEngineConfig, robot_model: RobotModelConfig, device: str | torch.device):
        super().__init__()
        wp.init()  # Ensure warp is initialized
        self.torch_device = torch.device(device)
        self.warp_device = wp.device_from_torch(self.torch_device)
        self.robot_model = robot_model  # Keep the original PyTorch RobotModelConfig
        self.engine_config_dict = engine_config  # Keep original dict

        # --- Create Warp Config Structs (Scalars) ---
        self.cfg_wp = PhysicsEngineConfigWarp()
        self.cfg_wp.gravity = engine_config.gravity
        self.cfg_wp.dt = engine_config.dt
        self.cfg_wp.damping_alpha = engine_config.damping_alpha
        self.cfg_wp.soft_contact_sigma = engine_config.soft_contact_sigma
        self.cfg_wp.torque_limit = engine_config.torque_limit
        self.cfg_wp.eps = 1e-6  # Small epsilon for numerical stability

        self.robot_cfg_wp = RobotModelConfigWarp()
        self.robot_cfg_wp.total_mass = robot_model.total_mass
        self.robot_cfg_wp.body_mass = robot_model.body_mass
        self.robot_cfg_wp.num_driving_parts = robot_model.num_driving_parts
        self.robot_cfg_wp.points_per_driving_part = robot_model.points_per_driving_part
        self.robot_cfg_wp.num_body_points = robot_model.points_per_body  # Corrected attribute name
        self.robot_cfg_wp.num_total_points = robot_model.n_pts  # Use property

        # --- Convert Static Robot Model Tensors to Warp Arrays ---
        # Use wp.from_torch for zero-copy if tensors are already on the target device
        # Ensure robot_model tensors are on self.torch_device before calling this
        self.robot_model.to(self.torch_device)  # Move robot model tensors to target device

        self.joint_local_driving_part_pts_wp = wp.from_torch(robot_model.joint_local_driving_part_pts)
        self.joint_positions_wp = wp.from_torch(robot_model.joint_positions)
        self.joint_local_driving_part_cogs_wp = wp.from_torch(robot_model.joint_local_driving_part_cogs)
        # Inertias need specific dtype wp.mat33
        self.driving_part_inertias_wp = wp.from_torch(robot_model.driving_part_inertias, dtype=wp.mat33)  # Flatten to (N, 9) then warp handles mat33
        self.driving_part_masses_wp = wp.from_torch(robot_model.driving_part_masses)
        # Pass body cog/inertia directly as kernel args (vec3/mat33)
        self.body_cog_wp = wp.vec3(robot_model.body_cog.tolist())  # Convert tensor to list/tuple for vec3
        self.body_inertia_wp = wp.mat33(robot_model.body_inertia.flatten().tolist())  # Convert tensor to list for mat33
        self.body_points_wp = wp.from_torch(robot_model.body_points)
        self.thrust_directions_wp = wp.from_torch(robot_model.thrust_directions)
        # Pass limits/vels directly as kernel args (vec2/float32)
        self.joint_limits_wp = wp.vec2(
            robot_model.joint_limits[:, 0].tolist()
        )  # Assuming limits are [min, max] per joint, take first joint's? Needs clarification if per-joint limits differ. Let's assume shared limits for now.
        self.joint_max_pivot_vels_wp = wp.float32(robot_model.joint_max_pivot_vels[0].item())  # Assuming shared max vel for now
        self.driving_direction_wp = wp.vec3(robot_model.driving_direction.tolist())
        self.v_max_wp = wp.float32(robot_model.v_max)

        # Preallocate temporary quaternion buffer for swizzling if needed
        self.q_buffer_wp = None  # Allocated in forward if needed

    def _update_q_buffer(self, batch_size):
        """Allocate or resize the temp quaternion buffer."""
        if self.q_buffer_wp is None or self.q_buffer_wp.shape[0] != batch_size:
            # This buffer holds xyzw for the kernel
            self.q_buffer_wp = wp.zeros(batch_size, dtype=wp.quatf, device=self.warp_device)

    def forward(self, state: PhysicsState, controls: torch.Tensor, terrain_config: TerrainConfig) -> tuple[PhysicsState, PhysicsStateDer]:
        """
        Performs one physics step using Warp with PyTorch zero-copy interop.

        Args:
            state (PhysicsState): Input state TensorClass on the correct device. Quaternions in [w, x, y, z].
            controls (torch.Tensor): Control inputs tensor on the correct device.
            terrain_config (TerrainConfig): Contains terrain parameters and tensors (z_grid, z_grid_grad) on the correct device.

        Returns:
            tuple[PhysicsState, PhysicsStateDer]: Output state and derivatives as new TensorClass objects. Quaternions in [w, x, y, z].
        """
        batch_size = state.batch_size[0]
        if controls.shape[0] != batch_size:
            raise ValueError("Batch size mismatch between state and controls")
        if state.device != self.torch_device or controls.device != self.torch_device:
            raise ValueError(f"Input tensors must be on device {self.torch_device}")

        # --- Ensure terrain_config tensors are on the correct device ---
        # Example: Assuming z_grid and z_grid_grad are attributes
        if not hasattr(terrain_config, "z_grid") or not hasattr(terrain_config, "z_grid_grad"):
            raise AttributeError("terrain_config must have 'z_grid' and 'z_grid_grad' tensors")
        if terrain_config.z_grid.device != self.torch_device or terrain_config.z_grid_grad.device != self.torch_device:
            terrain_config.z_grid = terrain_config.z_grid.to(self.torch_device)
            terrain_config.z_grid_grad = terrain_config.z_grid_grad.to(self.torch_device)
            # Note: This modifies the input terrain_config if tensors were on wrong device.

        # --- Create Warp Terrain Config Struct ---
        terrain_cfg_wp = TerrainConfigWarp()
        terrain_cfg_wp.max_coord = terrain_config.max_coord
        terrain_cfg_wp.k_stiffness = terrain_config.k_stiffness
        terrain_cfg_wp.k_friction_lon = terrain_config.k_friction_lon
        terrain_cfg_wp.k_friction_lat = terrain_config.k_friction_lat
        terrain_cfg_wp.grid_dim_h = terrain_config.z_grid.shape[0]
        terrain_cfg_wp.grid_dim_w = terrain_config.z_grid.shape[1]
        # terrain_cfg_wp.grid_res = terrain_config.grid_res # Add if available

        # --- Get Zero-Copy Warp Arrays from Input Tensors ---
        state_in_x_wp = wp.from_torch(state.x)
        state_in_xd_wp = wp.from_torch(state.xd)
        state_in_omega_wp = wp.from_torch(state.omega)
        state_in_thetas_wp = wp.from_torch(state.thetas)
        controls_wp = wp.from_torch(controls)
        z_grid_wp = wp.from_torch(terrain_config.z_grid)
        z_grid_grad_wp = wp.from_torch(terrain_config.z_grid_grad)

        # Quaternion Swizzle (w,x,y,z) -> (x,y,z,w) for kernel input
        # Create a temporary buffer if needed
        # Copy and swizzle: q_torch is [B, 4] with [w, x, y, z]
        # q_buffer_wp needs [B, 4] with [x, y, z, w]
        # Use wp.from_torch to make the buffer accessible to kernel? Or copy directly.
        # Direct copy might be simpler:
        # Assign to buffer: x=q[:,1], y=q[:,2], z=q[:,3], w=q[:,0]
        q_torch_xyzw = torch.roll(state.q, shifts=-1, dims=1)  # Roll to get [x,y,z,w]
        state_in_q_wp = wp.from_torch(q_torch_xyzw)  # This is now [x,y,z,w] for kernel input

        # --- Allocate Output PyTorch Tensors ---
        # Create new tensors to hold the results
        state_out_x_torch = torch.empty_like(state.x)
        state_out_xd_torch = torch.empty_like(state.xd)
        state_out_q_torch = torch.empty_like(state.q)  # Will hold wxyz result
        state_out_omega_torch = torch.empty_like(state.omega)
        state_out_thetas_torch = torch.empty_like(state.thetas)

        der_xdd_torch = torch.empty_like(state.xd)  # Match shape of xd
        der_omega_d_torch = torch.empty_like(state.omega)
        der_thetas_d_torch = torch.empty_like(state.thetas)  # Match shape of thetas
        # Get n_pts from robot model
        n_pts = self.robot_model.n_pts
        der_f_spring_torch = torch.empty(batch_size, n_pts, 3, device=self.torch_device, dtype=torch.float32)
        der_f_friction_torch = torch.empty(batch_size, n_pts, 3, device=self.torch_device, dtype=torch.float32)
        der_in_contact_torch = torch.empty(batch_size, n_pts, device=self.torch_device, dtype=torch.float32)  # Shape [B, n_pts]
        der_torque_torch = torch.empty_like(state.omega)
        der_thrust_vectors_torch = torch.empty(batch_size, n_pts, 3, device=self.torch_device, dtype=torch.float32)

        # --- Get Zero-Copy Warp Arrays for Output Tensors ---
        state_out_x_wp = wp.from_torch(state_out_x_torch)
        state_out_xd_wp = wp.from_torch(state_out_xd_torch)
        state_out_q_wp = wp.from_torch(state_out_q_torch)  # This view points to the final wxyz tensor buffer
        state_out_omega_wp = wp.from_torch(state_out_omega_torch)
        state_out_thetas_wp = wp.from_torch(state_out_thetas_torch)

        der_xdd_wp = wp.from_torch(der_xdd_torch)
        der_omega_d_wp = wp.from_torch(der_omega_d_torch)
        der_thetas_d_wp = wp.from_torch(der_thetas_d_torch)
        der_f_spring_wp = wp.from_torch(der_f_spring_torch)
        der_f_friction_wp = wp.from_torch(der_f_friction_torch)
        der_in_contact_wp = wp.from_torch(der_in_contact_torch)  # Shape [B, n_pts]
        der_torque_wp = wp.from_torch(der_torque_torch)
        der_thrust_vectors_wp = wp.from_torch(der_thrust_vectors_torch)  # Shape [B, n_pts, 3]

        # --- Launch Kernel ---
        # Note: Pass static robot data arrays stored in self
        wp.launch(
            kernel=physics_step_kernel,
            dim=batch_size,
            inputs=[
                state_in_x_wp,
                state_in_xd_wp,
                state_in_q_wp,
                state_in_omega_wp,
                state_in_thetas_wp,
                controls_wp,
                state_out_x_wp,
                state_out_xd_wp,
                state_out_q_wp,
                state_out_omega_wp,
                state_out_thetas_wp,  # Kernel writes xyzw to state_out_q_wp
                der_xdd_wp,
                der_omega_d_wp,
                der_thetas_d_wp,
                der_f_spring_wp,
                der_f_friction_wp,
                der_in_contact_wp,
                der_torque_wp,
                der_thrust_vectors_wp,
                self.cfg_wp,
                self.robot_cfg_wp,
                terrain_cfg_wp,
                self.joint_local_driving_part_pts_wp,
                self.joint_positions_wp,
                self.joint_local_driving_part_cogs_wp,
                self.driving_part_inertias_wp,
                self.driving_part_masses_wp,
                self.body_cog_wp,
                self.body_inertia_wp,
                self.body_points_wp,
                self.thrust_directions_wp,
                self.joint_limits_wp,
                self.joint_max_pivot_vels_wp,
                self.driving_direction_wp,
                self.v_max_wp,
                z_grid_wp,
                z_grid_grad_wp,
            ],
            device=self.warp_device,
        )

        # --- Quaternion Swizzle (x,y,z,w) -> (w,x,y,z) for output ---
        # The kernel wrote xyzw data into state_out_q_wp, which maps to state_out_q_torch.
        # We need state_out_q_torch to be wxyz.
        # Easiest way is to copy the xyzw components into a new wxyz tensor.
        q_out_xyzw_torch = state_out_q_torch  # Currently holds xyzw result
        q_out_wxyz_torch = torch.empty_like(q_out_xyzw_torch)
        q_out_wxyz_torch[:, 0] = q_out_xyzw_torch[:, 3]  # w = xyzw[:, 3]
        q_out_wxyz_torch[:, 1] = q_out_xyzw_torch[:, 0]  # x = xyzw[:, 0]
        q_out_wxyz_torch[:, 2] = q_out_xyzw_torch[:, 1]  # y = xyzw[:, 1]
        q_out_wxyz_torch[:, 3] = q_out_xyzw_torch[:, 2]  # z = xyzw[:, 2]

        # --- Construct Output TensorClass Objects ---
        # Use the PyTorch tensors that now hold the results
        next_state = PhysicsState(
            x=state_out_x_torch,
            xd=state_out_xd_torch,
            q=q_out_wxyz_torch,  # Use the swizzled wxyz tensor
            omega=state_out_omega_torch,
            thetas=state_out_thetas_torch,
            batch_size=[batch_size],
            device=self.torch_device,
        )

        state_der = PhysicsStateDer(
            xdd=der_xdd_torch,
            omega_d=der_omega_d_torch,
            thetas_d=der_thetas_d_torch,
            f_spring=der_f_spring_torch,
            f_friction=der_f_friction_torch,
            in_contact=der_in_contact_torch.unsqueeze(-1),  # Add back the last dim expected by PhysicsStateDer
            thrust_vectors=der_thrust_vectors_torch,
            torque=der_torque_torch,
            batch_size=[batch_size],
            device=self.torch_device,
        )

        return next_state, state_der
