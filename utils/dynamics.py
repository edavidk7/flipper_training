import torch

__all__ = ['inertia_tensor', 'vw_to_tracks_vel', 'generate_control_inputs']


@torch.compile
def inertia_tensor(mass: float | torch.Tensor, points: torch.Tensor):
    """
        Compute the inertia tensor for a rigid body represented by point masses.

        Parameters:

            mass (float | torch.Tensor): The total mass of the body or masses of the points.
            points (torch.Tensor): A tensor of shape (B, N, 3) representing the points of the body.
                                Each point contributes equally to the total mass.

        Returns:
            torch.Tensor: A 3x3 inertia tensor matrix.
        """
    n_points = points.shape[1]
    mass_per_point = mass / n_points
    points2 = points * points
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    x2 = points2[..., 0]
    y2 = points2[..., 1]
    z2 = points2[..., 2]
    Ixx = mass_per_point * (y2 + z2).sum(dim=-1)
    Iyy = mass_per_point * (x2 + z2).sum(dim=-1)
    Izz = mass_per_point * (x2 + y2).sum(dim=-1)
    Ixy = -mass_per_point * (x * y).sum(dim=-1)
    Ixz = -mass_per_point * (x * z).sum(dim=-1)
    Iyz = -mass_per_point * (y * z).sum(dim=-1)
    # Construct the inertia tensor matrix
    I = torch.stack([
        torch.stack([Ixx, Ixy, Ixz], dim=-1),
        torch.stack([Ixy, Iyy, Iyz], dim=-1),
        torch.stack([Ixz, Iyz, Izz], dim=-1)
    ], dim=-2)  #
    return I


def vw_to_tracks_vel(v, w, robot_size, n_tracks=2):
    """
        Converts the linear and angular velocities to the track velocities
        according to the differential drive model.
        v_l = v + r * w
        v_r = v - r * w

    Parameters:
        - v: Linear velocity.
        - w: Angular velocity.
        - robot_size: Size of the robot.
        - n_tracks: Number of tracks.

    Returns:
        - Tuple of the left and right track velocities.
    """
    s_x, s_y = robot_size
    r = s_y / 2
    if n_tracks == 2:
        v_l = v + r * w
        v_r = v - r * w
        controls = [v_l, v_r]
    elif n_tracks == 4:
        v_fl = v + r * w
        v_fr = v - r * w
        v_rl = v + r * w
        v_rr = v - r * w
        controls = [v_fl, v_fr, v_rl, v_rr]
    else:
        raise ValueError(f'Unsupported number of tracks: {n_tracks}. Supported values are 2 and 4.')
    return controls


def generate_control_inputs(n_trajs=10,
                            time_horizon=5.0, dt=0.01,
                            robot_base=1.0,
                            v_range=(-1.0, 1.0), w_range=(-1.0, 1.0),
                            n_tracks=2):
    """
    Generates control inputs for the robot trajectories.

    Parameters:
    - n_trajs: Number of trajectories.
    - time_horizon: Time horizon for each trajectory.
    - dt: Time step.
    - robot_base: Distance between the tracks.
    - v_range: Range of the forward speed.
    - w_range: Range of the rotational speed.
    - n_tracks: Number of tracks (2 or 4).

    Returns:
    - Control inputs for the robot trajectories.
    - Time stamps for the trajectories.
    """
    # rewrite the above code using torch instead of numpy
    time_steps = int(time_horizon / dt)
    time_stamps = torch.linspace(0, time_horizon, time_steps)

    # List to store control inputs (left and right wheel velocities)
    control_inputs = torch.zeros((n_trajs, time_steps, n_tracks))

    v = torch.rand(n_trajs) * (v_range[1] - v_range[0]) + v_range[0]  # Forward speed
    w = torch.rand(n_trajs) * (w_range[1] - w_range[0]) + w_range[0]  # Rotational speed

    if n_tracks == 2:
        v_L = v - (w * robot_base) / 2.0  # Left wheel velocity
        v_R = v + (w * robot_base) / 2.0  # Right wheel velocity
        control_inputs[:, :, 0] = v_L[:, None].repeat(1, time_steps)
        control_inputs[:, :, 1] = v_R[:, None].repeat(1, time_steps)
    elif n_tracks == 4:
        v_L = v - (w * robot_base) / 2.0
        v_R = v + (w * robot_base) / 2.0
        control_inputs[:, :, 0] = v_L[:, None].repeat(1, time_steps)
        control_inputs[:, :, 1] = v_R[:, None].repeat(1, time_steps)
        control_inputs[:, :, 2] = v_L[:, None].repeat(1, time_steps)
        control_inputs[:, :, 3] = v_R[:, None].repeat(1, time_steps)
    else:
        raise ValueError('n_tracks must be 2 or 4')

    return control_inputs, time_stamps
