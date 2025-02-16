import torch

__all__ = [
    'normalized',
    'skew_symmetric',
    'rot_X',
    'rot_Y',
    'rot_Z',
    'global_to_local',
    'local_to_global',
    'quaternion_multiply',
    'quaternion_to_rotation_matrix',
    'quaternion_conjugate',
    'rotate_vector_by_quaternion',
    'global_to_local_q',
    'local_to_global_q',
    'yaw_from_R',
    'planar_rot_from_R3',
    'planar_rot_from_q',
    'quaternion_to_yaw',
    'quaternion_to_pitch',
    'quaternion_to_roll',
    'points_in_oriented_box'
]


def normalized(x, eps=1e-6):
    """
    Normalizes the input tensor.

    Parameters:
    - x: Input tensor.
    - eps: Small value to avoid division by zero.

    Returns:
    - Normalized tensor.
    """
    norm = torch.norm(x, dim=-1, keepdim=True)
    norm.clamp_(min=eps)
    return x / norm


def skew_symmetric(v):
    """
    Returns the skew-symmetric matrix of a vector.

    Parameters:
    - v: Input vector.

    Returns:
    - Skew-symmetric matrix of the input vector.
    """
    assert v.dim() == 2 and v.shape[1] == 3
    U = torch.zeros(v.shape[0], 3, 3, device=v.device)
    U[:, 0, 1] = -v[:, 2]
    U[:, 0, 2] = v[:, 1]
    U[:, 1, 2] = -v[:, 0]
    U[:, 1, 0] = v[:, 2]
    U[:, 2, 0] = -v[:, 1]
    U[:, 2, 1] = v[:, 0]
    return U


def rot_X(theta):
    if isinstance(theta, float):
        theta = torch.tensor(theta)
    theta = theta.reshape(-1, 1)
    cos_ang = torch.cos(theta)
    sin_ang = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    return torch.stack([
        torch.cat([ones, zeros, zeros], dim=-1),
        torch.cat([zeros, cos_ang, -sin_ang], dim=-1),
        torch.cat([zeros, sin_ang, cos_ang], dim=-1)
    ], dim=1)  # Stack along new dimension to create (B, 3, 3)


def rot_Y(theta):
    if isinstance(theta, float):
        theta = torch.tensor(theta)
    theta = theta.reshape(-1, 1)
    cos_ang = torch.cos(theta)
    sin_ang = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    return torch.stack([
        torch.cat([cos_ang, zeros, sin_ang], dim=-1),
        torch.cat([zeros, ones, zeros], dim=-1),
        torch.cat([-sin_ang, zeros, cos_ang], dim=-1)
    ], dim=1)  # Stack along new dimension to create (B, 3, 3)


def rot_Z(theta):
    if isinstance(theta, float):
        theta = torch.tensor(theta)
    theta = theta.reshape(-1, 1)
    cos_ang = torch.cos(theta)
    sin_ang = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    return torch.stack([
        torch.cat([cos_ang, -sin_ang, zeros], dim=-1),
        torch.cat([sin_ang, cos_ang, zeros], dim=-1),
        torch.cat([zeros, zeros, ones], dim=-1)
    ], dim=1)  # Stack along new dimension to create (B, 3, 3)


def global_to_local(t: torch.Tensor, R: torch.Tensor, points: torch.Tensor):
    """
    Transforms the global coordinates to the local coordinates.

    Parameters:
    - t: Translation vector.
    - R: Rotation matrix in the global frame.
    - points: Global coordinates.

    Returns:
    - Local coordinates.
    """
    if points.dim() != 3:  # if points are not batched
        points = points.unsqueeze(0)  # (1, N, D)
    B, N, D = points.shape
    t = t.reshape(B, 1, D)
    R = R.reshape(B, D, D)
    return torch.bmm(points - t, R)  # Correspods to transposed rotation matrix -> inverse


def local_to_global(t: torch.Tensor, R: torch.Tensor, points: torch.Tensor):
    """
    Transforms the global coordinates to the local coordinates.

    Parameters:
    - t: Translation vector.
    - R: Rotation matrix in global frame.
    - points: Global coordinates.

    Returns:
    - Local coordinates.
    """
    if points.dim() != 3:  # if points are not batched
        points = points.unsqueeze(0)  # (1, N, D)
    B, N, D = points.shape
    t = t.reshape(B, 1, D)
    R = R.reshape(B, D, D)
    return torch.bmm(points, R.transpose(1, 2)) + t  # corresponds to original rotation matrix


def quaternion_multiply(q: torch.Tensor, r: torch.Tensor):
    """
    Multiplies two quaternions.
    q, r: Tensors of shape [B, 4]
    Returns: Tensor of shape [B, 4]
    """
    # q = [w, x, y, z]
    w1, x1, y1, z1 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w2, x2, y2, z2 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quaternion_to_rotation_matrix(q: torch.Tensor):
    """
    Converts a quaternion to a rotation matrix.
    q: Tensor of shape [B, 4]
    Returns: Tensor of shape [B, 3, 3]
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    B = q.shape[0]

    # Precompute products
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    R = torch.zeros(B, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = ww + xx - yy - zz
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = ww - xx + yy - zz
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = ww - xx - yy + zz

    return R


def quaternion_conjugate(q):
    """
    Compute the conjugate of a quaternion.
    q: Tensor of shape (N, 4)
    Returns: Tensor of shape (N, 4)
    """
    return q * torch.tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype)


def rotate_vector_by_quaternion(v, q):
    """
    Rotate vector(s) v by quaternion(s) q.
    v: Tensor of shape (..., 3)
    q: Tensor of shape (..., 4), must be unit quaternions
    Returns: Tensor of shape (..., 3)
    """
    # Normalize quaternion
    q = q / q.norm(dim=-1, keepdim=True)

    # Convert vector to quaternion with zero scalar part
    v_q = torch.zeros((*v.shape[:-1], 4), device=v.device, dtype=v.dtype)
    v_q[..., 1:] = v
    # Compute rotated quaternion
    q_conj = quaternion_conjugate(q)
    v_rot = quaternion_multiply(quaternion_multiply(q, v_q), q_conj)
    # Return vector part
    return v_rot[..., 1:]


def global_to_local_q(t: torch.Tensor, q: torch.Tensor, points: torch.Tensor):
    """
    Transforms the global coordinates to the local coordinates.

    Parameters:
    - t: Translation vector.
    - q: Rotation quaternion in the global frame.
    - points: Global coordinates.

    Returns:
    - Local coordinates.
    """
    if points.dim() != 3:  # if points are not batched
        points = points.unsqueeze(0)  # (1, N, D)
    B, N, D = points.shape
    t = t.reshape(B, 1, D)
    q = q.reshape(B, 4)
    return rotate_vector_by_quaternion(points - t, quaternion_conjugate(q))


def local_to_global_q(t: torch.Tensor, q: torch.Tensor, points: torch.Tensor):
    """
    Transforms the local coordinates to the global coordinates.

    Parameters:
    - t: Translation vector.
    - q: Rotation quaternion in global frame.
    - points: Global coordinates.

    Returns:
    - Global coordinates.
    """
    if points.dim() != 3:  # if points are not batched
        points = points.unsqueeze(0)  # (1, N, D)
    B, N, D = points.shape
    t = t.reshape(B, 1, D)
    q = q.reshape(B, 4)
    return rotate_vector_by_quaternion(points, q) + t


def yaw_from_R(R: torch.Tensor):
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])


def pitch_from_R(R: torch.Tensor):
    return torch.arcsin(-R[..., 2, 0])


def roll_from_R(R: torch.Tensor):
    return torch.atan2(R[..., 2, 1], R[..., 2, 2])


def rotation_matrix_to_euler_zyx(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert batch of rotation matrices to ZYX Euler angles (yaw, pitch, roll)
    Args:
        R: (B, 3, 3) batch of rotation matrices
    Returns:
        angles: (B, 3) tensor of Euler angles in radians
    """
    pitch = torch.asin(-R[:, 2, 0].clamp(-1 + eps, 1 - eps))
    # Safe cosine calculation
    cos_pitch = torch.cos(pitch)
    mask = torch.abs(cos_pitch) > eps
    yaw = torch.zeros_like(pitch)
    roll = torch.zeros_like(pitch)
    # Non-degenerate case
    yaw[mask] = torch.atan2(R[mask, 1, 0], R[mask, 0, 0])
    roll[mask] = torch.atan2(R[mask, 2, 1], R[mask, 2, 2])
    # Degenerate case (pitch near ±π/2)
    if torch.any(~mask):
        yaw[~mask] = 0.0
        roll[~mask] = torch.atan2(-R[~mask, 0, 1], R[~mask, 1, 1])
    return torch.stack([yaw, pitch, roll], dim=1)


def planar_rot_from_R3(R: torch.Tensor):
    ang = yaw_from_R(R)  # Extract yaw angle (rotation around Z axis)
    # Create the 2D rotation matrix for each batch element
    cos_ang = torch.cos(ang)
    sin_ang = torch.sin(ang)
    # Construct the planar rotation matrix (B, 2, 2)
    rot_matrix_2d = torch.stack([
        torch.stack([cos_ang, -sin_ang], dim=-1),
        torch.stack([sin_ang, cos_ang], dim=-1)
    ], dim=-2)  # Stack along new dimension to create (B, 2, 2)
    return rot_matrix_2d


def planar_rot_from_q(q: torch.Tensor):
    """
    Extracts the planar rotation matrix from a quaternion.
    q: Tensor of shape (N, 4)
    Returns: Tensor of shape (N, 2, 2)
    """
    # Extract the yaw angle from the quaternion
    yaw = quaternion_to_yaw(q)
    # Create the 2D rotation matrix for each batch element
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    # Construct the planar rotation matrix (B, 2, 2)
    rot_matrix_2d = torch.stack([
        torch.stack([cos_yaw, -sin_yaw], dim=-1),
        torch.stack([sin_yaw, cos_yaw], dim=-1)
    ], dim=-2)  # Stack along new dimension to create (B, 2, 2)
    return rot_matrix_2d


def quaternion_to_yaw(q):
    """
    Compute the yaw (psi) angle from a quaternion.

    Parameters:
    - q: Tensor of shape (..., 4), quaternion [w, x, y, z]

    Returns:
    - Yaw angle tensor in radians.
    """
    w, x, y, z = q.unbind(-1)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return yaw


def quaternion_to_pitch(q):
    """
    Compute the pitch (theta) angle from a quaternion.

    Parameters:
    - q: Tensor of shape (..., 4), quaternion [w, x, y, z]

    Returns:
    - Pitch angle tensor in radians.
    """
    w, x, y, z = q.unbind(-1)
    sinp = 2 * (w * y - z * x)
    sinp = sinp.clamp(-1, 1)
    pitch = torch.asin(sinp)
    return pitch


def quaternion_to_roll(q):
    """
    Compute the roll (phi) angle from a quaternion.

    Parameters:
    - q: Tensor of shape (..., 4), quaternion [w, x, y, z]

    Returns:
    - Roll angle tensor in radians.
    """
    w, x, y, z = q.unbind(-1)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    return roll


def points_in_oriented_box(points: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """
    Check if points are inside an oriented box.

    Args:
        points (torch.Tensor): Tensor of shape (N, 2), where N is the number of 2D points.
        box (torch.Tensor): Tensor of shape (4, 2), representing the 4 corner points of the box in order.

    Returns:
        torch.Tensor: Boolean mask of shape (N,), where True indicates the point is inside the box.
    """
    assert points.shape[1] == 2, "Points must be 2D (N, 2)."
    assert box.shape == (4, 2), "Box must have 4 corner points with shape (4, 2)."
    edge_vectors = box.roll(-1, dims=0) - box  # Edges: [p1->p2, p2->p3, p3->p4, p4->p1]
    point_vectors = points.unsqueeze(1) - box.unsqueeze(0)  # Shape: (N, 4, 2)
    dot_prods = torch.sum(edge_vectors * point_vectors, dim=2)  # Shape: (N, 4)
    inside_mask = (dot_prods >= 0).all(dim=1) | (dot_prods <= 0).all(dim=1)
    return inside_mask
