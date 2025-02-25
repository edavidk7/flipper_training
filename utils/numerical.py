import torch
from .geometry import skew_symmetric, normalized

__all__ = [
    "integrate_rotation",
    "condition_rotation_matrices",
    "integrate_quaternion",
]


def integrate_rotation(
    R: torch.Tensor,
    omega: torch.Tensor,
    dt: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Integrates the rotation matrix for the next time step using Rodrigues' formula.
    Parameters:
    - R: Tensor of rotation matrices. Shape [B, 3, 3]
    - omega: Tensor of angular velocities. Shape [B, 3]
    - dt: Time step. Float value.
    - eps: Small value to avoid division by zero. Float value.

    Returns:
    - Updated rotation matrices. Shape [B, 3, 3]
    """
    delta_omega = omega * dt  # Shape: [B, 3]
    theta = torch.norm(delta_omega, dim=1, keepdim=True)  # Rotation angle
    omega_skew = skew_symmetric(delta_omega)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)  # Shape: [1, 3, 3]
    theta_expand = torch.clamp(theta.unsqueeze(2), eps)
    sin_term = torch.sin(theta_expand) / theta_expand
    cos_term = (1 - torch.cos(theta_expand)) / (theta_expand**2)
    omega_skew_squared = torch.bmm(omega_skew, omega_skew)
    delta_R = I + sin_term * omega_skew + cos_term * omega_skew_squared
    return torch.bmm(delta_R, R)


def condition_rotation_matrices(R: torch.Tensor) -> torch.Tensor:
    """
    Condition the rotation matrices to prevent numerical instability.
    This is done by performing SVD on the rotation matrices and then reconstructing them.

    Parameters:
        - R: Rotation matrices. Shape [B, 3, 3].

    Returns:
        - torch.Tensor: Conditioned rotation matrices. Shape [B, 3, 3].
    """
    U, _, V = torch.svd(R)
    R = torch.bmm(U, V.transpose(-2, -1))
    return R


def integrate_quaternion(q: torch.Tensor, omega: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Integrate quaternion using the quaternion derivative.
    Parameters:
        - q: Quaternion. Shape [B, 4].
        - omega: Angular velocity. Shape [B, 3].
        - dt: Time step. Float value.
    Returns:
        - Updated quaternion. Shape [B, 4].
    """
    w, x, y, z = q.unbind(dim=-1)
    H = 0.5 * torch.stack(
        [
            torch.stack([-x, -y, -z], dim=-1),
            torch.stack([w, -z, y], dim=-1),
            torch.stack([z, w, -x], dim=-1),
            torch.stack([-y, x, w], dim=-1),
        ],
        dim=-2,
    )  # shape: [B, 4, 3]
    q_new = q + (dt * H @ omega.unsqueeze(-1)).squeeze(-1)
    return normalized(q_new)
