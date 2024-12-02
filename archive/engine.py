
def integrate_quaternion(q: torch.Tensor, omega: torch.Tensor, dt: float | torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
        Integrate quaternion over time given angular velocity.
        q: Tensor of shape [B, 4] (current quaternion)
        omega: Tensor of shape [B, 3] (angular velocity)
        dt: Scalar time step
        Returns: Tensor of shape [B, 4] (new quaternion)
        """
    # Compute the rotation angle
    theta = torch.norm(omega, dim=-1, keepdim=True) * dt  # [B, 1]

    # Compute half angle
    half_theta = 0.5 * theta  # [B, 1]

    # Avoid division by zero
    small_angle = theta.abs() < eps

    # Compute sin(half_theta) / theta
    sin_half_theta = torch.where(
        small_angle,
        0.5 * theta - (theta ** 3) / 48.0,
        torch.sin(half_theta)
    )  # [B, 1]

    # Normalize omega to get rotation axis
    omega_norm = torch.norm(omega, dim=-1, keepdim=True)
    axis = torch.where(
        omega_norm > eps,
        omega / omega_norm,
        torch.zeros_like(omega)
    )  # [B, 3]

    delta_q_vector = axis * sin_half_theta  # [B, 3]
    delta_q_scalar = torch.cos(half_theta)  # [B, 1]

    delta_q = torch.cat([delta_q_scalar, delta_q_vector], dim=-1)  # [B, 4]

    # Quaternion multiplication: q_new = delta_q * q
    q_new = quaternion_multiply(delta_q, q)

    # Normalize the new quaternion
    q_new = q_new / q_new.norm(dim=-1, keepdim=True)

    return q_new
