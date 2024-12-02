import torch
from typing import Callable, Literal
from .geometry import skew_symmetric


IntegrationMode = Literal["euler", "rk2", "rk4"]
Integrator = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]


def get_integrator_fn(mode: IntegrationMode) -> Integrator:
    """
    Returns the integrator function according to the integration mode.

    Parameters:
        - mode: Integration mode.

    Returns:
        - Function: Integrator function.
    """
    if mode == "euler":
        return euler_integrator
    elif mode == "rk2":
        return rk2_integrator
    elif mode == "rk4":
        return rk4_integrator
    else:
        raise ValueError(f'Unsupported integration mode: {mode}. Supported values')


def euler_integrator(x: torch.Tensor, xd: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Euler integrator.

    Parameters:
        - x: State.
        - xd: Derivative of the state.
        - dt: Time step.

    Returns:
        - torch.Tensor: Integrated state.
    """
    return x + xd * dt


def rk2_integrator(x: torch.Tensor, xd: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Second-order Runge-Kutta integrator.

    Parameters:
        - x: State.
        - xd: Derivative of the state.
        - dt: Time step.

    Returns:
        - torch.Tensor: Integrated state.
    """
    k1 = xd * dt
    k2 = dt * (xd + k1)
    return x + k2 / 2


def rk4_integrator(x: torch.Tensor, xd: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Fourth-order Runge-Kutta integrator.

    Parameters:
        - x: State.
        - xd: Derivative of the state.
        - dt: Time step.

    Returns:
        - torch.Tensor: Integrated state.
    """
    k1 = dt * xd
    k2 = dt * (xd + k1 / 2)
    k3 = dt * (xd + k2 / 2)
    k4 = dt * (xd + k3)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


@torch.compile
def integrate_rotation(R: torch.Tensor, omega: torch.Tensor, dt: float, eps: float = 1e-6, integrator: Integrator = rk4_integrator) -> torch.Tensor:
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
    omega_dt = integrator(torch.zeros_like(omega), omega, dt)
    theta = torch.norm(omega_dt, dim=1, keepdim=True)  # Rotation angle
    omega_skew = skew_symmetric(omega_dt)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)  # Shape: [1, 3, 3]
    theta_expand = torch.clamp(theta.unsqueeze(2), eps)
    sin_term = torch.sin(theta_expand) / theta_expand
    cos_term = (1 - torch.cos(theta_expand)) / (theta_expand ** 2)
    omega_skew_squared = torch.bmm(omega_skew, omega_skew)
    delta_R = I + sin_term * omega_skew + cos_term * omega_skew_squared
    return torch.bmm(delta_R, R)
