import torch
from typing import Iterable, Union
from tensordict import tensorclass

__all__ = ["PhysicsState", "PhysicsStateDer", "AuxEngineInfo", "vectorize_iter_of_states"]


@tensorclass
class PhysicsState:
    """Physics State

    Attributes:
        x (torch.Tensor): Position of the robot in the world frame. Shape (num_robots, 3).
        xd (torch.Tensor): Velocity of the robot in the world frame. Shape (num_robots, 3).
        R (torch.Tensor): Rotation matrix of the robot in the world frame. Shape (num_robots, 3, 3).
        omega (torch.Tensor): Angular velocity of the robot in the world frame. Shape (num_robots, 3).
        thetas (torch.Tensor): Angles of the movable joints. Shape (num_robots, num_driving_parts).
    """
    x: torch.Tensor
    xd: torch.Tensor
    R: torch.Tensor
    omega: torch.Tensor
    thetas: torch.Tensor

    @staticmethod
    def dummy_like(s: "PhysicsState") -> "PhysicsState":
        """Create a dummy PhysicsState object with zero tensors.

        Args:
            s (PhysicsState): PhysicsState object to mimic.

        Returns:
            PhysicsState: Dummy PhysicsState object with zero tensors.

        """
        return s.new_zeros()

    @staticmethod
    def dummy(**kwargs) -> "PhysicsState":
        """Create an empty dummy PhysicsState object with zero tensors.
           Some fields can be overridden by passing them as keyword arguments.
        """
        batch_size = kwargs.pop("batch_size", None)
        robot_model = kwargs.pop("robot_model", None)
        assert robot_model is not None, "Robot model must be provided."
        if kwargs:
            t = next(iter(kwargs.values()))
            if batch_size is not None and batch_size != t.shape[0]:
                raise ValueError("Specified batch size does not match the shape of the tensors.")
            else:
                batch_size = t.shape[0]
        elif batch_size is None:
            batch_size = 0
        base = dict(
            x=torch.zeros(batch_size, 3),
            xd=torch.zeros(batch_size, 3),
            R=torch.zeros(batch_size, 3, 3),
            omega=torch.zeros(batch_size, 3),
            thetas=torch.zeros(batch_size, robot_model.num_joints),
        )
        return PhysicsState(**base | kwargs)


@tensorclass
class PhysicsStateDer:
    """Physics State Derivative

    Attributes:
        xd (torch.Tensor): Derivative of the position of the robot in the world frame. Shape (num_robots, 3).
        xdd (torch.Tensor): Derivative of the velocity of the robot in the world frame. Shape (num_robots, 3).
        omega_d (torch.Tensor): Derivative of the angular velocity of the robot in the world frame. Shape (num_robots, 3).
        thetas_d (torch.Tensor): Angular velocities of the movable joints. Shape (num_robots, num_driving_parts).
    """
    xd: torch.Tensor
    xdd: torch.Tensor
    omega_d: torch.Tensor
    thetas_d: torch.Tensor


@tensorclass
class AuxEngineInfo:
    """
    Auxiliary Engine Information

    Attributes:
        F_spring (torch.Tensor): Spring forces. Shape (B, n_pts, 3).
        F_friction (torch.Tensor): Friction forces. Shape (B, n_pts, 3).
        in_contact (torch.Tensor): Contact status. Shape (B, n_pts).
        normals (torch.Tensor): Normals at the contact points. Shape (B, n_pts, 3).
        local_robot_points (torch.Tensor): Robot points in local coordinates. Shape (B, n_pts, 3).
        global_robot_points (torch.Tensor): Robot points in global coordinates. Shape (B, n_pts, 3).
        torque (torch.Tensor): Torque generated on the robot's CoG. Shape (B, 3).
        global_cog_coords (torch.Tensor): CoG coordinates in global frame. Shape (B, 3).
        cog_corrected_points (torch.Tensor): Corrected CoG points in global frame. Shape (B, n_pts, 3).
        I_global (torch.Tensor): Inertia tensor in global frame. Shape (B, 3, 3).
    """

    F_spring: torch.Tensor
    F_friction: torch.Tensor
    in_contact: torch.Tensor
    normals: torch.Tensor
    local_robot_points: torch.Tensor
    global_robot_points: torch.Tensor
    torque: torch.Tensor
    global_cog_coords: torch.Tensor
    cog_corrected_points: torch.Tensor
    I_global: torch.Tensor


StateLike = Union[PhysicsState, PhysicsStateDer, AuxEngineInfo]


def vectorize_iter_of_states(states: Iterable[StateLike], _device: str | None = None) -> StateLike:
    """Convert an iterable of state-like objects to a single state-like object.

    Args:
        states (Iterable[StateLike]): Iterable of state-like objects.

    Returns:
        StateLike: Vectorized state-like object.

    """
    device = _device or states[0].device
    return torch.stack([s.to(device) for s in states])
