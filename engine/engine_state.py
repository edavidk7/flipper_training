import torch
from typing import NamedTuple, Iterable

__all__ = ["PhysicsState", "PhysicsStateDer", "AuxEngineInfo", "vectorize_iter_of_tensor_tuples"]


class PhysicsState(NamedTuple):
    """Physics State

    Attributes:
        x (torch.Tensor): Position of the robot in the world frame. Shape (num_robots, 3).
        xd (torch.Tensor): Velocity of the robot in the world frame. Shape (num_robots, 3).
        R (torch.Tensor): Rotation matrix of the robot in the world frame. Shape (num_robots, 3, 3).
        local_robot_points (torch.Tensor): Robot points at a given timestep in local coordinates. Shape (num_robots, n_pts, 3).
        omega (torch.Tensor): Angular velocity of the robot in the world frame. Shape (num_robots, 3).
        thetas (torch.Tensor): Angles of the movable joints. Shape (num_robots, num_driving_parts).
    """
    x: torch.Tensor
    xd: torch.Tensor
    R: torch.Tensor
    local_robot_points: torch.Tensor
    omega: torch.Tensor
    thetas: torch.Tensor


class PhysicsStateDer(NamedTuple):
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


class AuxEngineInfo(NamedTuple):
    """
    Auxiliary Engine Information

    Attributes:
        F_spring (torch.Tensor): Spring forces. Shape (B, n_pts, 3).
        F_friction (torch.Tensor): Friction forces. Shape (B, n_pts, 3).
        in_contact (torch.Tensor): Contact status. Shape (B, n_pts).
        normals (torch.Tensor): Normals at the contact points. Shape (B, n_pts, 3).
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
    global_robot_points: torch.Tensor
    torque: torch.Tensor
    global_cog_coords: torch.Tensor
    cog_corrected_points: torch.Tensor
    I_global: torch.Tensor


def vectorize_iter_of_tensor_tuples(tuples: Iterable[NamedTuple]) -> NamedTuple:
    """
    Vectorize an iterable of Tensor Named Tuples into a single Named Tuple containing stacked tensors.

    Args:
        tuples (Iterable[Type[NamedTuple]]): Iterable of Named Tuples containing tensors.

    Returns:
        Type[NamedTuple]: Named Tuple containing stacked tensors.
    """
    nt = NamedTuple(f"Vectorized{tuples[0].__class__.__name__}", [(field_name, torch.Tensor) for field_name in tuples[0]._fields])
    fields = {field_name: torch.stack([getattr(t, field_name) for t in tuples], dim=0) for field_name in nt._fields}
    return nt(**fields)
