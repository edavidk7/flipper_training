import torch
from typing import NamedTuple


class PhysicsState(NamedTuple):
    """Physics State

    Attributes:
        x (torch.Tensor): Position of the robot in the world frame. Shape (num_robots, 3).
        xd (torch.Tensor): Velocity of the robot in the world frame. Shape (num_robots, 3).
        q (torch.Tensor): Orientation quaternion of the robot in the world frame. Shape (num_robots, 4).
        omega (torch.Tensor): Angular velocity of the robot in the world frame. Shape (num_robots, 3).
        robot_points (torch.Tensor): Position of the robot points in their local frame. Shape (num_robots, num_points, 3).
        I (torch.Tensor): Inertia tensor of the robot in the local frame. Shape (num_robots, 3, 3).
        joint_angles (torch.Tensor): Angles of the movable joints. Shape (num_robots, num_driving_parts).
    """
    x: torch.Tensor
    xd: torch.Tensor
    R: torch.Tensor
    omega: torch.Tensor
    robot_points: torch.Tensor
    I: torch.Tensor
    joint_angles: torch.Tensor | None


class PhysicsStateDer(NamedTuple):
    """Physics State Derivative

    Attributes:
        xd (torch.Tensor): Derivative of the position of the robot in the world frame. Shape (num_robots, 3).
        xdd (torch.Tensor): Derivative of the velocity of the robot in the world frame. Shape (num_robots, 3).
        omega_d (torch.Tensor): Derivative of the angular velocity of the robot in the world frame. Shape (num_robots, 3).
        joint_omegas (torch.Tensor): Angular velocities of the movable joints. Shape (num_robots, num_driving_parts).
    """
    xd: torch.Tensor
    xdd: torch.Tensor
    omega_d: torch.Tensor
    joint_omegas: torch.Tensor | None
