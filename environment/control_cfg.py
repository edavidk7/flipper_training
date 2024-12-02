# Description: Configuration file for the control system
import torch


class ControlConfig:
    flipper_kp: float = 3.0
    flipper_ki: float = 2.5
    flipper_kd: float = 0.0
    flipper_max_rot_vel: float = 0.3 * torch.pi  # rad/s
