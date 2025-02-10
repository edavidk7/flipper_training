import torch
from flipper_training.environment.torchrl_env import TorchRLEnv
from tensordict.nn import TensorDictModule


class TerrainEncoder(torch.nn.Module):
    def __init__(self, img_shape, output_size):
        super(TerrainEncoder, self).__init__()
        self.img_shape = img_shape
        self.output_size = output_size
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * (img_shape[0] // 16) * (img_shape[1] // 16), output_size)
        )
        self.forward = self.encoder.forward


class PolicyObservationEncoder(torch.nn.Module):
    def __init__(self, obs_spec, hidden_dim):
        super(PolicyObservationEncoder, self).__init__()
        self.ter_enc = TerrainEncoder(obs_spec["perception"].shape[2:], hidden_dim)
        self.state_enc = torch.nn.Sequential(
            torch.nn.Linear(3 + 9 + 3 + 4 + 3, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.LayerNorm(hidden_dim))
        self.shared_state_enc = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.LayerNorm(hidden_dim)
        )

    def forward(self,
                perception: torch.Tensor,
                velocity: torch.Tensor,
                rotation: torch.Tensor,
                omega: torch.Tensor,
                thetas: torch.Tensor,
                goal_vec: torch.Tensor,
                ):
        y_ter = self.ter_enc.forward(perception)
        x_state = torch.cat([velocity, rotation.flatten(start_dim=1), omega, thetas, goal_vec], dim=1)
        y_state = self.state_enc(x_state)
        y_shared = self.shared_state_enc(torch.cat([y_ter, y_state], dim=1))
        return y_shared


class Policy(torch.nn.Module):
    """
    Policy network for the flipper task. The policy network takes in the observation and goal vector and outputs
    """

    def __init__(self, obs_spec, act_spec, hidden_dim):
        super(Policy, self).__init__()
        self.encoder = PolicyObservationEncoder(obs_spec, hidden_dim)
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, act_spec.shape[1]),
        )

    def forward(self,
                perception: torch.Tensor,
                goal_vec: torch.Tensor,
                velocity: torch.Tensor,
                rotation: torch.Tensor,
                omega: torch.Tensor,
                thetas: torch.Tensor,
                ):
        y_shared = self.encoder.forward(perception, velocity, rotation, omega, thetas, goal_vec)
        return self.policy(y_shared)


def make_policy(env: TorchRLEnv, hidden_dim: int) -> TensorDictModule:
    net = Policy(env.observation_spec, env.action_spec, hidden_dim)
    policy = TensorDictModule(net,
                              in_keys=["perception", "goal_vec", "velocity", "rotation", "omega", "thetas"],
                              out_keys=["action"]
                              )
    return policy
