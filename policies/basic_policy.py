import torch
from flipper_training.environment.env import Env
from tensordict.nn import TensorDictModule
from torchrl.modules import NormalParamExtractor, ProbabilisticActor, TanhNormal, ValueOperator, ActorValueOperator

__all__ = ["make_actor_value_policy"]


class TerrainEncoder(torch.nn.Module):
    def __init__(self, img_shape, output_size):
        super(TerrainEncoder, self).__init__()
        self.img_shape = img_shape
        self.output_size = output_size
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(img_shape[0], 8, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8, track_running_stats=False),
            torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16, track_running_stats=False),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32, track_running_stats=False),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64, track_running_stats=False),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * (img_shape[1] // 16) * (img_shape[2] // 16), output_size)
        )
        self.forward = self.encoder.forward


class PolicyObservationEncoder(torch.nn.Module):
    def __init__(self, obs_spec, hidden_dim):
        super(PolicyObservationEncoder, self).__init__()
        self.ter_enc = TerrainEncoder(obs_spec["perception"].shape[1:], hidden_dim)
        self.hidden_dim = hidden_dim
        self.state_enc = torch.nn.Sequential(
            torch.nn.Linear(obs_spec["observation"].shape[-1], hidden_dim),
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
                observation: torch.Tensor,
                ):
        if perception.ndim > 4:
            B, T = perception.shape[:2]
            perception = perception.flatten(0, 1)
            y_ter = self.ter_enc.forward(perception)
            y_ter = y_ter.reshape((B, T, -1))
        else:
            y_ter = self.ter_enc.forward(perception)
        y_state = self.state_enc(observation)
        y_shared = self.shared_state_enc(torch.cat([y_ter, y_state], dim=-1))
        return y_shared


def make_mlp_layer_module(hidden_dim: int):
    return torch.nn.Sequential(
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.Tanh(),
        torch.nn.LayerNorm(hidden_dim),
    )


class ActorPolicy(torch.nn.Module):
    """
    Policy network for the flipper task. The policy network takes in the observation and goal vector and outputs
    """

    def __init__(self, hidden_dim, act_spec, actor_mlp_layers):
        super(ActorPolicy, self).__init__()
        self.policy = torch.nn.Sequential(
            *[make_mlp_layer_module(hidden_dim) for _ in range(actor_mlp_layers)],  # hidden layers
            torch.nn.Linear(hidden_dim, 2 * act_spec.shape[1]),
        )
        self.extractor = NormalParamExtractor()

    def forward(self, y_shared):
        mu_sigma = self.policy(y_shared)
        return self.extractor(mu_sigma)


class ValueFunction(torch.nn.Module):
    def __init__(self, hidden_dim, value_mlp_layers):
        super(ValueFunction, self).__init__()
        self.value = torch.nn.Sequential(
            *[make_mlp_layer_module(hidden_dim) for _ in range(value_mlp_layers)],  # hidden layers
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, y_shared):
        return self.value(y_shared)


def make_actor_value_policy(env: Env,
                            hidden_dim: int,
                            actor_mlp_layers: int,
                            value_mlp_layers: int,
                            ) -> ActorValueOperator:
    encoder = PolicyObservationEncoder(env.observation_spec, hidden_dim)
    encoder_module = TensorDictModule(encoder, in_keys=["perception", "observation"], out_keys=["y_shared"])
    actor = ActorPolicy(hidden_dim, env.action_spec, actor_mlp_layers)
    actor_td = TensorDictModule(actor, in_keys=["y_shared"], out_keys=["loc", "scale"])
    actor_module = ProbabilisticActor(module=actor_td,
                                      spec=env.action_spec,
                                      in_keys=["loc", "scale"],
                                      distribution_class=TanhNormal,
                                      distribution_kwargs={
                                          "low": env.action_spec.space.low[0],  # pass only the values without a batch dimension
                                          "high": env.action_spec.space.high[0],  # pass only the values without a batch dimension
                                      },
                                      return_log_prob=True)
    value = ValueFunction(hidden_dim, value_mlp_layers)
    value_module = ValueOperator(value, in_keys=["y_shared"])
    actor_value = ActorValueOperator(encoder_module, actor_module, value_module)
    return actor_value
