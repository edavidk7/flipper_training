import torch
from pathlib import Path
from flipper_training.environment.env import Env
from tensordict.nn import TensorDictModule
from torchrl.modules import NormalParamExtractor, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.data import TensorSpec
from . import make_mlp_layer_module

__all__ = ["make_policy", "make_value_function"]


class Policy(torch.nn.Module):
    """
    Policy network for the flipper task. The policy network takes in the observation and goal vector and outputs
    """

    def __init__(self, encoders: dict[str, torch.nn.Module], hidden_dim: int, num_hidden: int, act_spec: TensorSpec, ln: bool = False):
        super(Policy, self).__init__()
        self.encoders = torch.nn.ModuleDict(encoders)
        self.input_dim = sum([encoder.output_dim for encoder in encoders.values()])
        self.policy = torch.nn.Sequential(
            make_mlp_layer_module(self.input_dim, hidden_dim, ln),  # input dimension
            *[make_mlp_layer_module(hidden_dim, ln=ln) for _ in range(num_hidden)],  # hidden layers
            make_mlp_layer_module(hidden_dim, 2 * act_spec.shape[1], ln=ln),  # output dimension
        )
        self.extractor = NormalParamExtractor()

    def forward(self, **kwargs):
        y_encs = []
        for key, encoder in self.encoders.items():
            y_enc = encoder(kwargs[key])
            y_encs.append(y_enc)
        y_shared = torch.cat(y_encs, dim=-1)
        mu_sigma = self.policy(y_shared)
        return self.extractor(mu_sigma)


class ValueFunction(torch.nn.Module):
    def __init__(self, encoders: dict[str, torch.nn.Module], hidden_dim: int, num_hidden: int, ln: bool = False):
        super(ValueFunction, self).__init__()
        self.encoders = torch.nn.ModuleDict(encoders)
        self.input_dim = sum([encoder.output_dim for encoder in encoders.values()])
        self.value = torch.nn.Sequential(
            make_mlp_layer_module(self.input_dim, hidden_dim, ln),  # input dimension
            *[make_mlp_layer_module(hidden_dim, ln=ln) for _ in range(num_hidden)],  # hidden layers
            make_mlp_layer_module(hidden_dim, 1, ln=ln),  # output dimension
        )

    def forward(self, **kwargs):
        y_encs = []
        for key, encoder in self.encoders.items():
            y_enc = encoder(kwargs[key])
            y_encs.append(y_enc)
        y_shared = torch.cat(y_encs, dim=-1)
        value = self.value(y_shared)
        return value


def make_policy(
    env: Env,
    policy_opts: dict,
    encoders_opts: dict[str, dict],
    weights_path: str | Path | None = None,
    device: torch.device | None = None,
) -> ProbabilisticActor:
    action_spec = env.action_spec
    observations = env.observations
    encoders = {k: o.get_encoder(**encoders_opts[k]) for k, o in observations.items()}
    policy = Policy(encoders, **policy_opts, act_spec=action_spec)
    policy_td = TensorDictModule(policy, in_keys={k: k for k in observations}, out_keys=["loc", "scale"])
    policy_module = ProbabilisticActor(
        module=policy_td,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low[0],  # pass only the values without a batch dimension
            "high": env.action_spec.space.high[0],  # pass only the values without a batch dimension
        },
        return_log_prob=True,
    )
    if weights_path is not None:
        policy_module.load_state_dict(torch.load(weights_path, map_location=device if device is not None else "cpu"))
    if device is not None:
        policy_module.to(device)
    return policy_module


def make_value_function(
    env: Env,
    value_opts: dict,
    encoders_opts: dict[str, dict],
    weights_path: str | None = None,
    device: torch.device | None = None,
) -> ValueOperator:
    """
    Create the value function operator.
    """
    observations = env.observations
    encoders = {k: o.get_encoder(**encoders_opts[k]) for k, o in observations.items()}
    value_function = ValueFunction(encoders, **value_opts)
    value_operator = ValueOperator(
        module=value_function,
        in_keys={k: k for k in observations},  # pass the observations as input
    )
    if weights_path is not None:
        value_operator.load_state_dict(torch.load(weights_path, map_location=device if device is not None else "cpu"))
    if device is not None:
        value_operator.to(device)
    return value_operator
