import torch
import torch.nn as nn
from copy import deepcopy
from dataclasses import dataclass
from flipper_training.environment.env import Env
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import NormalParamExtractor, ProbabilisticActor, TanhNormal, ValueOperator, ActorCriticWrapper, ActorValueOperator
from flipper_training.utils.logging import get_terminal_logger
from . import PolicyConfig, EncoderCombiner, MLP

__all__ = ["MLPPolicyConfig"]


@dataclass
class MLPPolicyConfig(PolicyConfig):
    share_encoder: bool
    actor_mlp_opts: dict
    value_mlp_opts: dict
    actor_optimizer_opts: dict
    value_optimizer_opts: dict

    def __post_init__(self):
        self.logger = get_terminal_logger("MLPPolicyConfig")

    def create(self, env: Env, **kwargs):
        # Fetch the environment data
        action_spec = env.action_spec
        encoders = {o.name: o.get_encoder() for o in env.observations}
        # Create the encoder
        encoder = EncoderCombiner(encoders)
        if self.share_encoder:
            actor_value_wrapper = self._create_shared(action_spec, encoder)
            optim_groups = [
                {
                    "params": actor_value_wrapper.get_policy_operator().parameters(),
                    "name": "policy_and_encoder",
                    **self.actor_optimizer_opts,
                },
                {
                    "params": actor_value_wrapper.get_value_head().parameters(),  # this is only the value MLP head
                    "name": "value_head",
                    **self.value_optimizer_opts,
                },
            ]
            self.logger.info("Using shared encoder for actor and critic. Actor's optimizer settings will be used for the encoder.")
        else:
            actor_value_wrapper = self._create_separate(action_spec, encoder)
            optim_groups = [
                {
                    "params": actor_value_wrapper.get_policy_operator().parameters(),
                    "name": "policy_operator",
                    **self.actor_optimizer_opts,
                },
                {
                    "params": actor_value_wrapper.get_value_operator().parameters(),
                    "name": "value_operator",
                    **self.value_optimizer_opts,
                },
            ]
        if kwargs.get("device", None) is not None:
            actor_value_wrapper.to(kwargs["device"])
        if kwargs.get("weights_path", None) is not None:
            actor_value_wrapper.load_state_dict(torch.load(kwargs["weights_path"], map_location=actor_value_wrapper.device))
            self.logger.info(f"Loaded weights from {kwargs['weights_path']}")
        return actor_value_wrapper, optim_groups, None

    def _create_separate(self, action_spec, combined_encoder: EncoderCombiner):
        if "in_features" in self.actor_mlp_opts or "out_features" in self.actor_mlp_opts:
            raise ValueError(
                "in_features and out_features are not allowed in the policy MLP options. They are dictated by the environment and the encoder."
            )
        encoder_module = TensorDictModule(
            combined_encoder,
            in_keys={k: k for k in combined_encoder.encoders.keys()},
            out_keys=["y_shared"],
            out_to_in_map=True,
        )
        policy_module = TensorDictModule(
            module=nn.Sequential(
                MLP(in_dim=combined_encoder.output_dim, out_dim=2 * action_spec.shape[1], **self.actor_mlp_opts),
                NormalParamExtractor(),
            ),
            in_keys=["y_shared"],
            out_keys=["loc", "scale"],
        )
        policy_module = ProbabilisticActor(
            module=TensorDictSequential([deepcopy(encoder_module), policy_module]),
            spec=action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": action_spec.space.low[0],  # pass only the values without a batch dimension
                "high": action_spec.space.high[0],  # pass only the values without a batch dimension
            },
            return_log_prob=True,
        )
        if "in_features" in self.value_mlp_opts or "out_features" in self.value_mlp_opts:
            raise ValueError(
                "in_features and out_features are not allowed in the value MLP options. They are dictated by the environment and the encoder."
            )
        value_module = TensorDictModule(
            MLP(in_dim=combined_encoder.output_dim, out_dim=1, **self.value_mlp_opts),
            in_keys=["y_shared"],  # pass the observations as input
            out_keys=["state_value"],
        )
        value_operator = TensorDictSequential(
            [deepcopy(encoder_module), value_module],
        )
        wrapper = ActorCriticWrapper(
            policy_operator=policy_module,
            value_operator=value_operator,
        )
        return wrapper

    def _create_shared(self, action_spec, combined_encoder: EncoderCombiner):
        if "in_features" in self.actor_mlp_opts or "out_features" in self.actor_mlp_opts:
            raise ValueError(
                "in_features and out_features are not allowed in the policy MLP options. They are dictated by the environment and the encoder."
            )
        policy_td = TensorDictModule(
            nn.Sequential(
                MLP(in_dim=combined_encoder.output_dim, out_dim=2 * action_spec.shape[1], **self.actor_mlp_opts),
                NormalParamExtractor(),
            ),
            in_keys=["y_shared"],
            out_keys=["loc", "scale"],
        )
        policy_module = ProbabilisticActor(
            module=policy_td,
            spec=action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": action_spec.space.low[0],  # pass only the values without a batch dimension
                "high": action_spec.space.high[0],  # pass only the values without a batch dimension
            },
            return_log_prob=True,
        )
        value_operator = ValueOperator(
            module=MLP(in_dim=combined_encoder.output_dim, out_dim=1, **self.value_mlp_opts),
            in_keys=["y_shared"],  # pass the observations as input
        )
        encoder_module = TensorDictModule(
            combined_encoder,
            in_keys={k: k for k in combined_encoder.encoders.keys()},
            out_keys=["y_shared"],
            out_to_in_map=True,
        )
        return ActorValueOperator(
            policy_operator=policy_module,
            value_operator=value_operator,
            common_operator=encoder_module,
        )
