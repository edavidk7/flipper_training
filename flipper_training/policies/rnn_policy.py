import torch
import torch.nn as nn
from copy import deepcopy
from dataclasses import dataclass, field

from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import (
    NormalParamExtractor,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
    ActorValueOperator,
    ActorCriticWrapper,
)
# Assuming TensorDictModule can handle RNN state passing now
# If not, might need GRUModule or LSTMModule from torchrl.modules if available/needed
# Or a custom wrapper like in the tutorial. Let's try with TensorDictModule first.

# Import your existing classes
from flipper_training.environment.env import Env
from flipper_training.utils.logutils import get_terminal_logger
from . import PolicyConfig, EncoderCombiner, MLP  # Keep MLP for heads after RNN

# Add RNN types
from torch.nn import GRU, LSTM

__all__ = ["RNNPolicyConfig"]


@dataclass
class RNNPolicyConfig(PolicyConfig):
    share_encoder_rnn: bool  # Share both encoder and RNN?
    rnn_type: str  # 'gru' or 'lstm'
    rnn_hidden_size: int
    rnn_num_layers: int
    # MLP options for heads *after* the RNN
    actor_mlp_opts: dict
    value_mlp_opts: dict
    # --- Standard Opts ---
    actor_optimizer_opts: dict
    value_optimizer_opts: dict

    # --- Internal ---
    _actor_recurrent_spec: dict = field(init=False, default_factory=dict)
    _value_recurrent_spec: dict = field(init=False, default_factory=dict)
    _common_recurrent_spec: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.logger = get_terminal_logger("RNNPolicyConfig")
        if self.rnn_type.lower() not in ["gru", "lstm"]:
            raise ValueError(f"Unsupported rnn_type: {self.rnn_type}. Choose 'gru' or 'lstm'.")

        # Define recurrent keys based on RNN type
        # We use nested keys ("recurrent_state", "hidden") etc. as TorchRL often expects
        # tuple keys for nested TensorDict structures representing states.
        if self.rnn_type.lower() == "gru":
            self._common_recurrent_spec = {"in": ("recurrent_state", "hidden"), "out": ("next", "recurrent_state", "hidden")}
        else:  # LSTM
            self._common_recurrent_spec = {
                "in": (("recurrent_state", "hidden"), ("recurrent_state", "cell")),
                "out": (("next", "recurrent_state", "hidden"), ("next", "recurrent_state", "cell")),
            }
        # Create copies for separate actor/value if needed, though names might differ later
        self._actor_recurrent_spec = deepcopy(self._common_recurrent_spec)
        self._value_recurrent_spec = deepcopy(self._common_recurrent_spec)

    def _build_rnn_module(self, input_size: int) -> tuple[nn.Module, str]:
        """Builds the selected RNN module."""
        rnn_cls = GRU if self.rnn_type.lower() == "gru" else LSTM
        rnn_module = rnn_cls(
            input_size=input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True,  # TorchRL generally expects batch_first
        )
        # The feature output key from the RNN module
        rnn_feature_key = "rnn_features"
        return rnn_module, rnn_feature_key

    def create(self, env: Env, **kwargs):
        action_spec = env.action_spec
        encoders = {o.name: o.get_encoder() for o in env.observations}
        encoder_combiner = EncoderCombiner(encoders)
        encoder_feature_key = "encoder_features"  # Output key from combined encoder

        # Wrap the encoder
        encoder_module = TensorDictModule(
            encoder_combiner,
            in_keys=list(encoder_combiner.encoders.keys()),
            out_keys=[encoder_feature_key],
        )

        # Build the base RNN layer
        rnn_module, rnn_feature_key = self._build_rnn_module(input_size=encoder_combiner.output_dim)

        # --- Structure based on sharing ---
        if self.share_encoder_rnn:
            actor_value_wrapper = self._create_shared(action_spec, encoder_module, encoder_feature_key, rnn_module, rnn_feature_key)
            # Params: Encoder + RNN + Actor Head + Value Head
            optim_groups = [
                {
                    "params": list(actor_value_wrapper.get_policy_operator().parameters())
                    + list(actor_value_wrapper.common_operator.parameters()),  # ActorHead + RNN + Encoder
                    "name": "actor_common",
                    **self.actor_optimizer_opts,
                },
                {
                    "params": actor_value_wrapper.get_value_operator().parameters(),  # Value Head only
                    "name": "value_head",
                    **self.value_optimizer_opts,
                },
            ]
            self.logger.info("Using shared Encoder+RNN. Actor optimizer settings used for Encoder+RNN.")

        else:  # Separate Actor/Critic paths after encoder (each gets its own RNN)
            # Note: You might also want a `share_rnn_only` option later.
            actor_value_wrapper = self._create_separate(action_spec, encoder_module, encoder_feature_key, rnn_module, rnn_feature_key)
            # Params: Encoder + ActorRNN + ActorHead | ValueRNN + ValueHead
            optim_groups = [
                {  # Actor components: Encoder + ActorRNN + ActorHead
                    "params": actor_value_wrapper.get_policy_operator().parameters(),
                    "name": "policy_operator",
                    **self.actor_optimizer_opts,
                },
                {  # Value components: ValueRNN + ValueHead (shares encoder implicitly via wrapper)
                    # Note: If ActorCriticWrapper doesn't deepcopy the encoder passed implicitly,
                    # this needs adjustment. Let's assume it manages parameters correctly.
                    # Or more clearly: pass encoder explicitly to value if needed.
                    # Let's assume the structure implies separate RNNs *after* a shared encoder basis
                    # (if not, the separate creation logic needs full duplication).
                    # For now, assume ValueOperator includes its unique RNN + Head Params.
                    "params": actor_value_wrapper.get_value_operator().parameters(),
                    "name": "value_operator",
                    **self.value_optimizer_opts,
                },
            ]
            self.logger.info("Using separate RNNs for actor and critic (sharing encoder).")

        if kwargs.get("device", None) is not None:
            actor_value_wrapper.to(kwargs["device"])
        if kwargs.get("weights_path", None) is not None:
            # Loading needs care with RNN state shapes if saved differently
            self.logger.warning("Loading weights for RNN policies. Ensure saved state_dict matches.")
            actor_value_wrapper.load_state_dict(torch.load(kwargs["weights_path"], map_location=actor_value_wrapper.device))

        # Recurrent policies typically don't add env transforms here
        return actor_value_wrapper, optim_groups, None

    def _create_shared(self, action_spec, encoder_module: TensorDictModule, encoder_feature_key: str, rnn_module: nn.Module, rnn_feature_key: str):
        # Wrap the RNN module to handle state
        rnn_td_module = TensorDictModule(
            module=rnn_module,
            in_keys=[encoder_feature_key] + list(self._common_recurrent_spec["in"]),
            out_keys=[rnn_feature_key] + list(self._common_recurrent_spec["out"]),
        )

        # Common part: Encoder -> RNN
        common_operator = TensorDictSequential(encoder_module, rnn_td_module)

        # --- Actor Head ---
        if "in_dim" in self.actor_mlp_opts or "out_dim" in self.actor_mlp_opts:
            raise ValueError("in_dim/out_dim not allowed for actor head MLP opts.")
        actor_head_mlp = MLP(in_dim=self.rnn_hidden_size, out_dim=2 * action_spec.shape[-1], **self.actor_mlp_opts)
        actor_head = TensorDictModule(
            nn.Sequential(actor_head_mlp, NormalParamExtractor()),
            in_keys=[rnn_feature_key],
            out_keys=["loc", "scale"],
        )
        policy_operator = ProbabilisticActor(
            module=actor_head,  # Input is already processed by common_operator
            spec=action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": action_spec.space.low.squeeze(),  # Assuming low/high are shape (1, D) or (D,)
                "high": action_spec.space.high.squeeze(),
            },
            return_log_prob=True,
        )

        # --- Value Head ---
        if "in_dim" in self.value_mlp_opts or "out_dim" in self.value_mlp_opts:
            raise ValueError("in_dim/out_dim not allowed for value head MLP opts.")
        value_head_mlp = MLP(in_dim=self.rnn_hidden_size, out_dim=1, **self.value_mlp_opts)
        value_operator = ValueOperator(
            module=value_head_mlp,
            in_keys=[rnn_feature_key],  # Input processed by common_operator
            # out_keys=["state_value"] # Default output key for ValueOperator
        )

        return ActorValueOperator(
            common_operator=common_operator,
            policy_operator=policy_operator,
            value_operator=value_operator,
        )

    def _create_separate(self, action_spec, encoder_module: TensorDictModule, encoder_feature_key: str, rnn_module: nn.Module, rnn_feature_key: str):
        # --- Actor Branch ---
        actor_rnn_td_module = TensorDictModule(
            module=deepcopy(rnn_module),  # Separate RNN instance
            in_keys=[encoder_feature_key] + list(self._actor_recurrent_spec["in"]),
            out_keys=[rnn_feature_key] + list(self._actor_recurrent_spec["out"]),
        )
        actor_encoder_rnn = TensorDictSequential(deepcopy(encoder_module), actor_rnn_td_module)  # Separate Encoder+RNN path

        if "in_dim" in self.actor_mlp_opts or "out_dim" in self.actor_mlp_opts:
            raise ValueError("in_dim/out_dim not allowed for actor head MLP opts.")
        actor_head_mlp = MLP(in_dim=self.rnn_hidden_size, out_dim=2 * action_spec.shape[-1], **self.actor_mlp_opts)
        actor_head = TensorDictModule(
            nn.Sequential(actor_head_mlp, NormalParamExtractor()),
            in_keys=[rnn_feature_key],
            out_keys=["loc", "scale"],
        )
        policy_module = ProbabilisticActor(
            module=TensorDictSequential(actor_encoder_rnn, actor_head),  # Full path inside actor
            spec=action_spec,
            in_keys=["loc", "scale"],  # These are produced by the final module in sequence
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": action_spec.space.low.squeeze(),
                "high": action_spec.space.high.squeeze(),
            },
            return_log_prob=True,
            # Pass recurrent keys if ProbabilisticActor needs them explicitly?
            # Usually handled by the contained TensorDictSequential if keys match up.
        )

        # --- Value Branch ---
        # Define value recurrent keys if they need different names, otherwise use common
        value_rnn_td_module = TensorDictModule(
            module=deepcopy(rnn_module),  # Separate RNN instance
            in_keys=[encoder_feature_key] + list(self._value_recurrent_spec["in"]),
            out_keys=[rnn_feature_key] + list(self._value_recurrent_spec["out"]),
        )
        value_encoder_rnn = TensorDictSequential(deepcopy(encoder_module), value_rnn_td_module)  # Separate Encoder+RNN path

        if "in_dim" in self.value_mlp_opts or "out_dim" in self.value_mlp_opts:
            raise ValueError("in_dim/out_dim not allowed for value head MLP opts.")
        value_head_mlp = MLP(in_dim=self.rnn_hidden_size, out_dim=1, **self.value_mlp_opts)
        value_head_td_module = TensorDictModule(
            module=value_head_mlp,
            in_keys=[rnn_feature_key],
            out_keys=["state_value"],
        )
        value_operator = TensorDictSequential(value_encoder_rnn, value_head_td_module)

        # Wrap using ActorCriticWrapper
        # This wrapper expects policy and value operators that are self-contained
        return ActorCriticWrapper(
            policy_operator=policy_module,
            value_operator=value_operator,
        )
