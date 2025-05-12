from dataclasses import dataclass

import torch
from torchrl.data import Unbounded
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from . import Observation, ObservationEncoder
from flipper_training.policies import MLP


class PreviousActionEncoder(ObservationEncoder):
    def __init__(self, input_dim: int, output_dim: int, **mlp_kwargs):
        super(PreviousActionEncoder, self).__init__(output_dim)
        self.input_dim = input_dim
        self.mlp = MLP(**mlp_kwargs | {"in_dim": input_dim, "out_dim": output_dim, "activate_last_layer": True})

    def forward(self, x):
        return self.mlp(x)


@dataclass
class PreviousAction(Observation):
    supports_vecnorm = False

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
    ) -> torch.Tensor:
        return action.clone()

    @property
    def dim(self) -> int:
        """
        The dimension of the observation vector.
        """
        return 2 * self.env.robot_cfg.num_driving_parts

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.n_robots, self.dim),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )

    def get_encoder(self) -> PreviousActionEncoder:
        if self.encoder_opts is None:
            self.encoder_opts = {}
        return PreviousActionEncoder(
            input_dim=self.dim,
            **self.encoder_opts,
        )
