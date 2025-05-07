from dataclasses import dataclass

import torch
from torchrl.data import Unbounded
from flipper_training.engine.engine_state import PhysicsState, PhysicsStateDer
from flipper_training.rl_objectives.random_nav_latent import RandomNavigationWithLatentControl
from . import Observation, ObservationEncoder


class IdentityEncoder(ObservationEncoder):
    def __init__(self, output_dim: int):
        super(IdentityEncoder, self).__init__(output_dim)

    def forward(self, x):
        return x


@dataclass
class LatentControlParameter(Observation):
    supports_vecnorm = False

    def __post_init__(self):
        if not isinstance(self.env.objective, RandomNavigationWithLatentControl):
            raise ValueError("LatentControlParameter observation only works with RandomNavigationWithLatentControl")

    def __call__(
        self,
        prev_state: PhysicsState,
        action: torch.Tensor,
        prev_state_der: PhysicsStateDer,
        curr_state: PhysicsState,
    ) -> torch.Tensor:
        curr_params = getattr(self.env, "latent_control_params", None)
        if curr_params is None:
            raise ValueError("Latent control parameters not found in the environment. Make sure to set them before calling this observation.")
        return curr_params

    @property
    def dim(self) -> int:
        """
        The dimension of the observation vector.
        """
        return 1

    def get_spec(self) -> Unbounded:
        return Unbounded(
            shape=(self.env.n_robots, self.dim),
            device=self.env.device,
            dtype=self.env.out_dtype,
        )

    def get_encoder(self) -> IdentityEncoder:
        return IdentityEncoder(1)
