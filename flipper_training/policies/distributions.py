from torchrl.modules.distributions.continuous import SafeTanhTransform
import torch
from torch import distributions as D
from torchrl.modules.distributions.utils import (
    FasterTransformedDistribution,
)
from torch.distributions import constraints


class TanhGSDE(FasterTransformedDistribution):
    """
    A Tanh-transformed Gaussian distribution with state-dependent noise (gSDE).

    Args:
        loc (torch.Tensor): Mean of the action distribution.
        log_std (torch.Tensor): Log-standard deviation of the noise distribution.
        latent_sde (torch.Tensor): State-dependent features for noise modulation.
        low (torch.Tensor or float, optional): Minimum value of the distribution. Default is -1.0.
        high (torch.Tensor or float, optional): Maximum value of the distribution. Default is 1.0.
        full_std (bool, optional): If True, use a full covariance matrix for noise. Default is True.
        use_expln (bool, optional): Use expln function for std to ensure positivity. Default is False.
        squash_output (bool, optional): Apply tanh transformation to bound actions. Default is True.
        learn_features (bool, optional): If True, gradients flow through latent_sde. Default is False.
        upscale (float, optional): Scaling factor for loc. Default is 5.0.
        tanh_loc (bool, optional): Apply tanh scaling to loc. Default is False.
        safe_tanh (bool, optional): Use safe tanh transform to avoid numerical issues. Default is True.
        epsilon (float, optional): Small value to avoid numerical instability. Default is 1e-6.
    """

    arg_constraints = {
        "loc": constraints.real,
        "log_std": constraints.real,
        "latent_sde": constraints.real,
    }
    num_params = 3  # loc, log_std, latent_sde

    def __init__(
        self,
        loc: torch.Tensor,
        log_std: torch.Tensor,
        latent_sde: torch.Tensor,
        low: torch.Tensor | float = -1.0,
        high: torch.Tensor | float = 1.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = True,
        learn_features: bool = False,
        upscale: float = 5.0,
        tanh_loc: bool = False,
        safe_tanh: bool = True,
        epsilon: float = 1e-6,
    ):
        # Validate inputs
        if isinstance(high, torch.Tensor) or isinstance(low, torch.Tensor):
            if not (high > low).all():
                raise RuntimeError("high must be strictly greater than low")
        elif isinstance(high, float) and isinstance(low, float):
            if not high > low:
                raise RuntimeError("high must be strictly greater than low")

        self.device = loc.device
        self.low = torch.as_tensor(low, device=self.device)
        self.high = torch.as_tensor(high, device=self.device)
        self.non_trivial_min = (self.low != -1.0).any() if isinstance(self.low, torch.Tensor) else self.low != -1.0
        self.non_trivial_max = (self.high != 1.0).any() if isinstance(self.high, torch.Tensor) else self.high != 1.0
        self.full_std = full_std
        self.use_expln = use_expln
        self.squash_output = squash_output
        self.learn_features = learn_features
        self.upscale = torch.as_tensor(upscale, device=self.device)
        self.tanh_loc = tanh_loc
        self.epsilon = epsilon

        # Initialize transformation
        if squash_output:
            t = SafeTanhTransform() if safe_tanh else D.TanhTransform()
            if self.non_trivial_min or self.non_trivial_max:
                t = D.ComposeTransform([t, D.AffineTransform(loc=(self.high + self.low) / 2, scale=(self.high - self.low) / 2)])
        else:
            t = D.identity_transform
        self._t = t

        # Store dimensions
        self.latent_sde_dim = latent_sde.shape[-1]
        self.action_dim = loc.shape[-1]

        # Initialize distribution
        self.update(loc, log_std, latent_sde)

    def update(self, loc: torch.Tensor, log_std: torch.Tensor, latent_sde: torch.Tensor) -> None:
        """Update the distribution parameters."""
        if self.tanh_loc:
            loc = (loc / self.upscale).tanh() * self.upscale
        self.loc = loc
        self.log_std = log_std
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()

        # Compute standard deviation
        std = self.get_std(self.log_std)

        # Sample exploration matrix
        self.weights_dist = D.Normal(torch.zeros_like(std), std)
        self.exploration_mat = self.weights_dist.rsample()
        self.exploration_matrices = self.weights_dist.rsample((loc.shape[0],))

        # Compute variance
        variance = torch.mm(self._latent_sde**2, std**2)
        self.base_dist = D.Independent(D.Normal(loc, torch.sqrt(variance + self.epsilon)), 1)

        # Update transformed distribution
        super().__init__(self.base_dist, self._t)

    def get_std(self, log_std: torch.Tensor) -> torch.Tensor:
        """Compute standard deviation from log_std."""
        if self.use_expln:
            below_threshold = torch.exp(log_std) * (log_std <= 0)
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            std = torch.exp(log_std)

        if self.full_std:
            return std
        # Reduce to diagonal std
        return torch.ones(self.latent_sde_dim, self.action_dim, device=log_std.device) * std

    def get_noise(self, latent_sde: torch.Tensor) -> torch.Tensor:
        """Generate state-dependent noise."""
        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return torch.mm(latent_sde, self.exploration_mat)
        latent_sde = latent_sde.unsqueeze(1)
        noise = torch.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(1)

    @property
    def mode(self) -> torch.Tensor:
        """Return the mode (deterministic action)."""
        mode = self.base_dist.mean
        for t in self.transforms:
            mode = t(mode)
        return mode

    @property
    def deterministic_sample(self) -> torch.Tensor:
        """Return the deterministic sample (same as mode)."""
        return self.mode

    @property
    def min(self) -> torch.Tensor:
        return self.low

    @property
    def max(self) -> torch.Tensor:
        return self.high

    def log_prob(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute log-probability of the action."""
        if self.squash_output:
            gaussian_value = self._t.inv(value)
        else:
            gaussian_value = value
        log_prob = self.base_dist.log_prob(gaussian_value)
        if self.squash_output:
            log_prob -= torch.sum(torch.log(1.0 - value**2 + self.epsilon), dim=-1)
        return log_prob

    def entropy(self) -> torch.Tensor:
        """Compute entropy (None if squashed)."""
        if self.squash_output:
            raise NotImplementedError("Entropy is not implemented for squashed distributions.")
        return self.base_dist.entropy()

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """Sample actions with state-dependent noise."""
        noise = self.get_noise(self._latent_sde)
        actions = self.base_dist.mean + noise
        for t in self.transforms:
            actions = t(actions)
        return actions
