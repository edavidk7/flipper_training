import torch
from torch import distributions as D
from torch.distributions import constraints


class TransformedBeta(D.Independent):
    """Implements a Beta distribution transformed to the interval [low, high].

    The Beta distribution is naturally defined on [0, 1]. This class applies an affine transformation
    to map its samples and probability density to the action space [low, high]. The concentration
    parameters (alpha and beta) are assumed to be positive, typically ensured by the policy network.

    Args:
        concentration1 (torch.Tensor): Alpha parameter of the Beta distribution (must be positive).
        concentration0 (torch.Tensor): Beta parameter of the Beta distribution (must be positive).
        low (torch.Tensor or number, optional): Minimum value of the action space. Default = -1.0.
        high (torch.Tensor or number, optional): Maximum value of the action space. Default = 1.0.
    """

    num_params: int = 2

    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }

    def __init__(
        self,
        concentration1: torch.Tensor,
        concentration0: torch.Tensor,
        low: torch.Tensor | float = -1.0,
        high: torch.Tensor | float = 1.0,
    ):
        # Validate bounds
        err_msg = "TransformedBeta high values must be strictly greater than low values"
        if isinstance(high, torch.Tensor) or isinstance(low, torch.Tensor):
            if not (high > low).all():
                raise RuntimeError(err_msg)
        elif isinstance(high, (int, float)) and isinstance(low, (int, float)):
            if not high > low:
                raise RuntimeError(err_msg)
        else:
            if not all(high > low):
                raise RuntimeError(err_msg)

        # Check for non-trivial bounds
        if isinstance(high, torch.Tensor):
            self.non_trivial_max = (high != 1.0).any()
        else:
            self.non_trivial_max = high != 1.0

        if isinstance(low, torch.Tensor):
            self.non_trivial_min = (low != -1.0).any()
        else:
            self.non_trivial_min = low != -1.0

        # Store device and tensorize bounds
        self.device = concentration1.device
        self.low = torch.as_tensor(low, device=self.device)
        self.high = torch.as_tensor(high, device=self.device)

        # Update distribution with parameters
        self.update(concentration1, concentration0)

    @property
    def min(self):
        return self.low

    @property
    def max(self):
        return self.high

    @property
    def mean(self):
        """Compute the mean of the transformed Beta distribution."""
        # Mean of the base Beta distribution on [0, 1]
        mean = self.concentration1 / (self.concentration1 + self.concentration0)
        # Transform to [low, high]
        transformed_mean = self.low + (self.high - self.low) * mean
        return transformed_mean

    def update(self, concentration1: torch.Tensor, concentration0: torch.Tensor) -> None:
        """Update the distribution with new concentration parameters."""
        self.concentration1 = concentration1
        self.concentration0 = concentration0

        # Define the base Beta distribution on [0, 1]
        base_dist = D.Beta(concentration1, concentration0)

        # Apply affine transformation to map [0, 1] to [low, high]
        transform = D.AffineTransform(self.low, self.high - self.low)
        transformed_dist = D.TransformedDistribution(base_dist, transform)

        # Use Independent to interpret the last dimension as the event dimension
        super().__init__(transformed_dist, 1, validate_args=False)

    @property
    def mode(self):
        """Compute the mode of the transformed Beta distribution."""
        alpha = self.concentration1
        beta = self.concentration0
        # Mode of the base Beta distribution on [0, 1]
        mode = (alpha - 1) / (alpha + beta - 2)
        # Handle boundary cases: if alpha <= 1 or beta <= 1, mode may be at 0 or 1
        mode = torch.where((alpha > 1) & (beta > 1), mode, torch.where(alpha <= 1, torch.zeros_like(mode), torch.ones_like(mode)))
        # Transform to [low, high]
        transformed_mode = self.low + (self.high - self.low) * mode
        return transformed_mode

    @property
    def deterministic_sample(self):
        """Return the mean as the deterministic sample."""
        return self.mean

    def log_prob(self, value: torch.Tensor, **kwargs):
        """Compute the log probability, ensuring zero probability outside [low, high]."""
        # Check if values are outside bounds
        above_or_below = (value < self.low) | (value > self.high)
        # Clamp values to bounds for numerical stability in computation
        value_clamped = torch.clamp(value, self.low, self.high)
        # Compute log probability using the transformed distribution
        lp = super().log_prob(value_clamped, **kwargs)
        # Set log_prob to -inf for values outside bounds
        if above_or_below.any():
            if self.event_shape:
                above_or_below = above_or_below.flatten(-len(self.event_shape), -1).any(-1)
            lp = torch.masked_fill(
                lp,
                above_or_below.expand_as(lp),
                torch.tensor(-float("inf"), device=lp.device, dtype=lp.dtype),
            )
        return lp
