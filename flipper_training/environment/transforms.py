import torch
from tensordict import TensorDict
from torchrl.envs import Transform

class AdditiveGaussianActionNoise(Transform):
    """
    Add Gaussian noise to the action.
    """

    def __init__(self, scale: float | torch.Tensor):
        super().__init__()
        self.scale = scale

    def forward(self, tensordict: TensorDict) -> TensorDict:
        noise = torch.randn_like(tensordict["action"]) * self.std
        tensordict["action"] += noise
        return tensordict