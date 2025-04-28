import torch
from torchrl.envs import Transform


class RawRewardSaveTransform(Transform):
    """
    Save the raw reward in the tensordict.
    """

    def __init__(self):
        super().__init__(in_keys=["reward"], out_keys=["raw_reward"])

    def _apply_transform(self, obs: torch.Tensor) -> None:
        """
        Save the raw reward in the tensordict.
        """
        return obs
