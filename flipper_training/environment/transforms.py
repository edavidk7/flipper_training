import torch
from torchrl.envs import Transform
from tensordict import TensorDict


class RawRewardSaveTransform(Transform):
    """
    Save the raw reward in the tensordict.
    """

    def __init__(self):
        super().__init__(in_keys=["reward"], out_keys=["raw_reward"])

    def _call(self, tensordict):
        """
        Save the raw reward in the tensordict.
        """
        tensordict["raw_reward"] = tensordict["reward"]
        return tensordict
