import numpy as np
import torch


def set_device(device: str) -> torch.device:
    """
    Set the device for the torch module
    """
    if "cuda" in device and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return torch.device(device)
    elif device == "mps" and torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int) -> torch.Generator:
    """
    Seed all the random number generators
    """
    rng = torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.backends.cuda.is_built():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_built():
        torch.mps.manual_seed(seed)
    return rng
