import torch
import platform


def set_device(device: str):
    """
    Set the device for the torch module
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    elif device == "mps" and torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autodevice():
    if platform.system() == "Darwin":
        return set_device("cpu")
    return set_device("cuda")
