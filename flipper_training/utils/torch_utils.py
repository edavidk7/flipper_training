import torch
import platform


def set_device(device: str, high_performance: bool = False) -> torch.device:
    """
    Set the device for the torch module
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.set_float32_matmul_precision('high' if high_performance else 'highest')
        torch.backends.cuda.matmul.allow_tf32 = high_performance
        torch.backends.cudnn.allow_tf32 = high_performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = high_performance
        return torch.device("cuda")
    elif device == "mps" and torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autodevice(high_performance: bool = False):
    if platform.system() == "Darwin":
        return set_device("cpu", high_performance)
    return set_device("cuda", high_performance)
