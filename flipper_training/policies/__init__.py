import torch


def make_mlp_layer_module(d1: int, d2: int | None = None, ln: bool = False):
    if d2 is None:
        d2 = d1
    return torch.nn.Sequential(
        torch.nn.Linear(d1, d2),
        torch.nn.LayerNorm(d2) if ln else torch.nn.Identity(),
        torch.nn.Tanh(),
    )
