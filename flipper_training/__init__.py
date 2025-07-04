import os

os.environ["TORCHDYNAMO_REPRO_LEVEL"] = "4"

from pathlib import Path
import torch
from typing import Type
from lovely_tensors import monkey_patch
import math
from importlib import import_module
from omegaconf import OmegaConf

torch._inductor.config.fallback_random = True
torch._dynamo.config.cache_size_limit = 128


PACKAGE_ROOT = Path(__file__).parent
ROOT = PACKAGE_ROOT.parent


def resolve_class(typename: str) -> Type:
    module, class_name = typename.rsplit(".", 1)
    return getattr(import_module(module), class_name)


monkey_patch()


OmegaConf.register_new_resolver("add", lambda *args: sum(args))
OmegaConf.register_new_resolver("mul", lambda *args: math.prod(args))
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("intdiv", lambda a, b: a // b)
OmegaConf.register_new_resolver("cls", resolve_class)
OmegaConf.register_new_resolver("dtype", lambda s: getattr(torch, s))  # get a torch dtype
OmegaConf.register_new_resolver("tensor", lambda s: torch.tensor(s))
