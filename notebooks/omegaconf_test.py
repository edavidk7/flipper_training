# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from omegaconf import OmegaConf

# %%
from dataclasses import dataclass

class MySchizoClass:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
        
@dataclass
class MySchizoConfig:
    a: int
    b: int
    c: MySchizoClass
    
    
config = MySchizoConfig(a=1, b=2, c=MySchizoClass(3, 4))
# %%
def parse_my_schizo_class(c):
    # Resolver for MySchizoClass: expects input as a string "a,b"
    try:
        parts = c.split(',')
        if len(parts) != 2:
            raise ValueError(f"Expected 2 comma-separated values, got: {c}")
        a = int(parts[0].strip())
        b = int(parts[1].strip())
        return MySchizoClass(a, b)
    except Exception as e:
        raise ValueError(f"Error parsing MySchizoClass from input '{c}': {e}")
        
OmegaConf.register_new_resolver("my_schizo_class", parse_my_schizo_class)

# %%
OmegaConf.structured(config)
