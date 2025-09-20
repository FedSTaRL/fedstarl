import os
import random
import torch
import numpy as np

from typing import Optional
from omegaconf import DictConfig, OmegaConf


__all__ = ['seed_everything', 'OrderAction']


max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def seed_everything(seed: Optional[int] = None) -> int:
    r"""
    Function that sets the seed for pseudo-random number generators in: torch, numpy, and Python's random module.

    Args:
        seed: the integer value seed for global random state.
    """
    if not (min_seed_value <= seed <= max_seed_value):
        raise Warning(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


# For privacy 
class OrderAction:
    """
    Adjusted form https://github.com/apple/pfl-research/blob/develop/benchmarks/utils/argument_parsing.py#L268
    
    Order of the lp-norm for local norm clipping only.
    """
    def __init__(self, key):
        self.key = key

    def __call__(self, config: DictConfig, value: str):
        if value is None:
            return
        if value == 'inf':
            OmegaConf.update(config, self.key, np.inf)
        else:
            OmegaConf.update(config, self.key, float(value))