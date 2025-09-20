import torch
import numpy as np


from typing import List, Optional

from pfl.internal.ops.pytorch_ops import (
    get_default_device, get_pytorch_major_version, _get_ph
)


def add_gaussian_noise(tensors: List[np.ndarray], stddev: float,
                       seed: Optional[int]) -> List[torch.Tensor]:
    """
    Adjusted from https://github.com/apple/pfl-research/blob/develop/pfl/internal/ops/pytorch_ops.py#L351

    Add zero mean Gaussian noise to tensors.
    Transferring data to GPU, adding noise, and back to NumPy is faster than
    `np.random.normal`.

    :param tensors:
        A list of tensors to add noise to.
    :param stddev:
        Standard deviation of noise to add.
    :param seed:
        An integer for seed.
    :return:
        Same as `tensors` but with noise added.
    """
    if (get_default_device() == torch.device('mps')
            and get_pytorch_major_version() < 2):
        raise RuntimeError("You are trying to use gaussian noise with MPS, "
                           "please upgrade to torch>=2.0.0")
    g = torch.Generator(
        device=get_default_device()).manual_seed(int(seed)) if seed else None
    # This is a very fast in-memory way of adding noise. Only supported
    # for Gaussian noise.
    #
    # torch.tensor(v ... --> torch.tensor(v.detach().cpu() ....
    #
    return [
        torch.tensor(v.detach().cpu(), device=get_default_device()).add(
            _get_ph(v.shape).normal_(mean=0.0, std=stddev, generator=g))
        for v in tensors
    ]