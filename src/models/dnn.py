import torch.nn as nn
import numpy as np

from torch import Tensor
from typing import List, Tuple
from ..nn.module import BaseModule

from src.utils.models import image_classification_loss, image_classification_metrics

__all__ = ['DNN', 'SimpleDNN']


class DNN(BaseModule):
    def __init__(self, 
                 input_shape: Tuple[int, ...], 
                 hidden_dims: Tuple[int, ...],
                 num_outputs: int,
                 *args, **kwargs):
        super().__init__(loss_kwargs={}, *args, **kwargs)
    
        in_features = int(np.prod(input_shape))
        layers: List[nn.Module] = [nn.Flatten()]
        for dim in hidden_dims:
            layers.extend([nn.Linear(in_features, dim), nn.ReLU()])
            in_features = dim
        layers.append(nn.Linear(in_features, num_outputs))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs: Tensor, **kwargs):
        return self.model(inputs)

    def loss(self, inputs: Tensor, targets: Tensor, eval: bool=False, **kwargs):
        return image_classification_loss(self, inputs, targets, eval)

    def metrics(self, inputs: Tensor, targets: Tensor, eval: bool=False, **kwargs):
        return image_classification_metrics(self, inputs, targets, eval)


class SimpleDNN(DNN):
    def __init__(self, 
                 input_shape: Tuple[int, ...], 
                 num_outputs: int,
                 *args, **kwargs):
        super().__init__(input_shape=input_shape, hidden_dims=[200, 200], 
                         num_outputs=num_outputs,*args, **kwargs)
        """
        Feed-forward neural network with 2 hidden layers of size 200.
        This is the same architecture as used in McMahan et al. 2017 
        https://arxiv.org/pdf/1602.05629.pdf.
        """