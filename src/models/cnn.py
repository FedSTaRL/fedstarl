import torch
import torch.nn as nn

from torch import Tensor
from typing import Any, Dict, Tuple

from ..nn.module import BaseModule
from ..utils.models import image_classification_loss, image_classification_metrics


__all__ = ['SimpleCNN']


class Transpose2D(nn.Module):
    """
    Transpose Tensorflow style image to PyTorch compatible
    """

    def forward(self, inputs: Tensor):
        return inputs.permute((0, 3, 1, 2))


class SimpleCNN(BaseModule):
    def __init__(self, 
                 input_shape: Tuple[int, ...], 
                 num_outputs: int,
                 loss_kwargs: Dict[str, Any]={'loss_fn': 'bce'},
                 *args, **kwargs):
        super().__init__(loss_kwargs=loss_kwargs, *args, **kwargs)
        in_channels = input_shape[-1]
        maxpool_output_size = (input_shape[0] - 4) // 2
        flatten_size = maxpool_output_size * maxpool_output_size * 64
        self.model = nn.Sequential(*[
            Transpose2D(),
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_outputs),
        ])

        # Apply Glorot (Xavier) uniform initialization 
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
    
    def forward(self, inputs: Tensor, **kwargs):
        return self.model(inputs)

    def loss(self, inputs: Tensor, targets: Tensor, eval: bool=False, **kwargs):
        return image_classification_loss(self, inputs, targets, eval)

    def metrics(self, inputs: Tensor, targets: Tensor, eval: bool=False, **kwargs):
        return image_classification_metrics(self, inputs, targets, eval)
        
