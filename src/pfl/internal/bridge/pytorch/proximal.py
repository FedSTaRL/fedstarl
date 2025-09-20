from typing import Dict, Tuple, Any

import torch

from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import NNTrainHyperParams
from pfl.model.pytorch import PyTorchModel

from pfl.internal.bridge.base import SGDFrameworkBridge, FedProxFrameworkBridge
from pfl.internal.bridge.pytorch.utils import clip_norm_and_update
from pfl.metrics import Metrics

from src.utils.datasets import BaseDataObject
from src.pfl.data_class import LocalResultMetaData


def _proximal_train_step(pytorch_model, local_optimizer, raw_data,
                         train_kwargs, train_step_args, compute_metrics: bool=False,
                         **kwargs) -> None | Metrics:
    global_weights, mu = kwargs["global_weights"], kwargs["mu"]

    with train_step_args.amp_context:
        if isinstance(raw_data, Dict):
            loss = pytorch_model.loss(**{**raw_data, **train_kwargs})
        elif isinstance(raw_data, BaseDataObject):
            if compute_metrics:
                metrics = pytorch_model.metrics(**{
                        'x': raw_data.data, 'y': raw_data.label, 'N': raw_data.seq_len, 'batch_size': raw_data.batch_size,
                        **train_kwargs
                    })
                #loss_weighted: Weighted = metrics['loss']
                loss = metrics['loss_tensor']
            else:
                loss = pytorch_model.loss(raw_data.data, raw_data.label, raw_data.seq_len, raw_data.batch_size, forward=False, 
                                      **train_kwargs)

        else:
            if compute_metrics:
                metrics = pytorch_model.loss(**raw_data, **train_kwargs)
                loss = metrics['loss_tensor']
            else:
                loss = pytorch_model.loss(*raw_data, **train_kwargs)
        
        if 'loss_tensor' in list(metrics.keys()):
            metrics.pop('loss_tensor') 
            
        # Add proximal term (Definition 2)
        for name, param in pytorch_model.named_parameters():
            if param.requires_grad:
                loss += mu / 2 * torch.norm(param - global_weights[name])**2

        # Scale the loss to get the correct scale for the gradients.
        loss /= train_step_args.grad_accumulation_state.accumulation_steps

    if train_step_args.grad_scaler is None:
        loss.backward()
    else:
        train_step_args.grad_scaler.scale(loss).backward()
    train_step_args.grad_accumulation_state.increment()

    clip_norm_and_update(pytorch_model, local_optimizer, train_step_args)

    if compute_metrics:
        return metrics
    return None


class PyTorchFedProxBridge(FedProxFrameworkBridge[PyTorchModel,
                                                  NNTrainHyperParams]):
    """
    Concrete implementation of FedProx utilities in PyTorch, used by
    FedProx algorithm.
    """

    @staticmethod
    def do_proximal_sgd(model: PyTorchModel, user_dataset: AbstractDatasetType,
                        train_params: NNTrainHyperParams, mu: float,
                        compute_metrics: bool=False) -> Tuple[LocalResultMetaData, Metrics | Any]:
        global_weights = dict(model.get_parameters().items())
        local_results, metrics = model.do_multiple_epochs_of(user_dataset,
                                    train_params,
                                    _proximal_train_step,
                                    global_weights=global_weights,
                                    mu=mu,
                                    compute_metrics=compute_metrics)
        
        return local_results, metrics