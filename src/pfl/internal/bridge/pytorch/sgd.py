# Copyright © 2023-2024 Apple Inc.
from typing import Dict, Tuple, Any

from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import NNTrainHyperParams
from pfl.model.pytorch import PyTorchModel

from pfl.internal.bridge.base import SGDFrameworkBridge
from pfl.internal.bridge.pytorch.utils import clip_norm_and_update

from pfl.metrics import Metrics, Weighted

from src.utils.datasets import BaseDataObject
from src.pfl.data_class import LocalResultMetaData


def _sgd_train_step(pytorch_model, local_optimizer, raw_data, train_kwargs,
                train_step_args, compute_metrics: bool=False) -> None | Metrics:
    #print('compute_metrics', compute_metrics)
    with train_step_args.amp_context:
        if isinstance(raw_data, Dict):
            loss = pytorch_model.loss(**{**raw_data, **train_kwargs})
        elif isinstance(raw_data, BaseDataObject):
            #print("SGD Bridge", 
            #    raw_data.data.data.size(), raw_data.label.size())
            #.loss(x, label, N, batch_size)
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
            #print(type(loss), loss, dir(loss))
            #print('\n Metrics', metrics, '\n')
        else:
            if compute_metrics:
                loss = pytorch_model.loss(*raw_data, **train_kwargs)
                #print(loss)
                metrics = pytorch_model.metrics(*raw_data, **train_kwargs)
                #print(metrics)
                loss = metrics['loss_tensor']
            else:
                loss = pytorch_model.loss(*raw_data, **train_kwargs)
        # drop loss_tensor
        if compute_metrics and 'loss_tensor' in list(metrics.keys()):
            metrics.pop('loss_tensor') 
            #print('\n Metrics', metrics, '\n')
        
        # Scale the loss to get the correct scale for the gradients.
        #print(type(train_step_args.grad_accumulation_state.accumulation_steps), train_step_args.grad_accumulation_state.accumulation_steps)
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

from pfl.model.pytorch import PyTorchModel

class PyTorchSGDBridge(SGDFrameworkBridge[PyTorchModel, NNTrainHyperParams]):
    """
    Concrete PyTorch implementations of utils for stochastic gradient
    descent.
    """

    @staticmethod
    def do_sgd(model: PyTorchModel, user_dataset: AbstractDatasetType,
               train_params: NNTrainHyperParams, 
               compute_metrics: bool=False) -> Tuple[LocalResultMetaData, Metrics | Any]:
        #print('Train Client', user_dataset.user_id)
        #print(model, type(model))
        #if isinstance(model, PyTorchModel):
        #    metrics = model.do_multiple_epochs_of(user_dataset, train_params, _sgd_train_step)
        #    local_results = None
        #else:
        local_results, metrics = model.do_multiple_epochs_of(user_dataset, train_params,
                                    _sgd_train_step, compute_metrics)
        #print(metrics)
        metrics = Metrics() if metrics is None else metrics
        return local_results, metrics
