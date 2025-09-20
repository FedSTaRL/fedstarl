from typing import List, Dict, Any
from sklearn.metrics import fbeta_score, confusion_matrix, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from pfl.metrics import Weighted, MetricName, MetricValue

from .module import BaseModule


__all__ = ['RNNBaseModule']


class RNNBaseModule(BaseModule):
    def __init__(self, loss_kwargs: Dict[str, Any]={'loss_fn': 'bce'}, *args, **kwargs):
        assert loss_kwargs.get('loss_fn', None) is not None, f'`loss_kwargs` needs to have `loss_fn` attribute!'
        super().__init__(loss_kwargs, *args, **kwargs)
    
    def loss(self, x: Tensor | PackedSequence, y: Tensor, N: Tensor=None, batch_size: int=None, eval=False, forward: bool=False):
        self.eval() if eval else self.train()
        #x_dtype = x.data.dtype
        
        if self._loss_fn == 'bce':
            if not forward: 
                # x are raw inputs, hence need to be passed through forward pass     
                logits = self(x, N=N, batch_size=batch_size)
            else:
                # x are logits
                x = x.data if isinstance(x, PackedSequence) else x
                logits = x

            x_device = logits.device
            if y.device != x_device:
                y = y.to(x_device)

            x_dtype = x.data.dtype if isinstance(x, PackedSequence) else x.dtype
            assert y.device == x_device, f'{y.device, x_device}'
            return self.criterion[self._loss_fn](logits.flatten(), y.type(x_dtype).flatten())
        else:
            raise RuntimeError
    
    def metrics(self, x: Tensor | PackedSequence, y: Tensor, N: Tensor=None, batch_size: int=None, eval=False, cm_text_lables: List[str | int]=[0,1]):
        logits = self(x, N=N, batch_size=batch_size)
        num_samples = len(y)

        loss_value = self.loss(logits, y, eval=eval, forward=True)
        y_preds = self._compute_predictions_from_logits(logits)
        if len(y_preds.shape) > 1:
            y_preds = y_preds.flatten()
            y = y.flatten()
        
        device = y.device
        if device != logits.device:
            device = logits.device
            y = y.to(device)
        correct = torch.sum(torch.eq((y_preds > 0.5).float(), y.flatten()))
        #print('correct', correct.item() / num_samples)
        #print('loss', loss_value.item() / num_samples, loss_value)
        cm_matrix = confusion_matrix(
            y.flatten().detach().cpu().numpy(),
            y_preds.flatten().detach().cpu().numpy(),
            labels=cm_text_lables,
        )
        tn, fp, fn, tp = cm_matrix.ravel()
        f05 = fbeta_score(y.detach().cpu().numpy(), y_preds.detach().cpu().numpy(), average="macro", beta=0.5)
        f1 = f1_score(y.detach().cpu().numpy(), y_preds.detach().cpu().numpy(), average='macro')
        precision = precision_score(y.detach().cpu().numpy(), y_preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        recall = recall_score(y.detach().cpu().numpy(), y_preds.detach().cpu().numpy(), average='macro', zero_division=0.0)
        #print('acc', correct)
        #print('f05', f05, num_samples)
        #print('f1', f1, num_samples)
        #print('precison', precison, num_samples)
        #print('recall', recall, num_samples)
        return {
            'loss': Weighted(loss_value.item(), 1),
            'accuracy': Weighted(correct.item(), num_samples),
            'loss_tensor': loss_value,
            #'cm_matrix': cm_matrix.ravel(),
            'tn': tn, 
            'fp': fp, 
            'fn': fn, 
            'tp': tp,
            'f0.5': Weighted(f05, 1),
            'f1': Weighted(f1, 1),
            'precision': Weighted(precision, 1),
            'recall': Weighted(recall, 1),
            #'y': y.flatten().detach().cpu().numpy(),
            #'y_preds': y_preds.flatten().detach().cpu().numpy(),
        }