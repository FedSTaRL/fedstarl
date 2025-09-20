import torch
import torch.nn as nn

from typing import Literal, Callable, Union
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from ..nn import RNNBaseModule
from .output_heads import OutputHead, OutputHeadOptions


__all__ = ['LSTMNet']


class LSTMNet(RNNBaseModule):
    def __init__(self, 
                 # RNN
                 input_size: int,
                 hidden_size: int, # num of cell states
                 num_layers: int=1,
                 bias: bool=True,
                 batch_first: bool=False,
                 dropout: float=0.0,
                 bidirectional: bool=False,
                 # Output head
                 task: Literal['classification', 'regression', 'forecasting']='classification',
                 d_out: int=1, 
                 d_hidden: int=None, 
                 activation: Union[str, Callable[[Tensor], Tensor]] = nn.ReLU(),
                 reduced: bool=True, 
                 cls_method: Literal['cls_token', 'autoregressive', 'elementwise']='autoregressive',
                 # Base Class
                 loss_kwargs = {'loss_fn': 'bce'},
                *args, **kwargs):
        super().__init__(loss_kwargs, *args, **kwargs)
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._task = task 
        self._batch_first = batch_first
        self._hidden_activation = nn.ReLU() # activation function after Rnn

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                          bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=False) # add an LSTM layer
        
        self.output_head = OutputHead(task=task, d_model=hidden_size, d_out=d_out,
                                      d_hidden=d_hidden, activation=activation, reduced=reduced, 
                                      cls_method=cls_method)

    def forward(self, x: Tensor | PackedSequence, N: Tensor=None, batch_size: int=None) -> Tensor:
        assert isinstance(x, PackedSequence) or isinstance(x, Tensor), f'Input x is not a {PackedSequence} but {type(x)}.'
        assert batch_size is not None
        #if batch_size is None:
        #    batch_size = x.data.size(0) if self._batch_first else x.data.size(1)

        # x is a PackedSequence
        device = x.data.device
        if device != next(iter(self.lstm.parameters())).device:
            device = next(iter(self.lstm.parameters())).device
            x = x.to(device)
        
        dtype = x.data.dtype #if x.data.dtype == torch.float32 else torch.float32
        if dtype != next(iter(self.lstm.parameters())).dtype:
            dtype = next(iter(self.lstm.parameters())).dtype
            x = x.to(dtype)

        if torch.backends.mps.is_available() and x.data.dtype == torch.double:
            dtype = torch.float32
            x = x.to(dtype)

        # initial h0 and c0
        h0 = torch.zeros(self._num_layers, batch_size, self._hidden_size, dtype=dtype).to(device)
        c0 = torch.zeros(self._num_layers, batch_size, self._hidden_size, dtype=dtype).to(device)
        self.hidden = (h0, c0)
        # lstm
        
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # get last hidden state for autoregressive prediction
        last_hidden_state = self.hidden[0][-1]
        # output head 
        if self._task == OutputHeadOptions.classification:
            logits = self.output_head(x=last_hidden_state, N=N, batch_size=batch_size)
        else:
            raise NotImplementedError
        
        return logits
    