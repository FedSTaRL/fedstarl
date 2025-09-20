import torch
import pandas as pd
import logging

from typing import Optional, Any, Dict, List, Tuple, Callable, Type, Literal
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.nn.utils.rnn import PackedSequence, pad_sequence, pack_sequence
from pfl.data.pytorch import PyTorchDataDataset as PFLPyTorchDataDataset

from .base_pfl_central import BaseSequenceDataset
from src.utils.datasets import BaseDataObject
from abc import abstractmethod

__all__ = [
    "BaseSequenceFederatedDataset"
    "PyTorchFederatedSequenceDataset",
]


class BaseSequenceFederatedDataset(TorchDataset):
    """
    Extends the base torch.utils.data.Dataset to handle federated data for 
    the pfl framework.
    """
    def __init__(self,
                 federated_data: Dict[str | int, List[BaseDataObject] | Tuple[BaseDataObject]],
                 mode: str=None,
                 ) -> None: 
        self._mode = mode
        self._prep_data(federated_data=federated_data, mode=mode)
        super().__init__()

    @abstractmethod
    def _prep_data(self, federated_data, mode: Literal['train', 'val']=None):
        #
        # Example code
        #  
        # assert mode is not None, f'mode cannot be None'
        # assert mode in ['train', 'val'], f'{mode} should be either ["train", "val"]'
        # self._user_id_to_idx = {client: idx for idx, client in enumerate(federated_data.keys())}
        # self._idx_to_user_id = {idx: client for idx, client in enumerate(federated_data.keys())}
        # self._user_id_to_data = {self._user_id_to_idx[client]: BaseSequenceDataset(data=data[mode]) for client, data in federated_data.items()}
        raise NotImplementedError
    
    def __getitem__(self, index):
        """ index: int --> user_id """
        return self._user_id_to_data[index]

    def __len__(self):
        return len(self._user_id_to_data)
        
    @property
    def user_id_to_idx(self):
        return self._user_id_to_idx
    
    @property
    def idx_to_user_id(self):
        return self._idx_to_user_id
    
    def _configure_cli_logger(self, log_level=logging.INFO):
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(log_level)
        self.cli_logger = logger


class PyTorchFederatedSequenceDataset(PFLPyTorchDataDataset):
    def __init__(self, 
                 raw_data: torch.utils.data.Dataset, 
                 user_id: int | None = None,
                 user_str: str | None = None, 
                 metadata: Dict[str, Any] | None = None, 
                 train_kwargs: Dict[str, Any] | None = None, 
                 eval_kwargs: Dict[str, Any] | None = None, 
                 **dataloader_kwargs):
        if dataloader_kwargs.get('collate_fn', 0) == 0:
            raise AttributeError('collate_fn is required for this dataset!')
        self._user_str = user_str
        super().__init__(raw_data, user_id, metadata, train_kwargs, eval_kwargs, **dataloader_kwargs)

    def __getitem__(self, index):
        return self._raw_data[self._user_id][index]

    def __len__(self):
        return len(self._raw_data[self._user_id])

    def iter(self, batch_size: Optional[int]): 
        return iter(
            TorchDataLoader(self._raw_data[self._user_id],
                            batch_size=batch_size,
                            **self._dataloader_kwargs))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(data=[{len(self._raw_data[self._user_id])}], user_id={self._user_id})'
    
