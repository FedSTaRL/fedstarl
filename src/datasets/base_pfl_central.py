import logging
import torch

from typing import Optional, Any, Dict, List, Tuple
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader

from pfl.data.pytorch import PyTorchDataDataset as PFLPyTorchDataDataset

from src.utils.datasets import BaseDataObject


__all__ = [
    "BaseSequenceDataset",
    "PyTorchSequenceDataset"
]


class BaseSequenceDataset(TorchDataset):
    """
    Extends the base torch.utils.data.Dataset to handel custom sequence data objects.
    """
    def __init__(self, 
                 data: List[BaseDataObject] | Tuple[BaseDataObject]
                ) -> None:
        super(TorchDataset, self).__init__()
        self._data = data
        self._sequences = [sample.data for sample in data]
        self._labels = torch.concat([sample.label for sample in data])
        self._ids = [sample.seq_id for sample in data] if data[0].seq_id is not None else None 
        self._configure_cli_logger()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]
    
    @property
    def data(self):
        return self._data

    @property
    def sequences(self):
        return self._sequences
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def ids(self):
        return self._ids
    
    def _configure_cli_logger(self, log_level=logging.INFO):
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(log_level)
        self.cli_logger = logger


class PyTorchSequenceDataset(PFLPyTorchDataDataset):
    """
    Extends the base pfl PyTorchDataDataset to handel custom sequence data objects.
    """
    def __init__(self, 
                 raw_data: torch.utils.data.Dataset, 
                 user_id: str | None = None, 
                 metadata: Dict[str, Any] | None = None, 
                 train_kwargs: Dict[str, Any] | None = None, 
                 eval_kwargs: Dict[str, Any] | None = None, 
                 **dataloader_kwargs):
        if dataloader_kwargs.get('collate_fn', 0) == 0:
            raise AttributeError('collate_fn is required for this dataset!')
        self.__configure_cli_logger()
        super().__init__(raw_data, user_id, metadata, train_kwargs, eval_kwargs, **dataloader_kwargs)

    def __getitem__(self, index):
        return self._raw_data[index]

    def __len__(self):
        return len(self._raw_data)

    def iter(self, batch_size: Optional[int]):  # noqa: A003
        return iter(
            TorchDataLoader(self._raw_data,
                            batch_size=batch_size,
                            **self._dataloader_kwargs))

    def __configure_cli_logger(self, log_level=logging.INFO):
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(log_level)
        self.cli_logger = logger