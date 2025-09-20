import torch
import pandas as pd

from typing import Optional, List
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence


__all__ =  [
    "BaseDataObject",
    "SequenceData",
    "SequenceBatchData"
]


class BaseDataObject(object):
    def __init__(self, **kwargs) -> None:
        # Add additional features from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    
    def __repr__(self) -> str:
        """Returns a string representation of the Data object."""
        attr_str = []
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                attr_str.append(f"{k}={v.size()}")
            elif isinstance(v, list):
                attr_str.append(f'{k}=list([{len(v)}])')
            elif isinstance(v, pd.Series):
                attr_str.append(f'{k}=pd.Series([{len(v)}])')
            else:
                attr_str.append(f"{k}='{v}'")
        return f'{self.__class__.__name__}({", ".join(attr_str)})'


    def __getitem__(self, key):
        """Allows slicing the Data object as a dictionary."""
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise KeyError(f"Key '{key}' not found in Data object.")
    

    def __setitem__(self, key, value):
        """Allows adding attributes to the Data object."""
        setattr(self, key, value)


class SequenceData(BaseDataObject):
    def __init__(self,
                 seq_id: Optional[str]=None, 
                 data: Tensor=None, 
                 label: Tensor=None, 
                 **kwargs) -> None:
        self.seq_id = seq_id
        self.data = data
        self.label = label
        self.seq_len = torch.tensor(data.size(0))
        super().__init__(**kwargs)


class SequenceBatchData(BaseDataObject):
    def __init__(
        self,
        seq_id: Optional[List[str]]=None,
        data: Tensor | PackedSequence=None, 
        label: Tensor=None, 
        seq_len: Tensor | List[Tensor]=None,
        **kwargs
        ) -> None:
        self.seq_id = seq_id
        self.data = data
        self.label = label
        self.batch_size=len(seq_len) if seq_len.dim() > 0 else 1
        self.seq_len = seq_len.reshape(-1, 1)
        self.ptr = [torch.tensor([0])]
        if seq_id is not None or seq_len is not None:
        #    print(self.seq_len)
            for idx, sl in enumerate(self.seq_len):
                self.ptr.extend([self.ptr[idx] + sl])
        self.ptr = torch.concat(self.ptr)
        
        super().__init__(**kwargs)
    
    def pin_memory(self):
        self.data = self.data.pin_memory()
        self.label = self.label.pin_memory()
        self.seq_len = self.seq_len.pin_memory()
        return self