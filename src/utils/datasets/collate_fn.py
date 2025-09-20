import torch
from torch.nn.utils.rnn import pad_sequence, pack_sequence
from typing import List
from .data_object import BaseDataObject, SequenceBatchData


__all__ = [
    "collate_fn_default_padded",
    "collate_fn_default_packed"
]


def collate_fn_default_padded(batch: List[BaseDataObject]):
    """
    Default Collate Function for padding sequences - used for Transformer modesl
    """
    try:
        seq_ids, data, labels, seq_lens = [], [], [], []
        for data_item in batch:
            seq_ids.append(data_item.seq_id)
            data.append(data_item.data)
            labels.append(data_item.label)
            seq_lens.append(data_item.seq_len.reshape(1,-1))
    except:
        print(batch)
            
    data_padded = pad_sequence(sequences=data, batch_first=False, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float)
    seq_lens = torch.concat(seq_lens).squeeze()
    
    return SequenceBatchData(
        seq_id=seq_ids,
        data=data_padded,
        label=labels,
        seq_len=seq_lens,
    )


def collate_fn_default_packed(batch: List[BaseDataObject]):
    """
    Default Collate Function for packed sequences - used for RNN Models, i.e., LSTM
    """
    seq_ids, data, labels, seq_lens = [], [], [], []

    for data_item in batch:
        seq_ids.append(data_item.seq_id)
        data.append(data_item.data)
        labels.append(data_item.label)
        seq_lens.append(data_item.seq_len.reshape(1,-1))
    
    data_padded = pack_sequence(sequences=data, enforce_sorted=False)
    labels = torch.tensor(labels, dtype=torch.float)
    #print(torch.concat(seq_lens))
    seq_lens = torch.concat(seq_lens).reshape(-1)
    #print(seq_lens)
    return SequenceBatchData(
        seq_id=seq_ids,
        data=data_padded,
        label=labels,
        seq_len=seq_lens,
    )