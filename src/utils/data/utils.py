import os
import pickle 
import ujson
import torch
import logging
import numpy as np

from typing import Literal, Union, List, Dict, Tuple
from pathlib import Path

from src.utils.datasets import BaseDataObject
from src.utils.parameters import TrainingMethodOptions

__all__ = [
    "store_asset",
    "load_asset",
    "log_stats"
]

def store_asset(format: Literal['pickle', 'pt', 'json']='pickle', file_path: Union[str, Path, os.PathLike[str]]=None, obj: object=None, write: Literal['w', 'wb']='w') -> object:
    assert file_path is not None, f'file_path cannot be None'
    if format == 'pickle':
        with open(file_path, write) as file:
            pickle.dump(obj, file)
    elif format == 'pt':
        torch.save(obj, file_path)

    elif format == 'json':
        with open(file_path, write) as f:
            ujson.dump(obj, f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_asset(format: Literal['pickle', 'pt', 'json']='pickle', file_path: Union[str, Path, os.PathLike[str]]=None, buffer: bool=False, read: Literal['r', 'rb']='r') -> object:
    assert file_path is not None, f'file_path cannot be None'
    if format == 'pickle':
        with open(file_path, read) as file:
            if buffer:
                asset = pickle.loads(file)
            else:
                asset = pickle.load(file)

    elif format == 'pt':
        asset = torch.load(file_path)

    elif format == 'json':
        with open(file_path, read) as file:
            if buffer:
                asset = ujson.loads(file)
            else:
                asset = ujson.load(file)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return asset


def log_stats(
    method: str,
    cli_logger: logging.Logger,
    #data: List[Tuple]=None,
    train_data: List[BaseDataObject] | Tuple[BaseDataObject]=None,
    val_data: List[BaseDataObject] | Tuple[BaseDataObject]=None,
    test_data: List[BaseDataObject] | Tuple[BaseDataObject]=None,
    client_data: List[BaseDataObject] | Tuple[BaseDataObject]=None,
    c_identifier: bool=False,
    #ds_config: Dict=None
    ):
    if method == TrainingMethodOptions.central:
        log_stats_central(cli_logger=cli_logger, train_data=train_data,
        val_data=val_data, test_data=test_data)
    elif method == TrainingMethodOptions.federated:
        log_stats_federated(cli_logger=cli_logger, client_data=client_data, test_data=test_data, c_identifier=c_identifier)
    else:
        RuntimeError


def log_stats_federated(
    cli_logger: logging.Logger, 
    client_data: Union[dict[str, List[BaseDataObject]], dict[str, Dict[str, List[BaseDataObject]]]],
    test_data: List[BaseDataObject],
    c_identifier: bool=False,
    ):
    
    cli_logger.info("-" * 50)
    cli_logger.info("Client Statistics")
    cli_logger.info("-" * 50)

    tot = 0
    for idx, (client_key, client_data_i) in enumerate(client_data.items()):
        client_tot = 0
        #uniques = {}
        if not c_identifier: cli_logger.info(f'Client-{idx} | {client_key}\n')
        if not c_identifier: cli_logger.info(f'Dataset\t|  Size\t|  Distribution')
        if not c_identifier: cli_logger.info("-" * 35)
        for dataset_key, data in client_data_i.items():
            labels = [sample.label for sample in data]
            d = [sample.data for sample in data]
            #if bool(targets):
            if bool(labels):
                v, c = np.unique(torch.concat(labels).numpy(), return_counts=True)
                if dataset_key == 'train_data' and not c_identifier:
                    cli_logger.info(f"Train:\t| {len(d)}\t| {v} - [" + ', '.join([f"{(((cc / len(labels))*100)):.2f}" for cc in c]) + ']')

                if dataset_key == 'val_data' and not c_identifier:
                    cli_logger.info(f"Val:\t|  {len(d)}\t| {v} - [" + ', '.join([f"{(((cc / len(labels))*100)):.2f}" for cc in c]) + ']')
                
                client_tot += len(labels)
        
        if not c_identifier: cli_logger.info("-" * 35)
        if not c_identifier: cli_logger.info(f"Total:\t| {client_tot}\t|")
        tot += client_tot
        if not c_identifier: cli_logger.info("-" * 50)
        
    cli_logger.info("Global Statistics:\n")
    cli_logger.info(f"Size of Train Data:\t{tot}")
    cli_logger.info(f"Size of Test Data:\t{len(test_data)}")
    cli_logger.info("-" * 35)
    cli_logger.info(f"Total:\t\t\t{tot + len(test_data)}")
    cli_logger.info("-" * 50)


def log_stats_central(
    cli_logger: logging.Logger, 
    train_data: List[BaseDataObject] | Tuple[BaseDataObject]=None,
    val_data: List[BaseDataObject] | Tuple[BaseDataObject]=None,
    test_data: List[BaseDataObject] | Tuple[BaseDataObject]=None,
    ):
    cli_logger.info("-" * 50)
    cli_logger.info("Statistics")
    cli_logger.info("-" * 50)

    if train_data is not None: cli_logger.info(f"Size of Train Data:\t\t{len(train_data)}")
    if val_data is not None: cli_logger.info(f"Size of Val Data:\t\t{len(val_data)}")
    #if len(test_dataset.data) < 10000:
    #    cli_logger.info(f"Size of Test Data:\t {len(test_dataset.data)}")
    #else:
    cli_logger.info(f"Size of Test Data:\t\t{len(test_data)}")
    cli_logger.info("-" * 50)
    if train_data is not None and val_data is not None:
        cli_logger.info(f"Total Data Size:\t\t{len(train_data) + len(val_data) + len(test_data)}")
    elif val_data is not None :
        cli_logger.info(f"Total Data Size:\t\t{len(train_data) + len(test_data)}")
    cli_logger.info("-" * 50)

    cli_logger.info(f"\n")
    cli_logger.info(f"Label Distribution")
    if train_data is not None:
        train_labels = [sample.label for sample in train_data]
        v, c = np.unique(train_labels, return_counts=True)
        cli_logger.info("\t\t\t|" + "  |".join([f' {vv}' for vv in v]) + ' |')
        cli_logger.info(f"Train Dataset\t|" + " |".join([f" {(((vv / len(train_labels))*100)):.2f}" for vv in c]) + ' |')
    
    if val_data is not None:
        val_labels = [sample.label for sample in val_data]
        v, c = np.unique(val_labels, return_counts=True)
        cli_logger.info(f"Val Dataset\t\t|" + " |".join([f" {(((vv / len(val_labels))*100)):.2f}" for vv in c]) + ' |')
    
    test_labels = [sample.label for sample in test_data]
    v, c = np.unique(test_labels, return_counts=True)
    cli_logger.info(f"Test Dataset\t\t|" + " |".join([f" {(((vv / len(test_labels))*100)):.2f}" for vv in c]) + ' |')
    cli_logger.info("-" * 50)
    

