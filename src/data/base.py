import os.path as osp
import os
import pickle
import ujson
import torch
import logging
import boto3

from abc import ABC
from tqdm.auto import tqdm
from typing import List, Literal

from src.utils.data import load_asset, store_asset

__all__ = []


class DataPreprocessing(ABC):
    data_dir = osp.join("/".join(osp.abspath(__file__).split("/")[:-3]), "data")
    __config_filename = 'config.json'

    def __init__(self, dataset_name: str, seed: int=42, **kwargs) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.seed = seed
        self._data = None
        self.configure_cli_logger()
    
    def download(self, **kwargs):
        return NotImplementedError

    def process(self, **kwargs):
        return NotImplementedError

    def _process(self, **kwargs):
        dataset_dir = osp.join(self.data_dir, self.dataset_name)
        # check if dir exists
        if not osp.exists(dataset_dir):
            self.cli_logger.info(f"\nCreating {self.dataset_name} directory --> {dataset_dir}\n")
            os.makedirs(dataset_dir)
        
        # setup directories
        raw = osp.join(dataset_dir, "raw")
        processed = osp.join(dataset_dir, "processed")
        central = osp.join(processed, "central")
        federated = osp.join(processed, "federated")

        self.dataset_dir_paths = {
            "root": dataset_dir,
            "raw": raw,
            "processed": processed,
            "central": central,
            "federated": federated,
        }

        self.download(**kwargs)
        self.process(**kwargs)
    
    def preprocess(self, **kwargs):
        self._process(**kwargs)

    def store_asset(self, format: Literal['pickle', 'pt', 'json'] = 'pickle', file_path: str = None, obj: object = None, write: Literal['w', 'wb'] = 'w'):
        return store_asset(format=format, file_path=file_path, obj=obj, write=write)

    def load_asset(self, format: Literal['pickle', 'pt', 'json'] = 'pickle', file_path: str=None, buffer: bool=False, read: Literal['r', 'rb']='r'):
        return load_asset(format=format, file_path=file_path, buffer=buffer, read=read)
        
    def check_if_central_dataset_exists_on_s3(
        self, 
        myway_s3_client,
        s3_buckets: List,
        **kwargs):
        load_from_config = False
        ds_config = None

        # load configs from s3
        for key_summary in tqdm(s3_buckets, desc="Loading Configs from S3 Bucket"):
            if key_summary.key.split('/')[-1] == self.__config_filename: # check that config is loaded
                s3_object = myway_s3_client.get_object(Bucket=self.s3_bucket_path, Key=key_summary.key)
                binary = s3_object['Body'].read()
                tmp_config = ujson.loads(binary.decode('utf-8'))
                load_from_config, ds_config = self._check_if_central_dataset_exists(
                    load_from_config, ds_config, tmp_config,
                )
        
        return load_from_config, ds_config

    def check_if_central_dataset_exists(self, **kwargs):
        load_from_config = False
        ds_config = None
        
        for dir, folders, file in os.walk(self.dataset_dir_paths['central']):
            for folder in folders:
                if osp.exists(osp.join(dir, folder, self.__config_filename)):
                    with open(osp.join(dir, folder, self.__config_filename), 'r') as f:
                        tmp_config = ujson.load(f)
                        
                        load_from_config, ds_config = self._check_if_central_dataset_exists(
                            load_from_config, ds_config, tmp_config,
                        )
                    

        return load_from_config, ds_config
    
    def _check_if_central_dataset_exists(self, **kwargs):
        raise NotImplementedError

    def configure_cli_logger(self, log_level=logging.INFO):
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(log_level)
        self.cli_logger = logger
    
    def setup_aws_session(self):
        self.session = boto3.Session(profile_name=self.aws_profile)
        self.s3_client = self.session.client('s3')
        self.s3 = self.session.resource('s3')

    @property
    def data(self):
        return self._data

    @property
    def ds_config(self,):
        return self._ds_config

