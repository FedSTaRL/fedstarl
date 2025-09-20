import os
import boto3
import awswrangler 
import pandas as pd
import numpy as np
import torch 
import random

from typing import Union, List, Dict, Set, Tuple
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
from torch import Tensor

from src.utils.datasets import BaseDataObject
from src.utils import TrainingMethodOptions


def get_most_recent_date(dates_list) -> str:
    today = datetime.today().date()
    dates = [datetime.strptime(date, "%Y-%m-%d").date() for date in dates_list]
    most_recent_date = max(dates, key=lambda x: x if x <= today else datetime.date(1, 1, 1))

    return most_recent_date.strftime("%Y-%m-%d")

def load_most_recent_upload_date(
    s3_bucket_path: Union[str, Path, os.PathLike[str]]="cdh-vehicle-myway-preprocessed-src-b4se",
    prefix: str="artifact/",
    s3: boto3.Session=None,
    data_label: List[str]=['customer'],
    first_date: str = '2024-07-04',
    end_date: str = '2024-09-30',
    ) -> str:
    assert s3 != None
    data_upload_dates = []
    for key_summary in tqdm(list(s3.Bucket(s3_bucket_path).objects.filter(Prefix=prefix)), desc="Get Upload Dates"):
        if ".gzip" in key_summary.key:
            if key_summary.key.split('/')[1].split('_')[-1] != 'new':
                if key_summary.key.split('/')[2] in data_label and \
                    int(key_summary.key.split('/')[1].split('-')[1]) >= int(first_date.split('-')[1]) and \
                    int(key_summary.key.split('/')[1].split('-')[2]) >= int(first_date.split('-')[2]):
                    if end_date is not None:
                        if int(key_summary.key.split('/')[1].split('-')[1]) <= int(end_date.split('-')[1]) and \
                            int(key_summary.key.split('/')[1].split('-')[2]) <= int(end_date.split('-')[2]): 
                            data_upload_dates.append(key_summary.key.split('/')[1])
                    else:
                        data_upload_dates.append(key_summary.key.split('/')[1])

    return get_most_recent_date(data_upload_dates)


def download_data_from_s3(
    session: boto3.Session, 
    s3, 
    prefix: str='artifact/',
    s3_bucket_path: str = 'cdh-vehicle-myway-preprocessed-src-b4se',
    use_threads = False,
    data_label: List[str] = ['customer'],
    first_date: str = '2024-07-04',
    end_date: str = "2024-09-30",
    ):
    dfs = [] 
    data_upload_dates = []
    counter = 0   
    for key_summary in tqdm(list(s3.Bucket(s3_bucket_path).objects.filter(Prefix=prefix)), desc="Loading Data from S3 Bucket"):
        if ".gzip" in key_summary.key:
            if key_summary.key.split('/')[1].split('_')[-1] != 'new':
                if key_summary.key.split('/')[2] in data_label and \
                    int(key_summary.key.split('/')[1].split('-')[1]) >= int(first_date.split('-')[1]) and \
                    int(key_summary.key.split('/')[1].split('-')[2]) >= int(first_date.split('-')[2]):
                    if end_date is not None:
                        if int(key_summary.key.split('/')[1].split('-')[1]) <= int(end_date.split('-')[1]) and \
                            int(key_summary.key.split('/')[1].split('-')[2]) <= int(end_date.split('-')[2]):
                            print(counter, key_summary.key)
                            data_upload_dates.append(key_summary.key.split('/')[1])
                            path = "s3://"+s3_bucket_path+"/"+key_summary.key
                            df = awswrangler.s3.read_parquet(path, boto3_session=session, use_threads=use_threads)
                            dfs.append(df)
                            counter += 1
                    else:
                        print(counter, key_summary.key)
                        data_upload_dates.append(key_summary.key.split('/')[1])
                        path = "s3://"+s3_bucket_path+"/"+key_summary.key
                        df = awswrangler.s3.read_parquet(path, boto3_session=session, use_threads=use_threads)
                        dfs.append(df)
                        counter += 1

    most_recent_upload_date = get_most_recent_date(dates_list=data_upload_dates)
    if len(dfs) > 50:
        return dfs, most_recent_upload_date, key_summary.key.split('/')[1]
    #print('concatenating dfs')
    #df_full = pd.concat(dfs, ignore_index=True)    
    df_full = concat_dataframes_in_chunks(dfs, chunk_size=10)
    return df_full, most_recent_upload_date, key_summary.key.split('/')[1]


def concat_dataframes_in_chunks(dfs: List[pd.DataFrame], chunk_size=10):
    if len(dfs) > 10:
        concatenated_df = pd.DataFrame()
        for i in tqdm(range(0, len(dfs), chunk_size), desc='Concatenating pd.DataFrames'):
            chunk = dfs[i:i+chunk_size]
            concatenated_df_chunk = pd.concat(chunk, ignore_index=True)
            concatenated_df = pd.concat([concatenated_df, concatenated_df_chunk], ignore_index=True)
        return concatenated_df
    else:
        return pd.concat(dfs, ignore_index=True)
    

def prepare_dates_for_download_range(
    first_date: str, end_date: str, year: str, month_first_date: str, month_end_date: str,
    dates: Dict[str, Dict[str, str]],
    ):
    date_pairs = []
    for idx, month in enumerate(range(int(month_first_date), int(month_end_date)+1, 1)):
        if month < 10:
            month = f'0{month}'
        else:
            month = f'{month}'

        if idx == 0:
            if int(first_date.split('-')[-1]) > int(dates[month]["first_date"]):
                first_date = first_date
            else:
                first_date = f'{year}-{month}-{dates[month]["first_date"]}'
            end_date = f'{year}-{month}-{dates[month]["end_date"]}'
            
        elif idx == (int(month_first_date)-int(month_end_date)):
            first_date = f'{year}-{month}-{dates[month]["first_date"]}'
            if int(end_date.split('-')[-1]) < int(dates[month]["end_date"]):
                end_date = end_date
            else:
                end_date = f'{year}-{month}-{dates[month]["end_date"]}'
        else:
            first_date = f'{year}-{month}-{dates[month]["first_date"]}'
            end_date = f'{year}-{month}-{dates[month]["end_date"]}'
        
        date_pairs.append((first_date, end_date))
    return date_pairs


def shorten_trajectory(trajectory_data: torch.Tensor, max_trajectory_length: int=None):
    """
    Shortens a trajectory sequence to a specified length N, keeping the first and last elements
    and selecting the rest in a special manner.
    
    Args:
        trajectory (list or numpy array): The input trajectory sequence.
        N (int): The desired length of the shortened trajectory.
    
    Returns:
        numpy array: The shortened trajectory sequence.
    """
    assert max_trajectory_length is not None
    # Get the length of the input trajectory
    trajectory_length = trajectory_data.size(0)
    # Check if the trajectory is already shorter than the desired length
    if trajectory_length <= max_trajectory_length:
        return trajectory_data
    
    if 256 <= trajectory_length < 300:
        new_length = np.random.randint(175, 185)
    elif 300 <= trajectory_length < 350:
        new_length = np.random.randint(185, 195)
    elif 350 <= trajectory_length < 400:
        new_length = np.random.randint(195, 205)
    elif 400 <= trajectory_length < 450:
        new_length = np.random.randint(205, 215)
    elif 450 <= trajectory_length < 500:
        new_length = np.random.randint(215, 225)
    elif 500 <= trajectory_length < 550:
        new_length = np.random.randint(225, 235)
    elif 550 <= trajectory_length < 600:
        new_length = np.random.randint(235, 245)
    elif 550 <= trajectory_length < 1000:
        new_length = np.random.randint(245, max_trajectory_length+1)
    
    # Create the shortened trajectory array
    shortened_trajectory_data = torch.zeros(new_length, trajectory_data.size(1))

    # Set the first and last elements
    shortened_trajectory_data[0, :] = trajectory_data[0, :]
    shortened_trajectory_data[-1, :] = trajectory_data[-1, :]
    
    len_remain = shortened_trajectory_data.size(0) - 2
    
    # interpolate
    x_int = np.interp(
        np.linspace(1, trajectory_length, len_remain),
        np.arange(1, trajectory_length -1),
        trajectory_data[1:-1, 0]
        )
    y_int = np.interp(
        np.linspace(1, trajectory_length, len_remain),
        np.arange(1, trajectory_length -1),
        trajectory_data[1:-1, 1]
        )
    
    # get xy interpolated pos
    xy = np.concatenate([x_int[:, np.newaxis], y_int[:, np.newaxis]], axis=1) 
    pairwise_dist = np.sqrt(((xy[:, np.newaxis, :] - trajectory_data[:, :2].numpy())**2).sum(axis=2))
    # find nearest features from pairwise distances
    nearest_indices = pairwise_dist.argmin(axis=1)
    selected_features = trajectory_data[nearest_indices, 2:]
    
    shortened_trajectory_data[1:-1, :2] = torch.from_numpy(xy)
    shortened_trajectory_data[1:-1, 2:] = selected_features
    shortened_trajectory_data[0, :] = trajectory_data[0, :]
    shortened_trajectory_data[-1, :] = trajectory_data[-1, :] 

    return shortened_trajectory_data


def create_client_train_val_split(
    train_data: List[BaseDataObject], 
    train_val_split: dict=None,     
    ) -> Dict[str, List[BaseDataObject]]:
    assert train_val_split.get('train', None) is not None
    assert sum([v for v in train_val_split.values() if v is not None]) == 1.0

    random.shuffle(train_data)
    tot = len(train_data)
    N_val = int(tot*train_val_split.get('val', None)) if train_val_split.get('val', None) is not None else None
    N_train = int(tot*train_val_split['train']) if N_val is not None else tot
    
    if N_val is not None:
        training_data = train_data[:N_train]
        validation_data = train_data[N_train:(N_train+N_val)]

        random.shuffle(training_data)
        random.shuffle(validation_data)
    else:
        training_data = train_data
        validation_data = None
        random.shuffle(training_data)
    
    return {
        'train_data': training_data, 
        'val_data': validation_data,
    }


def create_train_test_split(
    data: List[BaseDataObject], 
    train_test_split: dict=None, 
    identical_training_class_label_distribution: bool=True,
    test_indices: Union[List[str], Set[str]]=None,
    training_method: str=None,
    ) -> Dict[str, List[BaseDataObject]]: 
    if test_indices is not None:
        assert training_method is not None
        assert train_test_split.get('train', None) is not None
        
        if training_method == TrainingMethodOptions.federated:
            train_data_static = []
            test_data = []
            test_indices_set = set(test_indices)
            for sample in tqdm(data, desc='Seperating train and test data'):
                if sample.seq_id in test_indices_set: 
                    test_data.append(sample)
                else:
                    train_data_static.append(sample)
            
            train_data = train_data_static
            val_data = None
            random.shuffle(train_data)
            random.shuffle(test_data)
            
        else:
            if train_test_split.get('val', None) is None:
                train_test_split['train'] = 1.0
            else:
                assert sum([v for k, v in train_test_split.items() if v is not None and k in ['train', 'val']]) == 1.0 # assert sum is 1.0

            train_data_static = []
            test_data = []
            for sample in tqdm(data, desc='Seperating train and test data'):
                if sample.seq_id in set(test_indices): 
                    test_data.append(sample)
                else:
                    train_data_static.append(sample)
                    
            random.shuffle(train_data_static)
            tot = len(train_data_static)
            N_train = int(tot*train_test_split.get('train', None))
            N_val = int(tot*train_test_split.get('val', None)) if train_test_split.get('val', None) is not None else None

            if N_val is not None:
                #for train_sample in train_data:
                #    assert train_sample.seq_id not in set(test_indices)
                
                train_data = train_data_static[:N_train]
                val_data = train_data_static[N_train:]
            
                random.shuffle(train_data)
                random.shuffle(val_data)
                random.shuffle(test_data)
            else:
                train_data = train_data_static #[:N_train]
                val_data = None

                random.shuffle(train_data)
                random.shuffle(test_data)
    
    else:
        assert train_test_split.get('train', None) is not None
        assert train_test_split.get('test', None) is not None
        assert sum([v for v in train_test_split.values() if v is not None]) == 1.0 # assert sum is 1.0

        if identical_training_class_label_distribution:
            random.shuffle(data)
            targets = [d.label.item() for d in data]
            target_keys = set(targets)

            data_sorted_by_label = {k: [] for k in target_keys}
            for d in data:    
                #_, _, t = d
                data_sorted_by_label[d.label.item()].append(d)

            v, c = np.unique(targets, return_counts=True)
            N_train_per_target = int(np.min(c)*(train_test_split.get('train')))

            if train_test_split.get('val', None) is not None:
                N_val_per_target = int(np.min(c)*(train_test_split.get('val'))) 
                train_data = []
                val_data = []
                test_data = []
                for k, v in data_sorted_by_label.items():
                    train_data.extend(v[:N_train_per_target])
                    val_data.extend(v[N_train_per_target:(N_train_per_target+N_val_per_target)])
                    test_data.extend(v[(N_train_per_target+N_val_per_target):])
                
                random.shuffle(train_data)
                random.shuffle(val_data)
                random.shuffle(test_data)

            else:
                train_data = []
                val_data = None
                test_data = []
                for k, v in data_sorted_by_label.items():
                    train_data.extend(v[:N_train_per_target])
                    test_data.extend(v[N_train_per_target:])
            
                random.shuffle(train_data)
                random.shuffle(test_data)
        else:
            random.shuffle(data)
            tot = len(data)
            N_train = int(tot*train_test_split.get('train', None))
            N_val = int(tot*train_test_split.get('val', None)) if train_test_split.get('val', None) is not None else None
            N_test = int(tot*train_test_split.get('test', None))

            #if tot != (N_train + N_val, N_test):
            #    diff = tot - (N_train + N_val, N_test)

            if N_val is not None:
                train_data = data[:N_train]
                val_data = data[N_train:(N_train+N_val)]
                test_data = data[(N_train+N_val):]

                random.shuffle(train_data)
                random.shuffle(val_data)
                random.shuffle(test_data)
            else:
                train_data = data[:N_train]
                val_data = None
                test_data = data[N_train:]

                random.shuffle(train_data)
                random.shuffle(test_data)

    return {
        'train_data': train_data, 
        'val_data': val_data,
        'test_data': test_data
    }

