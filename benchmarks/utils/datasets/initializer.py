import numpy as np

from typing import Any, Callable, Dict, Tuple
from omegaconf import DictConfig

from pfl.data.dataset import Dataset
from pfl.data.federated_dataset import FederatedDatasetBase


def parse_draw_num_datapoints_per_user(
        datapoints_per_user_distribution: str,
        mean_datapoints_per_user: float,
        minimum_num_datapoints_per_user: int = 1) -> Callable[[], int]:
    """
    Get a user dataset length sampler.

    :param datapoints_per_user_distribution:
        'constant' or 'poisson'.
    :param mean_datapoints_per_user:
        If 'constant' distribution, this is the value to always return.
        If 'poisson' distribution, this is the mean of the poisson
        distribution.
    :param minimum_num_datapoints_per_user:
        Only accept return values that are at least as large as this argument.
        If rejected, value is resampled until above the threshold, which will
        result in a truncated distribution.
    :return:
        A callable that samples lengths for artificial user datasets
        yet to be created.
    """
    if datapoints_per_user_distribution == 'constant':
        assert minimum_num_datapoints_per_user < mean_datapoints_per_user
        draw_num_datapoints_per_user = lambda: int(mean_datapoints_per_user)
    else:
        assert datapoints_per_user_distribution == 'poisson'

        def draw_truncated_poisson():
            while True:
                num_datapoints = np.random.poisson(mean_datapoints_per_user)
                # Try again if less than minimum specified.
                if num_datapoints >= minimum_num_datapoints_per_user:
                    return num_datapoints

        draw_num_datapoints_per_user = draw_truncated_poisson
    return draw_num_datapoints_per_user


def get_datasets(
    config: DictConfig,
    numpy_to_tensor: Callable,
) -> Tuple[FederatedDatasetBase, FederatedDatasetBase, Dataset, Dict[str, Any]]:
    """
    Create a federated dataset for training, a federated dataset for evalution
    and a central dataset for central evaluation.

    :param args:
        ``args.dataset`` specifies which dataset to load. Should be one of
        ``{cifar10,femnist,femnist_digits}``.
        ``args`` should also have any dataset-specific arguments added by
        ``add_dataset_arguments`` for the particular ``args.dataset`` chosen.
    :return:
        A tuple ``(fed_train, fed_eval, eval, metadata)``, where ``fed_train``
        is a federated dataset to be used for training, ``fed_eval`` is a
        federated dataset from a population separate for ``fed_train``,
        ``eval`` is a dataset for central evaluation, and ``metadata`` is
        a dictionary of metadata for the particular dataset.
    """
    # create federated training and val datasets from central training and val
    # data
    #numpy_to_tensor =  if config.dataset.numpy_to_tensor else lambda x: x
    
    numpy_to_tensor = getattr(config.dataset, "numpy_to_tensor", lambda x: x)
    datasets: Tuple[FederatedDatasetBase, FederatedDatasetBase, Dataset,
                    Dict[str, Any]]
    
    dataset_name: str = config.dataset.name

    if dataset_name.startswith('cifar10'):
        from .cifar10 import (
            dl_preprocess_and_dump,
            make_cifar10_datasets, 
            make_cifar10_iid_datasets
        )

        dl_preprocess_and_dump(output_dir=config.dataset.raw_data_dir) #,9
                               #proccesed_dir=config.dataset.processed_data_dir)

        user_dataset_len_sampler = parse_draw_num_datapoints_per_user(
            datapoints_per_user_distribution=config.dataset.datapoints_per_user_distribution,
            mean_datapoints_per_user=config.dataset.mean_datapoints_per_user,
            minimum_num_datapoints_per_user=config.dataset.minimum_num_datapoints_per_user)
        
        make_cifar10_dataset_params = {
            'data_dir': config.dataset.raw_data_dir,
            'user_dataset_len_sampler': user_dataset_len_sampler,
            'numpy_to_tensor': numpy_to_tensor,
        }
        if dataset_name == 'cifar10':
            make_cifar10_dataset_params['alpha'] = config.dataset.alpha
            datasets = make_cifar10_datasets(
                **make_cifar10_dataset_params
            )
        elif dataset_name == 'cifar10_iid':
            datasets = make_cifar10_iid_datasets(
                **make_cifar10_dataset_params
            )
    
    else:
        raise ValueError(f'{dataset_name} is not supported')
    return datasets