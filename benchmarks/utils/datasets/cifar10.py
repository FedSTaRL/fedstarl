#
# Adjusted form https://github.com/apple/pfl-research
#
import os
import pickle
import subprocess
import numpy as np
import urllib3

from glob import glob
from typing import Any, Callable, Dict, Optional, Tuple

from pfl.data import FederatedDataset
from pfl.data.dataset import Dataset
from pfl.data.partition import partition_by_dirichlet_class_distribution
from pfl.data.sampling import get_user_sampler


def load_and_preprocess(pickle_file_path: str,
                        channel_means: Optional[np.ndarray] = None,
                        channel_stddevs: Optional[np.ndarray] = None,
                        exclude_classes=None):
    images, labels = pickle.load(open(pickle_file_path, 'rb'))
    images = images.astype(np.float32)
    labels = labels.astype(np.int64)

    # Normalize per-channel.
    if channel_means is None:
        channel_means = images.mean(axis=(0, 1, 2), dtype='float64')
    if channel_stddevs is None:
        channel_stddevs = images.std(axis=(0, 1, 2), dtype='float64')
    images = (images - channel_means) / channel_stddevs

    if exclude_classes is not None:
        for exclude_class in exclude_classes:
            mask = (labels != exclude_class).reshape(-1)
            labels = labels[mask]
            images = images[mask]

    return images, labels, channel_means, channel_stddevs


def make_federated_dataset(images: np.ndarray,
                           labels: np.ndarray,
                           user_dataset_len_sampler: Callable,
                           numpy_to_tensor: Callable = lambda x: x,
                           alpha: float = 0.1) -> FederatedDataset:
    """
    Create a federated dataset from the CIFAR10 dataset.

    Users are created as proposed by Hsu et al. https://arxiv.org/abs/1909.06335,
    by sampling each user's class distribution from Dir(0.1).
    """
    data_order = np.random.permutation(len(images))
    images, labels = images[data_order], labels[data_order]
    users_to_indices = partition_by_dirichlet_class_distribution(
        labels, alpha, user_dataset_len_sampler)
    images = numpy_to_tensor(images)
    labels = numpy_to_tensor(labels)
    users_to_data = [(images[indices], labels[indices])
                     for indices in users_to_indices]

    user_sampler = get_user_sampler('random', range(len(users_to_data)))
    return FederatedDataset.from_slices(users_to_data, user_sampler)


def make_iid_federated_dataset(
        images: np.ndarray,
        labels: np.ndarray,
        user_dataset_len_sampler: Callable,
        numpy_to_tensor: Callable = lambda x: x) -> FederatedDataset:
    """
    Create a federated dataset with IID users from the CIFAR10 dataset.

    Users are created by first sampling the dataset length from
    ``user_dataset_len_sampler`` and then sampling the datapoints IID.
    """
    data_order = np.random.permutation(len(images))
    images, labels = images[data_order], labels[data_order]
    images = numpy_to_tensor(images)
    labels = numpy_to_tensor(labels)
    start_ix = 0
    users_to_data: Dict = {}
    while True:
        dataset_len = user_dataset_len_sampler()
        user_slice = slice(start_ix, start_ix + dataset_len)
        users_to_data[len(users_to_data)] = (images[user_slice],
                                             labels[user_slice])
        start_ix += dataset_len
        if start_ix >= len(images):
            break

    user_sampler = get_user_sampler('random', range(len(users_to_data)))
    return FederatedDataset.from_slices(users_to_data, user_sampler)


def make_central_dataset(images: np.ndarray, labels: np.ndarray) -> Dataset:
    """
    Create central dataset (represented as a ``Dataset``) from CIFAR10.
    This ``Dataset`` can be used for central evaluation with
    ``CentralEvaluationCallback``.
    """
    return Dataset(raw_data=[images, labels])


def make_cifar10_datasets(
    data_dir: str,
    user_dataset_len_sampler: Callable,
    numpy_to_tensor: Callable,
    alpha: float = 0.1,
) -> Tuple[FederatedDataset, FederatedDataset, Dataset, Dict[str, Any]]:
    """
    Create a train and val ``FederatedDataset`` as well as a
    central dataset from the CIFAR10 dataset.

    Here, users are created as proposed by Hsu et al. https://arxiv.org/abs/1909.06335,
    by sampling each user's class distribution from Dir(0.1).
    """
    train_images, train_labels, channel_means, channel_stddevs = (
        load_and_preprocess(os.path.join(data_dir, 'cifar10_train.p')))
    val_images, val_labels, _, _ = load_and_preprocess(
        os.path.join(data_dir, 'cifar10_test.p'), channel_means,
        channel_stddevs)

    # create artificial federated training and val datasets
    # from central training and val data.
    training_federated_dataset = make_federated_dataset(
        train_images, train_labels, user_dataset_len_sampler, numpy_to_tensor,
        alpha)
    val_federated_dataset = make_federated_dataset(val_images, val_labels,
                                                   user_dataset_len_sampler,
                                                   numpy_to_tensor, alpha)
    central_data = make_central_dataset(val_images, val_labels)

    #print(
    #    'make_cifar10_non-iid_datasets',
    #    central_data,
    #    training_federated_dataset,
    #    val_federated_dataset
    #)
    #exit(1)

    return training_federated_dataset, val_federated_dataset, central_data, {}


def make_cifar10_iid_datasets(
    data_dir: str, user_dataset_len_sampler: Callable,
    numpy_to_tensor: Callable
) -> Tuple[FederatedDataset, FederatedDataset, Dataset, Dict[str, Any]]:
    """
    Create a train and val ``FederatedDataset`` with IID users as well as a
    central dataset from the CIFAR10 dataset.

    Here, infinite users are created by continously sampling datapoints iid
    from full dataset whenever next user is requested.
    """
    train_images, train_labels, channel_means, channel_stddevs = (
        load_and_preprocess(os.path.join(data_dir, 'cifar10_train.p')))
    val_images, val_labels, _, _ = load_and_preprocess(
        os.path.join(data_dir, 'cifar10_test.p'), channel_means,
        channel_stddevs)

    # create artificial federated training and val datasets
    # from central training and val data.
    training_federated_dataset = make_iid_federated_dataset(
        train_images, train_labels, user_dataset_len_sampler, numpy_to_tensor)
    val_federated_dataset = make_iid_federated_dataset(
        val_images, val_labels, user_dataset_len_sampler, numpy_to_tensor)
    central_data = make_central_dataset(val_images, val_labels)

    #print(
    #    'make_cifar10_iid_datasets',
    #    central_data,
    #    training_federated_dataset,
    #    val_federated_dataset
    #)
    #print(len(central_data))
    #print(len([i for i in training_federated_dataset.get_cohort(10)]))
    #exit(1)

    return training_federated_dataset, val_federated_dataset, central_data, {}



URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

def dl_preprocess_and_dump(output_dir: str):
    #
    # Adjusted https://github.com/apple/pfl-research/benchmarks/cifar10/downlaod_preprocess.py
    #
    """
    Download the CIFAR10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html
    and preprocess into pickles, one for train set and one for test set.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_output_dir = os.path.join(output_dir, 'raw_data')
    os.makedirs(raw_output_dir, exist_ok=True)

    tar_path = os.path.join(raw_output_dir, "cifar-10-python.tar.gz")
    if not os.path.exists(tar_path):
        print(f'Downloading from {URL}.')
        http = urllib3.PoolManager()
        response = http.request('GET', URL)

        # Save the downloaded file.
        with open(tar_path, 'wb') as f:
            f.write(response.data)
        print(f'Saved raw data to {tar_path}.')
    else:
        print(f'Found {tar_path} on disk, skip download')

    # Extract tar file.
    subprocess.run(f"tar -zxf {tar_path} -C {raw_output_dir}".split(),
                   check=True)

    if not os.path.exists(os.path.join(output_dir, 'cifar10_train.p')) and \
        not os.path.exists(os.path.join(output_dir, 'cifar10_test.p')):
        # Merge all files into a pickle with train data and a pickle with test data.
        def merge_data_and_dump(data_paths, output_file_name):
            print(f'Merging files {data_paths}.')
            images, labels = [], []
            for train_file in data_paths:
                with open(train_file, 'rb') as f:
                    data = pickle.load(f, encoding='bytes')
                    images.append(data[b'data'])
                    labels.append(data[b'labels'])
            images = np.concatenate(images).reshape((-1, 32, 32, 3))
            labels = np.concatenate(labels).reshape((-1, 1))
            # This snippet was used to generate cifar10 for ci
            #images = images[:300]
            #labels = labels[:300]
            out_file_path = os.path.join(output_dir, output_file_name)
            with open(out_file_path, 'wb') as f:
                pickle.dump((images, labels), f)
            print(f'Saved processed data to {out_file_path}')

        merge_data_and_dump(glob(raw_output_dir + '/**/data_batch*'),
                            'cifar10_train.p')
        merge_data_and_dump(glob(raw_output_dir + '/**/test_batch*'),
                            'cifar10_test.p')
    else:
        print(f"Found {os.path.join(output_dir, 'cifar10_train.p')} and \
{os.path.join(output_dir, 'cifar10_test.p')} on disk, skip download")
