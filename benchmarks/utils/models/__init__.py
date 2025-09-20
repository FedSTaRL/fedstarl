#
# Adjusted from https://github.com/apple/pfl-research/benchmarks
#
from omegaconf import DictConfig
from typing import Tuple, Optional

try:
    from src.models import DNN, SimpleDNN, SimpleCNN
except ImportError:
    import sys
    import os.path as osp
    file_path = osp.abspath(__file__) # Get the absolute path of the current file
    dir_path = osp.normpath(osp.join(file_path, "../../../..")) # Get the directory path three levels up
    sys.path.append(dir_path) # Add the directory to the system path
    from src.models import DNN, SimpleDNN, SimpleCNN

def _get_model_dims_for_dataset(
        dataset_name: str) -> Tuple[Optional[Tuple[int, ...]], Optional[int]]:
    """
    Get the correct input shape and number of outputs for the
    specified dataset.
    """
    if dataset_name == 'femnist':
        input_shape = (28, 28, 1)
        num_outputs = 62
    elif dataset_name == 'femnist_digits':
        input_shape = (28, 28, 1)
        num_outputs = 10
    elif dataset_name in ['cifar10', 'cifar10_iid']:
        input_shape = (32, 32, 3)
        num_outputs = 10
    else:
        input_shape = None
        num_outputs = None

    return input_shape, num_outputs

def get_model_pytorch(config: DictConfig):
    """
    Initialize the PyTorch model specified by ``args.model_name`` with
    other required arguments also available in ``args``.
    Use ``add_model_arguments`` to dynamically add arguments required by
    the selected model.
    """
    assert config.model.get('name', None) is not None

    input_shape, num_outputs = _get_model_dims_for_dataset(
        dataset_name=config.dataset.name)

    model_name = config.model.name.lower()

    if model_name == 'dnn':
        model = DNN(input_shape=input_shape, 
                    hidden_dims=config.model.hidden_dims,
                    num_outputs=num_outputs
                    )
    elif model_name == 'simple_dnn':
        model = SimpleDNN(input_shape, num_outputs)
    elif model_name == 'simple_cnn':
        model = SimpleCNN(input_shape, num_outputs)
    else:
        raise TypeError(f'Model {model_name} not implemented for PyTorch.')
    return model