import os.path as osp
import logging

from omegaconf import DictConfig
from typing import List, Tuple
from pfl.algorithm import NNAlgorithmParams, FedProxParams
from pfl.algorithm import FederatedAveraging as PFLFederatedAveraging
from pfl.algorithm import FedProx as PFLFedProx
from pfl.algorithm import FedProxParams as PFLFedProxParams
from pfl.algorithm import AdaptMuOnMetricCallback as PFLAdaptMuOnMetricCallback

try:
    from src.pfl.algorithm.base import FederatedNNAlgorithm
    from src.pfl.algorithm import (
        FederatedAveraging,
        FedProx,
        FedProxParams,
        AdaptMuOnMetricCallback
    )
    from src.utils import AggregationModelOptions
except:
    import sys
    file_path = osp.abspath(__file__)
    dir_path = "/".join(file_path.split("/")[:-4])
    sys.path.append(dir_path)
    # datasets
    from src.pfl.algorithm.base import FederatedNNAlgorithm
    from src.pfl.algorithm import (
        FederatedAveraging,
        FedProx,
        FedProxParams,
        AdaptMuOnMetricCallback
    )
    from src.utils import AggregationModelOptions
   

logger = logging.getLogger(name=__name__)


def get_aggregation_algorithm(config: DictConfig):# -> Tuple[FederatedNNAlgorithm, NNAlgorithmParams | FedProxParams, List]:
    """
    Initialize the TensorFlow v2 model specified by ``args.model_name`` with
    other required arguments also available in ``args``.
    Use ``add_model_arguments`` to dynamically add arguments required by
    the selected model.
    """
    aggregation_algorithm_name = config.aggregation_algorithm.name.lower()
    logger.info(f'initializing algorithm {aggregation_algorithm_name}')
    callbacks = []
    lr_scheduler_monitor_val = config.callbacks.lr_scheduler.monitor if config.callbacks.lr_scheduler.apply else None

    algorithm_params = {
        'central_num_iterations': config.training.central.num_iterations,
        'evaluation_frequency': config.training.central.evaluation_frequency,
        'train_cohort_size': config.training.local.cohort_size,
        'val_cohort_size': config.training.local.val_cohort_size
    }
    if aggregation_algorithm_name == AggregationModelOptions.fedavg:
        logger.info(f'using algorithm parameters {algorithm_params}')
        algorithm_params = NNAlgorithmParams(**algorithm_params)
        #
        if config.use_pfl_internals:
            print('using `pfl` FederatedAveraging() algorithm')
            return PFLFederatedAveraging(), algorithm_params, callbacks
        else:
            print('using custom `src.pfl` FederatedAveraging() algorithm')
            return FederatedAveraging(lr_scheduler_monitor_val=lr_scheduler_monitor_val), algorithm_params, callbacks 

    elif aggregation_algorithm_name == AggregationModelOptions.fedprox:
        algorithm_params['mu'] = config.aggregation_algorithm.mu
        logger.info(f'using algorithm parameters {algorithm_params}')
        algorithm_params = FedProxParams(**algorithm_params)
        return FedProx(lr_scheduler_monitor_val=lr_scheduler_monitor_val), algorithm_params, callbacks

    elif aggregation_algorithm_name == AggregationModelOptions.adafedprox:
        mu = AdaptMuOnMetricCallback(
            metric_name=config.aggregation_algorithm.adafedprox_metric_name,
            adapt_frequency=config.aggregation_algorithm.adafedprox_adapt_frequency,
            decrease_mu_after_consecutive_improvements=config.aggregation_algorithm.
            adafedprox_decrease_mu_after_consecutive_improvements)
        callbacks.append(mu)
        algorithm_params['mu'] = mu
        logger.info(f'using algorithm parameters {algorithm_params}')
        algorithm_params = FedProxParams(**algorithm_params)
        return FedProx(lr_scheduler_monitor_val=lr_scheduler_monitor_val), algorithm_params, callbacks

    else:
        raise TypeError(f'Algorithm {aggregation_algorithm_name} not found.')
    #elif algorithm_name == 'scaffold':
        #user_state_storage: AbstractUserStateStorage
        #if args.scaffold_states_dir is None:
        #    user_state_storage = InMemoryUserStateStorage()
        #else:
        #    user_state_storage = DiskUserStateStorage(args.scaffold_states_dir)
        #algorithm_params = SCAFFOLDParams(
        #    central_num_iterations=args.central_num_iterations,
        #    evaluation_frequency=args.evaluation_frequency,
        #    train_cohort_size=args.cohort_size,
        #    val_cohort_size=args.val_cohort_size,
        #    population=args.scaffold_population,
        #    use_gradient_as_control_variate=args.
        #    scaffold_use_gradient_as_control_variate,
        #    user_state_storage=user_state_storage)
        #algorithm = SCAFFOLD()
