#
# adjusted form pfl source code
#
from typing import Tuple
#from pfl.algorithm.base import FederatedNNAlgorithm, NNAlgorithmParamsType
from pfl.algorithm.base import NNAlgorithmParamsType
from pfl.context import CentralContext
from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import ModelHyperParamsType
#from pfl.internal.bridge import FrameworkBridgeFactory as bridges
from ..internal.bridge import FrameworkBridgeFactory as bridges
from pfl.metrics import Metrics
from pfl.model.base import StatefulModelType
from pfl.stats import WeightedStatistics
from pfl.metrics import Metrics

from ..data_class import LocalResultMetaData
from .base import FederatedNNAlgorithm # import custom 

FedAvgCentralContextType = CentralContext[NNAlgorithmParamsType,
                                          ModelHyperParamsType]


class FederatedAveraging(FederatedNNAlgorithm[NNAlgorithmParamsType,
                                              ModelHyperParamsType,
                                              StatefulModelType,
                                              WeightedStatistics,
                                              AbstractDatasetType]):
    """
    Defines the `Federated Averaging <https://arxiv.org/abs/1602.05629>`_
    algorithm by providing the implementation as hooks into the training
    process.
    """
    def __init__(self, lr_scheduler_monitor_val: str=None):
        super().__init__(lr_scheduler_monitor_val)

    def process_aggregated_statistics(
            self, central_context: FedAvgCentralContextType,
            aggregate_metrics: Metrics, model: StatefulModelType,
            statistics: WeightedStatistics
    ) -> Tuple[StatefulModelType, Metrics]:
        """
        Average the statistics and update the model.
        """
        #print('statistics', statistics._data['lstm.weight_ih_l1'])
        statistics.average()
        #print('aggregate_metrics', aggregate_metrics)
        #print('statistics', statistics._data['lstm.weight_ih_l1'])
        monitor_val: Metrics = self._lr_scheduler_monitor_val

        return model.apply_model_update(statistics, monitor_val=monitor_val)

    def train_one_user(
        self, initial_model_state: WeightedStatistics,
        model: StatefulModelType, user_dataset: AbstractDatasetType,
        central_context: FedAvgCentralContextType,
        compute_metrics: bool,
    ) -> Tuple[WeightedStatistics, Metrics, LocalResultMetaData]:
        # Local training loop
        local_training, metrics = bridges.sgd_bridge().do_sgd(model, user_dataset,
                                    central_context.model_train_params, compute_metrics=compute_metrics)
        #assert True == False, f'{type(metrics), metrics}'
        training_statistics = model.get_model_difference(initial_model_state)
        #print('initial_model_state', initial_model_state._data['lstm.weight_ih_l1'])
        #print(training_statistics._data['lstm.weight_ih_l1'])
        #print(Metrics())
        # Don't reset model, will be used for evaluation after local training.

        #return training_statistics, Metrics(), local_training
        return training_statistics, metrics, local_training
