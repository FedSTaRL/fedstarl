from abc import abstractmethod
from typing import Optional, Tuple

from pfl.algorithm.base import FederatedAlgorithm as PFLFederatedAlgorithm
from pfl.algorithm.base import NNAlgorithmParamsType
from pfl.common_types import Population
from pfl.context import CentralContext
from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import ModelHyperParamsType
from pfl.metrics import Metrics, TrainMetricName
from pfl.model.base import StatefulModelType
from pfl.stats import StatisticsType

from src.pfl.data_class import LocalResultMetaData
import json
import logging
import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar

import numpy as np
import horovod.torch as hvd

from pfl.aggregate.base import Backend
from pfl.callback import TrainingProcessCallback
from pfl.common_types import Population, Saveable
from pfl.context import CentralContext
from pfl.data.dataset import AbstractDatasetType
from pfl.exception import CheckpointNotFoundError
from pfl.hyperparam.base import (
    AlgorithmHyperParams,
    AlgorithmHyperParamsType,
    HyperParamClsOrInt,
    ModelHyperParamsType,
    get_param_value,
)
from pfl.internal.platform.selector import get_platform
from pfl.metrics import Metrics, TrainMetricName
from pfl.model.base import ModelType, StatefulModelType
from pfl.stats import StatisticsType

from . import algorithm_utils

logger = logging.getLogger(__name__)


#NNAlgorithmParamsType = TypeVar('NNAlgorithmParamsType', bound=NNAlgorithmParams)


class FederatedAlgorithm(PFLFederatedAlgorithm[NNAlgorithmParamsType,
                                              ModelHyperParamsType,
                                              StatefulModelType,
                                              StatisticsType,
                                              AbstractDatasetType]):
    """
    Wraps the FederatedAlgorithm from pfl source code.
    """
    def __init__(self, lr_scheduler_monitor_val: str=None):
        self._lr_scheduler_monitor_val_str: str = lr_scheduler_monitor_val if lr_scheduler_monitor_val is not None else None
        self._lr_scheduler_monitor_val: Metrics = Metrics([(self._lr_scheduler_monitor_val_str, float("Inf"))]) if self._lr_scheduler_monitor_val_str is not None else None
        
        super().__init__()
    
    def run(self,
            algorithm_params: AlgorithmHyperParamsType,
            backend: Backend,
            model: ModelType,
            model_train_params: ModelHyperParamsType,
            model_eval_params: Optional[ModelHyperParamsType] = None,
            callbacks: Optional[List[TrainingProcessCallback]] = None,
            *,
            send_metrics_to_platform: bool = True,
            compute_metrics: bool=False,) -> ModelType:
        """
        Orchestrate the federated computation.

        :param backend:
            The :class:`~pfl.aggregate.base.Backend` that aggregates the
            contributions from individual users.
            This may be simulated (in which case the backend will call
            ``simulate_one_user``), or it may perform for live training (in
            which case your client code will be called).
        :param model:
            The model to train.
        :param callbacks:
            A list of callbacks for hooking into the training loop, potentially
            performing complementary actions to the model training, e.g. central
            evaluation or training parameter schemes.
        :param send_metrics_to_platform:
            Allow the platform to process the aggregated metrics after
            each central iteration.
        :returns:
            The trained model.
            It may be the same object as given in the input, or it may be
            different.

        """
        self._current_central_iteration = 0
        should_stop = False
        callbacks = list(callbacks or [])
        default_callbacks = get_platform().get_default_callbacks()
        for default_callback in default_callbacks:
            # Add default callback if it is not in the provided callbacks
            if all(
                    type(callback) != type(default_callback)
                    for callback in callbacks):
                logger.debug(f'Adding {default_callback}')
                callbacks.append(default_callback)
            else:
                logger.debug(f'Not adding duplicate {default_callback}')

        on_train_metrics = Metrics()
        for callback in callbacks:
            on_train_metrics |= callback.on_train_begin(model=model)

        central_contexts = None
        while True:
            # Step 1
            # Get instructions from algorithm what to run next.
            # Can be multiple queries to cohorts of devices.
            (new_central_contexts, model,
             all_metrics) = self.get_next_central_contexts(
                 model, self._current_central_iteration, algorithm_params,
                 model_train_params, model_eval_params)
            if new_central_contexts is None:
                break
            else:
                central_contexts = new_central_contexts

            if self._current_central_iteration == 0:
                all_metrics |= on_train_metrics
            
            #print('\nStep 2')
            # Step 2
            # Get aggregated model updates and
            # metrics from the requested queries.
            #print(central_contexts)
            results: List[Tuple[StatisticsType,
                                Metrics]] = algorithm_utils.run_train_eval(
                                    self, backend, model, central_contexts, callbacks=callbacks, 
                                    compute_metrics=compute_metrics)
            #print('\nStep 3')
            # Step 3
            # For each query result, accumulate metrics and
            # let model handle statistics result if query had any.
            stats_context_pairs = []
            for central_context, (stats,
                                  metrics) in zip(central_contexts, results):
                all_metrics |= metrics
                if stats is not None:
                    stats_context_pairs.append((central_context, stats))
            
            # 
            #print('Distributed local rank:', hvd.local_rank, all_metrics)
            #all_metrics = hvd.allgather()
            # Process statistics and get new model.
            
            (model, update_metrics
             ) = self.process_aggregated_statistics_from_all_contexts(
                 tuple(stats_context_pairs), all_metrics, model)

            all_metrics |= update_metrics
            #print('Step 4')
            # Step 4
            # End-of-iteration callbacks
            
            for callback in callbacks:
                stop_signal, callback_metrics = (
                    callback.after_central_iteration(
                        all_metrics,
                        model,
                        central_iteration=self._current_central_iteration))
                all_metrics |= callback_metrics
                should_stop |= stop_signal
                if self._lr_scheduler_monitor_val is not None:
                    metrics_dict = all_metrics.to_simple_dict()
                    monitor_metric = 'Central val | loss'
                    
                    monitor_val: float = metrics_dict[monitor_metric] if metrics_dict.get(monitor_metric, None) is not None else None
                    if monitor_val is not None:
                        self._lr_scheduler_monitor_val: Metrics = Metrics([(self._lr_scheduler_monitor_val_str, monitor_val)])
                        #print(f'[RUN]: {self._lr_scheduler_monitor_val}')

            if send_metrics_to_platform:
                get_platform().consume_metrics(
                    all_metrics, iteration=self._current_central_iteration)

            if should_stop:
                break
            self._current_central_iteration += 1

        for callback in callbacks:
            # Calls with central iteration configs used for final round.
            callback.on_train_end(model=model)

        return model


class FederatedNNAlgorithm(FederatedAlgorithm[NNAlgorithmParamsType,
                                              ModelHyperParamsType,
                                              StatefulModelType,
                                              StatisticsType,
                                              AbstractDatasetType]):
    """
    Inherits from new FederatedAlgorithm and not pfl.FederatedAlgorithm source code.

    Adjust from https://github.com/apple/pfl-research/pfl/algorithm/base (FederatedNNAlgorithm)
    """
    def __init__(self, lr_scheduler_monitor_val: str = None):
        super().__init__(lr_scheduler_monitor_val)
        # Just a placeholder of tensors to get_parameters faster.
        self._initial_model_state = None

    @abstractmethod
    def train_one_user(
        self, initial_model_state: StatisticsType, model: StatefulModelType,
        user_dataset: AbstractDatasetType,
        central_context: CentralContext[NNAlgorithmParamsType,
                                        ModelHyperParamsType]
    ) -> Tuple[StatisticsType, Metrics]:
        pass

    def get_next_central_contexts(
        self,
        model: StatefulModelType,
        iteration: int,
        algorithm_params: NNAlgorithmParamsType,
        model_train_params: ModelHyperParamsType,
        model_eval_params: Optional[ModelHyperParamsType] = None,
    ) -> Tuple[Optional[Tuple[CentralContext[NNAlgorithmParamsType,
                                             ModelHyperParamsType], ...]],
               StatefulModelType, Metrics]:
        if iteration == 0:
            self._initial_model_state = None

        # Stop condition for iterative NN federated algorithms.
        if iteration == algorithm_params.central_num_iterations:
            return None, model, Metrics()

        do_evaluation = iteration % algorithm_params.evaluation_frequency == 0
        static_model_train_params: ModelHyperParamsType = \
            model_train_params.static_clone()
        static_model_eval_params: Optional[ModelHyperParamsType]
        static_model_eval_params = None if model_eval_params is None else model_eval_params.static_clone(
        )

        configs: List[CentralContext[
            NNAlgorithmParamsType, ModelHyperParamsType]] = [
                CentralContext(
                    current_central_iteration=iteration,
                    do_evaluation=do_evaluation,
                    cohort_size=get_param_value(
                        algorithm_params.train_cohort_size),
                    population=Population.TRAIN,
                    model_train_params=static_model_train_params,
                    model_eval_params=static_model_eval_params,
                    algorithm_params=algorithm_params.static_clone(),
                    seed=self._get_seed())
            ]
        if do_evaluation and algorithm_params.val_cohort_size:
            configs.append(
                CentralContext(
                    current_central_iteration=iteration,
                    do_evaluation=do_evaluation,
                    cohort_size=algorithm_params.val_cohort_size,
                    population=Population.VAL,
                    model_train_params=static_model_train_params,
                    model_eval_params=static_model_eval_params,
                    algorithm_params=algorithm_params.static_clone(),
                    seed=self._get_seed()))

        return tuple(configs), model, Metrics()

    def simulate_one_user(
        self, model: StatefulModelType, user_dataset: AbstractDatasetType,
        central_context: CentralContext[NNAlgorithmParamsType,
                                        ModelHyperParamsType],
        compute_metrics: bool,
    ) -> Tuple[Optional[StatisticsType], Metrics]:
        """
        If population is ``Population.TRAIN``, trains one user and returns the
        model difference before and after training.
        Also evaluates the performance before and after training the user.
        Metrics with the postfix "after local training" measure the performance
        after training the user.
        If population is not ``Population.TRAIN``, does only evaluation.
        """
        # pytype: disable=duplicate-keyword-argument
        initial_metrics_format_fn = lambda n: TrainMetricName(
            n, central_context.population, after_training=False)
        final_metrics_format_fn = lambda n: TrainMetricName(
            n, central_context.population, after_training=True)
        # pytype: enable=duplicate-keyword-argument

        metrics = Metrics()
        # Train local user.

        if central_context.population == Population.TRAIN:
            #print('SIMULATE_ONE_USER:', central_context.population)
            # Evaluate before local training
            if central_context.do_evaluation:
                #print('simulate_one_user (Population.TRAIN:) Evaluate before local training')
                metrics |= model.evaluate(user_dataset,
                                          initial_metrics_format_fn,
                                          central_context.model_eval_params)
            # local training
            #print('simulate_one_user (Population.TRAIN:) local training')
            self._initial_model_state = model.get_parameters(
                self._initial_model_state)
            statistics, train_metrics, local_meta_data = self.train_one_user(
                self._initial_model_state, model, user_dataset,
                central_context, compute_metrics=compute_metrics)
            #print('train_metrics',train_metrics)
            metrics |= train_metrics

            # Evaluate after local training.
            if central_context.do_evaluation:
                #print('simulate_one_user (Population.TRAIN:) Evaluate before after training')
                metrics |= model.evaluate(user_dataset,
                                          final_metrics_format_fn,
                                          central_context.model_eval_params)

            model.set_parameters(self._initial_model_state)
            #print('simulate_one_user (Population.TRAIN:)',  statistics._data['lstm.weight_ih_l1'], metrics)
            return statistics, metrics
        else:
            #print('SIMULATE_ONE_USER:', central_context.population)
            metrics = model.evaluate(user_dataset, initial_metrics_format_fn,
                                     central_context.model_eval_params)
            #print('simulate_one_user', metrics)
            return None, metrics
