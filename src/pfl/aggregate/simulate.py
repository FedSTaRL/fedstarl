# Copyright © 2023-2024 Apple Inc.
import math
from typing import List, Optional, Tuple

import numpy as np

from pfl.aggregate.base import (
    Aggregator,
    Backend,
    SumAggregator,
    get_num_datapoints_weight_name,
    get_num_devices_weight_name,
    get_num_params_weight_name,
    get_total_weight_name,
)
#from pfl.algorithm import FederatedAlgorithm
from pfl.common_types import Population
from pfl.context import CentralContext, UserContext
from pfl.data.federated_dataset import FederatedDatasetBase
from pfl.internal.ops.numpy_ops import NumpySeedScope
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.internal.ops.selector import has_framework_module
from pfl.metrics import Metrics, Weighted, Zero, MetricName
from pfl.privacy import PrivacyMetricName
from pfl.model.base import Model
from pfl.postprocessor.base import Postprocessor
from pfl.stats import TrainingStatistics, WeightedStatistics

from src.pfl.algorithm import FederatedAlgorithm
from ..callbacks.base import TrainingProcessCallback


class SimulatedBackend(Backend):
    """
    Adjusted from https://github.com/apple/pfl-research/pfl/aggregate/simulate

    Backend that simulates an algorithm on the given federated datasets.
    The simulation performs the following:

    1. Perform the following steps on all users (can be parallelized with
       multiple workers):

       (a) Sample a user ``Dataset`` from the source ``FederatedDatasetBase``.
       (b) Train model on the user dataset and extract the model updates.
            The algorithm used for local training is injected by
            ``training_algorithm`` when calling ``gather_results``.
       (c) Apply local clipping to individual contribution.
       (d) Privatize individual contribution with ``local_privacy``.

    2. Apply central clipping of individual contribution.
    3. Sum all the individual contributions.
    4. Add central noise.


    :param training_data:
        Training dataset generator, derived from `FederatedDatasetBase`.
    :param val_data:
        Validation dataset generator, derived from `FederatedDatasetBase`.
    :param postprocessors:
        Steps for postprocessing statistics returned by users, e.g. weighting,
        sparsification, compression.
    :param aggregator:
        The aggregation method to use. If not specified, use the default
        :class:`pfl.aggregate.base.SumAggregator` which will sum all
        statistics of users trained in each process, then all-reduce sum
        across processes.
    :param max_overshoot_fraction:
        Simulate overshooting the number of results received from users. This
        is disabled by default, but is expected to be up to 0.05-0.10 when
        training live.
        The overshooting is assumed to be a fraction
        ``p ~ Uniform(0,max_overshoot_fraction)``.
    """

    def __init__(self,
                 training_data: FederatedDatasetBase,
                 val_data: FederatedDatasetBase,
                 postprocessors: Optional[List[Postprocessor]] = None,
                 aggregator: Optional[Aggregator] = None,
                 max_overshoot_fraction: float = 0.0):

        super().__init__()
        self._training_dataset = training_data
        self._val_dataset = val_data
        self._postprocessors = postprocessors or []
        if aggregator is None:
            self._aggregator: Aggregator = SumAggregator()
        else:
            self._aggregator = aggregator
        self._max_overshoot_fraction = max_overshoot_fraction
        self._local_step = 0

    async def async_gather_results(
        self,
        model: Model,
        training_algorithm: FederatedAlgorithm,
        *,
        central_context: CentralContext,
        callbacks: Optional[List[TrainingProcessCallback]] = None,
        compute_metrics: bool=False, # compute metrics in sgd bridge 
    ) -> Tuple[Optional[TrainingStatistics], Metrics]:
        """
        Run a simulation on a cohort of user datasets.
        Returns aggregated statistics and metrics from the procedure described
        in the class description.

        This implementation is not actually asynchronous and it yields execution
        only once it is completely done.
        The work may be split up amongst multiple workers.
        In this case, the order in which this method is called must be the same
        across workers, so that their synchronisation points line up.

        :param model:
            A model to be used for gathering the statistics.
        :param training_algorithm:
            An object that inherits from
            :class:`~pfl.algorithm.base.FederatedAlgorithm`,
            which injects the algorithm-specific behaviour.
        :param central_context:
            Settings to use for this round.
        :returns:
            A dictionary of the raw model updates and a metrics object
            of aggregated statistics. The model updates are weighted depending
            on the weighting strategy. The model update can be ``None`` if the
            algorithm does not train on the val population.

            If using local and central DP, useful privacy metrics with the
            category "local DP" and "central DP" are returned.
            Metrics from training are returned with the population as the
            category.
            ``number of devices`` is the number of users
            that the data came from.
            ``number of data points`` is self-explanatory.
            ``total weight`` is the denominator for a potentially weighted
            aggregation.
        """
        cohort_size = central_context.cohort_size
        population = central_context.population
        # Simulate overshooting the target number of users.
        # Keep consistent across workers with central seed.
        overshoot_seed = self._random_state.randint(0, 2**32, dtype=np.uint32)
        with NumpySeedScope(overshoot_seed):
            overshoot_cohort_size = cohort_size * (
                1. + self._max_overshoot_fraction)
            cohort_size = int(
                math.floor(
                    np.random.uniform(cohort_size, overshoot_cohort_size)))

        selected_dataset = self._training_dataset if population == Population.TRAIN else self._val_dataset
        num_users_trained = 0
        num_total_datapoints = Weighted(0, 0)
        total_weight = None
        user_metrics = Zero

        server_statistics = None

        for callback in callbacks:
            if hasattr(callback, 'on_local_train_begin'):
                callback.on_local_train_begin(**{'central_context': central_context})

        for user_dataset, local_seed in selected_dataset.get_cohort(
                cohort_size):
            
            for callback in callbacks:
                if hasattr(callback, 'on_local_user_train_begin'):
                    callback.on_local_user_train_begin(model=None)
            user_statistics, metrics_one_user = (
                training_algorithm.simulate_one_user(model, user_dataset,
                                                     central_context, compute_metrics=compute_metrics))
            #print('[BACKEND] -- metrics_one_user:', metrics_one_user)
            metric_data = [(metric_key, metric_value)
                for metric_key, metric_value in metrics_one_user.to_simple_dict().items()
                if metric_key != 'loss_tensor'
                ]
            metric_new = Metrics(metric_data)
            #assert True == False, f'{type(metrics_one_user), metrics_one_user, metric_new, metric_data}'
            metrics_one_user = metric_new # metrics_one_user.pop('loss_tensor') if 'loss_tensor' in list(metrics_one_user.keys()) else metrics_one_user
            for callback in callbacks:
                if hasattr(callback, 'on_local_user_train_end'):
                    logger_kwargs = {
                        'metrics': metrics_one_user,
                        #'local_iteration': self._local_step,
                        'central_iteration': central_context.current_central_iteration,
                        'user_id': user_dataset.user_id,
                        'cohort_size': cohort_size,
                    }
                    callback.on_local_user_train_end(logger_kwargs=logger_kwargs, central_context=central_context)

            user_context = UserContext(num_datapoints=len(user_dataset),
                                       seed=local_seed,
                                       user_id=user_dataset.user_id,
                                       metrics=metrics_one_user)

            num_total_datapoints += Weighted.from_unweighted(
                user_context.num_datapoints)

            if user_statistics is not None:
                #print(user_statistics.num_parameters)
                #_, weights = user_statistics.get_weights()
                #sig_norm = get_ops().global_norm(weights, order=2)
                #print(sig_norm)
                

                if isinstance(user_statistics, WeightedStatistics):
                    current_weight = Weighted.from_unweighted(
                        user_statistics.weight)
                    if total_weight is None:
                        total_weight = current_weight
                    else:
                        total_weight += current_weight

                old_user_statistics = user_statistics
                #print(self._postprocessors)
                for p in self._postprocessors:
                    #print(user_statistics.num_parameters)
                    (user_statistics,
                     postprocessor_metrics) = p.postprocess_one_user(
                         stats=user_statistics, user_context=user_context)
                    _, weights = user_statistics.get_weights()
                    sig_norm = get_ops().global_norm(weights, order=2)
                    print('after dp', sig_norm)
                    noise = abs(user_statistics.get_weights()[0] - old_user_statistics.get_weights()[0])
                    #print(noise)
                    metrics_one_user |= postprocessor_metrics
                #exit(1)
                if server_statistics is None:
                    user_statistics = user_statistics.apply_elementwise(
                        get_ops().clone)
                server_statistics = self._aggregator.accumulate(
                    accumulated=server_statistics, user_stats=user_statistics)

            user_metrics += metrics_one_user
            #print('metrics_one_user', metrics_one_user)
            #print('user_metrics', user_metrics)

            num_users_trained += 1
            self._local_step += 1

        #user_metrics_new = [
        #    (MetricName(f'{metric.lower()}', value)) if metric in ['Tp', 'Tn', 'Fp', 'Fn'] else (MetricName(f'{metric.lower()}', value / self._local_step))
        #    for metric, value in user_metrics.to_simple_dict().items()
        #]
        #print('\nuser_metrics\n', user_metrics)
        user_metrics_new = Metrics()
        for metric, value in user_metrics: #.to_simple_dict().items():
            if metric in ['Loss','Accuracy','Tn','Fp','Fn','Tp','F0.5','F1','Precision','Recall', 'Loss_ce', 'Loss_cl', 'Loss_cl_batch_sim', 'Loss_cl_class_sim']:
                          #'l2 norm bound', 'fraction of clipped norms', 'norm before clipping', 'DP noise std. dev.', 'signal-to-DP-noise ratio']:                    
                if metric in ['Tp', 'Tn', 'Fp', 'Fn']:
                    user_metrics_new[MetricName(f'{metric.lower()}', population)] = int(value)
                else:
                    user_metrics_new[MetricName(f'{metric.lower()}', population)] = value / central_context.cohort_size
            elif isinstance(metric, PrivacyMetricName):
                user_metrics_new[metric] = value
            else:
                #print(metric, ' || ', metric.split(' | ')[1].split(' ')[0])
                try:
                    if metric.split(' | ')[1].split(' ')[0] in ['tn', 'tp', 'fn', 'fp']:
                        #print(metric, ' || ', metric.split(' | ')[1].split(' ')[0], value)
                        user_metrics_new[metric] = int(value)
                    else:
                        user_metrics_new[metric] = value / central_context.cohort_size
                except Exception as e:
                    print(metric, value)
                    raise e

        #print('local step', self._local_step)

        
        user_metrics = user_metrics_new
        # MetricName('number of data points', category)
        #print('\nuser_metrics\n', user_metrics)
        #exit(0)
        worker_metrics = Metrics()
        #print('\nworker_metrics\n', worker_metrics)
        worker_metrics[get_num_devices_weight_name(
            population)] = num_users_trained
        worker_metrics[get_num_datapoints_weight_name(
            population)] = Weighted(int(num_total_datapoints.weighted_value), 1)
        #print('\nworker_metrics\n', worker_metrics)
        worker_metrics |= user_metrics
        #print('\nworker_metrics\n', worker_metrics)
        if total_weight is not None:
            worker_metrics[get_total_weight_name(population)] = total_weight

        # Only do reduce if multiple processes running.
        if has_framework_module() and get_ops().distributed.world_size > 1:
            if server_statistics is not None:
                model_update, total_metrics = self._aggregator.worker_reduce(
                    aggregated_worker_stats=server_statistics,
                    central_context=central_context,
                    aggregated_worker_metrics=worker_metrics)
            else:
                model_update = server_statistics
                total_metrics = self._aggregator.worker_reduce_metrics_only(
                    central_context=central_context,
                    aggregated_worker_metrics=worker_metrics)
        else:
            model_update = server_statistics
            total_metrics = worker_metrics

        if model_update is not None:
            # Apply central DP to aggregated statistics.
            # Reverse the order when postprocessing on server
            for p in self._postprocessors[::-1]:
                (model_update, postprocessor_metrics) = p.postprocess_server(
                    stats=model_update,
                    central_context=central_context,
                    aggregate_metrics=total_metrics)
                total_metrics |= postprocessor_metrics
            total_metrics[
                get_num_params_weight_name()] = model_update.num_parameters
        
        for callback in callbacks:
            #assert hasattr(callback, 'on_local_train_begin'), f'{callback}'
            if hasattr(callback, 'on_local_train_end'):
                callback.on_local_train_end(central_context=central_context, logger_kwargs={'metrics': total_metrics})
        
        #print('total_metrics', total_metrics)
        return model_update, total_metrics
