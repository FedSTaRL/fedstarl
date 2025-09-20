import logging
import operator
import os
import re
import subprocess
import time
import typing
from collections import OrderedDict
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, Union

from pfl.aggregate.base import get_num_datapoints_weight_name
from pfl.common_types import Population, Saveable
from pfl.data.dataset import AbstractDatasetType
from pfl.exception import CheckpointNotFoundError
from pfl.hyperparam.base import ModelHyperParams
from pfl.internal import ops
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.metrics import MetricName, MetricNamePostfix, Metrics, StringMetricName, get_overall_value
from pfl.model.base import EvaluatableModelType, ModelType, StatefulModel
from pfl.model.ema import CentralExponentialMovingAverage
from pfl.callback import WandbCallback

from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(name=__name__)


__all__ = [
    'WandbLoggerCallback'
]


class TrainingProcessCallback(Generic[ModelType]):
    """
    Adjusted from pfl.callbacks.TrainingProcessCallback by adding client training hooks.

    Base class for callbacks.
    """

    def on_train_begin(self, *, model: ModelType=None) -> Metrics:
        """
        Called before the first central iteration.
        """
        return Metrics()

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType=None, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        """
        Finalize any computations after each central iteration.

        :param aggregate_metrics:
            A :class:`~pfl.metrics.Metrics` object with aggregated metrics
            accumulated from local training on users and central updates
            of the model.
        :param model:
            A reference to the `Model` that is trained.
        :param central_iteration:
            The current central iteration number.
        :returns:
            A tuple.
            The first value returned is a boolean, signaling that training
            should be interrupted if ``True``.
            Can be useful for implementing features with early stopping or
            convergence criteria.
            The second value returned is new metrics.
            Do not include any of the aggregate_metrics!
        """
        return False, Metrics()

    def on_train_end(self, *, model: ModelType=None) -> None:
        """
        Called at the end of training.
        """
        pass

    def on_local_train_begin(self, **kwargs) -> None:
        """
        Called at the beginning of local training of all clients during 
        one communication round.
        """
        pass

    def on_local_user_train_begin(self, **kwargs) -> None:
        """
        Called at the beginning of local training of a client.
        """
        pass

    def on_local_user_train_end(self, **kwargs) -> None:
        """
        Called at the end of local training of a client.
        """
        pass

    def on_local_train_end(self, **kwargs) -> None:
        """
        Called at the end of local training of all clients during 
        one communication round.
        """
        pass