import contextlib
from abc import abstractmethod, ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Callable, Dict, Any, Tuple, Literal
from torch._C import dtype
from torch import Tensor

from pfl.context import LocalResultMetaData
from pfl.model.pytorch import PyTorchModel
from pfl.internal.ops.selector import get_framework_module
from pfl.internal.ops import pytorch_ops
from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam import NNEvalHyperParams, NNTrainHyperParams
from pfl.internal.ops.selector import get_framework_module
from pfl.metrics import Metrics, MetricsZero, StringMetricName, Zero
from pfl.metrics import Weighted, MetricValue
from pfl.stats import MappedVectorStatistics

from src.pfl.data_class import LocalResultMetaData
from src.utils import LossOptions
from src.utils.datasets import BaseDataObject


__all__ = [
   "PFLTorchModule",
   "BaseSequencePyTorchModel"
]


class PFLTorchModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def loss(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def metrics(self, *args, **kwargs):
        raise NotImplementedError

class BasePyTorchModel(PyTorchModel):
    def __init__(self, 
                 model, 
                 local_optimizer_create, 
                 central_optimizer, 
                 central_learning_rate_scheduler=None, 
                 amp_dtype: dtype | None = None, 
                 grad_scaling: bool = False, 
                 model_dtype_same_as_amp: bool = False, 
                 use_torch_compile: bool = False,
                 use_pfl_internals: bool = False
                 ):
        self._use_pfl_internals = use_pfl_internals
        super().__init__(model, local_optimizer_create, central_optimizer, central_learning_rate_scheduler, amp_dtype, grad_scaling, model_dtype_same_as_amp, use_torch_compile)


class BaseSequencePyTorchModel(BasePyTorchModel):
    def __init__(self, 
                 model, 
                 local_optimizer_create, 
                 central_optimizer, 
                 central_learning_rate_scheduler=None, 
                 amp_dtype: dtype | None = None, 
                 grad_scaling: bool = False, 
                 model_dtype_same_as_amp: bool = False, 
                 use_torch_compile: bool = False,
                 use_pfl_internals: bool = False,
                 local_optimizer_kwargs: Dict[str, Any]=None,
                 ):
        self._local_optimizer_kwargs = local_optimizer_kwargs if local_optimizer_kwargs is not None else {}
        if central_learning_rate_scheduler is not None:
            self._monitor_values = []
            self._learning_rates = []
            self._monitor_count = 0
            self._best_monitor = float('Inf') # i.e. lowest for loss
        super().__init__(model, local_optimizer_create, central_optimizer, central_learning_rate_scheduler, amp_dtype, grad_scaling, model_dtype_same_as_amp, use_torch_compile, use_pfl_internals)

    def new_local_optimizer(self, learning_rate) -> torch.optim.Optimizer:
        return self._local_optimizer_create(self._model.parameters(),
                                            lr=learning_rate,
                                            **self._local_optimizer_kwargs)
    

    def do_multiple_epochs_of(self, user_dataset: AbstractDatasetType,
                              train_params: NNTrainHyperParams,
                              train_step_fn: Callable,
                              compute_metrics: bool=False,
                              name_formatting_fn: Optional[Callable[
                                    [str], StringMetricName]] = lambda n: StringMetricName(n),
                              **kwargs) -> Tuple[LocalResultMetaData, Metrics | Any]:
        """
        Adjusted from source code 

        Perform multiple epochs of training. The customizable training
        function that will use a batch of data to update the local
        model state is defined by ``train_step_fn``.
        If you have specified an optimizer using the parameter
        `local_optimizer_create` in the constructor, a new optimizer will
        be initialized before training is performed in this method.

        :param user_dataset:
            Dataset of type ``Dataset`` to train on.
        :param train_params:
            An instance of :class:`~pfl.hyperparam.base.NNTrainHyperParams`
            containing configuration for training.
        :param train_step_fn:
            A function with the following arguments:
            * pytorch_model - the pytorch model object to train on.
            * local_optimizer - the optimizer to use for training.
            * raw_data - an iterable of tensors unpacked into the loss function
            ``pytorch_model.loss(*raw_data)``
            * train_kwargs - the ``train_kwargs`` property from the user
            dataset. With this, you can pass user-specific metadata to local
            training.
            * train_step_args - an instance of
            :class:`~pfl.internal.ops.pytorch_ops.PyTorchTrainStepArgs`
            that contains common arguments for PyTorch local training.
            * kwargs - other keyword arguments that a custom ``train_step_fn``
            might have.
        """
        if compute_metrics:
            assert name_formatting_fn is not None

        num_epochs = (1 if train_params.local_num_epochs is None else
                      train_params.get('local_num_epochs'))
        local_optimizer = self.new_local_optimizer(
            learning_rate=train_params.local_learning_rate)

        local_optimizer.zero_grad()
        # Common arguments used in `train_step_fn`
        local_num_steps = self._get_local_num_steps(train_params,
                                                    len(user_dataset))
        train_step_args = pytorch_ops.PyTorchTrainStepArgs(
            amp_context=self._amp_context or contextlib.nullcontext(),
            grad_scaler=self._grad_scaler,
            max_grad_norm=train_params.local_max_grad_norm,
            grad_accumulation_state=pytorch_ops.GradAccumulationState(
                local_num_steps, train_params.grad_accumulation_steps))

        metrics = Zero
        postprocess_fns = []

        for _ in range(num_epochs):
            for batch_ix, batch in enumerate(
                    user_dataset.iter(train_params.get('local_batch_size'))):

                if not isinstance(batch, list):
                    if batch.batch_size == 1: # skip
                        continue 

                if batch_ix == train_params.get('local_num_steps'):
                    break
                # initialize metric for a batch
                metrics_one_batch = Metrics()
                batch = self._prepare_batch(batch)
                # get compute metric per batch
                train_step_fn_args = [self._model, local_optimizer, batch,
                    user_dataset.train_kwargs, train_step_args]
                if not self._use_pfl_internals: train_step_fn_args.append(compute_metrics)

                metrics_outputs = train_step_fn(*train_step_fn_args, **kwargs)
                if compute_metrics and not self._use_pfl_internals:
                    for name, metric_value in metrics_outputs.items():
                        if isinstance(metric_value, tuple):
                            # Is tuple with metric postprocess function as 2nd
                            # argument.
                            metric_value, postprocess_fn = metric_value
                            allows_distributed_evaluation = False
                        else:
                            postprocess_fn = lambda x: x
                        if batch_ix == 0:
                            # Save for later when postprocessing.
                            postprocess_fns.append(postprocess_fn)

                        metrics_one_batch[name_formatting_fn(name)] = metric_value

                    metrics += metrics_one_batch
        
        if isinstance(metrics, MetricsZero):
            metrics = None
    
        return LocalResultMetaData(num_steps=local_num_steps, user_id=user_dataset.user_id, data_size=len(user_dataset)), metrics
        
    def evaluate(self,
                 dataset: AbstractDatasetType,
                 name_formatting_fn: Callable[
                     [str], StringMetricName] = lambda n: StringMetricName(n),
                 eval_params: Optional[NNEvalHyperParams] = None) -> Metrics:
        #print('\nevaluatung model\n')
        # Use mini-batches if local_batch_size is set.
        #print('Evaluate')
        batch_size = (len(dataset) if eval_params is None
                      or eval_params.local_batch_size is None else
                      eval_params.get('local_batch_size'))
        assert isinstance(batch_size, int)
        metrics = Zero

        postprocess_fns = []
        allows_distributed_evaluation = True
        amp_context = self._amp_context or contextlib.nullcontext()
        #print(dataset, eval_params)
        for batch_ix, batch in enumerate(dataset.iter(batch_size)):
            metrics_one_batch = Metrics()
            batch = self._prepare_batch(batch)
            #print('amp_context', amp_context)
            with amp_context:
                #print('amp_context', 'working')
                if isinstance(batch, Dict):
                    metrics_outputs = self._model.metrics(**{
                        **batch,
                        **dataset.eval_kwargs
                    })
                elif isinstance(batch, BaseDataObject):
                    metrics_outputs = self._model.metrics(**{
                        'x': batch.data, 'y': batch.label, 'N': batch.seq_len, 'batch_size': batch.batch_size,
                        **dataset.eval_kwargs
                    })
                    #print('Model', metrics_outputs)
                else:
                    metrics_outputs = self._model.metrics(
                        *batch, **dataset.eval_kwargs)
                #print('metrics', metrics_outputs)
            #print('amp_context', 'END TEST')
            
            for name, metric_value in metrics_outputs.items():
                if name != 'loss_tensor':
                    if isinstance(metric_value, tuple):
                        # Is tuple with metric postprocess function as 2nd
                        # argument.
                        metric_value, postprocess_fn = metric_value
                        allows_distributed_evaluation = False
                    else:
                        postprocess_fn = lambda x: x
                    if batch_ix == 0:
                        # Save for later when postprocessing.
                        postprocess_fns.append(postprocess_fn)

                    metrics_one_batch[name_formatting_fn(name)] = metric_value

            #print('metrics_one_batch', metrics_one_batch)
            metrics += metrics_one_batch

            #print('metrics', metrics)

        # Distributed evaluation is only allowed if no postprocess functions
        # are used.
        self._allows_distributed_evaluation = allows_distributed_evaluation
        if isinstance(metrics, MetricsZero):
            raise RuntimeError(  # noqa: TRY004
                f"Accumulated metrics were Zero for user with dataset {dataset}"
            )
        #print(metrics)
        #print('\n\n')
        #for (name,
        #         metric_value), postprocess_fn in zip(metrics, postprocess_fns):
        #    print('name, metric_value, postprocess_fn', name, metric_value, postprocess_fn)
        #print('\n\n')
        processed_metrics = [(name, postprocess_fn(metric_value))
            for (name,
                 metric_value), postprocess_fn in zip(metrics, postprocess_fns)]
        #print(hasattr(dataset, 'user_id'), dataset.user_id is not None)
        #if hasattr(dataset, 'user_id') and dataset.user_id is not None:
        #    processed_metrics += [ClientMetricName(description='user id', user_id=dataset.user_id)]
        #else:
        #    processed_metrics += [ClientMetricName(description='user id', user_id=None)]
        
        processed_metrics = Metrics(processed_metrics)
        #print('processed_metrics', processed_metrics)
        #if hasattr(dataset, 'user_id') and dataset.user_id is not None:
        #    print(dataset.user_id)
        #    processed_metrics['client id'] = [dataset.user_id]
        #print('processed_metrics_eddited', processed_metrics)

        #print('\n\nprocessed_metrics EVALUATE:', processed_metrics, '\n\n')
        return processed_metrics
    
    @staticmethod
    def _prepare_batch(batch):
        if isinstance(batch, Dict):
            return {
                k: get_framework_module().to_tensor(v)
                for k, v in batch.items()
            }
        elif isinstance(batch, BaseDataObject):
            #print('Hello')
            return batch

        else:
            return [get_framework_module().to_tensor(data) for data in batch]
    
    def apply_model_update(
            self, statistics: MappedVectorStatistics,
            monitor_val: Metrics=None,
    ) -> Tuple['PyTorchModel', Metrics]:
        assert isinstance(statistics, MappedVectorStatistics)
        metrics = Metrics()

        self._central_optimizer.zero_grad()
        for variable_name, difference in statistics.items():
            if self.variable_map[variable_name].grad is None:
                self.variable_map[variable_name].grad = torch.zeros_like(
                    self.variable_map[variable_name])
            # Interpret the model updates as gradients.

            self.variable_map[
                variable_name].grad.data.copy_(  # type: ignore[union-attr]
                    -1 * pytorch_ops.to_tensor(difference))

        self._central_optimizer.step()

        if self._use_pfl_internals:
            if self._central_learning_rate_scheduler is not None:
                if isinstance(self._central_learning_rate_scheduler, torch.optim.lr_scheduler.LambdaLR):
                    self._central_learning_rate_scheduler.step()
                else:
                    self._central_learning_rate_scheduler.step(metrics=monitor_val)
        else:
            monitor_key, monitor_val = list(monitor_val.to_simple_dict().items())[0]

            if self._central_learning_rate_scheduler is not None and monitor_val != None:
                self._monitor_values.append(monitor_val)
                self._learning_rates.append(self._central_optimizer.param_groups[0]['lr']) # append current lr
                #print(type(self._central_learning_rate_scheduler))
                if isinstance(self._central_learning_rate_scheduler, torch.optim.lr_scheduler.LambdaLR):
                    self._central_learning_rate_scheduler.step()
                else:
                    self._central_learning_rate_scheduler.step(metrics=monitor_val) # update current lr, i.e. current --> old, get new lr
                current_lr = self._central_optimizer.param_groups[0]['lr']
                if self._best_monitor > monitor_val:
                    self._best_monitor == monitor_val
                    self._monitor_count = 0
                if self._best_monitor < monitor_val:
                    self._monitor_count += 1
                    print(f'[Central Learning Rate Scheduler] -- Monitor Val: Has not improved in {self._monitor_count} step!')
                if len(self._monitor_values) >= 2:
                    print(f'[Central Learning Rate Scheduler] -- Monitor Val: {self._monitor_values[-2]} --> {self._monitor_values[-1]}')
                print(f'[Central Learning Rate Scheduler] -- LR: {self._learning_rates[-1]} --> {current_lr}')


        metrics[StringMetricName('learning rate')] = Weighted.from_unweighted(
            self._central_optimizer.param_groups[0]['lr'])

        return self, metrics


from dataclasses import dataclass, asdict
from pfl.metrics import StringMetricName


@dataclass(frozen=True, eq=False)
class ClientMetricName(StringMetricName):
    """
    A structured name for metrics which includes the population the metric was
    generated from and training descriptions.

    :param description:
        The metric name represented as a string.
    :param user_id:
        The user id used in training.
    """
    user_id: int
    
    def __repr__(self) -> str:
        return (f'ClientMetricName({self.description}={self.user_id})')
    
    def __str__(self) -> str:
        return (f'ClientMetricName({self.description}={self.user_id})')

    def __iter__(self):
        return iter([f'ClientMetricName(description={self.description}, user_id={self.user_id})', 
                     f'ClientMetricName(description={self.description}, user_id={self.user_id})'])
        #for key, value in asdict(self).items():
        #    yield key, value


class BaseModule(nn.Module, ABC):
    def __init__(self, loss_kwargs: Dict[str, Any]={'loss_fn': 'bce'}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if loss_kwargs.get('loss_fn', None) is not None:
            self._loss_fn: Literal['bce', 'ce', 'reconstruction', 'sscl'] = loss_kwargs['loss_fn']
            loss_kwargs.pop('loss_fn')
            self.configure_criterion(loss_kwargs=loss_kwargs)
    
    @abstractmethod
    def forward(self, ) -> Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def loss(self, **kwargs) -> Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def metrics(self, **kwargs) -> Dict[str, int | float | Tensor | MetricValue]:
        raise NotImplementedError
    
    def configure_criterion(self, **kwargs) -> Dict[str, nn.Module]:
        if self._loss_fn == LossOptions.bce:
            self.criterion = {self._loss_fn: nn.BCEWithLogitsLoss()}
        elif self._loss_fn == LossOptions.ce:
            self.criterion = {self._loss_fn: nn.CrossEntropyLoss()}
        elif self._loss_fn == 'sscl' or self._loss_fn == LossOptions.reconstruction: # not implemented
            raise NotImplementedError
        else:
            raise ValueError(f'{self._loss_fn=} is not implemented.')
    
    def _compute_predictions_from_logits(self, logits: Tensor) -> Tensor:
        if self._loss_fn == LossOptions.bce:
            probas = F.sigmoid(logits)
            return (probas > 0.5).int()
            
        elif self._loss_fn == LossOptions.ce:
            probas = F.softmax(logits, dim=1)
            outputs_max, outputs_argmax = torch.max(probas, dim=1)
            return outputs_argmax
        
        elif self._loss_fn == 'sscl': # not implemented
            probas = F.sigmoid(logits)
            return (probas > 0.5).int()
