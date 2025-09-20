import logging
import operator
import os
import re
import subprocess
import time
import typing
import wandb
import horovod.torch as hvd    
from sklearn.metrics import ConfusionMatrixDisplay
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

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
from pfl.privacy import PrivacyMetricName

from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(name=__name__)

from .base import TrainingProcessCallback

from src.utils import ModelOptions

__all__ = [
    'WandbLoggerCallback'
]

class WandbLoggerCallback(TrainingProcessCallback):
    """
    Adjusted from pfl.callbacks.WandbCallback and in inherits from custom callback class.

    Callback for reporting metrics to Weights&Biases dashboard for comparing
    different PFL runs.
    This callback has basic support for logging metrics. If you seek more
    advanced features from the Wandb API, you should make your own callback.

    See https://wandb.ai/ and https://docs.wandb.ai/ for more information on
    Weights&Biases.

    :param wandb_project_id:
        The name of the project where you're sending the new run. If the
        project is not specified, the run is put in an "Uncategorized" project.
    :param wandb_experiment_name:
        A short display name for this run. Generates a random two-word name
        by default.
    :param wandb_config:
        Optional dictionary (or argparse) of parameters (e.g. hyperparameter
        choices) that are used to tag this run in the Wandb dashboard.
    :param wandb_kwargs:
        Additional keyword args other than ``project``, ``name`` and ``config``
        that you can input to ``wandb.init``, see
        https://docs.wandb.ai/ref/python/init for reference.
    :param experiment_config: 
        Optional dictionary of complete parameters of experiment that is being executed.
    """

    # changec central format in centralevaluate callback
    __map_metric_keys_after_central_iteration = {
        # changed with format_fn argument
        'Central val | loss': 'central/loss (val)',
        'Central val | loss_ce': 'central/loss_ce (val)',
        'Central val | loss_cl': 'central/loss_cl (val)',
        'Central val | loss_cl_batch_sim': 'central/loss_cl_batch_sim (val)',
        'Central val | loss_cl_class_sim': 'central/loss_cl_class_sim (val)', 
        'Central val | accuracy': 'central/accuracy (val)', 
        'Central val | f0.5': 'central/f0.5 (val)', 
        'Central val | f1': 'central/f1 (val)', 
        'Central val | precision': 'central/precision (val)', 
        'Central val | recall': 'central/recall (val)', 
        'Central val | number of data points': 'central/number of data points (val)',
        #
        'Train population | loss before local training': 'train_population/loss before local training',
        'Train population | loss_ce before local training': 'train_population/loss_ce before local training',
        'Train population | loss_cl before local training': 'train_population/loss_cl before local training',
        'Train population | loss_cl_batch_sim before local training': 'train_population/loss_cl_batch_sim before local training',
        'Train population | loss_cl_class_sim before local training': 'train_population/loss_cl_class_sim before local training',
        'Train population | accuracy before local training': 'train_population/accuracy before local training',
        'Train population | f0.5 before local training': 'train_population/f0.5 before local training',
        'Train population | f1 before local training': 'train_population/f1 before local training',
        'Train population | precision before local training': 'train_population/precision before local training',
        'Train population | recall before local training': 'train_population/recall before local training',
        #
        'Train population | loss after local training': 'train_population/loss after local training',
        'Train population | loss_ce after local training': 'train_population/loss_ce after local training',
        'Train population | loss_cl after local training': 'train_population/loss_cl after local training',
        'Train population | loss_cl_batch_sim after local training': 'train_population/loss_cl_batch_sim after local training',
        'Train population | loss_cl_class_sim after local training': 'train_population/loss_cl_class_sim after local training',
        'Train population | accuracy after local training': 'train_population/accuracy after local training',
        'Train population | f0.5 after local training': 'train_population/f0.5 after local training',
        'Train population | f1 after local training': 'train_population/f1 after local training',
        'Train population | precision after local training': 'train_population/precision after local training',
        'Train population | recall after local training': 'train_population/recall after local training',
        #
        'Train population | total weight': 'train_population/total weight',
        'Train population | number of devices': 'train_population/number of devices', 
        'Train population | number of data points': 'train_population/number of data points',
        'Train population | loss': 'train_population/loss',
        'Train population | loss_ce': 'train_population/loss_ce',
        'Train population | loss_cl': 'train_population/loss_cl',
        'Train population | loss_cl_batch_sim': 'train_population/loss_cl_batch_sim',
        'Train population | loss_cl_class_sim': 'train_population/loss_cl_class_sim',
        'Train population | accuracy': 'train_population/accuracy',
        'Train population | f0.5': 'train_population/f0.5',
        'Train population | f1': 'train_population/f1',
        'Train population | precision': 'train_population/precision',
        'Train population | recall': 'train_population/recall',
        #
        'Val population | number of devices': 'val_population/number of devices', 
        'Val population | number of data points': 'val_population/number of data points',
        'Val population | loss_ce before local training': 'val_population/loss_ce before local training',
        'Val population | loss_cl before local training': 'val_population/loss_cl before local training',
        'Val population | loss_cl_batch_sim before local training': 'val_population/loss_cl_batch_sim before local training',
        'Val population | loss_cl_class_sim before local training': 'val_population/loss_cl_class_sim before local training',
        'Val population | accuracy before local training': 'val_population/accuracy before local training',
        'Val population | f0.5 before local training': 'val_population/f0.5 before local training',
        'Val population | f1 before local training': 'val_population/f1 before local training',
        'Val population | precision before local training': 'val_population/precision before local training',
        'Val population | recall before local training': 'val_population/recall before local training',
        #'Tn': 'train_population/tn',
        #'Fp': 'train_population/fp',
        #'Fn': 'train_population/fn',
        #'Tp': 'train_population/tp',
        #'F0.5': 'train_population/f0.5',
        #'F1': 'train_population/f1',
        #'Precision': 'train_population/precision',
        #'Recall': 'train_population/recall',
        'Number of parameters': 'Number of parameters',
        'Learning rate': 'Learning rate',
        'Overall time elapsed (min)': 'Overall time elapsed (min)',
        'Duration of iteration (s)': 'Duration of iteration (s)',
        'Overall average duration of iteration (s)': 'Overall average duration of iteration (s)',
        
        
        
    }
    def __init__(self,
                 wandb_project_id: str,
                 wandb_experiment_name: Optional[str] = None,
                 wandb_config=None,
                 experiment_config: DictConfig=None,
                 user_id_map: Dict[str, Any]=None,
                 config: DictConfig=None,
                 **wandb_kwargs):
        self._wandb_kwargs = {
            'entity': '', # add your entity  
            'project': wandb_project_id,
            'name': wandb_experiment_name,
            'config': wandb_config
        }
        self._wandb_kwargs.update(wandb_kwargs)
        self._experiment_config = experiment_config
        self._user_id_map = user_id_map

        self._config = config
        self._model_name = config.model.name if config is not None else None

    @property
    def wandb(self):
        # Not necessarily installed by default.
        import wandb
        return wandb
    
    def log_config_to_wandb(self) -> None:
        if self._experiment_config is not None:
            self.wandb.config.update(
                OmegaConf.to_container(self._experiment_config, resolve=True)
            )

    def on_train_begin(self, *, model: ModelType) -> Metrics:
        self._client_participation_table = {'train': {}, 'val': {}}
        self._cm_matrix = {'train': np.zeros([2, 2]), 'val': np.zeros([2, 2])}
        if get_ops().distributed.global_rank == 0:
            self.wandb.init(**self._wandb_kwargs)
            self.log_config_to_wandb()

        if self._user_id_map:
            self._client_participation_bar_plot = {'train': {}, 'val': {}}
            for client_id_int, user in self._user_id_map.items():
                self._client_participation_bar_plot['train'][client_id_int] = {
                    'user_id_str': user['user_id'],
                    'data_size': user['data_size'],
                    'participation': [], # record the participation per communincation round
                    'participation_size': []
                }

                self._client_participation_bar_plot['val'][client_id_int] = {
                    'user_id_str': user['user_id'],
                    'data_size': user['data_size'],
                    'participation': [], # record the participation per communincation round
                    'participation_size': []
                }
        else:
            self._client_participation_bar_plot = None
        
        return Metrics()

    def on_local_train_begin(self, *, model: ModelType=None, **kwargs):
        #print('\n Local Training \n')
        #print(f'\nLocal Training START')
        #if kwargs.get('central_context', None) is not None:
        #    print(kwargs['central_context'])
        pass

    def on_local_user_train_begin(self, *, model: ModelType=None, **kwargs):
        #print('\n Local User Training \n')
        pass

    def on_local_user_train_end(self, *, logger_kwargs: Dict=None, central_context=None, local_user_logging_enabled: bool=False, **kwargs):
        #print(f'{logger_kwargs["central_iteration"]} Local User Training END')
        assert logger_kwargs is not None
        assert logger_kwargs.get('central_iteration', None) is not None
        assert logger_kwargs.get('metrics', None) is not None
        assert logger_kwargs.get('user_id', None) is not None
        assert logger_kwargs.get('cohort_size', None) is not None
        key_map = {
            #
            'Train population | loss before local training': 'local/user_/train loss before local training',
            'Train population | accuracy before local training': 'local/user_/train accuracy before local training',
            'Train population | f0.5 before local training': 'local/user_/train f0.5 before local training',
            'Train population | f1 before local training': 'local/user_/train f1 before local training',
            'Train population | precision before local training': 'local/user_/train precision before local training',
            'Train population | recall before local training': 'local/user_/train recall before local training',
            #
            'Train population | loss after local training': 'local/user_/train loss after local training',
            'Train population | accuracy after local training': 'local/user_/train accuracy after local training',
            'Train population | f0.5 after local training': 'local/user_/train f0.5 after local training',
            'Train population | f1 after local training': 'local/user_/train f1 after local training',
            'Train population | precision after local training': 'local/user_/train precision after local training',
            'Train population | recall after local training': 'local/user_/train recall after local training',
            #
            'Val population | loss before local training': 'local/user_/val loss before local training',
            'Val population | accuracy before local training': 'local/user_/val accuracy before local training',
            'Val population | f0.5 before local training': 'local/user_/val f0.5 before local training',
            'Val population | f1 before local training': 'local/user_/val f1 before local training',
            'Val population | precision before local training': 'local/user_/val precision before local training',
            'Val population | recall before local training': 'local/user_/val recall before local training',
        }
        if central_context is None:
            local_logging = True
        else:
            local_logging = (central_context.do_evaluation and central_context.population == Population.VAL) or central_context.population == Population.TRAIN
        
        if local_logging:
            cm_matrix = {
                'tn': [],
                'fp': [],
                'fn': [],
                'tp': [],

            }
            #print('\n\n\n Hello World \n\n\n')
            central_iteration = central_context.current_central_iteration #logger_kwargs['central_iteration']
            mode = central_context.population.value
            #if mode == 'val': 
            #    print(logger_kwargs['metrics'])
            #print('\n', logger_kwargs['metrics'].to_simple_dict())
            #print('\nLogger', logger_kwargs['metrics'])
            for metric in logger_kwargs['metrics'].to_simple_dict().items():
                metric_key, metric_value = metric
                #print(metric_key in list(key_map.keys()), metric_key, list(key_map.keys()))
                if metric_key in list(key_map.keys()):
                    # metric_key --> {
                    # Train population | loss before local training, Val population | loss before local training 
                    # }
                    
                    key_parts = key_map[metric_key].split('/')
                    key_parts[1] = f'{key_parts[1]}{logger_kwargs["user_id"]}'
                    key = "/".join(key_parts)
                    #print(key, metric_value)
                    # only log in rank 0
                    if get_ops().distributed.global_rank == 0 and local_user_logging_enabled: 
                        self.wandb.log({key: metric_value}, step=logger_kwargs['central_iteration'])
                
                elif metric_key.lower() in ['tn', 'fp', 'fn', 'tp']:
                    
                    cm_matrix[metric_key.lower()] = metric_value
                    cm_log_title = f'local/user_{logger_kwargs["user_id"]}/{mode} confusion matrix'
                
                elif metric_key in ['Train population | tn before local training',
                                    'Train population | fp before local training',
                                    'Train population | fn before local training',
                                    'Train population | tp before local training',
                                    ]:
                    mapk = {
                        'Train population | tn before local training': 'tn',
                        'Train population | fp before local training': 'fp',
                        'Train population | fn before local training': 'fn',
                        'Train population | tp before local training': 'tp',
                    }
                    cm_matrix[mapk[metric_key]] = metric_value
                    cm_log_title = f'local/user_{logger_kwargs["user_id"]}/train confusion matrix before local training'
                
                elif metric_key in ['Train population | tn after local training',
                                    'Train population | fp after local training',
                                    'Train population | fn after local training',
                                    'Train population | tp after local training',
                                    ]:
                    mapk = {
                        'Train population | tn after local training': 'tn',
                        'Train population | fp after local training': 'fp',
                        'Train population | fn after local training': 'fn',
                        'Train population | tp after local training': 'tp',
                    }
                    cm_matrix[mapk[metric_key]] = metric_value
                    cm_log_title = f'local/user_{logger_kwargs["user_id"]}/train confusion matrix after local training'
                
                elif metric_key in ['Val population | tn before local training',
                                    'Val population | fp before local training',
                                    'Val population | fn before local training',
                                    'Val population | tp before local training',
                                    ]:
                    mapk = {
                        'Val population | tn before local training': 'tn',
                        'Val population | fp before local training': 'fp',
                        'Val population | fn before local training': 'fn',
                        'Val population | tp before local training': 'tp',
                    }
                    cm_matrix[mapk[metric_key]] = metric_value
                    cm_log_title = f'local/user_{logger_kwargs["user_id"]}/val confusion matrix before local training'
                
                elif isinstance(metric, PrivacyMetricName):
                    
                    local_privacy_metrics_map = {
                        "Local DP | l2 norm bound": "local_dp/l2 norm bound",
                        "Local DP | fraction of clipped norms": "local_dp/fraction of clipped norms",
                        "Local DP | norm before clipping": "local_dp/norm before clipping",
                        "Local DP | DP noise std. dev.": "local_dp/DP noise std. dev.",
                        "Local DP | signal-to-DP-noise ratio": "local_dp/signal-to-DP-noise ratio"
                    }
                    if get_ops().distributed.global_rank == 0:
                        self.wandb.log({local_privacy_metrics_map[metric_key]: cm_fig}, step=logger_kwargs['central_iteration'])

                
            try:
                cm_matrix = np.array(list(cm_matrix.values())).reshape(2, 2)
                self._cm_matrix[mode] += cm_matrix
                cm_fig = self.__create_cm_figure(cm_matrix=cm_matrix, display_labels=[0,1], plotly=True)
                #self.wandb.log({cm_log_title: wandb.Image(cm_fig)}, step=logger_kwargs['central_iteration'])
                # only log in rank 0
                if get_ops().distributed.global_rank == 0 and local_user_logging_enabled: 
                    self.wandb.log({cm_log_title: cm_fig}, step=logger_kwargs['central_iteration'])
            except Exception as e:
                print(f'[Warning] -- {cm_matrix}: Issue when logging Confusion Matrix, logging skipped!')

            #print(get_ops().distributed.global_rank == 0, logger_kwargs['metrics'].to_simple_dict())
            if bool(list(logger_kwargs['metrics'].to_simple_dict().items())) == True:
                #mode = key.split('/')[-1].split(' ')[0]
                #print(mode)
                if self._client_participation_table[mode].get(central_iteration, None) is None:
                    self._client_participation_table[mode][central_iteration] = []
                    #if self._user_id_map is not None:
                    #    self._client_participation_bar_plot[mode] = {}
                    #    for k, v in self._user_id_map.items():
                    #        self._client_participation_bar_plot[mode][k] = []

                
                #if len(self._client_participation_table[central_iteration]) == cohort_size:
                #    self._client_participation_table[central_iteration] = []
                
                self._client_participation_table[mode][central_iteration].append(logger_kwargs["user_id"])
                #if self._user_id_map is not None: 
                #    if logger_kwargs["user_id"] in list(self._user_id_map.keys()):
                #        uid = logger_kwargs["user_id"]
                #        print(uid)
                #        self._client_participation_bar_plot[mode][uid].append(self._user_id_map[uid]['data_size'])

    def on_local_train_end(self, *, central_context=None, logger_kwargs: Dict=None, **kwargs):
        assert central_context is not None
        local_logging = (central_context.do_evaluation and central_context.population == Population.VAL) or central_context.population == Population.TRAIN
        step = central_context.current_central_iteration
        mode = central_context.population.value
        
        # log privacy Params
        if self._config.privacy.local.mechanism is not None or self._config.privacy.central.mechanism is not None:
            assert logger_kwargs.get('metrics', None) is not None
            local_privacy_metrics_map = {
                "Local DP | l2 norm bound": "local_dp/l2 norm bound",
                "Local DP | fraction of clipped norms": "local_dp/fraction of clipped norms",
                "Local DP | norm before clipping": "local_dp/norm before clipping",
                "Local DP | DP noise std. dev.": "local_dp/DP noise std. dev.",
                "Local DP | signal-to-DP-noise ratio": "local_dp/signal-to-DP-noise ratio",
                "Central DP | l2 norm bound": "central_dp/l2 norm bound",
                "Central DP | fraction of clipped norms": "central_dp/fraction of clipped norms",
                "Central DP | norm before clipping": "central_dp/norm before clipping",
                #"Train population | total weight": "central_dp/"
                "Central DP | DP noise std. dev. on summed stats": "central_dp/DP noise std. dev. on summed stats",
                "Central DP | signal-to-DP-noise ratio on summed stats": "central_dp/signal-to-DP-noise ratio on summed stats",
            }
            for metric_key, metric_value in logger_kwargs.get('metrics', None).to_simple_dict().items():
                if metric_key in list(local_privacy_metrics_map.keys()):
                    if get_ops().distributed.global_rank == 0:
                        self.wandb.log({local_privacy_metrics_map[metric_key]: metric_value}, step=step)

        #print(mode, logger_kwargs['metrics'])

        import horovod.torch as hvd
        
        if local_logging:
            mode_table = self._client_participation_table[mode]
            #print(mode, mode_table)
            if bool(mode_table) == True and step in list(mode_table.keys()):
                #print(mode, mode_table, step)
                # participation table
                #print(hvd.local_rank(), mode_table[step])
                if hvd.is_initialized() and hvd.size() > 1:
                    import torch 
                    # if multgpu gather across all devices
                    mode_table_step = hvd.allgather(torch.tensor(mode_table[step])).tolist()
                    self._client_participation_table[mode][step] = mode_table_step
                    mode_table = self._client_participation_table[mode]
                    #print(mode_table)
                else:
                    mode_table_step = mode_table[step]
                
                
                
                columns = [f'client_{i}' for i in range(len(mode_table_step))]
                if self._user_id_map is not None:
                    data = [
                        [self._user_id_map[uid]['user_id'] for uid in v] 
                        for _, v in mode_table.items()
                    ]
                else:
                    data = [v for _, v in mode_table.items()]
                
                if get_ops().distributed.global_rank == 0: 
                    table = self.wandb.Table(columns=columns, data=data)
                    self.wandb.log({f'{mode}_population/Client Participation': table}, step=step)
                # cm 
                if hvd.is_initialized() and hvd.size() > 1:
                    import torch 
                    self._cm_matrix[mode] = hvd.allgather(torch.from_numpy(self._cm_matrix[mode])).numpy()
                    out = np.zeros((2, 2))
                    mrange = list(range(0, (hvd.size()+1)*2, 2)) # for size = 2 -> range() -> [0, 2, 4], for 3: [0,2,4,6]
                    for i in range(len(mrange)-1): 
                        out += self._cm_matrix[mode][mrange[i]:mrange[i+1]]
                    self._cm_matrix[mode] = out

                #self.wandb.log({f'{mode}_population/Confusion Matrix': wandb.Image(cm_fig)}, step=step)
                if get_ops().distributed.global_rank == 0: 
                    cm_fig = self.__create_cm_figure(plotly=True, cm_matrix=self._cm_matrix[mode], display_labels=[0,1])
                    self.wandb.log({f'{mode}_population/Confusion Matrix': cm_fig}, step=step)
                self._cm_matrix[mode] = np.zeros_like(self._cm_matrix[mode])
                
                # participation bar chart
                if self._client_participation_bar_plot is not None:
                    mode_bar_plt_data = self._client_participation_bar_plot[mode]
                    participating_cohort = mode_table[step]
                    for client_id_int in list(mode_bar_plt_data.keys()):
                        if client_id_int in participating_cohort:
                            mode_bar_plt_data[client_id_int]['participation'].append(1)
                            mode_bar_plt_data[client_id_int]['participation_size'].append(
                                self._user_id_map[client_id_int]['data_size']
                            )
                        else:
                            mode_bar_plt_data[client_id_int]['participation'].append(0)
                            mode_bar_plt_data[client_id_int]['participation_size'].append(0)
                    #print(f'Step: {step} | Mode: {mode} | {mode_bar_plt_data[0]["participation_size"]} | {len(mode_bar_plt_data[0]["participation_size"])}')
                    bar_fig = plot_client_participation_per_communcation_round(
                        participation=mode_bar_plt_data,
                        cmap_name='viridis',
                        fig_size=(12,8),
                        use_plotply=True,
                    )
                    if get_ops().distributed.global_rank == 0: self.wandb.log({f'{mode}_population/Client Participation (Bar Plot)': bar_fig}, step=step)
                    #self.wandb.log({f'{mode}_population/Client Participation (Bar Plot - Image)': wandb.Image(bar_fig)}, step=step)
            #print(f'{step} Local Training END \n')

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        """
        Submits metrics of this central iteration to Wandb experiment.
        """
        
        cm_matrix = {
            'tn': 0, 
            'fp': 0, 
            'fn': 0, 
            'tp': 0
        }
        cm_log_title = None
        #print('\n\naggregate_metrics Logging')
        #for metric, value in aggregate_metrics.to_simple_dict().items():
        #    print(f'{metric}: {value}')
        if get_ops().distributed.global_rank == 0:
            # Wandb package already uses a multithreaded solution
            # to submit log requests to server, such that this
            # call will not be blocking until server responds.
            metrics = {}
            for k, v in aggregate_metrics.to_simple_dict().items():
                if k in list(self.__map_metric_keys_after_central_iteration.keys()):
                    key = self.__map_metric_keys_after_central_iteration[k]
                    metrics[key] = v
                    self.wandb.log(metrics, step=central_iteration)
            
            for k, v in aggregate_metrics.to_simple_dict().items():    
                if k in ['Central val | tn', 'Central val | fp', 'Central val | fn', 'Central val | tp',
                         'central/tn (val)', 'central/fp (val)', 'central/ fn (val)', 'central/tp (val)']:
                    cm_matrix[k.split(' | ')[1]] = v
                    cm_log_title = f'central/confusion matrix (val)' 
            
            if cm_log_title is not None:
                cm_matrix = np.array(list(cm_matrix.values())).reshape(2, 2)
                cm_fig = self.__create_cm_figure(plotly=True, cm_matrix=cm_matrix, display_labels=[0,1])
                #self.wandb.log({cm_log_title: wandb.Image(cm_fig)}, step=central_iteration)
                #print(type(cm_fig))
                self.wandb.log({cm_log_title: cm_fig}, step=central_iteration)
            
        return False, Metrics()

    def on_train_end(self, *, model: ModelType) -> None:
        if get_ops().distributed.global_rank == 0:

            self.wandb.finish()
    

    def __create_cm_figure(self, 
        cm_matrix: np.ndarray, 
        display_labels: List=[0,1], 
        cmap: str='viridis',
        plotly: bool=False):
        if plotly:
            cm_matrix = cm_matrix.tolist()
            fig = px.imshow(cm_matrix,
                            labels=dict(x="Predicted Label", y="True Label", color="Count"),
                            x=display_labels,
                            y=display_labels,
                            text_auto=True, 
                            aspect="auto",
                            )
            return fig
        else:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_matrix, display_labels=display_labels)
            disp.plot(cmap=cmap)

            plt.close(disp.figure_)
            return disp.figure_
            

def plot_client_participation_per_communcation_round(
    participation: Dict[str | int, Dict[str, str | int | List[int]]],
    cmap_name: str = 'cividis',
    fig_size: Tuple[int, int]=(12, 8), 
    use_plotply: bool=True, 
    **plotting_kwargs):
    assert cmap_name in plt.colormaps(), f'{cmap_name} is not a valid colormap. Use any of the following: {plt.colormaps}'
    # 
    #participation = {
    # 0: {
    #        'user_id_str': str,
    #        'data_size': int,
    #        'participation': List[int],
    #        'participation_size': List[int]
    #    },
    # 1: .....
    #}
    # setting up arguments for plotting
    clients = list(participation.keys())
    num_clients = len(clients)
    communication_rounds = len(participation[0]['participation'])
    round_labels = [f'Communcation Round {i}' for i in range(communication_rounds)]
    bottom = np.zeros(communication_rounds)

    # Generate a colormap
    cmap = plt.get_cmap(cmap_name)
    colors = {client: cmap(i / num_clients) for i, client in enumerate(clients)}

    if not use_plotply:
        # Plotting Bar 
        fig, ax = plt.subplots(figsize=fig_size)
        for client, data in participation.items():
            data = participation[client]['participation_size']
            label = participation[client]['user_id_str']
            data_size = participation[client]['data_size']
            bars = ax.bar(round_labels, data, bottom=bottom, 
                        label=label, color=colors[client])
            # set text for each stacked element in bar plot 
            #for bar, value in zip(bars, data):
            #    if value > 0:
            #        ax.text(
            #            bar.get_x() + bar.get_width() / 2,
            #            bar.get_y() + bar.get_height() / 2,
            #            f'{label} ({data_size})',
            #            ha='center',
            #            va='center',
            #            fontsize=8,
            #            color='white'
            #        )
            bottom += np.array(data)
        
        
        xlabel = 'Communication Rounds' if plotting_kwargs.get('xlabel', None) is None else plotting_kwargs.get('xlabel', None)
        ylabel = 'Data Size Contribution' if plotting_kwargs.get('ylabel', None) is None else plotting_kwargs.get('ylabel', None)
        title = 'Client Participation and Contribution in Each Communication Round' if plotting_kwargs.get('title', None) is None else plotting_kwargs.get('title', None)
        xticks_rotation = 45 if plotting_kwargs.get('xticks_rotation', None) is None else plotting_kwargs.get('xticks_rotation', None)
        if isinstance(xticks_rotation, int) == False:
            raise TypeError(f'`xticks_rotation` should be an interger and not {type(xticks_rotation)}')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize='small')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.close()
        return fig
    else:
        # Plotting with Plotly
        fig = go.Figure()
        
        plotly_colors = {client: f'rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3]})' for client, color in colors.items()}

        for client, data in participation.items():
            data = participation[client]['participation_size']
            label = participation[client]['user_id_str']
            data_size = participation[client]['data_size']
            fig.add_trace(go.Bar(
                x=round_labels,
                y=data,
                name=label,
                marker=dict(color=plotly_colors[client]),
                text=[f'{label}({data_size})' if value > 0 else '' for value in data],
                textposition='inside'
            ))
            bottom += np.array(data)

        xlabel = 'Communication Rounds' if plotting_kwargs.get('xlabel', None) is None else plotting_kwargs.get('xlabel', None)
        ylabel = 'Data Size Contribution' if plotting_kwargs.get('ylabel', None) is None else plotting_kwargs.get('ylabel', None)
        title = 'Client Participation and Contribution in Each Communication Round' if plotting_kwargs.get('title', None) is None else plotting_kwargs.get('title', None)
        xticks_rotation = -45 if plotting_kwargs.get('xticks_rotation', None) is None else plotting_kwargs.get('xticks_rotation', None)
        if isinstance(xticks_rotation, int) == False:
            raise TypeError(f'`xticks_rotation` should be an interger and not {type(xticks_rotation)}')
        
        fig.update_layout(
            barmode='stack',
            title=title,
            xaxis=dict(title=xlabel, tickangle=xticks_rotation),
            yaxis=dict(title=ylabel),
            legend=dict(title='Clients', traceorder='normal'),
            width=fig_size[0]*100,  # Set the width of the figure
            height=fig_size[1]*100   # Set the height of the figure
        )

        return fig 
