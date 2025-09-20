import os
import sys
import hydra
import torch
import logging

from omegaconf import DictConfig, OmegaConf
from uuid import uuid4

from pfl.internal.ops.pytorch_ops import get_default_device, to_tensor
from pfl.hyperparam import NNEvalHyperParams, NNTrainHyperParams
from pfl.model.pytorch import PyTorchModel
from pfl.callback import (
    CentralEvaluationCallback, 
    TrackBestOverallMetrics, 
    ModelCheckpointingCallback,
    StopwatchCallback,
    AggregateMetricsToDisk,
    WandbCallback
)

from functools import partial

# Add the benchmark directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.privacy import parse_mechanism
from utils.datasets import get_datasets
from utils.models import get_model_pytorch
from utils.aggregation_algorithm import get_aggregation_algorithm

from torch.optim.lr_scheduler import LambdaLR

from pfl.aggregate.simulate import SimulatedBackend as PFLSimulatedBackend

try:
    from src.utils import seed_everything
    from src.pfl.aggregate.simulate import SimulatedBackend
except ImportError:
    import sys
    import os.path as osp
    file_path = osp.abspath(__file__) # Get the absolute path of the current file
    dir_path = osp.normpath(osp.join(file_path, "../../..")) # Get the directory path three levels up
    sys.path.append(dir_path) # Add the directory to the system path
    sys.path.append(dir_path)
    from src.utils import seed_everything
    from src.pfl.aggregate.simulate import SimulatedBackend

logger = logging.getLogger(name=__name__)

def get_polynomial_decay_schedule_with_warmup(optimizer,
                                              num_warmup_steps,
                                              num_training_steps,
                                              lr_end=1e-7,
                                              power=1.0,
                                              last_epoch=-1):
    """ polynomial LR decay schedule, implementation followed:
    https://huggingface.co/transformers/v4.6.0/_modules/transformers/optimization.html#get_polynomial_decay_schedule_with_warmup """

    lr_init = optimizer.defaults["lr"]
    assert lr_init >= lr_end, (f"lr_end ({lr_end}) must be be smaller than or "
                               f"equal to initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


@hydra.main(version_base=None, config_path="configs", config_name="cifar10")
def main(config: DictConfig):
    print('Start Training')
    print(f'Configs:\n{OmegaConf.to_yaml(config)}')
    seed_everything(seed=config.seed)

    use_pfl_internals = config.use_pfl_internals

    postprocessors = [] 
    if config.privacy.local.mechanism is not None and config.privacy.local.is_local:
        local_privacy = parse_mechanism(
            mechanism_name=config.privacy.local.mechanism,
            clipping_bound=config.privacy.local.clipping_bound,
            epsilon=config.privacy.local.epsilon,
            delta=config.privacy.local.delta,
            order=config.privacy.local.local_order
        )
        postprocessors.append(local_privacy)

    if config.privacy.central.mechanism is not None and config.privacy.central.is_central:
        central_privacy = parse_mechanism(
            mechanism_name=config.privacy.central.mechanism,
            clipping_bound=config.privacy.central.clipping_bound,
            epsilon=config.privacy.central.epsilon,
            delta=config.privacy.central.delta,
            order=config.privacy.central.order,
            cohort_size=config.privacy.central.cohort_size,
            noise_cohort_size=config.privacy.central.noise_cohort_size,
            num_epochs=config.privacy.central.num_epochs,
            population=config.privacy.central.population,
            min_separation=config.privacy.central.min_separation,
            is_central=config.privacy.central.is_central)
        postprocessors.append(central_privacy)

    numpy_to_tensor = partial(to_tensor, dtype=None)

    (training_federated_dataset, val_federated_dataset, central_data,
     metadata) = get_datasets(config, numpy_to_tensor=numpy_to_tensor)
    
    #num_classes = len(metadata["label_mapping"])
    #print('metadata', metadata["channel_mean"])
    #print('metadata', metadata["channel_stddevs"])
    #print('num_classes', num_classes)
    #arguments.channel_mean = metadata["channel_mean"]
    #arguments.channel_stddevs = metadata["channel_stddevs"]
    #arguments.num_classes = num_classes

    pytorch_model = get_model_pytorch(config=config)
    # Put on GPU if available.
    pytorch_model = pytorch_model.to(get_default_device())

    variables = [p for p in pytorch_model.parameters() if p.requires_grad]
    if config.optimizer.central.name == 'adam':
        central_optimizer = torch.optim.AdamW(
            variables,
            config.training.central.learning_rate,
            eps=0.01,
            betas=(0.9, 0.99),
            weight_decay=config.optimizer.central.weight_decay)
    else:
        central_optimizer = torch.optim.SGD(
            variables,
            config.training.central.learning_rate,
            weight_decay=config.optimizer.central.weight_decay)


    central_lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        central_optimizer,
        num_warmup_steps=30,
        num_training_steps=config.training.central.num_iterations,
        lr_end=0.0002)

    from src.nn import BaseSequencePyTorchModel

    model = BaseSequencePyTorchModel(model=pytorch_model,
                         local_optimizer_create=torch.optim.SGD,
                         central_optimizer=central_optimizer,
                         central_learning_rate_scheduler=central_lr_scheduler,
                         use_pfl_internals=use_pfl_internals)

    simulated_backend = PFLSimulatedBackend if use_pfl_internals else SimulatedBackend
    backend = simulated_backend(training_data=training_federated_dataset,
                               val_data=val_federated_dataset,
                               postprocessors=postprocessors)
    

    algorithm, algorithm_params, algorithm_callbacks = get_aggregation_algorithm(config=config)

    model_train_params = NNTrainHyperParams(
        local_learning_rate=config.training.local.learning_rate,
        local_num_epochs=config.training.local.num_epochs,
        local_batch_size=config.training.local.batch_size,
        local_max_grad_norm=10.0)

    model_eval_params = NNEvalHyperParams(
        local_batch_size=config.training.local.batch_size)

    # Central evaluation on dev data.
    callbacks = [
        CentralEvaluationCallback(central_data,
                                  model_eval_params=model_eval_params,
                                  frequency=config.training.central.evaluation_frequency),
        StopwatchCallback(),
        #AggregateMetricsToDisk('./metrics.csv'),
        #TrackBestOverallMetrics(
        #    higher_is_better_metric_names=['Central val | macro AP']),
    ]

    if config.callbacks.restore_training.model_path is not None:
        model.load(config.callbacks.restore_training.model_path)
        logger.info(f'Restored model from {config.callbacks.restore_training.model_path}')

    callbacks.extend(algorithm_callbacks)

    if config.callbacks.model_ckpt.save_model_path is not None:
        callbacks.append(ModelCheckpointingCallback(config.callbacks.model_ckpt.save_model_path))

    if config.logger.project_id:
        import wandb
        wandb.init(
            entity=None,#config.logger.entity, 
            project=config.logger.project_id,
            name=f"{config.model.name}_{config.aggregation_algorithm.name}_{os.environ.get('WANDB_TASK_ID', str(uuid4()))}"
        )
        wandb.config.update(
            OmegaConf.to_container(config, resolve=True)
        )
        callbacks.append(
            WandbCallback(
                wandb_project_id=config.logger.project_id,
                wandb_experiment_name=os.environ.get('WANDB_TASK_ID',
                                                     str(uuid4())),
                # List of dicts to one dict.
                #wandb_config=dict(vars(config.logger)),
                tags=os.environ.get('WANDB_TAGS', 'empty-tag').split(','),
                group=os.environ.get('WANDB_GROUP', None)))
    
    algo_params = {
        'algorithm_params': algorithm_params,
        'backend': backend,
        'model': model,
        'model_train_params': model_train_params,
        'model_eval_params': model_eval_params,
        'callbacks': callbacks
    }
    if not use_pfl_internals:
        algo_params['compute_metrics'] = True

    model = algorithm.run(**algo_params)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    main()



