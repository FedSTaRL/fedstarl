from typing import List
from types import NoneType


__all__ = [
    'AggregationModelOptions',
    'ClassificationOptions',
    'LossOptions',
    'MaskingOptions',
    'ModelOptions',
    'OutputHeadOptions',
    'PredictionOptions',
    'TrainingMethodOptions',
]


class OptionsBaseClass:
    @classmethod
    def get_options(cls) -> List[str | NoneType]:
        #return {key: value for key, value in cls.__dict__.items() if not key.startswith('__')}
        return [value for key, value in cls.__dict__.items() if not key.startswith('__') and key != 'get_options']


class AggregationModelOptions(OptionsBaseClass):
    fedavg: str='fedavg'
    fedprox: str='fedprox'
    adafedprox: str='adafedprox'


class ClassificationOptions(OptionsBaseClass):
    cls_token: str='cls_token'
    autoregressive: str='autoregressive'
    elementwise: str='elementwise' # per element in sequence


class LossOptions(OptionsBaseClass):
    bce: str='bce'
    ce: str='ce'
    reconstruction: str = 'reconstruction'


class MaskingOptions(OptionsBaseClass):
    random: str = 'random'
    darem: str = 'darem' # not implemented
    none: NoneType = None


class ModelOptions(OptionsBaseClass):
    rnn: str='rnn'
    lstm: str='lstm'
    gru: str='gru'


class OutputHeadOptions(OptionsBaseClass):
    classification: str='classification'
    regression: str='regression'
    forecasting: str='forecasting'


class PredictionOptions(OptionsBaseClass):
    binary: str='binary'
    multiclass: str='multiclass'


class TrainingMethodOptions(OptionsBaseClass):
    central: str = 'central'
    federated: str = 'federated'