from typing import Optional
from dataclasses import dataclass
from pfl.metrics import MetricName

@dataclass(frozen=True, eq=False)
class TrainMetricNameUserData(MetricName):
    """
    A structured name for metrics which includes the population the metric was
    generated from and training descriptions.

    :param description:
        The metric name represented as a string.
    :param population:
        The population the metric was generated from.
    :param after_training:
        `True` if the metric was generated after local training, `False`
        otherwise.
    :param local_partition:
        (Optional) The name of the local dataset partition this metric is
        related to.
        This is mainly only relevant when you have a train and val set locally
        on devices.
    """
    after_training: bool
    local_partition: Optional[str] = None
    user_id: Optional[int] = None
    user_data_size: Optional[int] = None

    def _get_user_data_info(self):
        id_msg = f'user id={self.user_id}' if self.user_id is not None else ''
        size_msg = f'user data size={self.user_data_size}' if self.user_data_size is not None else ''
        
        if not id_msg and not size_msg:
            return ''
    
        return f' | {id_msg} | {size_msg}'.strip(' | ')

    def __str__(self) -> str:
        partition = ('' if self.local_partition is None else
                     f'{self.local_partition} set | ')
        postfix = ' after local training' if self.after_training else ' before local training'

        return (f'{self._get_population_str()}{partition}'
                f'{self.description}{postfix}'
                f' | {self._get_user_data_info()}')