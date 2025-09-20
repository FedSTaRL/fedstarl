from dataclasses import dataclass


__all__ = [
    'LocalResultMetaData'
]


@dataclass(frozen=True)
class LocalResultMetaData:
    """
    Data that is typically returned by a model's local optimization procedure,
    e.g. ``PyTorchModel.do_multiple_epochs_of``. Can have useful information
    needed by the algorithm.

    :param num_steps:
        The number of local steps taken during the local optimization procedure.
    """
    num_steps: int
    user_id: int
    data_size: int