
__all__ = [
    'DatasetOptions',
    'DatasetPartitionOptions'
]


class DatasetOptions:
    default: str = 'default'


class DatasetPartitionOptions:
    # i.i.d
    iid: str = 'iid'  
    # non i.i.d
    shard: str = 'shard' 
    dirichlet: str = 'dirichlet'


