"""Source-data immutability without changing JSON consumer compatibility."""
from collections.abc import Mapping
import numpy as np


class FrozenDict(dict):
    """Read-only through normal mapping APIs; still serializable by json.dumps."""
    def _readonly(self, *args, **kwargs):
        raise TypeError('source evidence is read-only')

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def frozen_array(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True)
    if array.dtype.hasobject:
        raise ValueError('source arrays must not contain Python objects')
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def freeze(value):
    if isinstance(value, Mapping):
        return FrozenDict({key: freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(child) for child in value)
    if isinstance(value, np.ndarray):
        return frozen_array(value)
    return value


def thaw(value):
    if isinstance(value, Mapping):
        return {key: thaw(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
