"""
Utiliy functions for optimization functions
"""

import inspect
from functools import wraps
import numpy as np

from dreye.api.utils import (
    check_2darr, 
    check_value, 
    check_and_return_transformed_values
)


FAILURE_MESSAGE = "Optimization unsuccessful for {count} samples/batches."


def replace_numpy(module, func):
    """
    HACKY.
    Replace numpy module of a function
    with another module that uses the same attributes
    (e.g. torch or jax). 

    This decorator assumes that numpy was imported into
    the namespace as `numpy` or `np`.
    """
    if isinstance(func, np.ufunc):
        try:
            return getattr(module, func.__name__)
        except AttributeError:
            pass
    if not hasattr(func, '__globals__'):
        raise AttributeError("Function cannot be converted to a jax function.")
    namespace = func.__globals__.copy()
    namespace['np'] = namespace['numpy'] = module
    source = inspect.getsource(func)
    exec(source, namespace)
    return wraps(func)(namespace[func.__name__])


def get_batch_size(batch_size, total_size):
    if batch_size is None:
        return 1
    elif isinstance(batch_size, str):
        allowed = ('full', 'total')
        if batch_size in ('full', 'total'):
            return total_size
        else:
            raise NameError(f"batch_size name `{batch_size}` not one of: {allowed}.")
    else:
        return batch_size


def prepare_parameters_for_linear(A, B, lb, ub, W, K, baseline):
    """
    Check types and values
    """
    
    A, lb, ub, baseline = check_and_return_transformed_values(A, lb, ub, K, baseline)
    
    B = check_2darr(B)
    
    if isinstance(W, str):
        if W == 'inverse':
            W = 1/B
        else:
            raise NameError(f"W must be array or `inverse`, but is `{W}`")
    
    W = check_value(W, 1, A.shape[0])
    if W.ndim == 1:
        W = np.broadcast_to(W[None], (B.shape[0], W.shape[0]))

    assert A.shape[0] == B.shape[-1], "Channel number does not match."

    return A, B, lb, ub, W, baseline