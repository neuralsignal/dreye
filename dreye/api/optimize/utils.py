"""
Utiliy functions for optimization functions
"""

import inspect
from functools import wraps
from numbers import Number
import numpy as np


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


def check_value(value, default, size):
    if value is None:
        return np.ones(size) * default
    elif isinstance(value, Number):
        return np.ones(size) * value
    else:
        return np.asarray(value)


def check_2darr(arr, axis0_size=None, axis1_size=None):
    arr = np.atleast_2d(np.asarray(arr))
    if axis0_size is not None:
        assert axis0_size == arr.shape[0], "array size mismatch along first dimension"
    if axis1_size is not None:
        assert axis1_size == arr.shape[1], "array size mismat along second dimension"
    return arr


def check_2darrs(*arrs):
    return (np.atleast_2d(np.asarray(arr)) for arr in arrs)



def error_propagation(Epsilon, K):
    if K is not None:
        K = np.asarray(K) ** 2
        if K.ndim < 2:
            Epsilon = Epsilon * K[:, None]
        else:
            assert K.ndim == 2, "`K` must be one- or two-dimensional"
            assert K.shape[0] == K.shape[-1], "`K` must be square"
            Epsilon = K @ Epsilon
    return Epsilon


def linear_transform(A, K, baseline):
    """
    Helper function to reformulate `K(Ab+baseline)`
    """
    if K is not None:
        K = np.asarray(K)
        if K.ndim < 2:
            A = A * K[:, None]
            baseline = K * baseline
        else:
            assert K.ndim == 2, "`K` must be one- or two-dimensional"
            assert K.shape[0] == K.shape[-1], "`K` must be square"
            A = K @ A
            baseline = K @ baseline
    return A, baseline


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
    A, B = check_2darrs(A, B)

    lb = check_value(lb, 0, A.shape[-1])
    assert np.all(np.isfinite(lb)) or np.all(np.isneginf(lb)), "bounds must all be finite or inifinite."
    ub = check_value(ub, np.inf, A.shape[-1])
    assert np.all(np.isfinite(ub)) or np.all(np.isposinf(ub)), "bounds must all be finite or inifinite."
    W = check_value(W, 1, A.shape[0])
    if W.ndim == 1:
        W = np.broadcast_to(W[None], (B.shape[0], W.shape[0]))
    baseline = check_value(baseline, 0, A.shape[0])

    A, baseline = linear_transform(A, K, baseline)

    assert A.shape[0] == B.shape[-1], "Channel number does not match."

    return A, B, lb, ub, W, baseline