"""
Utiliy functions for optimization functions
"""

from types import ModuleType
from typing import Callable, Union, Optional, Tuple
import inspect
from functools import wraps
import numpy as np

from dreye.api.utils import ensure_2d_array, ensure_value, transform_values


FAILURE_MESSAGE = "Optimization unsuccessful for {count} samples/batches."


def replace_numpy_with(module: ModuleType, func: Callable) -> Callable:
    """
    Replaces the numpy module in a given function with another module
    that uses similar attributes (e.g. torch or jax).

    Parameters
    ----------
    module : module
        The module to replace numpy with. This module should ideally have similar attributes to numpy.
    func : function
        The function whose numpy module you want to replace.

    Returns
    -------
    function
        A function with the numpy module replaced with the specified module.

    Raises
    ------
    AttributeError
        If the function does not have a '__globals__' attribute or if the numpy function does not exist in the new module.

    Notes
    -----
    This function assumes that numpy was imported into the namespace of the function as `numpy` or `np`.
    If numpy was imported with a different name, this function will not work as expected.
    """

    if isinstance(func, np.ufunc):
        try:
            return getattr(module, func.__name__)
        except AttributeError:
            raise AttributeError("The numpy function does not exist in the new module.")
    if not hasattr(func, "__globals__"):
        raise AttributeError(
            "Function cannot be converted. It lacks a '__globals__' attribute."
        )

    namespace = func.__globals__.copy()
    if "np" in namespace:
        namespace["np"] = module
    if "numpy" in namespace:
        namespace["numpy"] = module

    source = inspect.getsource(func)
    exec(source, namespace)
    return wraps(func)(namespace.get(func.__name__, func))


def get_batch_size(batch_size: Union[None, int, str], total_size: int) -> int:
    """
    Determines the batch size based on the provided input.

    Parameters
    ----------
    batch_size : int or str, optional
        The desired batch size. If `None`, a batch size of 1 is used.
        If a string is provided, it must be either 'full' or 'total' to
        use the total_size as the batch_size. If an integer is provided,
        it's used directly as the batch size.
    total_size : int
        The total size of the data.

    Returns
    -------
    int
        The batch size to be used.

    Raises
    ------
    NameError
        If batch_size is a string but not 'full' or 'total'.

    """
    if batch_size is None:
        return 1
    elif isinstance(batch_size, str):
        allowed = ("full", "total")
        if batch_size in allowed:
            return total_size
        else:
            raise NameError(f"batch_size name `{batch_size}` not one of: {allowed}.")
    else:
        return batch_size


def prepare_parameters_for_linear(
    A: np.ndarray,
    B: np.ndarray,
    lb: Union[float, np.ndarray],
    ub: Union[float, np.ndarray],
    W: np.ndarray,
    K: Optional[np.ndarray],
    baseline: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare parameters for linear least squares problem.

    This function checks and transforms the input parameters to ensure
    they have the correct form and dimensionality.

    Parameters
    ----------
    A : np.ndarray
        The input matrix.
    B : np.ndarray
        The target matrix.
    lb : float or np.ndarray
        The lower bounds for the parameters.
    ub : float or np.ndarray
        The upper bounds for the parameters.
    W : np.ndarray
        The weight matrix.
    K : np.ndarray, optional
        Transformation matrix for `A`, `lb`, `ub`, and `baseline`, by default None.
    baseline : float or np.ndarray
        The baseline to subtract from the target matrix.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The checked and transformed input parameters.
    """

    A, lb, ub, baseline = transform_values(A, lb, ub, K, baseline)

    B = ensure_2d_array(B)

    if isinstance(W, str):
        if W == "inverse":
            W = 1 / B
        else:
            raise ValueError(f"W must be an array or `inverse`, but is `{W}`")

    W = ensure_value(W, 1, A.shape[0])
    if W.ndim == 1:
        W = np.broadcast_to(W[None], (B.shape[0], W.shape[0]))

    assert A.shape[0] == B.shape[-1], "Channel number does not match."

    return A, B, lb, ub, W, baseline
