"""
Utility functions for array and matrix operations.
"""

from numbers import Number

import numpy as np
from scipy.linalg import norm


def ensure_value(value, default, size):
    """
    Ensure a value is not None and return it as an array.

    If value is None, replace it with the default value.

    Parameters
    ----------
    value : int, float, or array-like
        Input value.
    default : int or float
        Default value to use if value is None.
    size : int
        Size of output array.

    Returns
    -------
    numpy.ndarray
        Value as an array.
    """
    if value is None:
        return np.ones(size) * default
    elif isinstance(value, Number):
        return np.ones(size) * value
    else:
        return np.asarray(value)


def ensure_2d_array(array, axis0_size=None, axis1_size=None):
    """
    Ensure input is a 2D array, and validate its shape.

    Parameters
    ----------
    array : array-like
        Input array.
    axis0_size : int, optional
        Expected size of the first dimension of the array.
    axis1_size : int, optional
        Expected size of the second dimension of the array.

    Returns
    -------
    numpy.ndarray
        Input as a 2D array.
    """
    array = np.atleast_2d(np.asarray(array))
    if axis0_size is not None:
        assert axis0_size == array.shape[0], "array size mismatch along first dimension"
    if axis1_size is not None:
        assert (
            axis1_size == array.shape[1]
        ), "array size mismatch along second dimension"
    return array


def apply_linear_transform(A, K, baseline):
    """
    Apply a linear transform `K(Ax+baseline)` to A and baseline.

    Parameters
    ----------
    A : array-like
        Input array to be transformed.
    K : array-like or None
        Transformation matrix or None.
    baseline : array-like
        Baseline to be transformed.

    Returns
    -------
    tuple of numpy.ndarray
        Transformed A and baseline.
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


def propagate_error(Epsilon, K):
    """
    Propagate error Epsilon through a transformation K.

    Parameters
    ----------
    Epsilon : array-like
        Input error to be propagated.
    K : array-like or None
        Transformation matrix or None.

    Returns
    -------
    numpy.ndarray
        Propagated error.
    """
    if K is not None:
        K = np.asarray(K) ** 2
        if K.ndim < 2:
            Epsilon = Epsilon * K[:, None]
        else:
            assert K.ndim == 2, "`K` must be one- or two-dimensional"
            assert K.shape[0] == K.shape[-1], "`K` must be square"
            Epsilon = K @ Epsilon
    return Epsilon


def ensure_bounds(lb, ub, size):
    """
    Ensure lower and upper bounds are finite or infinite arrays of correct size.

    Parameters
    ----------
    lb : array-like
        Lower bounds.
    ub : array-like
        Upper bounds.
    size : int
        Size of output arrays.

    Returns
    -------
    tuple of numpy.ndarray
        Lower and upper bounds as arrays.
    """
    lb = ensure_value(lb, 0, size)
    assert np.all(np.isfinite(lb)) or np.all(
        np.isneginf(lb)
    ), "bounds must all be finite or infinite."
    ub = ensure_value(ub, np.inf, size)
    assert np.all(np.isfinite(ub)) or np.all(
        np.isposinf(ub)
    ), "bounds must all be finite or infinite."
    return lb, ub


def transform_values(A, lb, ub, K, baseline):
    """
    Ensure input values are correctly formatted and apply a linear transform.

    Parameters
    ----------
    A : array-like
        Input array to be transformed.
    lb : array-like
        Lower bounds for A.
    ub : array-like
        Upper bounds for A.
    K : array-like or None
        Transformation matrix or None.
    baseline : array-like
        Baseline for A.

    Returns
    -------
    tuple of numpy.ndarray
        Transformed A, lb, ub, and baseline.
    """
    A = ensure_2d_array(A)
    lb, ub = ensure_bounds(lb, ub, A.shape[-1])
    baseline = ensure_value(baseline, 0, A.shape[0])
    A, baseline = apply_linear_transform(A, K, baseline)
    return A, lb, ub, baseline


def l2norm(arr, axis=-1, keepdims=False):
    """
    Calculate the L2-norm of an array along a given axis.

    Parameters
    ----------
    arr : ndarray
        Array to calculate the L2-norm of.
    axis : int, optional
        Axis to calculate the norm along, default is -1.
    keepdims : bool, optional
        Whether to keep the original number of dimensions, default is False.

    Returns
    -------
    ndarray
        L2-norm of the array along the given axis.
    """
    return norm(arr, ord=2, axis=axis, keepdims=keepdims)


def l1norm(arr, axis=-1, keepdims=False):
    """
    Calculate the L1-norm of an array along a given axis.

    Parameters
    ----------
    arr : ndarray
        Array to calculate the L1-norm of.
    axis : int, optional
        Axis to calculate the norm along, default is -1.
    keepdims : bool, optional
        Whether to keep the original number of dimensions, default is False.

    Returns
    -------
    ndarray
        L1-norm of the array along the given axis.
    """
    return norm(arr, ord=1, axis=axis, keepdims=keepdims)


def integral(arr, domain, axis=-1, keepdims=False):
    """
    Calculate the integral of an array over a given domain along an axis.

    Parameters
    ----------
    arr : ndarray
        The array to integrate.
    domain : float or ndarray
        The domain of integration. If a float, interpreted as the step size.
    axis : int, optional
        The axis of integration, default is -1.
    keepdims : bool, optional
        Whether to keep the original number of dimensions, default is False.

    Returns
    -------
    ndarray
        Integral of
    the array over the given domain.
    """
    kwargs = {"axis": axis}
    if isinstance(domain, Number):
        kwargs["dx"] = domain
    else:
        kwargs["x"] = domain
    integral_result = np.trapz(arr, **kwargs)
    if keepdims:
        return integral_result[_keepdims_slice_helper(arr.shape, axis)]
    else:
        return integral_result


def _keepdims_slice_helper(shape, axis):
    slices = len(shape) * [slice(None, None, None)]
    slices[axis] = None
    return tuple(slices)


def round_to_significant_digits(x, p=1):
    """
    Round an array to a specified number of round to significant digits.

    Parameters
    ----------
    x : array-like
        Array to round.
    p : int
        Number of significant digits to round to.

    Returns
    -------
    array-like
        Rounded array.
    """
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


# alias for backwards compatibility
signif = round_to_significant_digits


def round_to_precision(x, precision=1.0):
    """
    Round an array to a specified precision.

    Parameters
    ----------
    x : array-like
        Values to round.
    precision : float
        Precision to round to. For example, if `precision` is 0.5,
        then values are rounded to the nearest multiple of 0.5.

    Returns
    -------
    array-like
        Values of x rounded to the specified precision.
    """
    x = np.asarray(x)
    return np.round(x / precision, decimals=0) * precision


def arange_with_interval(
    start: float,
    stop: float,
    step: float,
    dtype=None,
    raise_on_step_change=False,
    return_interval=False,
):
    """
    Return evenly spaced values within a given interval.

    The function generates values within the closed interval ``[start, stop]``
    (i.e., the interval including `start` and `stop`). If necessary, the function
    adjusts the step size to ensure that `stop` is included in the output.

    Parameters
    ----------
    start : float
        Start of interval. The interval includes this value.
    stop : float
        End of interval. The interval includes this value.
    step : float
        Desired spacing between values. The actual spacing may be adjusted
        to include `stop` in the output.
    dtype : data-type, optional
        Desired output data-type.
    raise_on_step_change : bool, optional
        If True, raise an exception if the step size is adjusted. Default is False.
    return_interval : bool, optional
        If True, return the used step size. Default is False.

    Returns
    -------
    arr : ndarray
        Array of evenly spaced values.
    interval : float, optional
        The used step size. Returned only if `return_interval` is True.

    Raises
    ------
    ValueError
        If `raise_on_step_change` is True and the step size is adjusted.
    """
    num = int(np.around((stop - start) / step)) + 1
    arr, interval = np.linspace(start, stop, num, dtype=dtype, retstep=True)

    if interval != step and raise_on_step_change:
        raise ValueError(f"Step size was adjusted from {step} to {interval}.")

    if return_interval:
        return arr, interval
    return arr


# alias for backwards compatibility
arange = arange_with_interval


def predict_values(X: np.ndarray, A: np.ndarray, baseline: np.ndarray):
    """
    Return predicted values calculated as a linear combination of `X` and `A` added to the `baseline`.

    Parameters
    ----------
    X : np.ndarray
        Matrix of predictor variables.
    A : np.ndarray
        Weight matrix to apply to `X`.
    baseline : np.ndarray
        Baseline value to add to the result of `X` and `A`.

    Returns
    -------
    np.ndarray
        Predicted values.
    """
    return X @ A.T + baseline
