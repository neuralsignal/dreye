"""
Utils
"""

import warnings
from numbers import Number

import numpy as np
from scipy.linalg import norm


def check_value(value, default, size):
    """[summary]

    Parameters
    ----------
    value : [type]
        [description]
    default : [type]
        [description]
    size : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    if value is None:
        return np.ones(size) * default
    elif isinstance(value, Number):
        return np.ones(size) * value
    else:
        return np.asarray(value)


def check_2darr(arr, axis0_size=None, axis1_size=None):
    """[summary]

    Parameters
    ----------
    arr : [type]
        [description]
    axis0_size : [type], optional
        [description], by default None
    axis1_size : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    arr = np.atleast_2d(np.asarray(arr))
    if axis0_size is not None:
        assert axis0_size == arr.shape[0], "array size mismatch along first dimension"
    if axis1_size is not None:
        assert axis1_size == arr.shape[1], "array size mismat along second dimension"
    return arr


def linear_transform(A, K, baseline):
    """
    Helper function to reformulate `K(Ax+baseline)`

    Parameters
    ----------
    A : [type]
        [description]
    K : [type]
        [description]
    baseline : [type]
        [description]

    Returns
    -------
    [type]
        [description]
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


def error_propagation(Epsilon, K):
    """[summary]

    Parameters
    ----------
    Epsilon : [type]
        [description]
    K : [type]
        [description]

    Returns
    -------
    [type]
        [description]
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


def check_bounds(lb, ub, size):
    # default lower bound value is zero (non-negativity)
    lb = check_value(lb, 0, size)
    assert np.all(np.isfinite(lb)) or np.all(np.isneginf(lb)), "bounds must all be finite or infinite."
    # default upper bound value is infinity
    ub = check_value(ub, np.inf, size)
    assert np.all(np.isfinite(ub)) or np.all(np.isposinf(ub)), "bounds must all be finite or infinite."
    return lb, ub
    
def check_and_return_transformed_values(A, lb, ub, K, baseline):
    """[summary]

    Parameters
    ----------
    A : [type]
        [description]
    lb : [type]
        [description]
    ub : [type]
        [description]
    K : [type]
        [description]
    baseline : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # TODO K as string `norm`
    A = check_2darr(A)

    lb, ub = check_bounds(lb, ub, A.shape[-1])
    baseline = check_value(baseline, 0, A.shape[0])

    A, baseline = linear_transform(A, K, baseline)

    return A, lb, ub, baseline


def l2norm(arr, axis=-1, keepdims=False):
    """Calculate the L2-norm along a given axis

    Parameters
    ----------
    arr : ndarray
        Array used to calculate the L2-norm.
    axis : int, optional
        Axis to normalize along, by default -1
    keepdims : bool, optional
        Whether to keep the dimensionality of the array, by default False.

    Returns
    -------
    l2norm : ndarray
        L2-norm or array along axis.
    """
    return norm(arr, ord=2, axis=axis, keepdims=keepdims)


def l1norm(arr, axis=-1, keepdims=False):
    """Calculate the L1-norm along a given axis

    Parameters
    ----------
    arr : ndarray
        Array used to calculate the L1-norm.
    axis : int, optional
        Axis to normalize along, by default -1
    keepdims : bool, optional
        Whether to keep the dimensionality of the array, by default False.

    Returns
    -------
    l1norm : ndarray
        L1-norm or array along axis.
    """
    return norm(arr, ord=1, axis=axis, keepdims=keepdims)


def integral(arr, domain, axis=-1, keepdims=False):
    """Calculate the integral of an array given its domain along an axis

    Parameters
    ----------
    arr : ndarray of shape (..., n_domain, ...)
        The array to use to obtain the integral.
    domain : float or ndarray of shape (n_domain)
        The domain for `arr` along the given axis. 
        If a float, it is assumed to be the step size of the domain.
    axis : int, optional
        The axis of the domain dimension, by default -1
    keepdims : bool, optional
        Whether to keep the dimensionality or not, by default False

    Returns
    -------
    integral : ndarray
        Integral of the array along given axis.
    """
    kwargs = {'axis': axis}
    if isinstance(domain, Number):
        kwargs['dx'] = domain
    else:
        kwargs['x'] = domain
    trapz = np.trapz(arr, **kwargs)
    if keepdims:
        return trapz[_keepdims_slice_helper(arr.shape, axis)]
    else:
        return trapz
    
    
def _keepdims_slice_helper(shape, axis):
    slices = len(shape) * [slice(None, None, None)]
    slices[axis] = None
    return tuple(slices)


def signif(x, p=1):
    """
    Round an array to a significant digit. 

    Parameters
    ----------
    x : array-like
        Array to round.
    p : int
        Number of digits to round to.

    Returns
    -------
    x : array-like
        Rounded array.
    """
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def round(x, precision=1.0):
    """
    Round an array to a particular precision. 
    
    Parameters
    ----------
    x : array-like
        Values to round to a particular precision
    precision : float
        The precision to round to. For example, if `precision` is 0.5, 
        then values are round to the nearest multiple of 0.5
        
    Returns
    -------
    x : array-like
        Rounded values of x.
    """
    x = np.asarray(x)
    return np.round(x/precision, decimals=0) * precision


def arange(start, stop=None, step=1, dtype=None, error='ignore', return_interval=False):
    """
    Return evenly spaced values within a given interval.

    Values are generated within the closed interval ``[start, stop]``
    (in other words, the interval including `start` and `stop`).

    Parameters
    ----------
    start : number, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : numeric
        End of interval.  The interval includes this value.
    step : numeric, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    error : str
        Whether to raise an error or print a warning or ignore the case when 
        the found interval does not match `step`. 
    return_interval : bool
        Whether to return the new interval/stepsize or not.

    Returns
    -------
    arange : ndarray
        Array of evenly spaced values.
    interval : numeric
        The new interval, if `return_interval` is True.
    """

    if stop is None:
        stop = start
        start = 0

    num = int(np.around((step + stop - start) / step, 0))
    
    if not num:
        raise ValueError("Cannot form ranged array with "
                         f"start {start} and stop {stop} and step {step}.")

    arr, interval = np.linspace(start, stop, num, dtype=dtype, retstep=True)

    if interval != step:

        msg = "Interval had to change to {0} from {1}.".format(interval, step)

        if error == 'ignore':
            pass

        elif error == 'warn':
            warnings.warn(msg, RuntimeWarning)

        elif error == 'raise':
            raise ValueError(msg)

    if return_interval:
        return arr, interval
    
    return arr


def get_prediction(X : np.ndarray, A : np.ndarray, baseline : np.ndarray):
    """Return prediction

    Parameters
    ----------
    X : np.ndarray
        [description]
    A : np.ndarray
        [description]
    baseline : np.ndarray
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return (X @ A.T + baseline)