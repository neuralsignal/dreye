"""
Utils
"""

import warnings
import numpy as np
from scipy.linalg import norm


def l2norm(arr, axis=-1, keepdims=False):
    return norm(arr, ord=2, axis=axis, keepdims=keepdims)


def l1norm(arr, axis=-1, keepdims=False):
    return norm(arr, ord=1, axis=axis, keepdims=keepdims)


def signif(x, p=1):
    """
    Round an array to a significant digit. 

    Parameters
    ----------
    x : numpy.ndarray
        Array to round
    p : int
        Number of digits to round to

    Returns
    -------
    x : numpy.ndarray
        Rounded array
    """
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def round(x, precision=1):
    """
    Round an array to a precision
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
    return_interval : bool

    Returns
    -------
    arange : ndarray
        Array of evenly spaced values.
    interval : numeric
        The new interval.
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