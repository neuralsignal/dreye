"""
Utils
"""

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