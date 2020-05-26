"""
Array Utilities
===============

Defines array utilities objects.

References
----------
"""

import warnings

import numpy as np

from dreye.constants import (
    DEFAULT_FLOAT_DTYPE, RELATIVE_ACCURACY, ABSOLUTE_ACCURACY
)
from dreye.utilities.common import get_value


around = np.vectorize(np.round)
# vecotrized round function


def asarray(x, *args, **kwargs):
    """
    Always return array, but avoid unit stripping warning.
    """
    x = get_value(x)
    return np.asarray(x, *args, **kwargs)


def digits_to_decimals(x, digits):
    """
    Return decimal places for number of significant digits given x.
    """
    return -np.floor(np.log10(np.abs(x))) + digits - 1


def round_to_significant(x, digits):
    """
    Round x to significant digits
    """

    # strip units and all and return array
    x = asarray(x)

    if not np.all(np.isfinite(x)):
        raise ValueError('Cannot round to significant with non-finite value')

    if x.ndim:
        decimals = digits_to_decimals(x[x != 0], digits).astype(int)
        x[x != 0] = around(x[x != 0], decimals)
        return x
    elif x:
        decimals = int(digits_to_decimals(x, digits))
        return np.round(x, decimals)
    else:
        return x


def array_equal(x, y):
    """Determine if two arrays are equal
    """

    # asarrays
    x = asarray(x)
    y = asarray(y)

    if x.shape != y.shape:
        return False
    else:
        x[x == 0] = ABSOLUTE_ACCURACY
        y[y == 0] = ABSOLUTE_ACCURACY
        xtol = np.nanmax(digits_to_decimals(x, RELATIVE_ACCURACY))
        ytol = np.nanmax(digits_to_decimals(y, RELATIVE_ACCURACY))
        rtol = 10**-np.max([xtol, ytol])
        return np.allclose(
            x, y, rtol=rtol, atol=ABSOLUTE_ACCURACY,
            equal_nan=True
        )


def unique_significant(x, digits=RELATIVE_ACCURACY, **kwargs):
    """
    Return unique values of array within significant digit range
    """
    return np.unique(round_to_significant(x, digits), **kwargs)


def spacing(array, unique=True, sorted=False, axis=None):
    """
    Returns the spacing of given array.

    Parameters
    ----------
    array : array-like
        array to retrieve the spacing.
    unique : bool, optional
        Whether to return unique spacings if  the array is
        non-uniformly spaced or the complete spacings. The default is True.
    sorted : bool, optional
        Whether the array is already sorted. The default is False.
    axis : int or None, optional
        Whether to calculate the spacing sizes along an axis. If None,
        the array is flattened. The default is None.

    Returns
    -------
    differences : ndarray
        array spacing.

    Examples
    --------
    Uniformly spaced variable:

    >>> y = asarray([1, 2, 3, 4, 5])
    >>> spacing(y)
    array([ 1.])
    >>> spacing(y, False)
    array([ 1.,  1.,  1.,  1.])

    Non-uniformly spaced variable:

    >>> y = asarray([1, 2, 3, 4, 8])
    >>> spacing(y)
    array([ 1.,  4.])
    >>> spacing(y, False)
    array([ 1.,  1.,  1.,  4.])
    """

    array = asarray(array, DEFAULT_FLOAT_DTYPE)

    if axis is None:
        # set axis -1 for the rest
        axis = -1
        # flatten array
        array = np.ravel(array)

    # sort array if not sorted along given axis
    if not sorted:
        array = np.sort(array, axis=axis)

    differences = np.diff(array, axis=axis)

    if unique:
        return unique_significant(differences, axis=axis)
    else:
        return differences


def is_uniform(array, sorted=False, axis=None, is_array_diff=False):
    """
    Returns if given array is uniform.

    Parameters
    ----------
    array : array-like
        array to check for uniformity.
    sorted : bool, optional
        Whether the array is already sorted. The default is False.
    axis : int or None, optional
        Whether to assess uniformity along axis. If None,
        the array is flattened. The default is None.
    is_array_diff : bool, optional
        If the array is already the difference array. The default is False.

    Returns
    -------
    is_uniform : bool
        Is array uniform.

    Examples
    --------
    Uniformly spaced variable:

    >>> a = asarray([1, 2, 3, 4, 5])
    >>> is_uniform(a)
    True

    Non-uniformly spaced variable:

    >>> a = asarray([1, 2, 3.1415, 4, 5])
    >>> is_uniform(a)
    False
    """

    if is_array_diff:
        differences = unique_significant(array, axis=axis)
    else:
        differences = spacing(array, sorted=sorted, axis=axis)

    if axis is None:
        return True if differences.size == 1 else False
    else:
        return True if differences.shape[axis] == 1 else False


def array_domain(array, sorted=False, uniform=None, axis=None):
    """
    Returns the domain of an array with uniform spacing spacing.

    Parameters
    ----------
    array : ndarray
        array to retrieve the domain from.
    sorted : bool, optional
        Whether the array is sorted. The default is False.
    uniform : bool or None, optional
        Whether the array has uniform spacings. The default is None.
    axis : int or None, optional
        Whether to find the domain along a given axis. If None, the
        array is flattened. The default is None.

    Returns
    -------
    start : numeric or array-like
    end : numeric or array-like
    spacing : numeric or array-like

    Examples
    --------
    """

    if axis is None:
        # ravel array
        axis = -1
        array = np.ravel(array)

    if not sorted:
        array = np.sort(array, axis=axis)

    differences = spacing(array, sorted=True, axis=axis, unique=False)

    if uniform is None:
        uniform = is_uniform(differences, axis=axis, is_array_diff=True)

    if uniform:
        return (np.rollaxis(array, axis)[0], np.rollaxis(array, axis)[-1],
                np.rollaxis(differences, axis)[0])
    else:
        return (
            np.rollaxis(array, axis)[0],
            np.rollaxis(array, axis)[-1],
            differences  # TODO: test this behavior
        )


def arange(start, stop=None, step=1, dtype=None, error='ignore'):
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

    range_, interval = np.linspace(start, stop, num, dtype=dtype, retstep=True)

    if interval != step:

        msg = "Interval had to change to {0} from {1}.".format(interval, step)

        if error == 'ignore':
            pass

        elif error == 'warn':
            warnings.warn(msg, RuntimeWarning)

        elif error == 'raise':
            raise ValueError(msg)

    return range_, interval
