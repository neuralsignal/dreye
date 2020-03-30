"""
Array Utilities
===============

Defines array utilities objects.

References
----------
"""

import warnings

import numpy as np

from dreye.constants import DEFAULT_FLOAT_DTYPE, \
    RELATIVE_ACCURACY, ABSOLUTE_ACCURACY
from dreye.utilities.common import (
    round_to_significant, digits_to_decimals, dissect_units
)


def asarray(x, *args, **kwargs):
    """always return array, but dissect units before if necessary
    """
    x, _ = dissect_units(x)
    return np.asarray(x, *args, **kwargs)


def array_equal(x, y):
    """Determine if two arrays are equal
    """

    # TODO decide if to use allclose with global tolerance defined
    if x.shape != y.shape:
        return False
    else:
        xtol = np.min(digits_to_decimals(x, RELATIVE_ACCURACY))
        ytol = np.min(digits_to_decimals(y, RELATIVE_ACCURACY))
        rtol = 10**np.min([xtol, ytol])
        return np.allclose(x, y, rtol=rtol, atol=ABSOLUTE_ACCURACY)
    # return np.array_equal(x, y)


def unique_significant(x, digits=RELATIVE_ACCURACY, **kwargs):
    """
    Return unique values of array within significant digit range
    """

    return np.unique(round_to_significant(x, digits), **kwargs)


def closest_indexes(a, b):
    """
    Returns the :math:`a` variable closest element indexes to reference
    :math:`b` variable elements.
    Parameters
    ----------
    a : array_like
        Variable to search for the closest element indexes.
    b : numeric
        Reference variable.
    Returns
    -------
    numeric
        Closest :math:`a` variable element indexes.
    Examples
    --------
    >>> a = asarray([24.31357115, 63.62396289, 55.71528816,
    ...               62.70988028, 46.84480573, 25.40026416])
    >>> print(closest_indexes(a, 63))
    [3]
    >>> print(closest_indexes(a, [63, 25]))
    [3 5]
    """

    a = np.ravel(a)[:, np.newaxis]
    b = np.ravel(b)[np.newaxis, :]

    return np.abs(a - b).argmin(axis=0)


def diag_chunks(a, ravel=True):
    """
    Take diagonal chunks out of 2 dimensional array.
    """

    assert a.shape[0] <= a.shape[1]

    length = a.shape[-1]
    idcs = np.arange(length).reshape(a.shape[0], -1)

    # this should have the same shape as shape
    a = np.take_along_axis(a, idcs, axis=1)

    if ravel:
        return np.ravel(a)
    else:
        return a


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


def as_float(a):
    """
    Converts given :math:`a` variable to *numeric* using the type defined by
    :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute. In the event where
    :math:`a` cannot be converted, it is converted to *ndarray* using the type
    defined by :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.
    Parameters
    ----------
    a : object
        Variable to convert.
    Returns
    -------
    ndarray
        :math:`a` variable converted to *numeric*.
    Warnings
    --------
    The behaviour of this definition is different than
    :func:`colour.utilities.as_numeric` definition when it comes to conversion
    failure: the former will forcibly convert :math:`a` variable to *ndarray*
    using the type defined by :attr:`colour.constant.DEFAULT_FLOAT_DTYPE`
    attribute while the later will pass the :math:`a` variable as is.
    Examples
    --------
    >>> as_float(asarray([1]))
    1.0
    >>> as_float(np.arange(10))
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
    """

    try:
        return DEFAULT_FLOAT_DTYPE(a)
    except TypeError:
        return asarray(a, DEFAULT_FLOAT_DTYPE)
