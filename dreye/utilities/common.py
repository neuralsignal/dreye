"""
Common utilities
================
"""

from numbers import Number
from collections.abc import Sequence

import numpy as np
import pandas as pd
import json

from dreye.constants import ureg
from dreye.utilities.abstract import AbstractSequence

around = np.vectorize(np.round)


def is_jsoncompatible(a):
    """check if type is native python type
    """
    try:
        json.dumps(a)
    except TypeError:
        return False

    return True


def is_hashable(obj):
    return hasattr(obj, '__hash__') and hasattr(obj, '__eq__')


def digits_to_decimals(x, digits):
    """
    Return decimal places for number of significant digits given x.
    """

    return -np.floor(np.log10(np.abs(x))) + digits - 1


def round_to_significant(x, digits):
    """
    Round x to significant digits
    """

    if not np.all(np.isfinite(x)):
        raise ValueError('Only float and int types for rounding')

    if is_numeric(x):

        if x == 0:
            return x
        else:
            decimals = int(digits_to_decimals(x, digits))
            return np.round(x, decimals)

    else:

        x = np.nan_to_num(x)
        decimals = digits_to_decimals(x[x != 0], digits).astype(int)
        x[x != 0] = around(x[x != 0], decimals)

        return x


def has_units(value):
    """
    """

    return (
        hasattr(value, 'units')
        and hasattr(value, 'to')
        and hasattr(value, 'magnitude')
    )


def convert_units(value, units, optional=True):
    """
    """

    if has_units(value):
        value = value.to(units)
    elif not optional:
        raise TypeError('value does not have units')

    return value


def _convert_get_val_opt(value, units=None):

    if has_units(value) and units is not None:
        return value.to(units).magnitude
    elif has_units(value):
        return value.magnitude
    else:
        return value


def get_units(value):
    """
    """

    if has_units(value):
        return value.units
    else:
        return ureg(None)


def dissect_units(value):
    """
    """

    if isinstance(value, ureg.Quantity):
        return value.magnitude, value.units
    elif has_units(value):
        return value.magnitude, value.units
    else:
        return value, None


def dissect_units_null(value):
    """dissect units and return none for dimensionless units
    """

    value, units = dissect_units(value)

    if units == ureg(None).units:
        return value, None
    return value, units


def get_values(value):
    return dissect_units(value)[0]


def is_numeric(value):
    """
    """

    value, _ = dissect_units(value)
    return isinstance(value, Number)


def is_integer(value):
    """
    """

    value, _ = dissect_units(value)
    return isinstance(value, int)


def is_string(value):
    """
    """

    return isinstance(value, str)


def is_listlike(value):
    """
    """

    value, _ = dissect_units(value)
    return isinstance(
        value, (Sequence, np.ndarray, AbstractSequence, pd.Index)
    ) and not is_string(value)


def is_arraylike(value):
    """
    """

    value, _ = dissect_units(value)
    return isinstance(
        value,
        (
            pd.Series, pd.DataFrame,
            Sequence, np.ndarray,
            AbstractSequence, pd.Index
        )
    )
