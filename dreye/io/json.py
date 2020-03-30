"""
JSON
====

Defines functions to load and save json files
"""

import json
import inspect
import importlib
import cloudpickle
import pickle

import numpy as np
import pandas as pd

from dreye.utilities import AbstractSequence
from dreye.constants import UREG
from dreye.err import DreyeSerializerError


ARRAY_PREFIX = '#array#'
RECARRAY_PREFIX = '#recarray#'
DTYPE_PREFIX = '#dtype#'
SERIES_PREFIX = '#series#'
DFRAME_PREFIX = '#frame#'
DR_PREFIX = '#dreye#'
PINT_PREFIX = '#pint#'
FUNC_PREFIX = '#func#'
QUANT_PREFIX = '#quant#'
DICTABLE_PREFIX = '#dictable#'


def read_json(filename):
    """
    Read a JSON file.

    Parameters
    ----------
    filename : str
        location of file.

    Returns
    -------
    data : dict
        Data in JSON file.
    """

    with open(filename, 'r') as f:
        data = json.load(f, object_hook=deserializer)
    return data


def load_json(obj):
    """
    Read JSON string.
    """

    return json.loads(obj, object_hook=deserializer)


def write_json(filename, data):
    """
    Write to a JSON file.

    Parameters
    ----------
    filename : str
        location of file.
    data : dict
        Data to write to JSON file.
    """

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, default=serializer)


def dump_json(obj):
    """
    Return JSON string.
    """

    return json.dumps(obj, indent=4, default=serializer)


def serializer(obj):
    """serializer of numpy, pandas objects and dreye objects etc.
    """

    if isinstance(obj, np.ndarray) and obj.dtype.fields is not None:
        obj = {RECARRAY_PREFIX: str(cloudpickle.dumps(obj))}
    elif isinstance(obj, np.ndarray):
        obj = {ARRAY_PREFIX: obj.tolist()}
    elif isinstance(obj, pd.Series):
        obj = {SERIES_PREFIX: obj.to_dict()}
    elif isinstance(obj, pd.DataFrame):
        obj = {DFRAME_PREFIX: obj.to_dict()}
    elif isinstance(obj, np.dtype):
        obj = {DTYPE_PREFIX: str(obj)}
    elif isinstance(obj, UREG.Unit):
        obj = {PINT_PREFIX: str(obj)}
    elif isinstance(obj, UREG.Quantity):
        mag = obj.magnitude
        if isinstance(mag, np.ndarray):
            mag = mag.tolist()
        obj = {
            QUANT_PREFIX: [mag, str(obj.units)]
        }
    elif isinstance(obj, AbstractSequence):
        cls = obj.__class__
        obj = {
            DR_PREFIX:
            [str(cloudpickle.dumps(cls)), obj.to_dict()]
        }
    elif hasattr(obj, 'to_dict') and hasattr(obj, 'from_dict'):
        cls = obj.__class__
        obj = {
            DICTABLE_PREFIX:
            [str(cloudpickle.dumps(cls)), obj.to_dict()]
        }
    else:
        obj = {
            FUNC_PREFIX: str(cloudpickle.dumps(obj))
        }

    return obj


def deserializer(obj):
    """deserializer of numpy, pandas objects and dreye objects etc.
    """

    if len(obj) > 1:
        return obj

    for key, ele in obj.items():
        if key == RECARRAY_PREFIX:
            return pickle.loads(eval(ele))
        elif key == ARRAY_PREFIX:
            return np.array(ele)
        elif key == DTYPE_PREFIX:
            return np.dtype(ele)
        elif key == PINT_PREFIX:
            return UREG(ele).units
        elif key == FUNC_PREFIX:
            return pickle.loads(eval(ele))
        elif key == SERIES_PREFIX:
            return pd.Series(ele)
        elif key == DFRAME_PREFIX:
            return pd.DataFrame(ele)
        elif key == QUANT_PREFIX:
            return ele[0] * UREG(ele[1]).units
        elif (key == DR_PREFIX) or (key == DICTABLE_PREFIX):
            cls = pickle.loads(eval(ele[0]))
            return cls.from_dict(ele[1])

    return obj
