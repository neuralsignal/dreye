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

    if isinstance(obj, np.ndarray):
        obj = ARRAY_PREFIX + json.dumps(obj.tolist())
    elif isinstance(obj, pd.Series):
        obj = (
            SERIES_PREFIX,
            obj.to_dict()
        )
    elif isinstance(obj, pd.DataFrame):
        obj = (
            DFRAME_PREFIX,
            obj.to_dict()
        )
    elif isinstance(obj, np.dtype):
        obj = DTYPE_PREFIX + str(obj)
    elif isinstance(obj, UREG.Unit):
        obj = PINT_PREFIX + str(obj)
    elif isinstance(obj, UREG.Quantity):
        mag = obj.magnitude
        if isinstance(mag, np.ndarray):
            mag = mag.tolist()
        obj = (
            QUANT_PREFIX,
            mag,
            str(obj.units)
        )
    elif isinstance(obj, AbstractSequence):
        # TODO checking of module
        cls = obj.__class__
        obj = (
            DR_PREFIX,
            str(cloudpickle.dumps(cls)),
            obj.to_dict()
        )
    elif hasattr(obj, 'to_dict') and hasattr(obj, 'from_dict'):
        # TODO checking of module
        cls = obj.__class__
        obj = (
            DICTABLE_PREFIX,
            str(cloudpickle.dumps(cls)),
            obj.to_dict()
        )
    else:
        obj = (
            FUNC_PREFIX,
            str(cloudpickle.dumps(obj))
        )

    return obj


def deserializer(obj):
    """deserializer of numpy, pandas objects and dreye objects etc.
    """

    for key, ele in obj.items():
        if isinstance(ele, str):
            if ele.startswith(ARRAY_PREFIX):
                ele = np.array(json.loads(ele[len(ARRAY_PREFIX):]))
                obj[key] = ele
            elif ele.startswith(DTYPE_PREFIX):
                ele = np.dtype(ele[len(DTYPE_PREFIX):])
                obj[key] = ele
            elif ele.startswith(PINT_PREFIX):
                ele = UREG(ele[len(PINT_PREFIX):]).units
                obj[key] = ele

        elif isinstance(ele, (tuple, list)):
            if len(ele) == 2:
                if ele[0] == FUNC_PREFIX:
                    obj[key] = pickle.loads(eval(ele[1]))
                elif ele[0] == SERIES_PREFIX:
                    ele = pd.Series(ele[1])
                    obj[key] = ele
                elif ele[0] == DFRAME_PREFIX:
                    ele = pd.DataFrame(ele[1])
                    obj[key] = ele
            elif len(ele) == 3:
                if ele[0] == QUANT_PREFIX:
                    ele = ele[1] * UREG(ele[2]).units
                    obj[key] = ele
                elif ele[0] == DR_PREFIX:
                    cls = pickle.loads(eval(ele[1]))
                    ele = cls.from_dict(ele[2])
                    obj[key] = ele
                elif ele[0] == DICTABLE_PREFIX:
                    cls = pickle.loads(eval(ele[1]))
                    ele = cls.from_dict(ele[2])
                    obj[key] = ele

    return obj
