"""
Abstract Base Class for both Signal and Domain.
"""

from abc import abstractmethod
import copy
import operator

import numpy as np

from dreye.err import DreyeError
from dreye.utilities import (
    has_units, asarray,
    is_listlike, is_string,
    is_dictlike, is_hashable,
    array_equal, is_integer,
    is_numeric, get_units,
    get_value
)
from dreye.utilities.abstract import _AbstractArray
from dreye.constants import ureg
from dreye.io import read_json, write_json


class _UnitArray(_AbstractArray):
    """
    Attributes are assigned privately and attribute properties written
    for each attribute to prevent user-side redefinition.
    """

    _convert_attributes = ()
    _unit_mappings = {}
    # _enforce_same_shape = True
    _init_args = ('attrs', 'contexts', 'name')
    # all init arguments necessary to copy object except values and units
    _args_defaults = {}
    # dictionary of attributes defaults if None
    _unit_conversion_params = {}
    # attributes mapping passed to the "to" method of pint

    @property
    def _init_aligned_attrs(self):
        return {}

    @abstractmethod
    def _test_and_assign_values(self, values, kwargs):
        """
        """
        # values is always a numpy.ndarray/None/or other
        # assign new values
        # function can be used to also assign dependent attributes
        # values = np.array(values).astype(DEFAULT_FLOAT_DTYPE)
        # self._values = values
        # this is used in the init to assign new values and when
        # setting new values
        pass

    @abstractmethod
    def _equalize(self, other):
        """
        Should just return equalized other_magnitude or NotImplemented
        """
        pass

    @property
    @abstractmethod
    def _class_new_instance(self):
        """
        Class used to initialize a new instance (usually self).
        """
        pass

    def __init__(
        self, values, *, units=None,
        contexts=None, attrs=None, name=None,
        **kwargs
    ):
        convert_units = False
        # map the value
        units = self._unit_mappings.get(units, units)

        # run through unit array and copy over attributes
        if is_string(values):
            values = self.load(values)

        # handling of UnitArray type
        if isinstance(values, _UnitArray):
            # assign None values
            if name is None:
                name = values.name
            if attrs is None:
                attrs = values.attrs
            if contexts is None:
                contexts = values.contexts
            for key, value in kwargs.items():
                if value is None and hasattr(values, key):
                    kwargs[key] = copy.copy(getattr(values, key))
            values = values.values

        if has_units(values):
            if units is not None:
                convert_units = True
                convert_to = units
            units = values.units
            values = values.magnitude

        if (units is None) or isinstance(units, str):
            units = ureg(units).units
        elif has_units(units):
            units = units.units
        # units are assigned
        self._units = units
        # setup names and attributes and contexts
        self.contexts = contexts
        self.attrs = attrs
        self.name = name

        for key, value in kwargs.items():
            if value is None:
                value = self._args_defaults.get(key, None)
            if key in self._init_args:
                # these has an attribute property
                assert hasattr(type(self), key), \
                    f"Must provide property for attribute {key}"
                # set attribute
                setattr(self, '_'+key, value)
            else:
                # these do not have an attribute property
                kwargs[key] = value

        # pop all set keys
        for key in self._init_args:
            if key in ('name', 'attrs', 'contexts'):
                continue
            kwargs.pop(key)

        # this should assign the values
        self._test_and_assign_values(values, kwargs)

        # this ensure that things like the domain are carried over
        if convert_units:
            self.units = convert_to

    def _get_convert_attribute(self, attr):
        return getattr(self, attr)

    def _set_convert_attribute(self, attr, value):
        setattr(self, '_'+attr, value)

    @property
    def init_kwargs(self):
        return {arg: getattr(self, arg) for arg in self._init_args}

    def copy(self):
        """
        copy instance.
        """
        return copy.copy(self)

    def __copy__(self):
        """
        """
        return type(self)(self)

    @property
    def units(self):
        """
        The units attatched to the values.
        """
        return self._units

    @units.setter
    def units(self, value):
        """
        """
        self.to(value, copy=False)

    @property
    def contexts(self):
        """tuple contexts for unit conversion.
        """
        return self._contexts

    @contexts.setter
    def contexts(self, value):
        """reset context
        """
        # always flux context
        if value is None:
            self._contexts = ('flux',)
        elif is_string(value):
            self._contexts = (value, 'flux',)
        elif is_listlike(value):
            self._contexts = tuple(list(value) + ['flux'])
        else:
            raise DreyeError(
                "Context must be type tuple, str, or None, but "
                f"is of type {type(value)}"
            )

    @property
    def attrs(self):
        """
        Dictionary to hold arbitrary objects
        """
        return self._attrs

    @attrs.setter
    def attrs(self, value):
        """reset attribute
        """

        if value is None:
            self._attrs = {}
        elif is_dictlike(value):
            self._attrs = dict(value)
        else:
            raise DreyeError(
                "Attribute dictionary must be type dict or None, "
                f"but is of type {type(value)}"
            )

    @property
    def name(self):
        """
        Returns the name of the signal instance.
        """
        return self._name

    @name.setter
    def name(self, value):
        if not is_hashable(value):
            raise DreyeError("New name value of type "
                             f"{type(value)} is not hashable.")
        self._name = value

    @property
    def magnitude(self):
        """
        Array without units (numpy.ndarray object).
        """
        return self._values

    @property
    def values(self):
        """
        Array with units (pint.Quantity object).
        """
        return self.magnitude * self.units

    def to(self, units, *args, copy=True, **kwargs):
        """
        convert to new units
        """
        if copy:
            self = self.copy()

        # map the value
        units = self._unit_mappings.get(units, units)
        if is_string(units) or units is None:
            units = ureg(units).units

        if units == self.units:
            return self

        # converts values if possible
        values = self._convert_all_attrs(units, *args, **kwargs)
        self._units = values.units
        self._values = values.magnitude
        return self

    def _convert_values(self, values, units, *args, unitless=False, **kwargs):
        """
        convert values given the contexts
        """

        kws = self._unit_conversion_kws
        kws.update(kwargs)

        values = values.to(units, *(self.contexts + args), **kws)

        if unitless:
            return values.magnitude
        else:
            return values

    @property
    def _unit_conversion_kws(self):
        return {
            key: getattr(self, name)
            for key, name in self._unit_conversion_params.items()
        }

    def _convert_other_attrs(self, units, *args, **kwargs):
        """
        convert units of stored attributes
        """
        # any other attributes to be converted will be converted
        # these have to be accessible as self.attr and set by self._attr
        for attr in self._convert_attributes:
            # attr
            value = self._get_convert_attribute(attr)
            if value is None:
                continue
            unitless = False
            if not has_units(value):
                value = value * self.units
                unitless = True
            # convert value
            value = self._convert_values(
                value, units, *args, unitless=unitless, **kwargs
            )
            # set attr
            self._set_convert_attribute(attr, value)

    def _convert_all_attrs(self, units, *args, **kwargs):
        """
        convert units of all relevant attributes
        """
        self._convert_other_attrs(units, *args, **kwargs)
        return self._convert_values(
            self.values, units, *args, unitless=False, **kwargs
        )

    def _instance_handler(self, other, op, reverse=False):
        """
        standard instance handler.
        This is used by operators and
        should return a new instance of self
        """
        # special handling of string and None
        if is_string(other) or other is None:
            other = ureg(other)
        elif isinstance(other, ureg.Unit):
            other = other * 1
        # apply equalize
        other_magnitude, self = self._equalize(other)
        if other_magnitude is NotImplemented:
            return other_magnitude

        # unit handling # make common function?
        if has_units(other):
            other_units = other.units
        else:
            # assume dimensionless
            other_units = ureg(None).units
        # apply operation with units
        other_values = other_magnitude * other_units
        if reverse:
            new = getattr(operator, op)(other_values, self.values)
        else:
            new = getattr(operator, op)(self.values, other_values)
        # create new instance
        return self._class_new_instance(values=new, **self.init_kwargs)

    def __mul__(self, other):
        return self._instance_handler(other, 'mul')

    def __rmul__(self, other):
        return self._instance_handler(other, 'mul', True)

    def __truediv__(self, other):
        return self._instance_handler(other, 'truediv')

    def __rtruediv__(self, other):
        return self._instance_handler(other, 'truediv', True)

    def __floordiv__(self, other):
        return self._instance_handler(other, 'floordiv')

    def __rfloordiv__(self, other):
        return self._instance_handler(other, 'floordiv', True)

    def __add__(self, other):
        return self._instance_handler(other, 'add')

    def __radd__(self, other):
        return self._instance_handler(other, 'add', True)

    def __sub__(self, other):
        return self._instance_handler(other, 'sub')

    def __rsub__(self, other):
        return self._instance_handler(other, 'sub', True)

    def __pow__(self, other):
        return self._instance_handler(other, 'pow')

    def __rpow__(self, other):
        return self._instance_handler(other, 'pow', True)

    def __contains__(self, other):
        return self._instance_handler(other, 'contains')

    def __pos__(self):
        return self._class_new_instance(
            values=operator.pos(self.values), **self.init_kwargs
        )

    def __neg__(self):
        return self._class_new_instance(
            values=operator.neg(self.values), **self.init_kwargs
        )

    def __abs__(self):
        return self._class_new_instance(
            values=operator.abs(self.values), **self.init_kwargs
        )

    def __iter__(self):
        """
        Generate over values
        """
        return iter(self.values)

    def __len__(self):
        return len(self.magnitude)

    def __bool__(self):
        return True

    def __repr__(self):
        return str(self.values)

    def __getitem__(self, key):
        values = self.values[key]
        if hasattr(values, 'ndim'):
            if values.size == 0:
                return values
            # keep self
            if values.ndim == self.ndim:
                # if not is tuple make tuple
                if not isinstance(key, tuple):
                    key = (key,)

                for idx, ikey in enumerate(key):
                    if ikey is Ellipsis:
                        before = key[:idx]
                        if len(key) > idx+1:
                            after = key[idx+1:]
                        else:
                            after = ()
                        toadd = values.ndim - len(after + before)
                        key = (
                            before
                            + (slice(None, None, None),) * toadd
                            + after
                        )
                    elif isinstance(ikey, slice):
                        continue
                    else:
                        # only allow slices and ellipsis
                        return values
                # get new inits
                init_kwargs = self.init_kwargs
                # indices are assumed to be positive
                for idx, attrs in self._init_aligned_attrs.items():
                    for attr in attrs:
                        if idx is None:
                            init_kwargs[attr] = None
                        elif isinstance(idx, tuple):
                            # this assumes numpy.ndarray instances
                            if any(map(lambda x: x < 0), idx):
                                raise ValueError(
                                    "_init_aligned_attrs was set inproperly."
                                )
                            attr_value = getattr(self, attr)
                            ikey = tuple(
                                key[iidx]
                                if iidx < len(key)
                                else slice(None, None, None)
                                for iidx in idx
                            )
                            init_kwargs[attr] = attr_value[ikey]
                        else:
                            if idx < 0:
                                raise ValueError(
                                    "_init_aligned_attrs was set inproperly."
                                )
                            attr_value = getattr(self, attr)
                            # assumes same length and
                            # not broadcastable as in tuple case
                            if idx < len(key):
                                ikey = key[idx]
                                init_kwargs[attr] = attr_value[ikey]
                            else:
                                init_kwargs[attr] = attr_value
                return self._class_new_instance(
                    values=values, **init_kwargs
                )
        return values

    def __eq__(self, other):
        """
        Equality between self and other
        """

        if isinstance(other, type(self)):

            if self._init_args != other._init_args:
                return False

            if self.units != other.units:
                return False

            if not array_equal(self.magnitude, other.magnitude):
                return False

            for name in self._init_args:
                # convention that if attribute ends with underscore
                # do not compare!
                if name.endswith('_'):
                    continue
                sattr = getattr(self, name)
                oattr = getattr(other, name)
                if not isinstance(oattr, type(sattr)):
                    return False
                elif (
                    not is_hashable(sattr) and
                    (is_listlike(sattr))
                    or (is_numeric(sattr))
                ):
                    svalue, sunits = get_value(sattr), get_units(sattr)
                    ovalue, ounits = get_value(oattr), get_units(oattr)
                    if sunits != ounits:
                        return False

                    ovalue = np.array(ovalue)
                    svalue = np.array(svalue)
                    if ovalue.shape != svalue.shape:
                        return False
                    if (
                        np.issubdtype(svalue.dtype, np.number)
                        and np.issubdtype(ovalue.dtype, np.number)
                    ):
                        if not np.all(np.isnan(svalue) == np.isnan(ovalue)):
                            return False
                        truth = np.all(
                            svalue[~np.isnan(svalue)]
                            == ovalue[~np.isnan(ovalue)]
                        )
                    else:
                        truth = np.all(svalue == ovalue)
                else:
                    truth = sattr == oattr
                if not truth:
                    return False

            return True

        return False

    @property
    def ndim(self):
        return self.magnitude.ndim

    @property
    def shape(self):
        return self.magnitude.shape

    @property
    def size(self):
        return self.magnitude.size

    def asarray(self):
        """
        Return values as numpy array
        """
        return self.values.magnitude

    def __array__(self):
        return self.asarray()

    def to_dict(self):
        """
        convert object to dictionary
        """
        dictionary = {
            'values': self.magnitude.tolist(),
            'units': self.units,
            **self.init_kwargs
        }
        return dictionary

    @classmethod
    def from_dict(cls, data):
        """
        Create a class from dictionary.
        """
        return cls(**data)

    @classmethod
    def load(cls, filename):
        """
        Load object
        """
        return read_json(filename)

    def save(self, filename):
        """
        Save object
        """
        return write_json(filename, self)
