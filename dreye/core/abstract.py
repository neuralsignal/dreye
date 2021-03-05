"""
Abstract Base Class for both Signal and Domain.
"""

from abc import abstractmethod
import copy
import operator

import numpy as np

from dreye.err import DreyeError
from dreye.utilities import (
    has_units, is_listlike, is_string,
    is_dictlike, is_hashable, array_equal,
    is_numeric, get_units, get_value
)
from dreye.utilities.abstract import _AbstractArray
from dreye.constants import ureg, CONTEXTS
from dreye.io import read_json, write_json, read_pickle, write_pickle


class _UnitArray(_AbstractArray):
    """
    This abstract class is used by all signal-type classes
    and the `dreye.Domain` class.

    Attributes are assigned privately and attribute properties written
    for each attribute to prevent user-side redefinition.
    """

    _convert_attributes = ()
    _unit_mappings = {}
    # _enforce_same_shape = True
    _init_args = ('attrs', 'name')
    # all init arguments necessary to copy object except values and units
    _args_defaults = {}
    # dictionary of attributes defaults if None
    _unit_conversion_params = {}
    # attributes mapping passed to the "to" method of pint

    @property
    def _init_aligned_attrs(self):
        """
        This property defines attributes that are aligned with particular
        axis of the `values` array.
        """
        return {}

    @abstractmethod
    def _test_and_assign_values(self, values, kwargs):
        """
        Method implemented for each subclass.

        This method is used during initialzation to assign the `values`
        array and assign and check various other attributes.
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
        Equalize other to self for numerical operations.

        This method is used by various mathematical operators, and should
        always return `other_magnitude` and `self`, or `NotImplemented` and
        `self`.
        """
        pass

    @property
    @abstractmethod
    def _class_new_instance(self):
        """
        Class used to initialize a new instance (usually self).

        This can either be a property that returns the desired class
        for initialization after operations are performed, or it can
        be a method that accepts all arguments required for initialization
        """
        pass

    def __init__(
        self, values, *, units=None,
        attrs=None, name=None,
        **kwargs
    ):
        """
        The init always accepts a positional argument `values` and various
        designated keyword arguments: `units`, `attrs`, `name`.

        Parameters
        ----------
        values : array-like
        units : string or `pint.Unit`, optional
        attrs : dict, optional
        name : string or tuple, optional
        """
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
        # setup names and attributes
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
                setattr(self, '_' + key, value)
            else:
                # these do not have an attribute property
                kwargs[key] = value

        # pop all set keys
        for key in self._init_args:
            if key in ('name', 'attrs'):
                continue
            kwargs.pop(key)

        # this should assign the values
        self._test_and_assign_values(values, kwargs)

        # this ensure that things like the domain are carried over
        if convert_units:
            self.units = convert_to

    def _get_convert_attribute(self, attr):
        """
        Method to get an attribute whose units need to be converted
        the same way as the `values` array.
        """
        return getattr(self, attr)

    def _set_convert_attribute(self, attr, value):
        """
        Method to set an attribute whose units need to be converted
        the same way as the `values` array.
        """
        setattr(self, '_' + attr, value)

    @property
    def _init_kwargs(self):
        """
        Keyword arguments used to re-initializing/copying instance (does not
        include values and units).
        """
        return {arg: getattr(self, arg) for arg in self._init_args}

    def copy(self):
        """
        Returns copy of object.
        """
        return copy.copy(self)

    def __copy__(self):
        """
        Copy method.
        """
        return type(self)(self)

    @property
    def units(self):
        """
        The units associated with the object.
        """
        return self._units

    @units.setter
    def units(self, value):
        """
        Setting the units associated with the object
        """
        self.to(value, copy=False)

    @property
    def attrs(self):
        """
        Dictionary to hold arbitrary objects.

        The keys of the dictionary can be user-defined keys or keys specific to
        particular classes, such as for the `dreye.MeasuredSpectrum` class.
        """
        return self._attrs

    @attrs.setter
    def attrs(self, value):
        """
        Setting the attributes dictionary.
        """
        if value is None:
            self._attrs = {}
        elif is_dictlike(value):
            self._attrs = dict(value)
        else:
            raise DreyeError("Attribute dictionary must be type dict "
                             f"or None, but is of type {type(value)}.")

    @property
    def name(self):
        """
        Returns the name given to the object.

        The name is usually a string, tuple, or None.
        """
        return self._name

    @name.setter
    def name(self, value):
        """
        Setting the name of the object
        """
        if not is_hashable(value):
            raise DreyeError("New name value of type "
                             f"{type(value)} is not hashable.")
        self._name = value

    @property
    def magnitude(self):
        """
        Returns `numpy.ndarray` of values without units.
        """
        return self._values

    @property
    def values(self):
        """
        Returns `pint.Quantity` array (i.e. units are attached).
        """
        return self.magnitude * self.units

    def to(self, units, *args, copy=True, **kwargs):
        """
        Returns copy of self with converted units.

        Parameters
        ----------
        units : str
            Units to convert to
        args : tuple, optional
            Contexts to pass to `pint.Quantity.to` method.
        copy : bool, optional
            If self will be copied. Defaults to True.
        kwargs : dict, optional
            Keyword arguments passed to `pint.Quantity.to` method.

        Returns
        -------
        obj : instance of self
            Returns self or a copy of self with units converted

        Raises
        ------
        pint.errors.DimensionalityError
            If self cannot be converted to the given units.

        Notes
        -----
        This function essentially uses `pint`'s unit conversion system.
        Besides converting the `values` array, this method will also convert
        any attributes that have the same dependent units as the `values`
        array.
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
        Convert values.

        This method is indirectly used by the `to` method
        and the directly by the `_convert_all_attrs` and
        `_convert_other_attrs`.
        """

        kws = self._unit_conversion_kws
        kws.update(kwargs)

        values = values.to(units, *CONTEXTS, *args, **kws)

        if unitless:
            return values.magnitude
        else:
            return values

    @property
    def _unit_conversion_kws(self):
        """
        A set of keywords that are passed to the unit conversion automatically.

        This property is used by the `to` method.
        """
        return {
            key: getattr(self, name)
            for key, name in self._unit_conversion_params.items()
        }

    def _convert_other_attrs(self, units, *args, **kwargs):
        """
        Convert units of stored attributes that have the same
        units as the `values` array.

        This method is indirectly used by the `to` method and
        directly used by `_convert_all_attrs`.
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
        Convert units of all relevant attributes
        (i.e. `values` and `_convert_attributes`).

        This method is used by the `to` method.
        """
        self._convert_other_attrs(units, *args, **kwargs)
        return self._convert_values(
            self.values, units, *args, unitless=False, **kwargs
        )

    def _instance_handler(self, other, op, reverse=False):
        """
        Standard instance handler for various mathematical operations.

        This method is used by various mathematical operators and
        requires that the `_equalize` method is implemented.

        The mathematical operations are performed on the `pint.Quantity`
        arrays of self and other for unit tracking.
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
        return self._class_new_instance(values=new, **self._init_kwargs)

    def __mul__(self, other):
        """
        Multiply self and other.

        This method keeps track of units and attempts to return
        an instance of the `self` class.
        """
        return self._instance_handler(other, 'mul')

    def __rmul__(self, other):
        """
        Multiply self and other.

        This method keeps track of units and attempts to return
        an instance of the `self` class.
        """
        return self._instance_handler(other, 'mul', True)

    def __truediv__(self, other):
        """
        Divide self by other.

        This method keeps track of units and attempts to return
        an instance of the `self` class.
        """
        return self._instance_handler(other, 'truediv')

    def __rtruediv__(self, other):
        """
        Divide other by self.

        This method keeps track of units and attempts to return
        an instance of the `self` class.
        """
        return self._instance_handler(other, 'truediv', True)

    def __floordiv__(self, other):
        """
        Divide self by other.

        This method keeps track of units and attempts to return
        an instance of the `self` class.
        """
        return self._instance_handler(other, 'floordiv')

    def __rfloordiv__(self, other):
        """
        Divide other by self.

        This method keeps track of units and attempts to return
        an instance of the `self` class.
        """
        return self._instance_handler(other, 'floordiv', True)

    def __add__(self, other):
        """
        Divide self and other.

        This method keeps track of units and attempts to return
        an instance of the `self` class.
        """
        return self._instance_handler(other, 'add')

    def __radd__(self, other):
        """
        Divide self and other.

        This method keeps track of units and attempts to return
        an instance of the `self` class.
        """
        return self._instance_handler(other, 'add', True)

    def __sub__(self, other):
        """
        Subtract other from self.

        This method keeps track of units and attempts to return
        an instance of the `self` class.
        """
        return self._instance_handler(other, 'sub')

    def __rsub__(self, other):
        """
        Subtract self from other.

        This method keeps track of units and attempts to return
        an instance of the `self` class.
        """
        return self._instance_handler(other, 'sub', True)

    def __pow__(self, other):
        """
        Raise self to a power.

        This method keeps track of units and attempts to return
        an instance of the `self` class.
        """
        return self._instance_handler(other, 'pow')

    def __rpow__(self, other):
        """
        Raise other to a power.

        This method keeps track of units and attempts to return
        an instance of the `self` class.
        """
        return self._instance_handler(other, 'pow', True)

    def __contains__(self, other):
        """
        Applies `numpy.ndarray` __contains__ method with self and other.

        This method keeps track of units and attempts to return
        an instance of the `self` class.
        """
        return self._instance_handler(other, 'contains')

    def __pos__(self):
        """
        Positive self.

        This method keeps track of units and attempts to return
        an instance of the `self` class.
        """
        return self._class_new_instance(
            values=operator.pos(self.values), **self._init_kwargs
        )

    def __neg__(self):
        """
        Negative self.

        This method keeps track of units and attempts to return
        an instance of the `self` class.
        """
        return self._class_new_instance(
            values=operator.neg(self.values), **self._init_kwargs
        )

    def __abs__(self):
        """
        Absulate self.

        This method keeps track of units and attempts to return
        an instance of the `self` class.
        """
        return self._class_new_instance(
            values=operator.abs(self.values), **self._init_kwargs
        )

    def __iter__(self):
        """
        Iterate over values
        """
        return iter(self.values)

    def __len__(self):
        """
        Length of `values`; Same as len(`self.magnitude`).
        """
        return len(self.magnitude)

    def __bool__(self):
        """
        Returns True
        """
        return True

    def __repr__(self):
        """
        Represenation of self
        """
        return str(self.values)

    def __getitem__(self, key):
        """
        Returns copy of self if key is a tuple of slices or a slice instance.
        Otherwise `__getitem__` returns a `pint.Quantity`.
        """
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
                        if len(key) > idx + 1:
                            after = key[idx + 1:]
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
                _init_kwargs = self._init_kwargs
                # indices are assumed to be positive
                for idx, attrs in self._init_aligned_attrs.items():
                    for attr in attrs:
                        if idx is None:
                            _init_kwargs[attr] = None
                        elif isinstance(idx, tuple):
                            # this assumes numpy.ndarray instances
                            if any(map(lambda x: x < 0), idx):
                                raise ValueError(
                                    "`_init_aligned_attrs` was set inproperly."
                                )
                            attr_value = getattr(self, attr)
                            ikey = tuple(
                                key[iidx]
                                if iidx < len(key)
                                else slice(None, None, None)
                                for iidx in idx
                            )
                            _init_kwargs[attr] = attr_value[ikey]
                        else:
                            if idx < 0:
                                raise ValueError(
                                    "`_init_aligned_attrs` was set inproperly."
                                )
                            attr_value = getattr(self, attr)
                            # assumes same length and
                            # not broadcastable as in tuple case
                            if idx < len(key):
                                ikey = key[idx]
                                _init_kwargs[attr] = attr_value[ikey]
                            else:
                                _init_kwargs[attr] = attr_value
                return self._class_new_instance(
                    values=values, **_init_kwargs
                )
        return values

    def __eq__(self, other):
        """
        Equality between self and other.

        Self and other are equal if all attributes match and the `values`
        arrays are equal. Units have to match as well.
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
        """
        Dimensionality of `values` array.

        See Also
        --------
        numpy.ndarray.ndim
        """
        return self.magnitude.ndim

    @property
    def shape(self):
        """
        Shape of `values` array.

        See Also
        --------
        numpy.ndarray.shape
        """
        return self.magnitude.shape

    @property
    def size(self):
        """
        Size of `values` array.

        See Also
        --------
        numpy.ndarray.size
        """
        return self.magnitude.size

    def asarray(self):
        """
        Return values as `numpy.ndarray` object.
        """
        return self.values.magnitude

    def __array__(self):
        """
        Method for converting to `numpy.ndarray` object.
        """
        return self.values.magnitude

    def to_dict(self):
        """
        Convert object to dictionary, containing values, units,
        and other attributes necessary for initialization.
        """
        dictionary = {
            'values': self.magnitude.tolist(),
            'units': self.units,
            **self._init_kwargs
        }
        return dictionary

    @classmethod
    def from_dict(cls, data):
        """
        Create class instance from dictionary.
        """
        return cls(**data)

    @classmethod
    def load(cls, filename):
        """
        Load JSON or JSON-compressed object.

        Parameters
        ----------
        filename : str
            Filename with stored object.

        Returns
        -------
        object : instance of `cls`
            Instance of `cls`.
        """
        if '.pkl' in filename:
            return read_pickle(filename)
        else:
            return read_json(filename)

    def save(self, filename):
        """
        Save JSON or JSON-compressed object object.

        Parameters
        ----------
        filename : str
            Filename to store object.
        """
        if '.pkl' in filename:
            return write_pickle(filename, self)
        else:
            return write_json(filename, self)
