"""
Abstract Base Class for both Signal and Domain.
"""

from abc import abstractmethod
import copy

import numpy as np
from pint import DimensionalityError

from dreye.err import DreyeError, DreyeUnitError
from dreye.utilities import (
    dissect_units, has_units,
    convert_units, is_listlike, is_string,
    is_dictlike, is_hashable, _convert_get_val_opt,
    asarray
)
from dreye.utilities.abstract import _AbstractArray
from dreye.constants import ureg, DEFAULT_FLOAT_DTYPE


class _UnitArray(_AbstractArray):

    _convert_attributes = ()
    _unit_mappings = {}
    _init_args = ()

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
        if value is None:
            return

        # map the value
        value = self._unit_mappings.get(value, value)

        if value == self.units:
            return

        # converts values if possible
        try:
            values = self._convert_all_attrs(value)
        except DimensionalityError:
            raise DreyeUnitError(
                str(value), str(self.units),
                ureg(str(value)).dimensionality,
                self.units.dimensionality,
                f' for instance of type {type(self).__name__}.'
            )
        self._units = values.units
        self.values = values.magnitude

    @property
    def contexts(self):
        """contexts for unit conversion.
        """

        if self._contexts is None:
            return ()
        elif isinstance(self._contexts, str):
            return (self._contexts, )
        else:
            return tuple(self._contexts)

    @contexts.setter
    def contexts(self, value):
        """reset context
        """
        if is_listlike(value) or is_string(value) or value is None:
            self._contexts = value
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
        if self._attrs is None:
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self, value):
        """reset attribute
        """

        if is_dictlike(value) or value is None:
            self._attrs = value
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
            raise DreyeError(
                f'New name value of type {type(value)} is not hashable.')
        self._name = value

    @property
    def magnitude(self):
        """
        Array without units (numpy.ndarray object).
        """
        return self._values

    @magnitude.setter
    def magnitude(self, values):
        """
        alias for setting values, but throws an error if has units
        """
        if has_units(values):
            raise DreyeError(
                "When setting magnitude new magnitude cannot have units."
            )
        self.values = values

    @property
    def values(self):
        """
        Array with units (pint.Quantity object).
        """
        return self.magnitude * self.units

    @values.setter
    def values(self, values):
        """
        Setting new values array
        """

        values = _convert_get_val_opt(values, self.units)
        values = asarray(values).astype(DEFAULT_FLOAT_DTYPE)

        if not values.shape == self.shape:
            raise DreyeError('Array for values assignment must be same shape.')

        self._test_and_assign_new_values(values)

    def _test_and_assign_new_values(self, values):
        # assign new values
        # function can be used to also assign dependent attributes
        self._values = values

    def to(self, units, copy=True):
        """
        """

        if copy:
            self = self.copy()

        self.units = units
        return self

    def _convert_values(self, values, units, unitless=False):
        """
        convert values given the contexts
        """

        try:
            values = values.to(units, *self.contexts)
        except TypeError:
            if not hasattr(self, 'domain'):
                raise

            domain = self.domain.values
            if values.ndim == 2:
                domain = np.expand_dims(
                    domain.magnitude, axis=self.other_axis
                ) * domain.units
            elif values.ndim > 2:
                raise DreyeError("Values must be 1- or 2-dimensional.")

            values = values.to(units, *self.contexts, domain=domain)

        if unitless:
            return values.magnitude
        else:
            return values

    def _convert_other_attrs(self, units):
        """
        convert units of stored attributes
        """
        # any other attributes to be converted will be converted
        # these have to be accessible as self.attr and set by self._attr
        for attr in self._convert_attributes:
            # attr
            if hasattr(self, attr):
                value = getattr(self, attr)
            else:
                value = self.attrs.get(attr, None)

            if value is None:
                continue

            unitless = False
            if not has_units(value):
                value = value * self.units
                unitless = True

            # convert value
            value = self._convert_values(value, units, unitless=unitless)

            if hasattr(self, attr):
                setattr(self, '_'+attr, value)
            else:
                self.attrs[attr] = value

    def _convert_all_attrs(self, units):
        """
        convert units of all relevant attributes
        """
        self._convert_other_attrs(units)
        return self._convert_values(self.values, units)

    @abstractmethod
    def equalize_domains(self, other):
        pass

    @property
    @abstractmethod
    def _class_new_instance(self):
        pass

    @staticmethod
    def _other_handler(other):
        """handles other objects
        """
        if isinstance(other, str):
            other = ureg(other)
        return dissect_units(other)

    def _instance_handler(self, other):
        """
        standard instance handler (overwritten by Signal class)
        """
        return self, other, {}

    def _factor_function(self, other, operation, **kwargs):
        """
        operator function for multiplications and divisions
        """
        # handle instance
        self, other, kws = self._instance_handler(other)
        kwargs.update(kws)
        # handle units
        other_magnitude, other_units = self._other_handler(other)

        new = getattr(self.magnitude, operation)(other_magnitude)

        if other_units is None:
            new_units = self.units
        else:
            new_units = getattr(self.units, operation)(other_units)

        return self._create_new_instance(new * new_units, **kwargs)

    def _sum_function(self, other, operation, **kwargs):
        """
        operator function for additions and subtractions
        """
        # handle instance
        self, other, kws = self._instance_handler(other)
        kwargs.update(kws)
        # handle units
        other_magnitude, other_units = self._other_handler(other)

        if other_units is None or (other_units == self.units):
            pass
        else:
            # convert units if possible
            other_magnitude = other.to(self.units).magnitude

        new = getattr(self.magnitude, operation)(other_magnitude)

        return self._create_new_instance(new * self.units, **kwargs)

    def _simple_function(self, other, operation, **kwargs):
        """
        operator function for opeartions without unit handling, or
        other has unit None
        """
        # handle instance
        self, other, kws = self._instance_handler(other)
        kwargs.update(kws)
        # handle units
        other_magnitude, other_units = self._other_handler(other)

        if other_units is None:
            pass
        elif other_units != ureg(None).units:
            raise DreyeError(
                f'{operation} requires unitless values.'
            )

        new = getattr(self.magnitude, operation)(other_magnitude)
        new_units = getattr(self.units, operation)(other_magnitude)

        return self._create_new_instance(new * new_units, **kwargs)

    def _single_function(self, operation, **kwargs):
        """
        performs operation and multiplies units
        """
        return self._create_new_instance(
            getattr(self.magnitude, operation)() * self.units,
            **kwargs
        )

    def _create_new_instance(self, values, **kwargs):
        """create new instance given a numpy.array
        """
        return self._class_new_instance(
            values, **{**self.init_kwargs, **kwargs}
        )

    def __mul__(self, other):
        return self._factor_function(other, '__mul__')

    def __div__(self, other):
        return self._factor_function(other, '__div__')

    def __rdiv__(self, other):
        return self._factor_function(other, '__rdiv__')

    def __truediv__(self, other):
        return self._factor_function(other, '__truediv__')

    def __rtruediv__(self, other):
        return self._factor_function(other, '__rtruediv__')

    def __floordiv__(self, other):
        return self._factor_function(other, '__floordiv__')

    def __add__(self, other):
        return self._sum_function(other, '__add__')

    def __sub__(self, other):
        return self._sum_function(other, '__sub__')

    def __pow__(self, other):
        return self._simple_function(other, '__pow__')

    def __mod__(self, other):
        return self._sum_function(other, '__mod__')

    def __pos__(self):
        return self._single_function('__pos__')

    def __neg__(self):
        return self._single_function('__neg__')

    def __abs__(self):
        return self._single_function('__abs__')

    def __iter__(self):
        """
        Returns a generator for the domain values.

        Returns
        -------
        generator
            domain values generator.
        """

        return iter(self.values)

    def __contains__(self, other):
        """
        Returns if the domain contains given value.
        """
        # TODO tolerance for floating point overflow

        return np.all(np.in1d(other, self.values))

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return str(self.values)

    def __getitem__(self, key):
        return self.values.__getitem__(key)

    def __setitem__(self, key, value):
        value = convert_units(value, self.units)
        self._values[key] = value

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
