"""
Abstract Base Classes for Continuous

======

Defines the class for implementing continuous signals:
"""

# standard python packages
from abc import abstractmethod
import copy

# third party library imports
import numpy as np

# package imports
from dreye.err import DreyeError, DreyeUnitError
from dreye.utilities import (
    dissect_units, AbstractSequence, is_arraylike, has_units,
    convert_units
)
from dreye.constants import UREG


class AbstractDomain(AbstractSequence):

    convert_attributes = ()

    def copy(self):
        """
        """

        return copy.copy(self)

    def __copy__(self):
        return self.__class__(self, dtype=self.dtype)

    @property
    def units(self):
        """
        """

        return self._units

    @units.setter
    def units(self, value):
        """
        """

        if value is None:
            return

        if value == self.units:
            return

        # converts values if possible
        values = self._converting_values(value)
        self._units = values.units
        self.values = values.magnitude

    def to(self, units, copy=True):
        """
        """

        if copy:
            self = self.copy()

        self.units = units
        return self

    @staticmethod
    def _convert_values(
        values, units, contexts, domain=None, axis=None, unitless=False
    ):

        if contexts is None:
            contexts = ()
        elif isinstance(contexts, str):
            contexts = (contexts, )

        try:
            values = values.to(units, *contexts)
        except TypeError:
            if values.ndim == 1:
                values = values.to(units, *contexts, domain=domain)
            else:
                values = values.to(
                    units,
                    *contexts,
                    domain=(
                        np.expand_dims(domain.magnitude, axis=axis)
                        * domain.units),
                )

        if unitless:
            return values.magnitude
        else:
            return values

    def _convert_other_attrs(self, units):

        # any other attributes to be converted will be converted
        for attr in self.convert_attributes:
            # attr
            if getattr(self, attr) is None:
                continue

            value = getattr(self, attr)
            unitless = False
            if not has_units(value):
                value = value * self.units
                unitless = True

            setattr(
                self, '_' + attr,
                self._convert_values(
                    value, units, self.contexts, unitless=unitless
                )
            )

    def _converting_values(self, units):
        """
        """

        self._convert_other_attrs(units)

        return self._convert_values(self.values, units, self.contexts)

    @property
    def contexts(self):
        """
        """

        if self._contexts is None:
            return ()
        elif isinstance(self._contexts, str):
            return (self._contexts, )
        else:
            return tuple(self._contexts)

    @contexts.setter
    def contexts(self, value):
        """
        """

        self._contexts = value

    @property
    def magnitude(self):
        """
        """

        return self._values

    @abstractmethod
    def equalize_domains(self, other):
        """returns an equalized domain
        """
        pass

    @property
    @abstractmethod
    def dtype(self):
        pass

    @dtype.setter
    @abstractmethod
    def dtype(self):
        pass

    @property
    @abstractmethod
    def values(self):
        pass

    @values.setter
    @abstractmethod
    def values(self):
        pass

    @staticmethod
    def _other_handler(other):

        if isinstance(other, str):
            other = UREG(other)

        return dissect_units(other)

    def _factor_function(self, other, operation, **kwargs):
        """operator function for multiplications and divisions
        """

        other_magnitude, other_units = self._other_handler(other)

        new = getattr(self.magnitude, operation)(other_magnitude)

        if other_units is None:
            new_units = self.units
        else:
            new_units = getattr(self.units, operation)(other_units)

        return self._create_new_instance(new * new_units, **kwargs)

    def _sum_function(self, other, operation, **kwargs):
        """operator function for additions and subtractions
        """

        other_magnitude, other_units = self._other_handler(other)

        if other_units is None or (other_units == self.units):
            pass
        else:
            # convert units if possible
            other_magnitude = other.to(self.units).magnitude

        new = getattr(self.magnitude, operation)(other_magnitude)

        return self._create_new_instance(new * self.units, **kwargs)

    def _simple_function(self, other, operation, **kwargs):
        """operator function for opeartions without unit handling.
        That is other must have unit None
        """

        other_magnitude, other_units = self._other_handler(other)

        if other_units is None:
            pass
        elif other_units != UREG(None).units:
            raise DreyeUnitError(
                f'{operation} requires unitless values.'
            )

        new = getattr(self.magnitude, operation)(other_magnitude)
        new_units = getattr(self.units, operation)(other_magnitude)

        return self._create_new_instance(new * new_units, **kwargs)

    def _single_function(self, operation, **kwargs):
        """
        """

        return self._create_new_instance(
            getattr(self.magnitude, operation)() * self.units,
            **kwargs
        )

    @property
    @abstractmethod
    def _class_new_instance(self):
        pass

    def _create_new_instance(self, values, **kwargs):
        """create new instance given a numpy.array
        """

        return self._class_new_instance(values, **kwargs)

    def __mul__(self, other):
        """
        """

        return self._factor_function(other, '__mul__')

    def __div__(self, other):
        """
        """

        return self._factor_function(other, '__div__')

    def __rdiv__(self, other):
        """
        """

        return self._factor_function(other, '__rdiv__')

    def __truediv__(self, other):
        """
        """

        return self._factor_function(other, '__truediv__')

    def __rtruediv__(self, other):
        """
        """

        return self._factor_function(other, '__rtruediv__')

    def __floordiv__(self, other):
        """
        """

        return self._factor_function(other, '__floordiv__')

    def __add__(self, other):
        """
        """

        return self._sum_function(other, '__add__')

    def __sub__(self, other):
        """
        """

        return self._sum_function(other, '__sub__')

    def __pow__(self, other):
        """
        """

        return self._simple_function(other, '__pow__')

    def __mod__(self, other):
        """
        """

        return self._sum_function(other, '__mod__')

    def __pos__(self):
        """
        """

        return self._single_function('__pos__')

    def __neg__(self):
        """
        """

        return self._single_function('__neg__')

    def __abs__(self):
        """
        """

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
        """
        Return number of values in domain
        """

        return len(self.values)

    def __repr__(self):
        """
        Returns a str representation of the domain.
        """

        return str(self.values)

    def __getitem__(self, key):
        """
        """

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


class AbstractSignal(AbstractDomain):
    """Abstract class for signal class
    """

    def _converting_values(self, units):
        """
        """

        self._convert_other_attrs(units)

        return self._convert_values(
            self.values,
            units,
            self.contexts,
            domain=self.domain.values,
            axis=self.other_axis
        )

    @abstractmethod
    def domain_class(self):
        pass

    @property
    @abstractmethod
    def domain(self):
        pass

    @domain.setter
    @abstractmethod
    def domain(self):
        pass

    @abstractmethod
    def interpolate(self):
        """returns a new instance of Signal interpolated along the domain.
        """
        pass

    @abstractmethod
    def __call__(self):
        """same as interpolate
        """
    @property
    @abstractmethod
    def interpolator(self):
        pass

    @interpolator.setter
    @abstractmethod
    def interpolator(self):
        pass

    @property
    @abstractmethod
    def interpolator_kwargs(self):
        pass

    @property
    @abstractmethod
    def labels(self):
        pass

    @property
    @abstractmethod
    def domain_axis(self):
        pass

    @property
    @abstractmethod
    def other_axis(self):
        pass

    @property
    @abstractmethod
    def other_len(self):
        pass

    @property
    @abstractmethod
    def domain_len(self):
        pass

    @property
    @abstractmethod
    def T(self):
        pass

    @abstractmethod
    def init_args(self):
        pass

    @property
    def init_kwargs(self):

        return {arg: getattr(self, arg) for arg in self.init_args}

    def _create_new_instance(self, values, **kwargs):
        """create instance from numpy.array
        """

        return self._class_new_instance(
            values, **{**self.init_kwargs, **kwargs}
        )

    def _instance_handler(self, other):

        # check if subclass of AbstractSignal
        # check if domains are equal
        # check if interpolator is equal

        labels = self.labels

        if isinstance(other, AbstractSignal):

            assert other.interpolator == self.interpolator, (
                "Interpolators must be equal, "
                "if domains have different intervals")

            self, other, labels = self.equalize_domains(other,
                                                        return_labels=True)

        elif isinstance(other, AbstractDomain) and self.ndim == 2:
            other = (
                np.expand_dims(other.magnitude, axis=self.other_axis)
                * other.units
            )

        return self, other, labels

    def _broadcast(self, other, shared_axis=None):
        """
        """

        if (
            (np.ndim(other) == 0)
            or (self.ndim == 1)
            or isinstance(other, AbstractDomain)
            or (np.ndim(other) == self.ndim)
        ):
            return other

        else:
            if shared_axis is None:
                shared_axis = self.domain_axis

            other_axis = (shared_axis + 1) % 2

            other, units = dissect_units(other)

            other = np.expand_dims(other, other_axis)

            assert self.shape[shared_axis] == other.shape[shared_axis]

            if units is None:
                return other

            else:
                return other * units

    def equalize_domains(
        self,
        other,
        interval=None,
        start=None,
        end=None,
        equalize_dimensions=True,
        return_labels=False,
        **kwargs
    ):
        """equalize domains for both instances
        """

        labels = self.labels

        if equalize_dimensions:
            self, other, labels = self._equalize_dimensionality(other)

        if self.domain != other.domain:

            domain = self.domain.equalize_domains(
                other.domain,
                interval=interval,
                start=start,
                end=end,
                **kwargs
            )

            self = self(domain)
            other = other(domain)

        if return_labels:
            return self, other, labels
        else:
            return self, other

    def _equalize_dimensionality(self, other):
        """
        """

        labels = self.labels

        if self.ndim == 1:
            if other.ndim == 1:
                pass
            else:
                self = self._expand_dims(other.other_axis)
                labels = other.labels

        if other.ndim == 1:
            if self.ndim == 1:
                pass
            else:
                other = other._expand_dims(self.other_axis)

        if self.domain_axis != other.domain_axis:
            other = other.moveaxis(other.domain_axis, self.domain_axis)

        return self, other, labels

    def _expand_dims(self, axis):
        """
        """

        assert self.ndim == 1, \
            'can only expand dimension of one dimensional signal.'

        if axis == 0:
            domain_axis = 1
        elif axis == 1:
            domain_axis = 0
        else:
            raise DreyeError('can only expand dimension for axis 0 or 1.')

        values = np.expand_dims(self.magnitude, axis)

        self = self.copy()
        self._values = values
        self._labels = (self.labels,)
        self._domain_axis = domain_axis
        return self

    @abstractmethod
    def moveaxis(self, source, destination):
        pass

    def _factor_function(self, other, operation):
        """operator function for multiplications and divisions
        """

        self, other, labels = self._instance_handler(other)

        return AbstractDomain._factor_function(
            self,
            other,
            operation,
            labels=labels
        )

    def _sum_function(self, other, operation):
        """operator function for additions and subtractions
        """

        self, other, labels = self._instance_handler(other)

        return AbstractDomain._sum_function(
            self,
            other,
            operation,
            labels=labels)

    def _simple_function(self, other, operation):
        """operator function for opeartions without unit handling.
        That is other must have unit None
        """

        self, other, labels = self._instance_handler(other)

        return AbstractDomain._simple_function(
            self,
            other,
            operation,
            labels=labels
        )

    def _single_function(self, operation):
        """
        """

        return super()._simple_function(operation)

    def _flip_axes_assignment(self):
        """only used internally
        """
        self._domain_axis = self.other_axis

    @abstractmethod
    def concat_labels(self, labels, left=False):
        """concatenating labels
        """
