"""Domain class
"""

from numbers import Number
import warnings

import numpy as np
from pint import DimensionalityError

from dreye.utilities import (
    is_uniform, array_domain, dissect_units,
    is_numeric, array_equal, has_units, is_arraylike,
    asarray
)
from dreye.io import read_json, write_json
from dreye.constants import DEFAULT_FLOAT_DTYPE
from dreye.core.abstract import AbstractDomain
from dreye.core.mixin import UnpackDomainMixin, CheckClippingValueMixin


class Domain(AbstractDomain, UnpackDomainMixin):
    """
    Defines the base class for domains.

    Parameters
    ----------
    start : numeric, optional
    end : numeric, optional
    interval : numeric or array-like, optional
    units : str, optional
    values : array-like or str, optional
    dtype : type, optional

    Attributes
    ----------
    start
    end
    interval
    boundaries
    values
    dtype
    span

    Methods
    -------
    __str__
    __repr__
    __iter__
    __contains__
    __len__
    __getitem__
    __getattr__
    [arithmetical operations]
    load
    save
    unpack
    asarray

    Examples
    --------
    >>> Domain(0, 1, 0.1, 's')
    Domain(start=0, end=1, interval=0.1, units=second)
    """

    _required = ('start', 'end', 'interval', 'units', 'dtype', 'contexts')
    _all = _required + ('values', )

    def __init__(self,
                 start=None,
                 end=None,
                 interval=None,
                 units=None,
                 values=None,
                 dtype=DEFAULT_FLOAT_DTYPE,
                 contexts=None,
                 **kwargs):

        if isinstance(dtype, str):
            dtype = np.dtype(dtype).type

        values, kwargs = self.unpack(values=values,
                                     start=start,
                                     end=end,
                                     interval=interval,
                                     units=units,
                                     dtype=dtype,
                                     contexts=contexts,
                                     **kwargs)

        self._values = values
        self._units = kwargs.pop('units')

        for key, value in kwargs.items():
            if key in self.convert_attributes:
                if value is None:
                    pass
                elif has_units(value):
                    value = value.to(self.units)
                else:
                    value = value * self.units
            setattr(self, '_' + key, value)

    @property
    def dtype(self):
        """
        """

        return self._dtype

    @dtype.setter
    def dtype(self, value):
        """
        """

        if value is None:
            pass

        elif isinstance(value, str):
            value = np.dtype(value).type

        elif not hasattr(value, '__call__'):
            raise AttributeError('dtype attribute: must be callable.')

        self._dtype = value
        self._start = self._dtype(self.start)
        self._end = self._dtype(self.end)
        self._values, self._interval = self.create_values(
            self.start, self.end, self.interval, self._dtype)

    @property
    def start(self):
        """
        """

        return self._start

    @start.setter
    def start(self, value):
        """
        """

        value, units = dissect_units(value)

        if units is not None and units != self.units:
            raise DimensionalityError(units, self.units)

        if value is None:
            pass

        elif not is_numeric(value):
            raise TypeError(
                'start attribute: {0} is not numeric'.format(value))

        # elif value < self._start:
        #     raise ValueError((
        #         'start attribute: {0} value '
        #         'below previous start value {1}'
        #     ).format(value, self._start))

        else:
            self._start = self.dtype(value)
            self._values, self._interval = self.create_values(
                self._start, self.end, self.interval, self.dtype)

    @property
    def end(self):
        """
        """

        return self._end

    @end.setter
    def end(self, value):
        """
        """

        value, units = dissect_units(value)

        if units is not None and units != self.units:
            raise DimensionalityError(units, self.units)

        if value is None:
            pass

        elif not is_numeric(value):
            raise TypeError('end attribute: {0} is not numeric'.format(value))

        # elif value > self._end:
        #     raise ValueError((
        #         'end attribute: {0} value '
        #         'above previous end value {1}'
        #     ).format(value, self._end))

        else:
            self._end = self.dtype(value)
            self._values, self._interval = self.create_values(
                self.start, self._end, self.interval, self.dtype)

    @property
    def interval(self):
        """
        """

        return self._interval  # * self.units

    @interval.setter
    def interval(self, value):
        """
        """

        value, units = dissect_units(value)

        if units is not None and units != self.units:
            raise DimensionalityError(units, self.units)

        if value is None:
            pass

        else:

            self._values, self._interval = self.create_values(
                self.start, self.end, value, self.dtype)

    @property
    def values(self):
        """
        """

        if self._values is None:
            self._values, self._interval = self.create_values(
                self.start, self.end, self.interval, self.dtype)

        return self._values * self.units

    @values.setter
    def values(self, values):
        """
        """

        values, units = dissect_units(values)
        values = asarray(values)

        if units is not None and units != self.units:
            raise DimensionalityError(units, self.units)

        start, end, interval = array_domain(values, uniform=is_uniform(values))

        self._start = start
        self._end = end

        self._values, self._interval = self.create_values(
            start, end, interval,
            np.dtype(values.dtype).type)

    @property
    def boundaries(self):
        """
        """

        return (self.start, self.end)

    @property
    def span(self):
        """
        """

        return self.end - self.start

    @property
    def is_uniform(self):
        """
        """

        return is_numeric(self.interval)

    def __eq__(self, other):
        """
        Returns the Domain equality with given other Domain.

        Parameters
        ----------
        other : Domain
            Domain to compare for equality.

        Returns
        -------
        bool
            Domain equality.

        Examples
        --------
        >>> Domain(0, 20, 0.1) == Domain(0, 20, 0.1)
        True
        >>> Domain(0, 20, 0.1) == Domain(0, 20, 0.1, 'seconds')
        False
        >>> Domain(0, 20, 0.1, 'seconds') == Domain(0, 20, 0.2, 'seconds')
        False
        """

        if isinstance(other, self.__class__):

            return (
                array_equal(self.asarray(), other.asarray())
                and (self.units == other.units)
            )

        else:
            return False

    def __str__(self):
        """
        Returns a str representation of the domain.
        """

        return (
            "Domain(start={0}, end={1}, interval={2}, units={3}, dtype={4})"
        ).format(*(self.get_standard(key) for key in self._required))

    def enforce_uniformity(self, method=np.mean, on_gradient=True, copy=True):
        """
        """

        if copy:
            self = self.copy()

        if on_gradient:
            self.interval = method(self.gradient.magnitude)
        else:
            self.interval = method(self.interval)

        return self

    def equalize_domains(self, other, interval=None, start=None, end=None):
        """return an equalized domain
        """

        if self == other:

            return self.copy()

        domain_class = self.__class__

        assert issubclass(
            other.__class__,
            domain_class), ("Both domains must be the same class")

        # test if units are equal
        if self.units != other.units:
            try:
                other = other.copy()
                other.units = self.units
            except Exception:
                raise DimensionalityError(
                    self.units,
                    other.units,
                    extra_msg=(' Domain units must equal for'
                               ' equalization operation'))
        # check that both domains are uniform
        if not (self.is_uniform and other.is_uniform) and interval is None:
            raise TypeError(
                'Domains must be uniform for operations on signal classes, '
                'if domains are not equal.')

        # Choose the bigger interval, bigger start, and smaller end
        # This works only if domains are uniform
        if interval is None:
            interval = np.max([self.interval, other.interval])
        if start is None:
            start = np.max([self.start, other.start])
        if end is None:
            end = np.min([self.end, other.end])

        # ignores dtype for now
        domain = domain_class(
            start=start,
            end=end,
            interval=interval,
            units=self.units,
        )

        if not domain.is_uniform:
            warnings.warn('Equalized domain is not uniform.', RuntimeWarning)

        return domain

    @classmethod
    def load(cls, filename, dtype=None):
        """
        Load domain instance.

        Parameters
        ----------
        filename : str
            location of JSON file
        dtype : type, optional
        """

        data = read_json(filename)

        return cls.from_dict(data, dtype)

    @classmethod
    def from_dict(cls, data, dtype=None):
        """build class from dictionary
        """

        if dtype is not None:
            data['dtype'] = dtype

        set_keys = set(data.keys())
        set_required = set(cls._required)

        assert not set_required - set_keys, (
            "data in JSON does not contain all required keys:"
            " must contain {0}, but only contains {1}").format(
                set_required, set_keys)

        return cls(**data)

    def to_dict(self, json_compatible=True):
        """
        Return instance as a dictionary.

        Parameters
        ----------
        json_compatible : bool, optional
            If True, will return dictionary that is json compatible and only
            contains these keys: {0}. The default is False.

        Returns
        -------

        Examples
        --------
        """.format(self._required)

        if json_compatible:
            return {key: self.get_standard(key) for key in self._required}

        else:
            return {key: getattr(self, key) for key in self._all}

    def get_standard(self, name):
        """
        Return attribute as standard python type.

        Parameters
        ----------
        name : str
            name of attribute.
        """

        attribute = getattr(self, name)
        return self.to_standard(attribute)

    @staticmethod
    def to_standard(attribute):
        """
        Return attribute as standard python type (i.e. without units and
        not as a ndarray).

        Parameters
        ----------
        attribute : quantity or unit
            Quantity or unit type to convert.
        """

        if hasattr(attribute, 'magnitude'):
            attribute = attribute.magnitude

        if isinstance(attribute, np.ndarray):
            return attribute.tolist()
        elif attribute is None:
            return attribute
        elif isinstance(attribute, (Number, list, tuple, dict)):
            return attribute
        elif isinstance(attribute, type):
            return attribute.__name__
        else:
            return str(attribute)

    def to_tuple(self):
        """
        Return {0} as a tuple and standard pythonic types.
        """.format(self._required)

        return (self.get_standard(key) for key in self._required)

    def save(self, filename):
        """
        Save domain instance.

        Parameters
        ----------
        filename : str
            location of JSON file.
        """

        data = self.to_dict(True)

        write_json(filename, data)

    def append(self, domain, left=False, copy=True):
        """
        """

        if isinstance(domain, AbstractDomain):
            domain = domain.convert_to(self.units)
            domain = asarray(domain)
        elif is_arraylike(domain):
            domain = asarray(domain)
            assert domain.ndim == 1, "domain must be one-dimensional."
        elif is_numeric(domain):
            domain = asarray([domain])
        else:
            raise TypeError(f'appending of type: {type(domain)}.')

        if copy:
            self = self.copy()

        if left:
            values = np.concatenate([domain, self.magnitude])
        else:
            values = np.concatenate([self.magnitude, domain])

        self.values = values

        return self

    def extend(self, length, left=False, copy=True):
        """
        """

        assert isinstance(length, int)

        add_domain = self._add_length(length, left)

        return self.append(add_domain, left=left, copy=copy)

    def _add_length(self, length, left=False):
        """
        """

        assert self.is_uniform, "domain must be uniform"

        if left:
            return asarray([
                self.start - (length - idx) * self.interval
                for idx in range(length)
            ])
        else:
            return asarray([
                self.end + self.interval * (idx+1)
                for idx in range(length)
            ])

    @property
    def gradient(self):
        """
        """

        return np.gradient(self.magnitude) * self.units


class ClippedDomain(Domain, CheckClippingValueMixin):
    """
    """

    _required = Domain._required + (
        'domain_min', 'domain_max')
    convert_attributes = Domain.convert_attributes + (
        'domain_min', 'domain_max')

    def __init__(self,
                 start=None,
                 end=None,
                 interval=None,
                 units=None,
                 values=None,
                 dtype=DEFAULT_FLOAT_DTYPE,
                 contexts=None,
                 domain_min=None,
                 domain_max=None):

        super().__init__(
            start=start,
            end=end,
            interval=interval,
            units=units,
            values=values,
            dtype=dtype,
            contexts=contexts,
            domain_min=domain_min,
            domain_max=domain_max
        )

        self._check_clip_value(self.domain_min)
        self._check_clip_value(self.domain_max)

    @property
    def domain_min(self):

        return self._domain_min

    @property
    def domain_max(self):

        return self._domain_max
