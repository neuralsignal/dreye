"""Domain class
"""

import warnings

import numpy as np
from pint import DimensionalityError

from dreye.utilities import (
    is_uniform, array_domain, dissect_units,
    is_numeric, array_equal, is_arraylike,
    asarray
)
from dreye.err import DreyeError
from dreye.io import read_json, write_json
from dreye.constants import DEFAULT_FLOAT_DTYPE
from dreye.core.abstract import AbstractDomain
from dreye.core.unpack_mixin import UnpackDomainMixin


class Domain(AbstractDomain, UnpackDomainMixin):
    """
    Defines the base class for domains. Includes a range of values sorted from
    min to max, with some units attatched.

    Parameters
    ----------
    start : numeric, optional
        Start of Domain.
    end : numeric, optional
        End of Domain.
    interval : numeric or array-like, optional
        The interval between values in Domain. If interval is not uniform, will return a
        list of interval values of length n-1.
    values : array-like or str, optional
        The numpy array multiplied by the units (quantity instance).
    dtype : type, optional
        Data type of Domain.
    span:
        Span of Domain from start to end.
    gradient:
        Implements the numpy.gradient function. If the gradient is non uniform,
        the instantanous gradient will vary.
    magnitude:
        The numpy array without units attatched.
    units: str, optional
        Units attatched to the values in Domain.

    Attributes
    ----------
    start
    end
    interval
    boundaries
    values
    dtype
    span
    gradient
    magnitude
    units

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
    _unpack
    asarray

    Examples
    --------
    >>> Domain(0, 1, 0.1, 's')
    Domain(start=0, end=1, interval=0.1, units=second)
    """

    @property
    def _class_new_instance(self):
        return Domain

    def __init__(
        self,
        start=None,
        end=None,
        interval=None,
        units=None,
        values=None,
        dtype=DEFAULT_FLOAT_DTYPE,
        contexts=None,
    ):

        if isinstance(dtype, str):
            dtype = np.dtype(dtype).type

        values, start, end, interval, contexts, dtype, units = self._unpack(
            values=values,
            start=start,
            end=end,
            interval=interval,
            units=units,
            dtype=dtype,
            contexts=contexts,
        )

        self._values = values
        self._units = units
        self._start = start
        self._end = end
        self._contexts = contexts
        self._interval = interval
        self._dtype = dtype

    @property
    def dtype(self):
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
        self._values, self._interval = self._create_values(
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

        elif value < self._start and not self.is_uniform:
            raise DreyeError((
                'start attribute: {0} value '
                'below previous start value {1}'
            ).format(value, self._start))

        else:
            self._start = self.dtype(value)
            self._values, self._interval = self._create_values(
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

        elif value > self._end and not self.is_uniform:
            raise DreyeError((
                'end attribute: {0} value '
                'above previous end value {1}'
            ).format(value, self._end))

        else:
            self._end = self.dtype(value)
            self._values, self._interval = self._create_values(
                self.start, self._end, self.interval, self.dtype)

    @property
    def interval(self):
        """
        """

        return self._interval

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

            self._values, self._interval = self._create_values(
                self.start, self.end, value, self.dtype)

    @property
    def values(self):
        """
        """

        if self._values is None:
            self._values, self._interval = self._create_values(
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

        self._values, self._interval = self._create_values(
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

        if isinstance(other, Domain):

            return (
                array_equal(self.magnitude, other.magnitude)
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
        ).format(self.start, self.end, self.interval, self.units, self.dtype)

    def enforce_uniformity(self, method=np.mean, on_gradient=True, copy=True):
        """
        Returns the domain with a uniform interval, calculated from the average
        of all original interval values.
        """

        if copy:
            self = self.copy()

        if on_gradient:
            self.interval = method(self.gradient.magnitude)
        else:
            self.interval = method(self.interval)

        return self

    def equalize_domains(self, other, interval=None, start=None, end=None):
        """
        Equalizes the range and the interval between two domains. Domains must
        be uniform for this to succeed. Takes the most common denominator for 
        the domain range (largest Start value and smallest End value), and takes
        the largest interval from the original two domains.
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
            raise DreyeError(
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
    def load(cls, filename, dtype=DEFAULT_FLOAT_DTYPE):
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
    def from_dict(cls, data, dtype=DEFAULT_FLOAT_DTYPE):
        """build class from dictionary
        """

        return cls(
            data['values'],
            units=data['units'],
            dtype=dtype
        )

    def to_dict(self):
        """
        Return instance as a dictionary.

        Returns
        -------

        Examples
        --------
        """

        return {
            'values': self.magnitude.tolist(),
            'units': str(self.units)
        }

    def save(self, filename):
        """
        Save domain instance.

        Parameters
        ----------
        filename : str
            location of JSON file.
        """

        data = self.to_dict()

        write_json(filename, data)

    def append(self, domain, left=False, copy=True):
        """
        """

        if isinstance(domain, AbstractDomain):
            domain = domain.to(self.units)
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
