"""Domain class
"""

import warnings

import numpy as np
import pandas as pd

from dreye.utilities import (
    is_listlike, asarray, array_domain, is_uniform,
    optional_to, is_numeric, arange, get_value,
    is_integer, has_units
)
from dreye.utilities.abstract import inherit_docstrings
from dreye.err import DreyeError
from dreye.constants import DEFAULT_FLOAT_DTYPE
from dreye.core.abstract import _UnitArray


@inherit_docstrings
class Domain(_UnitArray):
    """
    Ascending or descending range of values with particular units.

    Parameters
    ----------
    start : numeric or array-like
        Start of domain if numeric or all values of domain in ascending or
        descending order if array-like.
    end : numeric, optional
        End of domain.
    interval : numeric or array-like, optional
        The interval between values in domain. If interval is not uniform,
        will return a list of interval values of length n-1.
    units: str, optional
        Units associated with the domain.
    name : str
        Name for the domain object.
    attrs : dict
        User-defined dictionary storage.

    Notes
    -----
    The :obj:`~Domain` object is used for various signal-type classes to specify
    the domain range of a signal. For example, for a spectral distribution
    of light the domain range will be various wavelengths in nanometers.

    Examples
    --------
    >>> Domain(0, 1, 0.1, units='s')
    Domain(start=0.0, end=1.0, interval=0.1, units=second)
    >>> Domain([0, 1, 2, 3], units='V')
    Domain(start=0.0, end=3.0, interval=1.0, units=volts)
    """

    _convert_attributes = (
        'start', 'end', 'interval'
    )
    _init_args = ('attrs', 'name', 'interval_')
    _deprecated_kws = {
        **_UnitArray.deprecated_kws,
        'interval_': '_interval_'
    }

    @property
    def _class_new_instance(self):
        return Domain

    def __init__(
        self,
        start=None,
        end=None,
        interval=None,
        *,
        units=None,
        values=None,
        attrs=None,
        name=None,
        _interval_=None
    ):
        if is_listlike(start) and values is None:
            values = start
            start = None
        if units is None:
            if has_units(values):
                units = values.units
            elif has_units(start):
                units = start.units
            elif has_units(end):
                units = end.units
            elif has_units(interval):
                units = interval.units

        super().__init__(
            values=values,
            units=units,
            attrs=attrs,
            name=name,
            start=start,
            end=end,
            interval=interval,
            _interval_=_interval_
        )

    @property
    def _backup_interval_(self):
        """
        This property is used internally.

        This backup interval is used in the case when values is of size 1,
        the property is used in the internal method `_test_and_assign_values`.
        """
        # always ensures that it is a float
        if hasattr(self, '_interval'):
            return np.mean(self._interval)
        return self._interval_

    @property
    def ndim(self):
        """
        Dimensionality of a domain is always 1.
        """
        return 1

    def to_index(self, name=None):
        """
        Return :obj:`~pandas.Index` instance of domain.

        Parameters
        ----------
        name : str, optional
            The name for instantiating :obj:`~pandas.Index`.

        Returns
        -------
        index : :obj:`~pandas.Index`
        """
        name = (self.name if name is None else name)
        return pd.Index(self.magnitude, name=name)

    def _test_and_assign_values(self, values, kwargs):
        """
        This assigns all the necessary attributes to the domain instance.
        """
        if values is None:
            # this is the case when values was not passed during __init__
            start = kwargs.get('start', None)
            end = kwargs.get('end', None)
            interval = kwargs.get('interval', None)

            if start is None or end is None or interval is None:
                raise DreyeError("Unable to create Domain, define either "
                                 "a range of values or pass start, end, "
                                 "and interval.")

            start = optional_to(start, self.units)
            end = optional_to(end, self.units)
            interval = optional_to(interval, self.units)

            start = DEFAULT_FLOAT_DTYPE(start)
            end = DEFAULT_FLOAT_DTYPE(end)
            reverse = (start > end) or np.all(interval < 0)
            # always positive interval and ascending
            if reverse:
                start, end = end, start
            interval = np.abs(interval)
        else:
            values = asarray(values, DEFAULT_FLOAT_DTYPE)
            assert values.ndim == 1, "Array must be 1-dimensional"
            reverse = np.all(np.sort(values) == values[::-1])

            # check if domain is sorted
            if not reverse and not np.all(np.sort(values) == values):
                raise DreyeError("Values for domain initialization "
                                 f"must be sorted: {values}.")

            if values.size == 1:
                # start and end equal
                start = values[0]
                end = values[0]
                interval = kwargs.get('interval', None)
                # take backup interval
                if interval is None:
                    interval = self._backup_interval_
                assert is_numeric(interval), \
                    "Need to supply numeric interval if domain is of size 1."
                interval = optional_to(interval, self.units)
            else:
                # returns start and end in ascending order
                start, end, interval = array_domain(
                    values, uniform=is_uniform(values)
                )

            # always positive interval
            interval = np.abs(interval)

        if reverse:
            self._start, self._end = end, start
        else:
            self._start, self._end = start, end

        self._values, self._interval = self._create_values(
            start, end, interval, reverse
        )

    @staticmethod
    def _create_values(start, end, interval, reverse):
        """
        Create values from start, end, and interval.
        """
        if is_listlike(interval):
            # TODO: dealing with tolerance
            interval = asarray(interval)
            interval_diff = (end - start) - np.sum(interval)
            if interval_diff > 0:
                # TODO: raise warning
                # raise ValueError('Intervals given smaller than range')
                interval = np.append(interval, interval_diff)
            elif interval_diff < 0:
                raise DreyeError("Sum of intervals are larger "
                                 "than span of domain.")

            values = np.append(start, start + np.cumsum(interval))
            values = values.astype(DEFAULT_FLOAT_DTYPE)
            interval = interval.astype(DEFAULT_FLOAT_DTYPE)

            if values.size != np.unique(values).size:
                raise DreyeError('values are non-unique: {0}'.format(values))

        elif is_numeric(interval):
            if end == start:
                values = np.array([start]).astype(DEFAULT_FLOAT_DTYPE)
            elif interval > (end - start):
                raise DreyeError(f"Interval Attribute value '{interval}' "
                                 f"bigger than span '{start-end}'.")
            else:
                values, interval = arange(
                    start, end, interval, dtype=DEFAULT_FLOAT_DTYPE)
            interval = DEFAULT_FLOAT_DTYPE(interval)

        else:
            raise DreyeError("Interval not correct type, "
                             f"but type '{type(interval)}'.")

        if reverse:
            values = values[::-1]
            interval = -interval
        return values, interval

    def _equalize(self, other):
        """
        Equalize other in order to do mathematical operations.
        """
        if is_numeric(other):
            return get_value(other), self
        elif isinstance(other, _UnitArray):
            if isinstance(other, Domain):
                return other.magnitude, self
            else:
                return NotImplemented, self
        elif is_listlike(other):
            other = asarray(other)
            if other.ndim > 1:
                return NotImplemented, self
            else:
                return other, self
        else:
            return NotImplemented, self

    @property
    def start(self):
        """
        Returns start of Domain without units.
        """
        return self._start

    @property
    def end(self):
        """
        Returns the end of Domain without units.
        """
        return self._end

    @property
    def interval(self):
        """
        Returns the Domain interval without units.
        """
        return self._interval

    @property
    def is_uniform(self):
        """
        Domain has a uniform interval.

        An interval is uniform, if it is numeric instead of array-like.
        """
        # if np.nan then True
        return is_numeric(self.interval)

    @property
    def has_interval(self):
        """
        Domain has a defined interval.

        This property should always be True.
        """
        return np.all(np.isfinite(self.interval))

    @property
    def is_descending(self):
        """
        Is the domain in descending order.
        """
        return np.all(self.interval < 0)

    @property
    def is_sorted(self):
        """
        Domain is sorted.

        This property should always be True.
        """
        ascending = np.diff(self.magnitude) > 0
        return np.all(ascending) or not np.any(ascending)

    def __str__(self):
        """
        Returns a str representation of the domain.
        """
        return (
            "Domain(start={0}, end={1}, interval={2}, units={3})"
        ).format(self.start, self.end, self.interval, self.units)

    def enforce_uniformity(self):
        """
        Returns a new domain with uniform intervals if `self` does not have
        uniform intervals.

        This method takes the mean of the non-uniform intervals
        to enforce uniformity.

        Returns
        -------
        domain : :obj:`~Domain`
            A new domain with uniform intervals, or a copy of self if self
            is already uniform.
        """

        if self.is_uniform:
            return self.copy()

        # change values and interval
        return self._class_new_instance(
            start=self.start,
            end=self.end,
            interval=np.mean(self.interval),
            units=self.units,
            **self._init_kwargs
        )

    def equalize_domains(self, other):
        """
        Equalizes the range and the interval between two domains.

        Parameters
        ----------
        other : :obj:`~Domain` object or array-like
            Domain object to equalize to self.

        Returns
        -------
        domain : :obj:`~Domain`
            A new domain corresponding the "most common denominator" between
            self and other.

        Notes
        -----
        If domains do not have uniform intervals, uniformity will be enforced
        using :obj:`~Domain.enforce_uniformity`.

        The "most common denominator" domain between self and other is
        chosen as follows:

        * largest `start` value
        * smalled `end` value
        * largest `interval` value
        """

        # handles only one-dimensional uniform arrays.
        if not self.is_uniform:
            warnings.warn("Enforcing uniformity in self.", RuntimeWarning)
            self = self.enforce_uniformity()

        self = (self[::-1] if self.is_descending else self)
        if self == other:
            return self.copy()

        if isinstance(other, Domain):
            if not other.is_uniform:
                warnings.warn("Enforcing uniformity in other.", RuntimeWarning)
                other = other.enforce_uniformity()
            other = (other[::-1] if other.is_descending else other)
            other = other.to(self.units)
            start = other.start
            end = other.end
            interval = other.interval
        elif is_listlike(other):
            other_magnitude = optional_to(other, self.units)

            if other_magnitude.ndim != 1:
                raise DreyeError("Other array is not one-dimensional.")

            if not is_uniform(other_magnitude):
                raise DreyeError("Other array does not "
                                 "have a unique interval!")

            start, end, interval = array_domain(
                other_magnitude, uniform=True
            )
        else:
            raise DreyeError("Other is of wrong type "
                             f"'{type(other)}' to equalize domains.")

        if (start > self.end) or (end < self.start):
            raise DreyeError("Cannot equalize domains with boundaries "
                             f"({start}, {end}) and boundaries "
                             f"({self.start}, {self.end}).")

        if start < self.start:
            start = self.start
        if end > self.end:
            end = self.end
        if interval < self.interval:
            interval = self.interval

        # create domain class
        domain = self._class_new_instance(
            start=start,
            end=end,
            interval=interval,
            units=self.units,
            **self._init_kwargs
        )

        if not domain.is_uniform:
            warnings.warn("Equalized domain is not uniform.", RuntimeWarning)

        return domain

    def append(self, domain, left=False):
        """
        Append to domain.

        Parameters
        ----------
        domain : :obj:`~Domain` or array-like
            Domain values to append to self.
        left : bool, optional
            Whether to append to the left side of the domain.
            Defaults to False.

        Returns
        -------
        domain : :obj:`~Domain`
            Returns appended domain.

        Examples
        --------
        >>> domain1 = Domain(0, 1, 0.1, units='s')
        >>> domain2 = Domain(1, 2, 0.1, units='s')
        >>> domain1.append(domain2)
        Domain(start=0.0, end=2.0, interval=0.1, units=second)
        """

        # Works with reverse
        if isinstance(domain, Domain):
            domain = domain.to(self.units)
            domain = domain.magnitude
        elif is_listlike(domain):
            domain = optional_to(domain, self.units)
            assert domain.ndim == 1, "Domain must be one-dimensional."
        elif is_numeric(domain):
            domain = optional_to(domain, self.units)
            domain = np.array([domain])
        else:
            raise DreyeError(f"Appending type '{type(domain)}' impossible.")

        if left:
            values = np.concatenate([domain, self.magnitude])
        else:
            values = np.concatenate([self.magnitude, domain])

        return self._class_new_instance(
            values=values,
            units=self.units,
            **self._init_kwargs
        )

    def extend(self, length, left=False):
        """
        Extend domain by a certain length.

        Parameters
        ----------
        length : int
            Number of indices to extend by.
        left : bool, optional
            Whether to append to the left side of the domain.
            Defaults to False.

        Returns
        -------
        domain : :obj:`~Domain`
            Returns extended domain.

        Examples
        --------
        >>> domain = Domain(0, 1, 0.1, units='s')
        >>> domain.extend(5)
        Domain(start=0.0, end=1.5, interval=0.1, units=second)
        """
        # works with reversed/descending values
        assert is_integer(length), "Length must be integer type."
        assert self.is_uniform, \
            "Domain must be uniform to use method 'extend'."

        length = get_value(length)

        if left:
            add_domain = np.array([
                self.start - (length - idx) * self.interval
                for idx in range(length)
            ])
        else:
            add_domain = np.array([
                self.end + self.interval * (idx + 1)
                for idx in range(length)
            ])
        # add domain
        return self.append(add_domain, left=left)

    @property
    def gradient(self):
        """
        Calculates gradient between points.

        Returns
        -------
        gradient : :obj:`~pint.Quantity`
            The gradient of the domain in units of the domain.

        See Also
        --------
        numpy.gradient
        """

        return np.gradient(self.magnitude) * self.units

    @property
    def span(self):
        """
        Span of the domain (max-min).

        Returns
        -------
        span : float
            Span of the domain
        """
        return np.max(self.magnitude) - np.min(self.magnitude)

    @property
    def boundaries(self):
        """
        Tuple of minimum and maximum value.

        Returns
        -------
        boundaries : two-tuple of floats
            Minimum and maximum of domain.
        """
        return (np.min(self.magnitude), np.max(self.magnitude))
