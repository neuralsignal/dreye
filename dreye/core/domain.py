"""Domain class
"""

import warnings

import numpy as np

from dreye.utilities import (
    is_listlike, asarray, array_domain, is_uniform,
    optional_to, is_numeric, arange, get_value,
    is_integer, has_units
)
from dreye.err import DreyeError
from dreye.constants import DEFAULT_FLOAT_DTYPE
from dreye.core.abstract import _UnitArray


class Domain(_UnitArray):
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
        The interval between values in Domain. If interval is not uniform,
        will return a list of interval values of length n-1.
    units: str, optional
        Units attatched to the values in Domain.
    values : array-like or str, optional
        The numpy array multiplied by the units (quantity instance).

    Examples
    --------
    >>> Domain(0, 1, 0.1, 's')
    Domain(start=0, end=1, interval=0.1, units=second)
    """

    _convert_attributes = (
        'start', 'end', 'interval'
    )
    _init_args = ('attrs', 'contexts', 'name', 'interval_')

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
        contexts=None,
        attrs=None,
        name=None,
        interval_=None
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
            contexts=contexts,
            attrs=attrs,
            name=name,
            start=start,
            end=end,
            interval=interval,
            interval_=interval_
        )

    @property
    def interval_(self):
        """
        Used Internally.

        Backup interval used in the case when values is of size 1.

        See _test_and_assign_values.
        """
        # always ensures that it is a float
        if hasattr(self, '_interval'):
            return np.mean(self._interval)
        return self._interval_

    @property
    def ndim(self):
        """
        Dimensionality of domain is always 1.
        """
        return 1

    def _test_and_assign_values(self, values, kwargs):
        if values is None:
            # this is the case when values was not passed during __init__
            start = kwargs.get('start', None)
            end = kwargs.get('end', None)
            interval = kwargs.get('interval', None)

            if start is None or end is None or interval is None:
                raise DreyeError("Unable to create Domain, define either "
                                 "a range of values or pass start, end, "
                                 "and interval.")

            start = optional_to(start, self.units, *self.contexts)
            end = optional_to(end, self.units, *self.contexts)
            interval = optional_to(interval, self.units, *self.contexts)

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
                    interval = self.interval_
                assert is_numeric(interval), \
                    "Need to supply numeric interval if domain is of size 1."
                interval = optional_to(interval, self.units, *self.contexts)
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
        """create values from start, end, and interval
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
        Should just return equalized other_magnitude or NotImplemented
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
        Returns start of Domain.
        """
        return self._start

    @property
    def end(self):
        """
        Returns the end of Domain.
        """
        return self._end

    @property
    def interval(self):
        """
        Returns the Domain interval.
        """
        return self._interval

    @property
    def is_uniform(self):
        """
        Check if distribution is uniform.
        """
        # if np.nan then True
        return is_numeric(self.interval)

    @property
    def has_interval(self):
        """
        Does the domain have a defined interval
        """
        return np.all(np.isfinite(self.interval))

    @property
    def is_descending(self):
        """
        Is the domain in descending order
        """
        return np.all(self.interval < 0)
        # return np.all(np.diff(self.magnitude) < 0)

    @property
    def is_sorted(self):
        """
        Is the domain sorted
        """
        return True
        # ascending = np.diff(self.magnitude) > 0
        # return np.all(ascending) or not np.any(ascending)

    def __str__(self):
        """
        Returns a str representation of the domain.
        """

        return (
            "Domain(start={0}, end={1}, interval={2}, units={3})"
        ).format(self.start, self.end, self.interval, self.units)

    def enforce_uniformity(self, method=np.mean, on_gradient=False):
        """
        Returns the domain with a uniform interval, calculated from the average
        of all original interval values.
        """

        if self.is_uniform:
            return self

        if on_gradient:
            value = method(self.gradient.magnitude)
        else:
            value = method(self.interval)

        # change values and interval
        return self._class_new_instance(
            start=self.start,
            end=self.end,
            interval=value,
            units=self.units,
            **self.init_kwargs
        )

    def equalize_domains(self, other):
        """
        Equalizes the range and the interval between two domains. Domains must
        be uniform for this to succeed. Takes the most common denominator for
        the domain range (largest Start value and smallest End value), and
        takes the largest interval from the original two domains.
        """

        # handles only one-dimensional uniform arrays.
        if not self.is_uniform:
            self = self.enforce_uniformity()
            warnings.warn("Enforcing uniformity in self.")

        self = (self[::-1] if self.is_descending else self)
        if self == other:
            return self.copy()

        if isinstance(other, Domain):
            if not other.is_uniform:
                warnings.warn("Enforcing uniformity in other.")
                other = other.enforce_uniformity()
            other = (other[::-1] if other.is_descending else other)
            other = other.to(self.units)
            start = other.start
            end = other.end
            interval = other.interval
        elif is_listlike(other):
            other_magnitude = optional_to(other, self.units, *self.contexts)

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
            **self.init_kwargs
        )

        if not domain.is_uniform:
            warnings.warn("Equalized domain is not uniform.", RuntimeWarning)

        return domain

    def append(self, domain, left=False):
        """
        Append a domain to domain.
        """

        # Works with reverse
        if isinstance(domain, Domain):
            domain = domain.to(self.units)
            domain = domain.magnitude
        elif is_listlike(domain):
            domain = optional_to(domain, self.units, *self.contexts)
            assert domain.ndim == 1, "Domain must be one-dimensional."
        elif is_numeric(domain):
            domain = optional_to(domain, self.units, *self.contexts)
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
            **self.init_kwargs
        )

    def extend(self, length, left=False):
        """
        Extend Domain by a certain number of index length.
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
                self.end + self.interval * (idx+1)
                for idx in range(length)
            ])
        # add domain
        return self.append(add_domain, left=left)

    @property
    def gradient(self):
        """
        Calculates gradient between points (difference).
        """

        return np.gradient(self.magnitude) * self.units

    @property
    def span(self):
        """
        Span of the domain
        """
        return np.max(self.magnitude) - np.min(self.magnitude)

    @property
    def boundaries(self):
        """
        Tuple of start and end.
        """
        return (np.min(self.magnitude), np.max(self.magnitude))
        # return (self.start, self.end)
