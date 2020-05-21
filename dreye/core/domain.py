"""Domain class
"""

import warnings

import numpy as np

from dreye.utilities import (
    is_uniform, array_domain, dissect_units,
    is_numeric, array_equal, is_arraylike,
    asarray, _convert_get_val_opt
)
from dreye.err import DreyeError, DreyeUnitError
from dreye.io import read_json, write_json
from dreye.constants import DEFAULT_FLOAT_DTYPE, ureg
from dreye.core.abstract import _UnitArray
from dreye.core.unpack_mixin import _UnpackDomain


class Domain(_UnitArray, _UnpackDomain):
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
    values : array-like or str, optional
        The numpy array multiplied by the units (quantity instance).
    span:
        Span of Domain from start to end.
    gradient:
        Implements the numpy.gradient function. If the gradient is non uniform,
        the instantanous gradient will vary.
    magnitude:
        The numpy array without units attatched.
    units: str, optional
        Units attatched to the values in Domain.

    Examples
    --------
    >>> Domain(0, 1, 0.1, 's')
    Domain(start=0, end=1, interval=0.1, units=second)
    """

    _init_args = (
        'contexts', 'attrs', 'name'
    )
    _convert_attributes = ('start', 'end', 'interval')

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
        name=None,
        attrs=None
    ):

        container = self._unpack(
            values=values,
            start=start,
            end=end,
            interval=interval,
            units=units,
            contexts=contexts,
            name=name,
            attrs=attrs
        )

        for key, value in container.items():
            setattr(self, '_'+key, value)

    @property
    def start(self):
        """
        Returns start of Domain.
        """
        return self._start

    # @start.setter
    # def start(self, value):
    #     """
    #     Change start value of domain.
    #     """
    #
    #     value = _convert_get_val_opt(value, units=self.units)
    #
    #     if value is None:
    #         pass
    #
    #     elif not is_numeric(value):
    #         raise DreyeError("Start attribute is not a numeric "
    #                          f"type, but type '{type(value)}'.")
    #
    #     elif value < self.start and not self.is_uniform:
    #         raise DreyeError(f"Start attribute '{value}' is below "
    #                          f"previous start value '{self.start}' "
    #                          "and domain is not uniform.")
    #
    #     else:
    #         self._start = DEFAULT_FLOAT_DTYPE(value)
    #         self._values, self._interval = self._create_values(
    #             self.start, self.end, self.interval)

    @property
    def end(self):
        """
        Returns the end of Domain.
        """
        return self._end

    # @end.setter
    # def end(self, value):
    #     """
    #     Change end value of domain.
    #     """
    #     value = _convert_get_val_opt(value, units=self.units)
    #
    #     if value is None:
    #         pass
    #
    #     elif not is_numeric(value):
    #         raise DreyeError("End attribute is not a numeric "
    #                          f"type, but type '{type(value)}'.")
    #
    #     elif value > self.end and not self.is_uniform:
    #         raise DreyeError(f"End attribute '{value}' is below "
    #                          f"previous start value '{self.end}' "
    #                          "and domain is not uniform.")
    #
    #     else:
    #         self._end = DEFAULT_FLOAT_DTYPE(value)
    #         self._values, self._interval = self._create_values(
    #             self.start, self.end, self.interval)

    @property
    def interval(self):
        """
        Returns the Domain interval.
        """
        return self._interval

    # @interval.setter
    # def interval(self, value):
    #     """
    #     Change interval value of domain.
    #     """
    #     value = _convert_get_val_opt(value, units=self.units)
    #
    #     if value is None:
    #         pass
    #     else:
    #         self._values, self._interval = self._create_values(
    #             self.start, self.end, value)

    def _test_and_assign_new_values(self, values):
        """
        assign start, end, interval to new values.
        """
        start, end, interval = array_domain(values, uniform=is_uniform(values))
        self._start = start
        self._end = end
        self._values, self._interval = self._create_values(
            start, end, interval)

    @property
    def boundaries(self):
        """
        Tuple of start and end
        """
        return (self.start, self.end)

    @property
    def span(self):
        """
        Returns the span of Domain from start to end.
        """
        return self.end - self.start

    @property
    def is_uniform(self):
        """
        Check if distribution is uniform.
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
            "Domain(start={0}, end={1}, interval={2}, units={3})"
        ).format(self.start, self.end, self.interval, self.units)

    def enforce_uniformity(self, method=np.mean, on_gradient=True):
        """
        Returns the domain with a uniform interval, calculated from the average
        of all original interval values.
        """

        self = self.copy()

        if on_gradient:
            value = method(self.gradient.magnitude)
        else:
            value = method(self.interval)

        self._values, self._interval = self._create_values(
            self.start, self.end, value)

        return self

    def equalize_domains(self, other, interval=None, start=None, end=None):
        """
        Equalizes the range and the interval between two domains. Domains must
        be uniform for this to succeed. Takes the most common denominator for
        the domain range (largest Start value and smallest End value), and
        takes the largest interval from the original two domains.
        """

        if self == other:

            return self.copy()

        domain_class = type(self)

        assert issubclass(
            other.__class__,
            domain_class), ("Both domains must be the same class")

        # test if units are equal
        if self.units != other.units:
            try:
                other = other.copy()
                other.units = self.units
            except Exception:
                raise DreyeUnitError(
                    self.units,
                    other.units,
                    self.units.dimensionality,
                    other.units.dimensionality,
                    '. Domain units must equal for equalization operation'
                )
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

        # create domain class
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
    def load(cls, filename):
        """
        Load domain instance.

        Parameters
        ----------
        filename : str
            location of JSON file
        """

        return read_json(filename)

    @classmethod
    def from_dict(cls, data):
        """build class from dictionary
        """

        return cls(
            data['values'],
            units=data['units'],
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

        write_json(filename, self)

    def append(self, domain, left=False, copy=True):
        """
        Append domains.
        """
        if isinstance(domain, Domain):
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

    @property
    def gradient(self):
        """
        Returns the calculated gradient.
        """

        return np.gradient(self.magnitude) * self.units

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
