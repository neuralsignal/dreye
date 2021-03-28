"""
Numpy mixin for the Signal class
"""


import numpy as np

from dreye.utilities import (
    get_value, get_units
)
from dreye.constants import ureg


class _NumpyMixin:
    """
    Mixin class for signal class to use functions such as mean, sum, etc.
    """

    def _numpy_reduce(
        self,
        func,
        units,
        axis=None,
        **kwargs
    ):
        """
        General method for using mean, sum, etc.
        """
        values = func(self.magnitude, axis=axis, **kwargs)
        return values * units

    def mean(self, *args, **kwargs):
        """
        Compute the arithmetic mean along the specified axis.

        Returns
        -------
        mean : :obj:`~pint.Quantity`
        """

        return self._numpy_reduce(
            np.mean, self.units, *args, **kwargs
        )

    def nanmean(self, *args, **kwargs):
        """
        Compute the arithmetic mean along the specified axis, ignoring NaNs.

        Returns
        -------
        mean : :obj:`~pint.Quantity`
        """

        return self._numpy_reduce(
            np.nanmean, self.units, *args, **kwargs
        )

    def sum(self, *args, **kwargs):
        """
        Sum of array elements over a given axis.

        Returns
        -------
        sum : :obj:`~pint.Quantity`
        """

        return self._numpy_reduce(
            np.sum, self.units, *args, **kwargs
        )

    def nansum(self, *args, **kwargs):
        """
        Return the sum of array elements over a given axis treating
        NaNs as zero.

        Returns
        -------
        sum : :obj:`~pint.Quantity`
        """

        return self._numpy_reduce(
            np.nansum, self.units, *args, **kwargs
        )

    def std(self, *args, **kwargs):
        """
        Compute the standard deviation along the specified axis.

        Returns
        -------
        std : :obj:`~pint.Quantity`
        """

        return self._numpy_reduce(
            np.std, self.units, *args, **kwargs
        )

    def nanstd(self, *args, **kwargs):
        """
        Compute the standard deviation along the specified axis, while ignoring
        NaNs.

        Returns
        -------
        std : :obj:`~pint.Quantity`
        """

        return self._numpy_reduce(
            np.nanstd, self.units, *args, **kwargs
        )

    def var(self, *args, **kwargs):
        """
        Compute the variance along the specified axis.

        Returns
        -------
        var : :obj:`~pint.Quantity`
        """

        return self._numpy_reduce(
            np.var, self.units ** 2, *args, **kwargs
        )

    def nanvar(self, *args, **kwargs):
        """
        Compute the variance along the specified axis, while ignoring NaNs.

        Returns
        -------
        var : :obj:`~pint.Quantity`
        """

        return self._numpy_reduce(
            np.nanvar, self.units ** 2, *args, **kwargs
        )

    def min(self, *args, **kwargs):
        """
        Return the minimum along a given axis.

        Returns
        -------
        min : :obj:`~pint.Quantity`
        """

        return self._numpy_reduce(
            np.min, self.units, *args, **kwargs
        )

    def nanmin(self, *args, **kwargs):
        """
        Return minimum of an array or minimum along an axis, ignoring any NaNs.

        When all-NaN slices are encountered a RuntimeWarning is raised and Nan
        is returned for that slice.

        Returns
        -------
        min : :obj:`~pint.Quantity`
        """

        return self._numpy_reduce(
            np.nanmin, self.units, *args, **kwargs
        )

    def max(self, *args, **kwargs):
        """
        Element-wise maximum of array elements.

        Returns
        -------
        max : :obj:`~pint.Quantity`
        """

        return self._numpy_reduce(
            np.max, self.units, *args, **kwargs
        )

    def nanmax(self, *args, **kwargs):
        """
        Return the maximum of an array or maximum along an axis, ignoring any
        NaNs.

        When all-NaN slices are encountered a RuntimeWarning is raised
        and NaN is returned for that slice.

        Returns
        -------
        max : :obj:`~pint.Quantity`
        """

        return self._numpy_reduce(
            np.nanmax, self.units, *args, **kwargs
        )

    def dot(self, other, transpose=False):
        """
        Returns the dot product of array-like instances.

        Returns
        -------
        array : :obj:`~pint.Quantity`
        """

        if transpose:
            self_values = self.magnitude.T
        else:
            self_values = self.magnitude

        other_values = get_value(other)
        other_units = get_units(other)

        new_units = self.units * other_units

        dot_array = np.dot(self_values, other_values)

        return dot_array * new_units

    def cov(self, *args, **kwargs):
        """
        Calculate covariance matrix of signal.

        Returns
        -------
        array : :obj:`~pint.Quantity`
        """

        values = np.cov(self.magnitude, *args, **kwargs)
        return values * (self.units ** 2)

    def corrcoef(self, *args, **kwargs):
        """
        Calculate pearson's correlation matrix containing correlation
        coefficients (variance/variance squared).

        Returns
        -------
        array : :obj:`~pint.Quantity`
        """

        values = np.corrcoef(self.magnitude, *args, **kwargs)
        return values * ureg(None).units
