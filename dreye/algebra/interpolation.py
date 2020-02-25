"""
Interpolation
=============

Integratorinterpolator
"""

import numpy as np
from scipy.interpolate import interp1d

from dreye.utilities import spacing, is_uniform


class IntegratorInterpolator(interp1d):
    """
    Works like scipy.interpolate.interp1d, but instead of performing
    a usual interpolation IntegratorInterpolator makes sure that the
    sum within the given range is the same as before.
    This will require uniform spacing along the x axis.
    """

    def __init__(self, x, y, kind='linear', axis=-1, **kwargs):

        self.diff = self.check_uniformity(x)

        super().__init__(x, y, kind=kind, axis=axis, **kwargs)

    def __call__(self, x):

        diff = self.check_uniformity(x)

        y = super().__call__(x)

        return y * diff / self.diff

    @staticmethod
    def check_uniformity(x):
        """method to check uniformity of x array.
        Returns the unique difference.
        """

        # always assume sorted for uniformity test of x array.
        diffs = spacing(x, unique=False, sorted=True)

        if not is_uniform(diffs, is_array_diff=True):
            raise ValueError(
                'For IntegratorInterpolator x '
                'axis must have uniform spacing.')

        return np.mean(diffs)
