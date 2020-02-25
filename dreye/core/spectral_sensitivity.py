"""Class to define Spectral Sensitivity function

TODO Govardoskii and Stavenga fitting of spectrum
"""

import numpy as np

from dreye.core.spectrum import AbstractSpectrum
from dreye.constants import UREG


class AbstractSensitivity(AbstractSpectrum):
    pass


class RelativeOpsinSensitivity(AbstractSensitivity):
    """
    Subclass of Signal. Sets max of spectrum to one.
    """

    @property
    def units(self):
        """
        """

        return UREG(None).units

    @property
    def values(self):
        """
        """

        # TODO include a way to make sure domain is not shrunk

        return (
            self._values
            / np.max(self._values, axis=self.domain_axis, keepdims=True)
        ) * self.units


class AbsoluteOpsinSensitivity(AbstractSensitivity):
    """
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('AbsoluteOpsinSensitivity')
