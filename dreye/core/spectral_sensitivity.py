"""Class to define Spectral Sensitivity function

TODO Govardoskii and Stavenga fitting of spectrum
"""

import numpy as np

from dreye.core.spectrum import AbstractSpectrum
from dreye.constants import UREG, RELATIVE_ACCURACY


class AbstractSensitivity(AbstractSpectrum):
    """
    Same as AbstractSpectrum, but assigns domain min and max,
    if it is not given. This ensures the values around the peak
    sensitivity are always considered.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_sig_domain_bounds()

    def _set_sig_domain_bounds(self):

        maxima = np.nanmax(
            self.magnitude, axis=self.domain_axis, keepdims=True
        )
        # significant intensities
        sig = self.magnitude > (maxima * RELATIVE_ACCURACY)
        if sig.ndim == 2:
            sig = np.any(sig, axis=self.other_axis)

        # conditions for domain min
        # first value is significant
        if self._domain_min is None:
            if sig[0]:
                self._domain_min = self.domain.start * self.domain.units
            # no value is significant
            elif not np.any(sig):
                self._domain_min = None
            # some other value is significant
            else:
                # get first significant value
                self._domain_min = self.domain[np.flatnonzero(sig)[0]]

        # conditions for domain min
        # last value is significant
        if self._domain_max is None:
            if sig[-1]:
                self._domain_max = self.domain.end * self.domain.units
            # no value is significant
            elif not np.any(sig):
                self._domain_max = None
            # some other value is significant
            else:
                # get last significant value
                self._domain_max = self.domain[np.flatnonzero(sig)[-1]]


class RelativeOpsinSensitivity(AbstractSensitivity):
    """
    Subclass of Signal. Sets max of spectrum to one.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._values = (
            self.magnitude
            / np.max(self.magnitude, axis=self.domain_axis, keepdims=True)
        )
        self._units = UREG(None).units


class AbsoluteOpsinSensitivity(AbstractSensitivity):
    """Absolute Opsin Sensitivity in user-defined units
    """
