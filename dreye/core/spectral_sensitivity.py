"""Class to define Spectral Sensitivity function

TODO Govardoskii and Stavenga fitting of spectrum
"""

import numpy as np

from dreye.core.spectrum import Spectra
from dreye.constants import RELATIVE_ACCURACY
from dreye.err import DreyeError


class Sensitivity(Spectra):
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
        sig = self.magnitude > (maxima * 10 ** -RELATIVE_ACCURACY)
        sig = np.any(sig, axis=self.labels_axis)

        # conditions for domain min
        # first value is significant
        if np.isnan(self.domain_min.magnitude):
            if sig[0]:
                self.domain_min = self.domain.start
            # no value is significant
            elif not np.any(sig):
                raise DreyeError(
                    "Couldn't find significant bound for spectral sensitivity"
                )
            # some other value is significant
            else:
                # get first significant value
                self.domain_min = self.domain.magnitude[
                    np.flatnonzero(sig)[0]
                ]

        # conditions for domain min
        # last value is significant
        if np.isnan(self.domain_max.magnitude):
            if sig[-1]:
                self.domain_max = self.domain.end
            # no value is significant
            elif not np.any(sig):
                raise DreyeError(
                    "Couldn't find significant bound for spectral sensitivity"
                )
            # some other value is significant
            else:
                # get last significant value
                self.domain_max = self.domain.magnitude[
                    np.flatnonzero(sig)[-1]
                ]
