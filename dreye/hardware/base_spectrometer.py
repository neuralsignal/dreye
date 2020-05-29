"""
"""

from abc import abstractmethod, ABC
import time

import numpy as np

from dreye.core.spectrum import Spectra
from dreye.core.measurement_utils import convert_measurement
from dreye.err import DreyeError


class AbstractSpectrometer(ABC):

    @property
    @abstractmethod
    def calibration(self):
        pass

    @property
    def cal(self):
        return self.calibration

    @property
    def integration_time(self):
        return self.current_it

    @property
    @abstractmethod
    def original_it(self):
        pass

    @property
    @abstractmethod
    def current_it(self):
        pass

    @property
    @abstractmethod
    def min_it(self):
        pass

    @property
    @abstractmethod
    def max_it(self):
        pass

    @abstractmethod
    def set_it(self, it):
        pass

    @property
    @abstractmethod
    def max_photon_count(self):
        pass

    @property
    def ideal_upper_bound(self):
        return self.max_photon_count - 0.15 * self.max_photon_count

    @property
    def ideal_lower_bound(self):
        return self.max_photon_count - 0.2 * self.max_photon_count

    @property
    def ideal_mid_point(self):
        return (self.ideal_upper_bound + self.ideal_lower_bound)/2

    @property
    @abstractmethod
    def wavelengths(self):
        pass

    @property
    def size(self):
        return self.wavelengths.size

    @property
    @abstractmethod
    def intensities(self):
        pass

    @property
    def wls(self):
        return self.wavelengths

    @property
    def int(self):
        return self.intensities

    @property
    def ints(self):
        return self.intensities

    @property
    def maxi(self):
        return np.max(self.int)

    @property
    def intensity(self):
        return self.intensities

    @property
    def signal(self):
        return convert_measurement(
            Spectra(
                self.intensity,
                domain=self.wls,
            ),
            self.cal,
            self.current_it,
        )

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def open(self):
        pass

    def maxi_within_bounds(self):
        maxi = self.maxi
        return (
            (maxi < self.ideal_upper_bound)
            & (maxi > self.ideal_lower_bound)
        )

    def perform_measurement(
        self, n, sleep=None, return_spectrum=True, return_it=False,
        verbose=0, **kwargs
    ):
        """perform measurement by finding
        best integration time and averaging across
        multiple single measurements (and converting by creating a Spectrum
        or MeasuredSpectrum instance)
        """

        it = self.find_best_it()
        ints = self.avg_ints(n, sleep)

        if return_spectrum:
            ints = Spectra(
                ints,
                domain=self.wls
            )
            ints = convert_measurement(
                ints, calibration=self.cal,
                integration_time=self.current_it,
                **kwargs
            )

        if return_it:
            return ints, it
        else:
            return ints

    def find_best_it(self, return_intensity=False):
        """find best integration time and return
        new integration time (and intensity measured)
        """

        found_it = self.maxi_within_bounds()

        while not found_it:
            # current integration time and max intensity
            maxi = self.maxi
            it = self.current_it
            # multiply integration time by a factor
            factor = self.ideal_mid_point / maxi
            it *= factor

            if it > self.max_it:
                it = self.max_it
                self.set_it(it)

                maxi = self.maxi
                found_it = maxi < self.ideal_upper_bound

            elif it < self.min_it:
                it = self.min_it
                found_it = True
                self.set_it(it)

                if self.maxi > self.ideal_upper_bound:
                    raise DreyeError(
                        "Integration time cannot be made smaller, but "
                        "the spectrophotometer is saturating."
                    )

            else:
                self.set_it(it)
                found_it = self.maxi_within_bounds()

        if return_intensity:
            return self.current_it, self.intensity
        else:
            return self.current_it

    def avg_ints(self, n, sleep=None):
        """average over multiple samples of intensities
        """

        ints = np.zeros((n,)+self.wls.shape)

        for idx in range(n):
            ints[idx] = self.int
            if sleep is not None:
                time.sleep(sleep)
        # average
        return ints.mean(axis=0)
