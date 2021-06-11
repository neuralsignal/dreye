"""
"""

from abc import abstractmethod, ABC
import time
import warnings

import numpy as np

from dreye.core.signal import Signal
from dreye.core.measurement_utils import convert_measurement
from dreye.err import DreyeError


class AbstractSpectrometer(ABC):
    """
    Abstract base class for a spectrophotometer.

    Notes
    -----
    The following methods should be implemented for each subclass:

    * `__init__`
    * `calibration` property
    * `original_it` property
    * `current_it` property
    * `min_it` property
    * `max_it` property
    * `set_it` method
    * `max_photon_count` property
    * `wavelengths` property
    * `intensities` property
    * `open` method
    * `close` method
    """

    @property
    @abstractmethod
    def calibration(self):
        """
        The `dreye.CalibrationSpectrum` object.

        See Also
        --------
        cal
        """
        pass

    @property
    def cal(self):
        """
        The `dreye.CalibrationSpectrum` object.

        See Also
        --------
        calibration
        """
        return self.calibration

    @property
    def integration_time(self):
        """
        The current integration time.
        """
        return self.current_it

    @property
    @abstractmethod
    def original_it(self):
        """
        The original integration time used during initialization.
        """
        pass

    @property
    @abstractmethod
    def current_it(self):
        """
        The current integration time.
        """
        pass

    @property
    @abstractmethod
    def min_it(self):
        """
        The minimum allowed integration time.
        """
        pass

    @property
    @abstractmethod
    def max_it(self):
        """
        The maximum allowed integratino time.
        """
        pass

    @abstractmethod
    def set_it(self, it):
        """
        Set a new current integration time.

        Parameters
        ----------
        it : numeric
            The new integration time
        """
        pass

    @property
    @abstractmethod
    def max_photon_count(self):
        """
        The maximum photon count of the spectrophotometer.
        """
        pass

    @property
    def ideal_upper_bound(self):
        """
        The ideal upper bound for measuring spectra.

        This correspond to :math:`max_count - 0.15 * max_count`.

        See Also
        --------
        max_photon_count
        ideal_lower_bound
        ideal_mid_point
        """
        return self.max_photon_count - 0.15 * self.max_photon_count

    @property
    def ideal_lower_bound(self):
        """
        The ideal lower bound for measuring spectra.

        This correspond to :math:`max_count - 0.2 * max_count`.

        See Also
        --------
        max_photon_count
        ideal_upper_bound
        ideal_mid_point
        """
        return self.max_photon_count - 0.2 * self.max_photon_count

    @property
    def ideal_mid_point(self):
        """
        The mid point between `ideal_upper_bound` and `ideal_lower_bound`.

        See Also
        --------
        max_photon_count
        ideal_upper_bound
        ideal_lower_point
        """
        return (self.ideal_upper_bound + self.ideal_lower_bound)/2

    @property
    @abstractmethod
    def wavelengths(self):
        """
        A `numpy.ndarray` of all wavelength values that are measured.
        """
        pass

    @property
    def size(self):
        """
        The size of the `wavelengths` array.
        """
        return self.wavelengths.size

    @property
    @abstractmethod
    def intensities(self):
        """
        A `numpy.ndarray` of all photon counts for each wavelength in
        `wavelengths`.

        See Also
        --------
        wavelengths
        """
        pass

    @property
    def wls(self):
        """
        Alias for `wavelengths`.

        See Also
        --------
        wavelengths
        """
        return self.wavelengths

    @property
    def int(self):
        """
        Alias for `intensities`.

        See Also
        --------
        intensities
        """
        return self.intensities

    @property
    def ints(self):
        """
        Alias for `intensities`.

        See Also
        --------
        intensities
        """
        return self.intensities

    @property
    def maxi(self):
        """
        The max of the `intensities` array.

        See Also
        --------
        intensities
        """
        return np.max(self.int)

    @property
    def intensity(self):
        """
        Alias for `intensities`.

        See Also
        --------
        intensities
        """
        return self.intensities

    @property
    def signal(self):
        """
        The `intensity` and `wavelength` array converted to a `dreye.Spectrum`
        instance, given the `calibration`.
        """
        return convert_measurement(
            Signal(
                self.intensity,
                domain=self.wls,
                domain_units='nm',
            ),
            self.cal,
            self.current_it,
        )

    @abstractmethod
    def close(self):
        """
        Close the spectrometer.
        """
        pass

    @abstractmethod
    def open(self):
        """
        Open the spectrometer
        """
        pass

    def maxi_within_bounds(self):
        """
        Is `maxi` within the `ideal_lower_bound` and
        `ideal_upper_bound` bound.
        """
        maxi = self.maxi
        return (
            (maxi < self.ideal_upper_bound)
            & (maxi > self.ideal_lower_bound)
        )

    def perform_measurement(
        self, n=1, sleep=None,
        return_spectrum=True, return_it=False, return_sd=False,
        verbose=0, optimize_it=True, error='raise', **kwargs
    ):
        """
        Perform measurement with optimal integration time.

        Parameters
        ----------
        n : int, optional
            Number of averages.
        sleep : float, optional
            Algorithm sleep between each measurement in seconds.
        return_spectrum : bool, optional
            Whether to return a `dreye.Spectrum` instance, converted
            to the proper intensity units using the available `calibration`
            attribute.
        return_it : bool, optional
            Whether to return the integration time used.
        verbose : int, optional
            Verbosity level.
        optimize_it : bool, optional
            Whether to optimize the integration time.
        error : str {'raise', 'warn', 'ignore'}, optional
            Whether to raise, warn, or ignore when an optimal
            integration time cannot be found and the photometer is
            saturated.
        kwargs : dict, optional
            Keyword arguments passed to `dreye.Spectrum`

        Returns
        -------
        intensities : numpy.ndarray or dreye.Spectrum
            The intensities measured.
        integration_time : float
            If `return_it` is set to True

        Notes
        -----
        Measurements are performed in three steps:]

        1. Find the best integration time.
        2. Average over samples
        3. (Optional) Create `dreye.Spectrum` instance.
        """

        if optimize_it:
            it = self.find_best_it(error=error)
        else:
            it = self.current_it
        if return_sd:
            ints, ints_sd = self.avg_ints(n, sleep, return_sd=return_sd)
        else:
            ints = self.avg_ints(n, sleep, return_sd=return_sd)

        if return_spectrum:
            ints = Signal(
                ints,
                domain=self.wls, 
                domain_units='nm'
            )
            ints = convert_measurement(
                ints, calibration=self.cal,
                integration_time=self.current_it,
                **kwargs
            )

        if return_it and return_sd:
            return ints, ints_sd, it
        elif return_sd:
            return ints, ints_sd
        elif return_it:
            return ints, it
        else:
            return ints

    def find_best_it(self, return_intensity=False, error='raise'):
        """
        Find best integration time and return new integration time
        (and intensity measured).

        Notes
        -----
        The best integration time is found by iteratively measuring the
        spectrum and changing the integration time until the measurement
        is within the `ideal_lower_bound` and `ideal_upper_bound`.
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
                    if error == 'ignore':
                        pass
                    elif error == 'warn':
                        warnings.warn(
                            "Integration time cannot be made smaller, but "
                            "the spectrophotometer is saturating."
                        )
                    else:
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

    def avg_ints(self, n, sleep=None, return_sd=False):
        """
        Average over multiple samples of intensities

        Parameters
        ----------
        n : int, optional
            Number of averages.
        sleep : float, optional
            Algorithm sleep between each measurement in seconds.

        Returns
        -------
        intensities : numpy.ndarray
            Mean across samples
        """

        ints = np.zeros((n,)+self.wls.shape)

        for idx in range(n):
            ints[idx] = self.int
            if sleep is not None:
                time.sleep(sleep)
        # average
        if return_sd:
            return ints.mean(axis=0), ints.std(axis=0, ddof=1)
        return ints.mean(axis=0)
