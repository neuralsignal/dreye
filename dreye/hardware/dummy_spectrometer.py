"""Dummy spectrophotometer
"""

import numpy as np

from dreye.core.spectrum import Spectra
from dreye.core.spectral_measurement import CalibrationSpectrum
from dreye.hardware.base_spectrometer import AbstractSpectrometer
from dreye.hardware.dummy_system import DummySystem
from dreye.utilities import asarray, get_value


class DummySpectrometer(AbstractSpectrometer):
    """Dummy Spectrometer class
    """

    def __init__(
        self, wavelengths, dummy_leds, dummy_system,
        noise_scale=0.01
    ):

        assert isinstance(dummy_leds, Spectra)
        assert isinstance(dummy_system, DummySystem)
        assert dummy_leds.ndim == 2
        assert dummy_leds.shape[dummy_leds.domain_axis-1] == len(dummy_system)
        leds = dummy_leds.to('spectralirradiance')
        leds.domain_axis = 0
        self.leds = leds.magnitude
        self.system = dummy_system
        self.noise_scale = noise_scale
        self._wavelengths = asarray(wavelengths)
        self._calibration = CalibrationSpectrum(
            np.ones(self._wavelengths.shape),
            wavelengths,
            area=100
        )

    @property
    def calibration(self):
        return self._calibration

    @property
    def original_it(self):
        return 0.001

    @property
    def current_it(self):
        return 0.001

    @current_it.setter
    def current_it(self, value):
        pass

    @property
    def min_it(self):
        return 0.001

    @property
    def max_it(self):
        return 0.001

    def set_it(self, it):
        """set integration time in seconds
        """
        pass

    @property
    def max_photon_count(self):
        return 1000

    @property
    def wavelengths(self):
        return self._wavelengths

    @property
    def intensities(self):
        for idx, output in enumerate(self.system):
            if output._open:
                value = output._current_value
                zero_intensity_bound = get_value(output.zero_intensity_bound)
                max_intensity_bound = get_value(output.max_intensity_bound)
                assert value is not None
                break
        else:
            idx = 0
            output = self.system[0]
            zero_intensity_bound = get_value(output.zero_intensity_bound)
            max_intensity_bound = get_value(output.max_intensity_bound)
            value = max_intensity_bound
        led = self.leds[:, idx]
        led = led + np.random.normal(0, self.noise_scale, size=led.shape)
        led = led / np.max(led)
        led = led * self.ideal_mid_point

        rel_value = np.abs(value - zero_intensity_bound)
        rel_value /= np.abs(zero_intensity_bound - max_intensity_bound)
        led = led * rel_value
        return asarray(led)

    def close(self):
        pass

    def open(self):
        pass
