"""Dummy spectrophotometer
"""

import numpy as np

from dreye.core.spectrum import Spectra
from dreye.core.spectral_measurement import CalibrationSpectrum
from dreye.hardware.base_spectrometer import AbstractSpectrometer
from dreye.hardware.dummy_system import DummySystem
from dreye.utilities import asarray, get_value
from dreye.utilities.abstract import inherit_docstrings


@inherit_docstrings
class DummySpectrometer(AbstractSpectrometer):
    """
    Dummy Spectrometer class that can be used to test
    `dreye.hardware.MeasurementRunner`.
    """

    def __init__(
        self, wavelengths, dummy_leds, dummy_system,
        background=None,
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
        self.background = background
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
            value = zero_intensity_bound
        led = self.leds[:, idx]
        led = led / np.max(led)
        led = led * self.ideal_mid_point

        rel_value = np.abs(value - zero_intensity_bound)
        rel_value /= np.abs(zero_intensity_bound - max_intensity_bound)
        led = led * rel_value
        led = led + np.random.normal(0, self.noise_scale, size=led.shape)
        if self.background is not None:
            led = led + self.background
        return asarray(led)

    def close(self):
        pass

    def open(self):
        pass
