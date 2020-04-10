"""
"""

import time
import os
import datetime

import numpy as np
import pandas as pd

from dreye.constants import ureg
from dreye.core.spectrum import AbstractSpectrum
from dreye.core.spectral_measurement import CalibrationSpectrum
from dreye.core.measurement_utils import (
    create_calibration_spectrum, convert_measurement,
    create_measured_spectrum, create_measured_spectra
)
from dreye.hardware.base_spectrometer import AbstractSpectrometer
from dreye.err import DreyeError
from dreye.utilities import is_numeric

#HARDWARE API IMPORTS
try:
    import seabreeze.spectrometers as sb
except ImportError as e:
    raise DreyeError(f"You need to install seabreeze: {e}")


def read_calibration_file(
    filename, return_integration_time=False,
    create_spectrum=True
):
    """read lamp calibration data
    """
    assert isinstance(filename, str)
    if filename.endswith('.cal'):
        area_text = 'Collection-area (cm^2)'
        integration_time_text = 'Integration Time (sec):'
    elif filename.endswith('.IRRADCAL'):
        area_text = 'Fiber (micron)'
        integration_time_text = 'Int. Time(usec)'
    else:
        raise DreyeError(f"Incorrect file extension for filename '{filename}'"
                         "; file extension must be .cal or .IRRADCAL. "
                         "If the file ends with .cal, the line indicating the "
                         "area should start with 'Collection-area (cm^2)' "
                         "and the line indicating the integration time "
                         "should start with 'Integration Time (sec):'. "
                         "If the file ends with .IRRADCAL, the line "
                         "indicating the area should start with 'Fiber "
                         "(micron)', and the line indicating the integration "
                         "time should start with 'Int. Time(usec)'.")

    area = None
    integration_time = None
    cal_data = np.loadtxt(filename, skiprows=9)

    with open(filename, 'r') as f:
        # just check the first nine lines
        for n in range(9):
            line = next(f)
            line = line.rstrip()
            if line.startswith(area_text):
                area = eval(
                    line.replace(area_text, ''))
            elif line.startswith(integration_time_text):
                integration_time = eval(
                    line.replace(integration_time_text, ''))

    if (area is None) or (integration_time is None):
        raise DreyeError(
            'Could not find area or integration time in lamp file.')

    # convert to seconds and cm2
    if filename.endswith('.IRRADCAL'):
        integration_time = integration_time * ureg('us')
        integration_time = integration_time.to('s')
        area = area * ureg('um')  # actually diameter
        area = np.pi * (area/2) ** 2
        area = area.to('cm**2')
    else:
        integration_time = integration_time * ureg('s')
        area = area * ureg('cm**2')

    if create_spectrum:
        cal = create_calibration_spectrum(
            cal_data[:, 1], cal_data[:, 0], area
        )
        if return_integration_time:
            return cal, integration_time
        else:
            return cal
    elif return_integration_time:
        return cal_data[:, 0], cal_data[:, 1], area, integration_time

    return cal_data[:, 0], cal_data[:, 1], area


class Spectrometer(AbstractSpectrometer):
    """Basic Spectrometer class for OceanView Spectrometer
    """

    def __init__(
        self, calibration, sb_device=None, integration_time=1.0,
        correct_dark_counts=True, correct_nonlinearity=False,
        min_it=np.nan, max_it=np.nan
    ):
        # TODO open first one if sb_device is None
        try:
            if sb_device is None:
                self.spec = sb.Spectrometer.from_first_available()
            elif isinstance(sb_device, sb.Spectrometer):
                self.spec = sb_device
            elif isinstance(sb_device, str):
                self.spec = sb.Spectrometer.from_serial_number(sb_device)
            else:
                self.spec = sb.Spectrometer(sb_device)
        except Exception as e:
            raise DreyeError(
                "Unable to connect to spectrometer; "
                "ensure all other software that uses the spectrometer "
                f"is closed. Error message: {e}"
            )

        if isinstance(calibration, CalibrationSpectrum):
            self._calibration = calibration
        else:
            self._calibration = read_calibration_file(calibration)

        self.correct_dark_counts = correct_dark_counts
        self.correct_nonlinearity = correct_nonlinearity
        self._original_it = integration_time
        self._current_it = integration_time
        self._min_it = min_it
        self._max_it = max_it
        self.set_it(integration_time)

    @property
    def calibration(self):
        return self._calibration

    @property
    def original_it(self):
        return self._original_it

    @property
    def current_it(self):
        return self._current_it

    @current_it.setter
    def current_it(self, value):
        assert is_numeric(value)
        self._current_it = value

    @property
    def min_it(self):
        return np.nanmax([
            self._min_it,
            self.spec.integration_time_micros_limits[0] * 10 ** -6
        ])

    @property
    def max_it(self):
        return np.nanmin([
            self._max_it,
            self.spec.integration_time_micros_limits[1] * 10 ** -6
        ])

    def set_it(self, it):
        """set integration time in seconds
        """
        self.spec.integration_time_micros(it * 10 ** 6)
        self.current_it = it

    @property
    def max_photon_count(self):
        return self.spec.max_intensity

    @property
    def wavelengths(self):
        return self.spec.wavelengths()

    @property
    def intensities(self):
        return self.spec.intensities(
            correct_dark_counts=self.correct_dark_counts,
            correct_nonlinearity=self.correct_nonlinearity
        )

    def close(self):
        return self.spec.close()

    def open(self):
        return self.spec.open()
