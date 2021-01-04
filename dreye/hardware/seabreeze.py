"""
"""

import numpy as np

from dreye.constants import ureg
from dreye.core.spectral_measurement import CalibrationSpectrum
from dreye.hardware.base_spectrometer import AbstractSpectrometer
from dreye.err import DreyeError
from dreye.utilities import is_numeric
from dreye.utilities.abstract import inherit_docstrings

# HARDWARE API IMPORTS
try:
    import seabreeze.spectrometers as sb
    SEABREEZE = True
except ImportError:
    SEABREEZE = False


def read_calibration_file(
    filename, return_integration_time=False,
    create_spectrum=True
):
    """
    Read lamp calibration data for a OceanView Spectrometer.

    Parameters
    ----------
    return_integration_time : bool
        Whether to return the integration time used.
    create_spectrum : bool
        Whether to create a `dreye.CalibrationSpectrum` object.

    Returns
    -------
    (wavelengths) : numpy.ndarray
        Wavelength domain of calibration. Only returned,
        if `create_spectrum` is False.
    calibration : numpy.ndarray or dreye.CalibrationSpectrum
        Calibration instance.
    (area) : `pint.Quantity`
        Area of spectrometer used. Only returned,
        if `create_spectrum` is False.
    (integration_time) : `pint.Quantity`
        Integration time used for calibration.
        Only returned, if `return_integration_time` is True.
    """
    assert isinstance(filename, str)
    area_texts = {
        'Collection-area(cm^2)': ('area', 'cm**2'),
        'Fiber(micron)': ('diameter', 'micrometer'),
        'Fiber(cm)': ('diameter', 'cm'),
        'Collection-area(um^2)': ('area', 'micrometer**2')
    }
    integration_time_texts = {
        'Int.Time(usec)': 'microsecond',
        'IntegrationTime(sec)': 'second',
        'Int.Time(sec)': 'second',
        'IntegrationTime(usec)': 'microsecond',
    }

    area = None
    integration_time = None
    cal_data = np.loadtxt(filename, skiprows=9)

    with open(filename, 'r') as f:
        # just check the first nine lines
        for n in range(9):
            line = next(f)
            # removes all spaces
            line = line.replace(' ', '').replace(':', '')
            if area is None:
                for area_text, area_type in area_texts.items():
                    if area_text in line:
                        area = float(line.replace(area_text, ''))
                        break

            if integration_time is None:
                for it_text, it_units in integration_time_texts.items():
                    if it_text in line:
                        integration_time = float(line.replace(it_text, ''))
                        break

    if (area is None) or (integration_time is None):
        raise DreyeError("Could not find area or "
                         "integration time in lamp file.")

    if area_type[0] == 'diameter':
        area = area * ureg(area_type[1])
        area = np.pi * (area/2) ** 2
        area = area.to('cm**2')
    elif area_type[0] == 'area':
        area = (area * ureg(area_type[1])).to('cm**2')
    else:
        raise DreyeError("Area type {area_type} not recognized.")

    integration_time = (integration_time * ureg(it_units)).to('s')

    if create_spectrum:
        cal = CalibrationSpectrum(
            values=cal_data[:, 1],
            domain=cal_data[:, 0],
            area=area
        )
        if return_integration_time:
            return cal, integration_time
        else:
            return cal
    elif return_integration_time:
        return cal_data[:, 0], cal_data[:, 1], area, integration_time

    return cal_data[:, 0], cal_data[:, 1], area


@inherit_docstrings
class OceanSpectrometer(AbstractSpectrometer):
    """
    Basic Spectrometer class for OceanView Spectrometer.

    This requires installation of the `seabreeze` package.

    Parameters
    ----------
    calibration : dreye.CalibrationSpectrum or str, optional
        Calibration filename or calibration instance used for
        converting measurements to intensity units.
    sb_device : seabreeze.Spectrometer or str, optional
        The OceanView device for measurements. If None,
        it will use the first device available.
    integration_time : numeric, optional
        The initial integration time used.
    correct_dark_counts : bool, optional
        Whether to correct for dark counts during measurements.
    correct_nonlinearity : bool, optional
        Whether to correct for a nonlinearity during measurements.
    min_it : numeric, optional
        The minimum integration time allowed. If not given,
        the `min_it` will be taken from the `sb_device`.
    max_it : numeric, optional
        The maximum integration time allowed. If not given,
        the `max_it` will be taken from the `sb_device`.
    """

    def __init__(
        self, calibration=None, sb_device=None, integration_time=1.0,
        correct_dark_counts=True, correct_nonlinearity=False,
        min_it=np.nan, max_it=np.nan
    ):
        if not SEABREEZE:
            raise DreyeError(f"You need to install seabreeze.")
        # opens any device that is a seabreeze device
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

        if calibration is None:
            self._calibration = None
        elif isinstance(calibration, CalibrationSpectrum):
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
