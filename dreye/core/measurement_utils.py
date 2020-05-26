"""
utility functions for spectrum measurements
"""

import numpy as np

from dreye.utilities import has_units, is_numeric
from dreye.constants import ureg
from dreye.core.domain import Domain
from dreye.core.signal import _SignalMixin
from dreye.core.spectrum import IntensitySpectra, DomainSpectrum
from dreye.core.spectral_measurement import (
    CalibrationSpectrum, MeasuredSpectrum,
    MeasuredSpectraContainer
)


def convert_measurement(
    signal, calibration, integration_time,
    area=None,
    units='microspectralphotonflux',
    spectrum_cls=IntensitySpectra,
    **kwargs
):
    """
    function to convert photon count signal into spectrum.
    """

    assert isinstance(signal, _SignalMixin)

    if area is None:
        assert isinstance(calibration, CalibrationSpectrum)
        area = calibration.area
    else:
        CalibrationSpectrum(
            calibration,
            domain=signal.domain,
            area=area
        )
        area = calibration.area

    if not has_units(integration_time):
        integration_time = integration_time * ureg('s')

    if not is_numeric(integration_time):
        integration_time = np.expand_dims(
            integration_time.magnitude, signal.domain_axis
        ) * integration_time.units

    # units are tracked
    spectrum = (signal * calibration)
    spectrum = spectrum / (integration_time * area)
    spectrum = spectrum.piecewise_gradient

    return spectrum_cls(spectrum, units=units, **kwargs)


def create_measured_spectrum(
    spectrum_array, inputs,
    wavelengths,
    calibration,
    integration_time,
    area=None,
    units='uE',
    input_units='V',
    is_mole=False,
    assume_contains_input_bounds=True,
    resolution=None
):
    """
    Parameters
    ----------
    spectrum_array : array-like
        array of photon counts across wavelengths for each input
        (wavelength x input labels).
    inputs : array-like
        array of inputs in ascending order.
    wavelengths : array-like
        array of wavelengths in nanometers in ascending order.
    calibration : CalibrationSpectrum or array-like
        Calibration spectrum
    integration_times : array-like
        integration times in seconds.
    axis : int
        axis of wavelengths in spectrum_array
    units : str
        units to convert to
    input_units : str
        units of inputs.
    """
    # create labels
    labels = Domain(inputs, units=input_units)
    spectrum = DomainSpectrum(
        spectrum_array,
        domain=wavelengths,
        labels=labels
    )
    if assume_contains_input_bounds:
        intensities = spectrum.magnitude.sum(0)
        if intensities[0] > intensities[-1]:
            zero_boundary = spectrum.labels.end
            max_boundary = spectrum.labels.start
        else:
            zero_boundary = spectrum.labels.start
            max_boundary = spectrum.labels.end
    if is_mole:
        spectrum = spectrum * ureg('mole')

    return convert_measurement(
        spectrum,
        calibration=calibration,
        integration_time=integration_time,
        units=units,
        area=area,
        spectrum_cls=MeasuredSpectrum,
        zero_boundary=zero_boundary,
        max_boundary=max_boundary,
        resolution=resolution
    )


def create_measured_spectra(
    spectrum_arrays,
    input_arrays,
    wavelengths,
    calibration,
    integration_time,
    area=None,
    units='uE',
    input_units='V',
    is_mole=False,
    assume_contains_input_bounds=True,
    resolution=None
):
    """convenience function
    """

    measured_spectra = []
    for spectrum_array, inputs in zip(spectrum_arrays, input_arrays):
        measured_spectrum = create_measured_spectrum(
            spectrum_array, inputs, wavelengths,
            calibration, integration_time, area=area,
            units=units, input_units=input_units,
            is_mole=is_mole,
            resolution=resolution,
            assume_contains_input_bounds=assume_contains_input_bounds
        )
        measured_spectra.append(measured_spectrum)

    return MeasuredSpectraContainer(measured_spectra)
