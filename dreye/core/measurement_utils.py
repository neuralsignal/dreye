"""
utility functions for spectrum measurements
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from dreye.utilities import has_units, is_numeric, asarray
from dreye.constants import ureg
from dreye.err import DreyeError
from dreye.core.domain import Domain
from dreye.core.signal import _SignalMixin, _Signal2DMixin
from dreye.core.spectrum import IntensitySpectra, DomainSpectrum
from dreye.core.spectral_measurement import (
    CalibrationSpectrum, MeasuredSpectrum,
    MeasuredSpectraContainer
)


def convert_measurement(
    signal, calibration=None, integration_time=None,
    area=None,
    units='uE',
    spectrum_cls=IntensitySpectra,
    **kwargs
):
    """
    Convert photon count signal into `Spectrum` or `DomainSpectrum` subclass.

    Parameters
    ----------
    signal : `Signal` instance
        Signal instance with dimensionless values
    calibration : `CalibrationSpectrum`, optional
        Calibration spectrum. If None, assume a flat calibration spectrum
        at 1 mircojoules.
    integration_time : numeric or array-like, optional
        Integration times for each measurement in `signal`. If None,
        assumes an integration time of 1 second.
    area : numeric, optional
        Area for `calibration` instance. If None and `calibration` is
        `CalibrationSpectrum` instance, area is 1 cm^2.
    units : string or `pint.Unit`.
        The units to convert the spectrum to. Defaults to 'uE', which is
        microspectralphotonflux.
    spectrum_cls : object
        Spectrum class to use to create measurement. Default to
        `IntensitySpectra`
    kwargs : dict
        Keyword arguments passed to the `spectrum_cls`.

    Returns
    -------
    object : `spectrum_cls`
        Returns `spectrum_cls` object given `signal`.
    """

    assert isinstance(signal, _SignalMixin)

    if calibration is None:
        calibration = CalibrationSpectrum(
            np.ones(signal.domain.size),
            signal.domain,
            area=area
        )

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

    if integration_time is None:
        integration_time = ureg('s')  # assumes 1 seconds integration time

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
    spectrum_array, output,
    wavelengths,
    calibration=None,
    integration_time=None,
    area=None,
    units='uE',
    output_units='V',
    is_mole=False,
    zero_intensity_bound=None,
    max_intensity_bound=None,
    assume_contains_output_bounds=True,
    resolution=None,
    name=None,
    **kwargs
):
    """
    Create `MeasuredSpectrum` instance from numpy.ndarray objects.

    Parameters
    ----------
    spectrum_array : array-like
        Array of photon counts across wavelengths for each output
        (wavelength x output).
    output : array-like
        array of output in ascending or descending order.
    wavelengths : array-like, optional
        array of wavelengths in nanometers in ascending or descending order.
    calibration : CalibrationSpectrum or array-like
        Calibration spectrum instance used for conversion of `spectrum_array`.
    integration_times : numeric or array-like, optional
        Integration times in seconds for each measurement in `spectrum_array`.
    area : numeric, optional
        Area measured over.
    units : str, optional
        Units to convert spectrum to. Defaults to 'uE', which is
        microspectralphotonflux.
    output_units : str, optional
        Units of output (e.g. 'V' for volts applied).
    is_mole : bool, optional
        If `spectrum_array` is in units of moles instead of photon counts.
    zero_intensity_bound : numeric, optional
        The output value, which corresponds to zero intensity.
    max_intensity_bound : numeric, optional
        The output value, which corresponds to the maximum intensity.
    assume_contains_output_bounds: bool, optional
        If True, assume `output` contains the zero and max intensity bound;
        thus inferring `zero_intensity_bound` and `max_intensity_bound`, if
        not given.
    resolution : array-like, optional
        Array of each value the output hardware can resolve.
    name : str, optional
        Name given to measured spectra.
    kwargs : dict, optional
        Keyword arguments passed to `MeasuredSpectrum` instance.

    Returns
    -------
    object : `MeasuredSpectrum`
        `MeasuredSpectrum` instance containing `spectrum_array`.
    """
    # create labels
    spectrum = DomainSpectrum(
        spectrum_array,
        domain=wavelengths,
        labels=Domain(output, units=output_units)
    )
    if assume_contains_output_bounds:
        intensities = spectrum.magnitude.sum(0)
        if intensities[0] > intensities[-1]:
            if zero_intensity_bound is not None:
                zero_intensity_bound = spectrum.labels.end
            if max_intensity_bound is not None:
                max_intensity_bound = spectrum.labels.start
        else:
            if zero_intensity_bound is not None:
                zero_intensity_bound = spectrum.labels.start
            if max_intensity_bound is not None:
                max_intensity_bound = spectrum.labels.end
    if is_mole:
        spectrum = spectrum * ureg('mol')

    return convert_measurement(
        spectrum,
        calibration=calibration,
        integration_time=integration_time,
        units=units,
        area=area,
        spectrum_cls=MeasuredSpectrum,
        zero_intensity_bound=zero_intensity_bound,
        max_intensity_bound=max_intensity_bound,
        resolution=resolution,
        name=name,
        **kwargs
    )


def create_measured_spectra(
    spectrum_arrays,
    output_arrays,
    wavelengths,
    calibration,
    integration_times,
    area=None,
    units='uE',
    output_units='V',
    is_mole=False,
    assume_contains_output_bounds=True,
    resolution=None
):
    """
    Create `MeasuredSpectraContainer` out of a set of arrays.

    Parameters
    ----------
    spectrum_arrays : array-like
    output_arrays : array-like
    wavelengths : array-like
    calibration : `CalibrationSpectrum`
    integration_times : array-like
    area : numeric, optional
    units : str or `pint.Unit`, optional
    output_units : str or `pint.Unit`, optional
    is_mole: bool, optional
    assume_contains_output_bounds: bool, optional
    resolution : array-like, optional

    Returns
    -------
    object : `MeasuredSpectraContainer`

    See Also
    --------
    create_measured_spectrum
    """

    measured_spectra = []
    for spectrum_array, output, integration_time in zip(
        spectrum_arrays, output_arrays, integration_times
    ):
        measured_spectrum = create_measured_spectrum(
            spectrum_array, output, wavelengths,
            calibration=calibration,
            integration_time=integration_time, area=area,
            units=units, output_units=output_units,
            is_mole=is_mole,
            resolution=resolution,
            assume_contains_output_bounds=assume_contains_output_bounds
        )
        measured_spectra.append(measured_spectrum)

    return MeasuredSpectraContainer(measured_spectra)


def get_led_spectra_container(
    led_spectra=None,  # wavelengths x LED (ignores units)
    intensity_bounds=(0, 100),  # two-tuple of min and max intensity
    wavelengths=None,  # wavelengths (two-tuple or array-like)
    output_bounds=None,  # two-tuple of min and max output
    resolution=None,  # array-like
    intensity_units=None,  # units
    output_units=None,
    transform_func=None,  # callable
    steps=10,
    names=None,
):
    """
    Convenience function to created `MeasuredSpectraContainer` from
    LED spectra and intensity bounds.

    Parameters
    ----------
    led_spectra : array-like, optional
    intensity_bounds : two-tuple of numeric or array-like, optional
    wavelengths : array-like, optional
    output_bounds : two-tuple of numeric or array-like, optional
    resolution : array-like, optional
    intensity_units : str or `pint.Unit`, optional
    output_units : str or `pint.Unit`, optional
    transform_func : callable, optional
    steps : int, optional
    names : list of str, optional

    Returns
    -------
    object : `MeasuredSpectraContainer`
    """
    # create fake LEDs
    if led_spectra is None or is_numeric(led_spectra):
        if wavelengths is None:
            wavelengths = np.arange(300, 700.1, 0.5)
        if led_spectra is None:
            centers = np.arange(350, 700, 50)[None, :]  # 7 LEDs
        else:
            centers = np.linspace(350, 650, int(led_spectra))[None, :]
        led_spectra = norm.pdf(wavelengths[:, None], centers, 20)
    # wavelengths
    if isinstance(led_spectra, _Signal2DMixin) and wavelengths is not None:
        led_spectra = led_spectra(wavelengths)
        led_spectra.domain_axis = 0
    # check if we can obtain wavelengths (replace wavelengths)
    if hasattr(led_spectra, 'domain'):
        wavelengths = led_spectra.domain
    elif hasattr(led_spectra, 'wavelengths'):
        wavelengths = led_spectra.wavelengths
    elif isinstance(led_spectra, (pd.DataFrame, pd.Index)):
        wavelengths = led_spectra.index
    elif wavelengths is None:
        raise DreyeError("Must provide wavelengths.")

    led_spectra = asarray(led_spectra)
    led_spectra /= np.trapz(led_spectra, asarray(wavelengths), axis=0)

    intensities = np.broadcast_to(
        np.linspace(*intensity_bounds, steps).T,
        (led_spectra.shape[1], steps)
    )

    # handle intenisty units
    if isinstance(intensity_units, str):
        units = ureg(intensity_units).units / ureg('nm').units
    elif has_units(intensity_units):
        units = intensity_units.units / ureg('nm').units
    elif intensity_units is None:
        units = 'uE'  # assumes in microspectralphotonflux
        intensity_units = 'microphotonflux'
    else:
        # assumes is ureg.Unit
        units = intensity_units / ureg('nm').units

    if output_bounds is None:
        # assume output bounds is in same units
        if output_units is None:
            output_units = intensity_units
        output = intensities.copy()
    elif transform_func is not None:
        output = transform_func(intensities)
    else:
        output = np.linspace(*output_bounds, steps)
    output = np.broadcast_to(output.T, (led_spectra.shape[1], steps))

    measured_spectra = []
    for idx, led_spectrum in enumerate(led_spectra.T):
        # always do 100 hundred steps
        led_spectrum = (
            led_spectrum[:, None]
            * intensities[idx, None]
        )

        measured_spectrum = MeasuredSpectrum(
            values=led_spectrum,
            domain=wavelengths,
            labels=output[idx],
            labels_units=output_units,
            units=units,
            resolution=resolution,
            name=(None if names is None else names[idx])
        )
        measured_spectra.append(measured_spectrum)

    return MeasuredSpectraContainer(measured_spectra)
