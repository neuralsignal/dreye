"""
utility functions for spectrum measurements
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from dreye.utilities import has_units, is_numeric, asarray, optional_to
from dreye.utilities.common import is_signallike
from dreye.constants import ureg
from dreye.err import DreyeError
from dreye.core.domain import Domain
from dreye.core.signal import (
    Signal, Signals, DomainSignal
)
from dreye.core.spectral_measurement import (
    CalibrationSpectrum, MeasuredSpectrum,
    MeasuredSpectraContainer
)
from dreye.constants.common import DEFAULT_WL_RANGE


def convert_measurement(
    signal, calibration=None, integration_time=None,
    area=None,
    units='uE',
    spectrum_cls=None,
    background=None,
    **kwargs
):
    """
    Convert photon count signal into
    :obj:`~dreye.Spectrum` or :obj:`~dreye.DomainSpectrum` subclass.

    Parameters
    ----------
    signal : Signal-type instance
        Signal instance with dimensionless values
    calibration : :obj:`~CalibrationSpectrum`, optional
        Calibration spectrum. If None, assume a flat calibration spectrum
        at 1 mircojoules.
    integration_time : numeric or array-like, optional
        Integration times for each measurement in `signal`. If None,
        assumes an integration time of 1 second.
    area : numeric, optional
        Area for `calibration` instance. If None and `calibration` is
        :obj:`~CalibrationSpectrum` instance, area is 1 cm^2.
    units : string or `pint.Unit`.
        The units to convert the spectrum to. Defaults to 'uE', which is
        microspectralphotonflux.
    spectrum_cls : object
        Spectrum class to use to create measurement. Defaults to
        :obj:`~IntensitySpectra`
    background : Signal-type instance
        This signal-type instance is subtracted from the signal class,
        once the signal class has been converte to irradiance units.
    kwargs : dict
        Keyword arguments passed to the `spectrum_cls`.

    Returns
    -------
    object : `spectrum_cls`
        Returns `spectrum_cls` object given `signal`.
    """
    assert is_signallike(signal)

    if spectrum_cls is None:
        if signal.ndim == 1:
            spectrum_cls = Signal
        elif signal.ndim == 2:
            spectrum_cls = Signals
        else:
            raise DreyeError("Signal instance of unknown dimensionality.")

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
    spectrum = spectrum / spectrum.domain.gradient
    # subtract background
    if background is not None:
        spectrum = spectrum - background.to(spectrum.units)

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
    background=None,
    **kwargs
):
    """
    Create :obj:`~dreye.MeasuredSpectrum` instance from a :obj:`~numpy.ndarray`.

    Parameters
    ----------
    spectrum_array : array-like
        Array of photon counts across wavelengths for each output
        (wavelength x output).
    output : array-like
        array of output in ascending or descending order.
    wavelengths : array-like, optional
        array of wavelengths in nanometers in ascending or descending order.
    calibration : :obj:`~CalibrationSpectrum` or array-like
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
    background : Signal-type instance
        This signal-type instance is subtracted from the signal class,
        once the signal class has been converte to irradiance units.
    kwargs : dict, optional
        Keyword arguments passed to :obj:`~MeasuredSpectrum` instance.

    Returns
    -------
    object : :obj:`~MeasuredSpectrum`
        `MeasuredSpectrum` instance containing `spectrum_array`.
    """
    # create labels
    spectrum = DomainSignal(
        spectrum_array,
        domain=wavelengths,
        domain_units='nm',
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
        background=background,
        **kwargs
    )


def create_measured_spectra_container(
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
    Convenience function to created :obj:`~dreye.MeasuredSpectraContainer` from
    LED spectra and intensity bounds.

    Parameters
    ----------
    led_spectra : array-like or numeric, optional
        A two-dimensional array of LED spectral distributions. The rows
        must correspond to individual wavelengths and the columns must
        correspond to different LEDs. If led_spectra are one-dimensional,
        it is assumed that they correspond to the peaks of LEDs that
        have a gaussian-like spectral distribution.
    intensity_bounds : two-tuple of numeric or array-like, optional
        The intensity bounds of each or all LEDs. If two-tuple of array-like,
        each array must be the same length as the number of columns in
        `led_spectra`.
    wavelengths : array-like, optional
        The wavelength values. The size of this array should match the size of
        the rows in `led_spectra`.
    output_bounds : two-tuple of numeric or array-like, optional
        The output value bounds of each or all LEDs. If two-tuple of
        array-like, each array must be the same length as the number of
        columns in `led_spectra`. The output values correspond to values
        that can be sent to a hardware piece to set a particular intensity.
        If you only care about the intensity, keep this as None.
    resolution : array-like, optional
        The resolution of the hardware piece. An array of each step that the
        hardware piece can resolve.
    intensity_units : str or :obj:`~pint.Unit`, optional
        The units of intensity. Defaults to `microphotonflux`.
    output_units : str or :obj:`~pint.Unit`, optional
        The units of output values. Defaults to None or the units of
        intensity.
    transform_func : callable, optional
        If this function is supplied, the intensity values are transformed
        with this function to give the desired output values.
    steps : int, optional
        The number of intensity steps. Defaults to 10.
    names : list of str, optional
        The names to give to each LED. If None, the names will be assigned
        automatically according to the peak in the `led_spectra`.
        This list must be the same lenght as the columns in `led_spectra`.

    Returns
    -------
    object : :obj:`~dreye.MeasuredSpectraContainer`
        A container that can map intensity values to output values, and
        which can be used for various estimators and building stimuli.

    Notes
    -----
    This function was made for convenience, but a better way to create
    a :obj:`~dreye.MeasuredSpectraContainer` instance is to first create
    :obj:`~dreye.MeasuredSpectrum` instance for each LED and then to pass a list
    of these instances to initialize a :obj:`~dreye.MeasuredSpectraContainer`
    instance.
    """
    hard_std = 20.  # STD of gaussians if led_spectra not given
    # create fake LEDs
    if led_spectra is None or is_numeric(led_spectra):
        if wavelengths is None:
            wavelengths = DEFAULT_WL_RANGE
        if led_spectra is None:
            centers = np.arange(350, 700, 50)[None, :]  # 7 LEDs
        else:
            centers = np.linspace(350, 650, int(led_spectra))[None, :]
        led_spectra = norm.pdf(wavelengths[:, None], centers, hard_std)
    elif asarray(led_spectra).ndim == 1:
        centers = optional_to(led_spectra, 'nm')
        if wavelengths is None:
            wavelengths = DEFAULT_WL_RANGE
        led_spectra = norm.pdf(wavelengths[:, None], centers, hard_std)
    # wavelengths
    if is_signallike(led_spectra) and (wavelengths is not None):
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

    if intensity_bounds is None:
        # some very high max intensity
        intensity_bounds = (0, 10000)

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
    else:
        if transform_func is not None:
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
