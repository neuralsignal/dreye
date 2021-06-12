"""
Functions for creating or fitting particular spectra
"""

import numpy as np
from scipy.stats import norm

from dreye.utilities import optional_to, is_integer, asarray
from dreye.utilities.common import is_signallike
from dreye.core.signal import Signal, Signals
from dreye.constants import ureg


def create_spectrum(
    intensities=None,
    wavelengths=None,
    **kwargs
):
    """
    Convenience function to create a `Spectrum` instance
    """

    if intensities is None:
        if wavelengths is None:
            wavelengths = np.arange(300, 700.1, 0.5)
        intensities = np.ones(len(wavelengths))
        intensities /= np.trapz(intensities, wavelengths)
    elif wavelengths is None:
        wavelengths = np.linspace(300, 700, len(intensities))

    domain_units = kwargs.pop('domain_units', 'nm')

    return Signal(
        values=intensities,
        domain=wavelengths,
        domain_units=domain_units,
        **kwargs
    )


def create_max_normalized_gaussian_spectra(
    intensities=None,
    wavelengths=None,
    **kwargs
):
    """
    Convenience function to create a max-normalized Gaussian
    `Spectra` instance.
    """
    kwargs['units'] = kwargs.get('units', None)
    if intensities is None or is_integer(intensities):
        if wavelengths is None:
            wavelengths = np.arange(300, 700.1, 0.5)
        if intensities is None:
            intensities = 10
        spectra = create_gaussian_spectrum(
            wavelengths, np.linspace(300, 700, intensities),
            **kwargs
        )
        return spectra.max_normalized
    elif asarray(intensities).ndim == 1:
        if wavelengths is None:
            wavelengths = np.arange(300, 700.1, 0.5)
        spectra = create_gaussian_spectrum(
            wavelengths, intensities,
            **kwargs
        )
        return spectra.max_normalized
    elif wavelengths is None:
        wavelengths = np.linspace(300, 700, len(intensities))

    domain_units = kwargs.pop('domain_units', 'nm')

    return Signals(
        values=intensities,
        domain=wavelengths,
        domain_units=domain_units,
        **kwargs
    ).max_normalized


def create_gaussian_spectrum(
    wavelengths, centers, std=10,
    intensity=1,
    units='uE',
    cdf=None,
    background=None,
    filter=False,
    add_background=False,
    zero_cutoff=True,
    max_normalized=False,
    **kwargs
):
    """
    Get `Spectra` instance from Gaussian probability distribution.

    wavelengths : array-like
        Wavelength array in nm.
    centers : float or array-like
        The centers for each single wavelength Gaussian in nm.
    std : float or array-like, optional
        The standard deviation for each single wavelength Gaussian in nm.
    intensity : float, optional
        The intensity of the single wavelength Gaussians.
    units : str, optional
        The units for `Spectra` instance.
    cdf : bool, optional
        Allows changes to the shape of the spectral distribution.
        If `cdf` is None, the spectral distribution will correspond to the pdf.
        If `cdf` is True, the spectral distribution will correspond to the cdf.
        If `cdf` is False, the spectral distribution will correspond to the
        1-cdf.
    background : array-like, optional
        A background spectral distribution that can be added
        and/or used for filtering.
    filter : bool, optional
        If True, filters the spectral distribution by the background.
    add_background : bool, optional
        If True, adds the background and spectral distribution together.
    max_normalized : bool, optional
        Whether to max-normalize the spectral distribution.
    zero_cutoff : bool, optional
        If True, spectral distribution clipped at 0 and negative values
        are discarded.
    kwargs : dict, optional
        Keyword arguments passed to the `Spectra` class.
    """
    if is_signallike(background):
        background = background(wavelengths).to(units)

    if isinstance(units, str) or units is None:
        units = ureg(units).units

    wavelengths = optional_to(wavelengths, 'nm')
    centers = optional_to(centers, 'nm')
    std = optional_to(std, 'nm')
    intensity = optional_to(intensity, units * ureg('nm').units)
    assert wavelengths.ndim == 1

    if background is not None:
        background = optional_to(background, units)
        assert background.ndim == 1
        assert background.shape == wavelengths.shape

    centers = np.atleast_2d(centers)
    std = np.atleast_2d(std)
    wavelengths = wavelengths[:, None]
    if background is not None:
        background = background[:, None]

    if cdf is None:
        spectrum_array = norm.pdf(wavelengths, centers, std)
    elif cdf:
        spectrum_array = norm.cdf(wavelengths, centers, std)
    else:
        spectrum_array = 1 - norm.cdf(wavelengths, centers, std)

    if filter and background is not None:
        # here spectrum array acts as a filter of the background illuminant
        spectrum_array = intensity * (
            background
            * spectrum_array
            / np.max(spectrum_array, axis=0, keepdims=True)
        )
    elif max_normalized:
        spectrum_array = intensity * (
            spectrum_array / np.max(spectrum_array, axis=0, keepdims=True)
        )
    else:
        spectrum_array *= (
            intensity
            # integral to ensure it integrate to intensity
            / np.trapz(spectrum_array, wavelengths, axis=0)
        )

    if add_background and background is not None:
        spectrum_array += background

    if zero_cutoff:
        spectrum_array = np.clip(spectrum_array, 0, None)

    domain_units = kwargs.pop('domain_units', 'nm')

    return Signals(
        spectrum_array,
        domain=np.squeeze(wavelengths),
        units=units,
        domain_units=domain_units,
        **kwargs
    )
