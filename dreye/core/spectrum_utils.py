"""
Functions for creating or fitting particular spectra
"""

import numpy as np
from scipy.stats import norm

from dreye.utilities import optional_to
from dreye.core.spectrum import IntensitySpectra, Spectrum
from dreye.core.signal import _SignalMixin
from dreye.constants import ureg


def get_spectrum(
    intensities=None,
    wavelengths=None,
    **kwargs
):
    """
    convenience function to create a Spectrum instance
    """

    if intensities is None:
        if wavelengths is None:
            wavelengths = np.arange(300, 700.1, 0.5)
        intensities = np.ones(len(wavelengths))
        intensities /= np.trapz(intensities, wavelengths)
    elif wavelengths is None:
        wavelengths = np.linspace(300, 700, len(wavelengths))

    return Spectrum(
        values=intensities,
        domain=wavelengths,
        **kwargs
    )


def create_gaussian_spectrum(
    wavelengths, centers, std=10,
    intensity=1,
    units='uE',
    cdf=None,
    background=None,
    filter=False,
    add_background=False,
    zero_cutoff=True,
    **kwargs
):
    """
    Get IntensitySpectra instance from Gaussian probability distribution.

    wavelengths : array-like
        Wavelength array in nm.
    centers : float, array-like
        The centers for each single wavelength Gaussian in nm.
    std : float, array-like
        The standard deviation for each single wavelength Gaussian in nm.
    intensity : float
        The intensity of the single wavelength Gaussians.
    units : str
        The unit and scale specified if in photon flux, e.g.
        'microspectralphotonflux' or simply 'spectralirradiance'.
    cdf : bool
        Allows changes to the distribution function. If True, the cdf will ; if
        cdf=None, the pdf will be plotted; if cdf=False, 1-cdf will be plotted
    background : array-like
        A numpy array which specifies the background distribution. Added to the
        wl array, and therefore must be the same length.
    filter : bool
        If True, filters the wavelength array by the background spectrum
        instead of adding background as a pure spectral distribution.
    add_background : array-like
        A numpy array of same length as the wl array which is added to the
        background distribution.
    zero_cutoff : bool
        If True, spectral distribution clipped at 0 and negative values
        discarded.
    """
    if isinstance(background, _SignalMixin):
        background = background(wavelengths).to(units)

    units = IntensitySpectra._unit_mappings.get(units, units)
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

    return IntensitySpectra(
        spectrum_array,
        domain=np.squeeze(wavelengths),
        units=units,
        **kwargs
    )
