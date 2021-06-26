"""
Utility functions
"""

import numpy as np

from dreye.constants import ureg
from dreye.core.spectral_measurement import MeasuredSpectraContainer
from dreye.core.photoreceptor import Photoreceptor, create_photoreceptor_model
from dreye.utilities.common import (
    is_dictlike, is_listlike, is_signallike, is_string, optional_to, 
    is_numeric
)
from dreye.core.measurement_utils import create_measured_spectra_container
from dreye.core.spectrum_utils import create_spectrum



def check_measured_spectra(
    measured_spectra,
    size=None, photoreceptor_model=None,
    change_dimensionality=True, 
    wavelengths=None, intensity_bounds=None
):
    """
    check and create measured spectra container if necessary
    """
    if isinstance(measured_spectra, MeasuredSpectraContainer):
        pass
    elif is_dictlike(measured_spectra):
        if photoreceptor_model is not None:
            # assumes Photoreceptor instance
            measured_spectra['wavelengths'] = measured_spectra.get(
                'wavelengths', photoreceptor_model.wavelengths
            )
        measured_spectra['led_spectra'] = measured_spectra.get(
            'led_spectra', size
        )
        measured_spectra['intensity_bounds'] = measured_spectra.get(
            'intensity_bounds', intensity_bounds
        )
        measured_spectra = create_measured_spectra_container(
            **measured_spectra
        )
    elif is_listlike(measured_spectra):
        measured_spectra = create_measured_spectra_container(
            led_spectra=measured_spectra, 
            intensity_bounds=intensity_bounds, 
            wavelengths=(
                wavelengths
                if photoreceptor_model is None
                else photoreceptor_model.wavelengths 
                if wavelengths is None else
                wavelengths
            )
        )
    elif measured_spectra is None:
        measured_spectra = create_measured_spectra_container(
            size, 
            intensity_bounds=intensity_bounds, 
            wavelengths=(
                wavelengths
                if photoreceptor_model is None
                else photoreceptor_model.wavelengths 
                if wavelengths is None else
                wavelengths
            )
        )
    else:
        raise ValueError("Measured Spectra must be Spectra "
                            "container or dict, but is type "
                            f"'{type(measured_spectra)}'.")

    # enforce photon flux for photoreceptor models
    if (
        photoreceptor_model is not None
        and (
            measured_spectra.units.dimensionality
            != ureg('uE').units.dimensionality
        )
        and change_dimensionality
    ):
        return measured_spectra.to('uE')

    return measured_spectra


def check_photoreceptor_model(
    photoreceptor_model, size=None, 
    wavelengths=None, capture_noise_level=None
):
    """
    check and create photoreceptor model if necessary
    """
    if isinstance(photoreceptor_model, Photoreceptor):
        photoreceptor_model = type(photoreceptor_model)(
            photoreceptor_model, wavelengths=wavelengths, capture_noise_level=capture_noise_level
        )
    elif is_dictlike(photoreceptor_model):
        photoreceptor_model['sensitivity'] = photoreceptor_model.get(
            'sensitivity', size
        )
        photoreceptor_model['capture_noise_level'] = photoreceptor_model.get(
            'capture_noise_level', capture_noise_level
        )
        photoreceptor_model['wavelengths'] = photoreceptor_model.get(
            'wavelengths', wavelengths
        )
        photoreceptor_model = create_photoreceptor_model(
            **photoreceptor_model
        )
    elif is_listlike(photoreceptor_model):
        photoreceptor_model = create_photoreceptor_model(
            sensitivity=photoreceptor_model, 
            wavelengths=wavelengths, 
            capture_noise_level=capture_noise_level
        )
    elif photoreceptor_model is None:
        photoreceptor_model = create_photoreceptor_model(
            size, 
            wavelengths=wavelengths, 
            capture_noise_level=capture_noise_level
        )
    else:
        raise ValueError("Photoreceptor model must be Photoreceptor "
                            "instance or dict, but is type "
                            f"'{type(photoreceptor_model)}'.")

    return photoreceptor_model


def check_background(background, measured_spectra, wavelengths=None):
    """
    check and create background if necessary
    """
    # enforce measured_spectra units
    if is_dictlike(background):
        background['wavelengths'] = background.get(
            'wavelengths', measured_spectra.wavelengths
        )
        background['units'] = measured_spectra.units
        background = create_spectrum(**background)
    elif background is None:
        return
    elif is_string(background) and (background == 'null'):
        return
    elif is_string(background) and (background == 'mean'):
        return background
    elif is_signallike(background):
        background = background.to(measured_spectra.units)
    elif is_listlike(background):
        background = optional_to(background, measured_spectra.units)
        # check size requirements
        if wavelengths is None:
            assert background.size == measured_spectra.normalized_spectra.shape[
                measured_spectra.normalized_spectra.domain_axis
            ], "array-like object for `background` does not match wavelength shape of `measured_spectra` object."
        background = create_spectrum(
            intensities=background,
            wavelengths=(measured_spectra.domain if wavelengths is None else wavelengths),
            units=measured_spectra.units
        )
    else:
        raise ValueError(
            "Background must be Spectrum instance or dict-like, but"
            f"is of type {type(background)}."
        )

    return background


def get_background_from_bg_ints(bg_ints, measured_spectra):
    """
    Get background from background intensity array
    """
    # sanity check
    assert measured_spectra.normalized_spectra.domain_axis == 0
    try:
        # assume scipy.interpolate.interp1d
        return measured_spectra.ints_to_spectra(
            bg_ints, bounds_error=False, fill_value='extrapolate'
        )
    except TypeError:
        return measured_spectra.ints_to_spectra(bg_ints)


def get_bg_ints(bg_ints, measured_spectra, rtype=None):
    """
    Get background intensity values and convert units if necessary
    """
    if rtype is None:
        default = 0
    elif rtype in {'absolute', 'diff'}:
        default = 0
    else:  # relative types
        default = 1

    # set background intensities to default
    if bg_ints is None:
        bg_ints = np.ones(len(measured_spectra)) * default
    elif is_numeric(bg_ints):
        bg_ints = np.ones(
            len(measured_spectra)
        ) * optional_to(bg_ints, measured_spectra.intensities.units)
    else:
        if is_dictlike(bg_ints):
            names = measured_spectra.names
            bg_ints = [bg_ints.get(name, default) for name in names]
        bg_ints = optional_to(
            bg_ints,
            measured_spectra.intensities.units
        )
        assert len(bg_ints) == len(measured_spectra)
        assert np.all(bg_ints >= 0)
    return bg_ints


def get_ignore_bounds(ignore_bounds, measured_spectra, intensity_bounds):
    # ignore bounds depending on logic
    if ignore_bounds is None:
        return (
            not isinstance(measured_spectra, MeasuredSpectraContainer) 
            and intensity_bounds is None
        )
    return ignore_bounds


def estimate_bg_ints_from_background(
    Xbg,
    photoreceptor_model,
    background, 
    measured_spectra,  
    wavelengths=None,
    intensity_bounds=None,
    fit_weights=None, 
    max_iter=None, 
    ignore_bounds=None, 
    lsq_kwargs=None, 
):
    """
    Estimate background intensity for light sources given a background spectrum to fit to.
    """
    # internal import required here
    from dreye import IndependentExcitationFit
    # fit background and assign bg_ints_
    # build estimator
    # if subclasses should still use this fitting procedure
    est = IndependentExcitationFit(
        photoreceptor_model=photoreceptor_model,
        fit_weights=fit_weights,
        background=background,
        measured_spectra=measured_spectra,
        max_iter=max_iter,
        unidirectional=False,  # always False for fitting background
        bg_ints=None,
        fit_only_uniques=False,
        ignore_bounds=ignore_bounds,
        lsq_kwargs=lsq_kwargs,
        background_external=False, 
        intensity_bounds=intensity_bounds, 
        wavelengths=wavelengths
    )
    est._lazy_background_estimation = True
    est.fit(np.atleast_2d(Xbg))
    # assign new bg_ints_
    return est.fitted_intensities_[0]