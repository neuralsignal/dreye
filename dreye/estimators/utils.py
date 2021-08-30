"""
Utility functions
"""

from itertools import combinations, product
import numpy as np
from sklearn.preprocessing import normalize

from dreye.constants import ureg
from dreye.core.spectral_measurement import MeasuredSpectraContainer
from dreye.core.photoreceptor import CAPTURE_STRINGS, Photoreceptor, create_photoreceptor_model
from dreye.utilities.common import (
    is_dictlike, is_listlike, is_signallike, is_string, optional_to, 
    is_numeric
)
from dreye.core.signal import Signal
from dreye.core.measurement_utils import create_measured_spectra_container
from dreye.core.spectrum_utils import create_spectrum
from dreye.utilities.array import asarray
from dreye.utilities.convex import in_hull
from dreye.err import DreyeError


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
        # TODO - intensity_bounds?, photoreceptor_model interpolate wavelengths
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
            photoreceptor_model, 
            wavelengths=wavelengths, 
            capture_noise_level=capture_noise_level
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


def check_background(background, measured_spectra, wavelengths=None, photoreceptor_model=None):
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
    elif is_string(background) and (background in CAPTURE_STRINGS):
        return background
    elif is_signallike(background):
        background = background.to(measured_spectra.units)
    elif is_listlike(background):
        background = optional_to(background, measured_spectra.units)
        if (photoreceptor_model is None) or (photoreceptor_model.n_opsins != background.size):
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
    **kwargs
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
        wavelengths=wavelengths, 
        **kwargs
    )
    est._lazy_background_estimation = True
    est.fit(np.atleast_2d(Xbg))
    # assign new bg_ints_
    return est.fitted_intensities_[0]


# functions related to light sources and indexing measured_spectra


def get_source_idcs_from_k(n, k):
    idcs = np.array(list(combinations(np.arange(n), k)))
    source_idcs = np.zeros((len(idcs), n)).astype(bool)
    source_idcs[
        np.repeat(np.arange(len(idcs)), k),
        idcs.ravel()
    ] = True
    return source_idcs


def get_source_idcs(names, combos):
    n = len(names)
    if is_numeric(combos):
        combos = int(combos)
        source_idcs = get_source_idcs_from_k(
            n, combos
        )
    elif is_listlike(combos):
        combos = asarray(combos)
        if combos.ndim == 1:
            if is_string(combos[0]):  # if first element is string all should be
                source_idcs = np.array([
                    get_source_idx(names, source_idx, asbool=True)
                    for source_idx in combos
                ])
            else:  # else assume it is a list of ks
                combos = combos.astype(int)
                source_idcs = []
                for k in combos:
                    source_idx = get_source_idcs_from_k(n, k)
                    source_idcs.append(source_idx)
                source_idcs = np.vstack(source_idcs)
        elif combos.ndim == 2:
            source_idcs = combos.astype(bool)
        else:
            raise ValueError(
                f"`combos` dimensionality is `{combos.ndim}`, "
                "but needs to be 1 or 2."
            )
    else:
        raise TypeError(
            f"`combos` is of type `{type(combos)}`, "
            "but must be numeric or array-like."
        )

    return source_idcs


def get_source_idx(names, source_idx, asbool=False):
    if is_string(source_idx):
        source_idx = [
            names.index(name)
            for name in source_idx.split('+')
        ]
    source_idx = asarray(source_idx)
    if source_idx.dtype == np.bool and not asbool:
        source_idx = np.flatnonzero(source_idx)
    elif asbool and source_idx.dtype != np.bool:
        _source_idx = np.zeros(len(names)).astype(bool)
        _source_idx[source_idx] = True
        source_idx = _source_idx
    return source_idx


def get_source_idx_string(names, source_idx):
    if is_string(source_idx):
        return source_idx
    source_idx = asarray(source_idx)
    return '+'.join(asarray(names)[source_idx])


def get_spanning_intensities(
    intensity_bounds,
    ratios=np.linspace(0., 1., 11), 
    compute_ratios=False
):
    """
    Get intensities given intensity bounds that span 
    capture space appropriately.
    """
    n = len(intensity_bounds[0])
    samples = np.array(list(product(*([[0, 1]] * n)))).astype(float)
    samples *= (intensity_bounds[1] - intensity_bounds[0]) + intensity_bounds[0]
    if compute_ratios:
        samples_ = []
        for (idx, isample), (jdx, jsample) in product(enumerate(samples), enumerate(samples)):
            if idx >= jdx:
                continue
            s_ = (ratios[:, None] * isample) + ((1-ratios[:, None]) * jsample)
            samples_.append(s_)
        samples_ = np.vstack(samples_)
        samples = np.vstack([samples, samples_])
    samples = np.unique(samples, axis=0)
    return samples


def get_optimal_capture_samples(
    photoreceptor_model : Photoreceptor, 
    background : Signal, 
    ratios : np.ndarray = np.linspace(0., 1., 11), 
    compute_isolation : bool = False,
    compute_ratios : bool = False
) -> np.ndarray:
    """
    Get optimal capture samples for the chromatic hyperplane
    """
    dirac_delta_spectra = np.eye(photoreceptor_model.wls.size)
    captures = photoreceptor_model.capture(
        dirac_delta_spectra,
        background=background,
        return_units=False
    )
    # normalize for chromatic plane (proportional captures)
    captures = normalize(captures, norm='l1', axis=1)
    if compute_isolation:
        # fast approximation of isolating captures (max of ratios)
        isolating_captures = captures[np.argmax(captures, axis=0)]
        captures = np.vstack([captures, isolating_captures])
        if compute_ratios:
            qs = []
            for idx, jdx in product(range(photoreceptor_model.n_opsins), range(photoreceptor_model.n_opsins)):
                if idx >= jdx:
                    continue
                qs_ = (
                    isolating_captures[idx] * ratios[:, None] 
                    + isolating_captures[jdx] * (1-ratios[:, None])
                )
                qs.append(qs_)
            qs = np.vstack(qs)
            captures = np.vstack([captures, qs])
        captures = np.unique(captures, axis=0)
    return captures


def range_of_solutions(
    A, x, bounds, check=True
):
    """
    Range of solutions for underdetermined matrix

    A - opsins x leds - normalized capture matrix
    x - opsin captures
    bounds - (min, max)
    """

    if check:
        assert A.shape[0] < A.shape[1], "System is not underdetermined."
        samples = get_spanning_intensities(bounds)
        points = samples @ A.T  # n_samples x n_opsins
        assert in_hull(points, x), "No perfect solutions exist for system."

    maxs = bounds[0].copy()
    mins = bounds[1].copy()

    # difference between number of opsins and leds
    n_diff = A.shape[1] - A.shape[0]
    
    # combinations x n_diff
    omat = np.array(list(product(*[[0, 1]]*n_diff)))

    idcs = np.arange(A.shape[1])    
    for ridcs in product(*[idcs]*n_diff):
        # always use ascending order idcs
        if not np.all(np.diff(np.argsort(ridcs)) < 0):
            # ascending always
            continue
        
        ridcs = list(ridcs)
        Arest = A[:, ridcs]
        # combinations x n_diff
        offsets = omat * (
            bounds[1][ridcs] - bounds[0][ridcs]
        ) + bounds[0][ridcs]
        # filter non-real values
        offsets[np.isnan(offsets)] = 0.0
        offsets = offsets[np.isfinite(offsets).all(axis=1)]
        # combinations x opsins
        offset = (offsets @ Arest.T)
        
        Astar = np.delete(A, ridcs, axis=1)
        
        sols = np.linalg.solve(
            Astar, 
            (x - offset).T
        ).T
        # combinations x used leds

        # allowed bounds for included 
        b0 = np.delete(bounds[0], ridcs)
        b1 = np.delete(bounds[1], ridcs)
        
        # all within bounds
        psols = np.all((sols >= b0) & (sols <= b1), axis=1)
        
        if not np.any(psols):
            # continue if no solutions exist
            continue

        # add minimum and maximum to global minimum and maximum
        rbool = np.isin(idcs, ridcs)
        _mins = np.zeros(A.shape[1])
        _maxs = np.zeros(A.shape[1])

        _mins[rbool] = offsets[psols].min(axis=0)
        _maxs[rbool] = offsets[psols].max(axis=0)
        _mins[~rbool] = sols[psols].min(axis=0)
        _maxs[~rbool] = sols[psols].max(axis=0)

        mins = np.minimum(mins, _mins)
        maxs = np.maximum(maxs, _maxs)
        
    return mins, maxs


def spaced_solutions(
    wmin, wmax,
    A, x, n=20, 
    eps=1e-7
):
    """
    Spaced intensity solutions from `wmin` to `wmax`.
    If the difference between the number
    of light sources and the number of opsins exceeds 1 
    than the number of samples cannot be predetermined 
    but will be greate than `n` ** `n_diff`.
    """
    n_diff = A.shape[1] - A.shape[0]

    if n_diff > 1:
        # go through first led -> remove -> find new range of solution
        # restrict by wmin and wmax
        # repeat until only one extra LED
        # then get equally spaced solutions
        ws = []

        # for numerical stability
        eps_ = (wmax - wmin) * eps
        idcs = np.arange(A.shape[1])

        for idx in idcs:
            # for indexing
            not_idx = ~(idcs == idx)
            argsort = np.concatenate([[idx], idcs[not_idx]])
            
            for iw in np.linspace(wmin[idx]+eps_[idx], wmax[idx]-eps_[idx], n):
                Astar = A[:, not_idx]
                offset = A[:, idx] * iw
                xstar = x - offset
                boundsstar = (
                    wmin[not_idx], 
                    wmax[not_idx]
                )
                wminstar, wmaxstar = range_of_solutions(
                    Astar, xstar, boundsstar, check=False
                )
                if (wminstar > wmaxstar).any():
                    continue

                wsstar = spaced_solutions(
                    wminstar, wmaxstar, Astar, xstar, n=n, 
                    eps=eps
                )
                w_ = np.hstack([
                    np.ones((wsstar.shape[0], 1)) * iw, 
                    wsstar
                ])[:, argsort]
                ws.append(w_)

        return np.vstack(ws)

    else:
        # create equally space solutions
        ws = np.zeros((n, A.shape[1]))
        
        # for numerical stability
        eps_ = (wmax[0] - wmin[0]) * eps
        
        idx = 0
        for iw in np.linspace(wmin[0]+eps_, wmax[0]-eps_, n):
            Astar = A[:, 1:]
            offset = A[:, 0] * iw
            sols = np.linalg.solve(
                Astar, 
                (x - offset)
            )
            ws[idx] = np.concatenate([[iw], sols])
            idx += 1
            
        return ws
