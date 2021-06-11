"""
Excitation models
"""

import inspect
import warnings

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import lsq_linear, least_squares
from sklearn.utils.validation import check_array, check_is_fitted

from dreye.utilities import (
    optional_to, asarray, is_listlike, is_callable
)
from dreye.constants import ureg
from dreye.core.spectral_measurement import MeasuredSpectraContainer
from dreye.estimators.base import _SpectraModel, OptimizeResultContainer
from dreye.err import DreyeError
from dreye.utilities.abstract import inherit_docstrings


@inherit_docstrings
class IndependentExcitationFit(_SpectraModel):
    """
    Class to fit (relative) photoreceptor excitations for each sample
    independently.

    Parameters
    ----------
    photoreceptor_model : dreye.Photoreceptor, optional
        A photoreceptor model that implements the `capture`, `excitation`,
        `excitefunc` and `inv_excitefunc` methods. If None,
        a fake photoreceptor model will be created with three different
        photoreceptor types.
    fit_weights : array-like, optional
        Weighting of the importance of each photoreceptor type in the model.
        If None, weighting will be equal between all photoreceptor types.
        Must be same length as the number of photoreceptor types
    background : dreye.Signal, optional
        The spectral distribution of the background illuminant. Assumes
        that it contains the whole added spectrum including the LED intensities,
        if they are non-zero.
    measured_spectra : dreye.MeasuredSpectraContainer, optional
        Container with all available LEDs and their measured spectra. If
        None, a fake LED measurement will be created with intensities
        ranging from 0 to 100 microphotonflux.
    max_iter : int, optional
        The number of maximum iterations. This is passed directly to
        `scipy.optimize.lsq_linear` and `scipy.optimize.least_squares`.
    hard_separation : bool or list-like, optional
        An array of LED intensities.
        If given and all capture values are below or above `hard_sep_value`,
        then do not allow the LED intensities to go above or below
        these intensities. If True, first estimate the optimal LED
        intensities that correspond to the relative capture
        of `hard_sep_value`. If True and `bg_ints` are given and
        all `hard_sep_value` are 1, then `hard_separation` is set to `bg_ints`.
    hard_sep_value : numeric or numpy.ndarray, optional
        The capture value for `hard_separation`. Defaults to 1, which
        corresponds to the relative capture when the illuminant equals
        the background.
    bg_ints : numpy.ndarray, optional
        The intensity values for each LED, when the relative capture of each
        photoreceptor equals one (i.e. background intensity).
        This will prevent fitting of the
        LED intensities if the background LED intensities
        are preset and the relative capture is 1.
    fit_only_uniques : bool, optional
        If True, use `numpy.unique` to select only the unique samples
        for fitting before transforming X back to the full array.
    ignore_bounds : bool, optional
        If True, ignore the bounds of the LED intensities. Howerver, a zero
        LED intensity bound will always exist.
    lsq_kwargs : dict, optional
        Keyword arguments passed directly to `scipy.optimize.least_squares`.

    Attributes
    ----------
    photoreceptor_model_ : dreye.Photoreceptor
        The photoreceptor model used for fitting. This will be the same if
        as `photoreceptor_model` if it is a `dreye.Photoreceptor` instance.
    measured_spectra_ : dreye.MeasuredSpectraContainer (n_leds)
        Measured spectrum container used for fitting. This will be the same
        if as `measured_spectra` if a `dreye.MeasuredSpectraContainer` instance
        was passed.
    bounds_ : numpy.ndarray (n_leds)
        The LED intensity bounds used for fitting.
    background_ : dreye.Spectrum
        The background used for calculating the relative photon capture.
    normalized_spectra_ : dreye.Spectra
        The normalizes LED spectra. Each spectrum integrates to 1.
    A_ : numpy.ndarray (n_prs, n_leds)
        The relative photon capture of each normalized LED spectrum.
    sep_bound_ : numpy.ndarray (n_leds)
        The LED intensities used to as new bounds, if `hard_separation`
        was set.
    sep_result_ : scipy.optimize.OptimizeResult
        The result if `hard_separation` was set to `True` for fitting.
    capture_X_ : numpy.ndarray (n_samples, n_prs)
        The current relative photon capture values used for fitting.
    excite_X_ : numpy.ndarray (n_samples, n_prs)
        The current photoreceptor excitation values used for fitting.
    fitted_intensities_ : numpy.ndarray
        Intensities fit in units of `measured_spectra_.intensities.units`
    fitted_capture_X_ : numpy.ndarray (n_samples, n_prs)
        The recalculated relative photon capture values after fitting.
    fitted_excite_X_ : numpy.ndarray (n_samples, n_prs)
        The recalculated photoreceptor excitations after fitting.
    """

    # same length as X but not X or fitted X
    _X_length = [
        'fitted_intensities_'
    ]
    _deprecated_kws = {
        **_SpectraModel._deprecated_kws,
        "photoreceptor_fit_weights": "fit_weights",
        "q1_ints": "bg_ints",
        "smoothing_window": None
    }
    fit_to_transform = False
    _skip_bg_ints = True
    _lazy_clone = False
    _requirements_set = False

    def __init__(
        self,
        *,
        photoreceptor_model=None,  # dict or Photoreceptor class
        fit_weights=None,
        background=None,  # dict or Spectrum instance or array-like
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        max_iter=None,
        hard_separation=False,  # bool or list-like (same length as number of LEDs)
        hard_sep_value=None,  # float in capture units (1 relative capture)
        bg_ints=None,
        fit_only_uniques=False,
        ignore_bounds=None,
        lsq_kwargs=None,
        ignore_capture_units=True,
        background_only_external=False, 
        intensity_bounds=None, 
        wavelengths=None
    ):
        self.photoreceptor_model = photoreceptor_model
        self.measured_spectra = measured_spectra
        self.background = background
        self.max_iter = max_iter
        self.hard_separation = hard_separation
        self.hard_sep_value = hard_sep_value
        self.fit_weights = fit_weights
        self.fit_only_uniques = fit_only_uniques
        self.lsq_kwargs = lsq_kwargs
        self.ignore_bounds = ignore_bounds
        self.bg_ints = bg_ints
        self.ignore_capture_units = ignore_capture_units
        self.background_only_external = background_only_external
        self.intensity_bounds = intensity_bounds
        self.wavelengths = wavelengths

    def _set_required_objects(self, size=None):
        # TODO split into separate routines
        if self._requirements_set:
            return self
        # create photoreceptor model
        self.photoreceptor_model_ = self._check_photoreceptor_model(
            self.photoreceptor_model, size=size, 
            wavelengths=self.wavelengths
        )
        self.n_features_ = self.photoreceptor_model_.pr_number
        self.channel_names_ = self.photoreceptor_model_.names

        # create measured_spectra_
        self.measured_spectra_ = self._check_measured_spectra(
            self.measured_spectra,
            photoreceptor_model=self.photoreceptor_model_, 
            wavelengths=self.wavelengths, 
            intensity_bounds=self.intensity_bounds
        )
        self.n_leds_ = len(self.measured_spectra_)
        # fit isotonic regression
        self.measured_spectra_._assign_mapper()

        # set background intensities if exist
        self.bg_ints_ = self._get_bg_ints(
            self.bg_ints, self.measured_spectra_,
            skip=self._skip_bg_ints,  # skip if bg_ints is None
            rtype=getattr(self, 'rtype', None)
        )

        ignore_bounds = self._get_ignore_bounds()

        # measured_spectra attributes
        # intensity bounds as two-tuple
        if ignore_bounds:
            self.bounds_ = (
                np.zeros(self.n_leds_),
                np.inf * np.ones(self.n_leds_)
            )
        else:
            self.bounds_ = self.measured_spectra_.intensity_bounds
        # normalized spectra
        self.normalized_spectra_ = self.measured_spectra_.normalized_spectra
        # sanity checks
        assert self.normalized_spectra_.domain_axis == 0

        # create background
        self.background_ = self._check_background(
            self.background, self.measured_spectra_, 
            wavelengths=self.wavelengths
        )

        self.is_measurement_background_ = True
        if self.background_ is not None and self.bg_ints_ is not None:
            background_ = self._get_background_from_bg_ints(
                self.bg_ints_, self.measured_spectra_, skip_zero=False
            )
            if self.background_only_external:
                self.background_ = self.background_ + background_
            if not np.allclose(self.background_.magnitude, background_.magnitude):
                if not self.background_only_external:
                    warnings.warn(
                        "`background` differs from `bg_ints`. "
                        "Assuming background is the sum of "
                        "an external source and `bg_ints`."
                    )
                self.is_measurement_background_ = False
                self.bg_ints_background_ = background_

                assert np.all(
                    background_.magnitude <= self.background_.magnitude
                ), "`background` spectrum does not add up with `bg_ints`."
        elif self.background_ is not None and self.bg_ints_ is None:
            if self.background_only_external:
                # warnings.warn(
                #     "Assuming `bg_ints` are all zero, "
                #     "since `background_only_external` set to True."
                # )
                self.bg_ints_ = np.zeros(self.n_leds_)
            elif (
                not self._lazy_clone
                and not np.allclose(self.background_.magnitude, 0)
            ):
                # fit background and assign bg_ints_
                # get params from init
                params = inspect.signature(
                    IndependentExcitationFit.__init__
                ).parameters
                # build estimator
                # if subclasses should still use this fitting procedure
                est = IndependentExcitationFit(
                    **{
                        param: getattr(self, param)
                        for param in params
                        if param != 'self'
                    }
                )
                est._lazy_clone = True
                X = np.ones((1, self.n_features_))
                X = self.photoreceptor_model_.excitefunc(X)
                est.fit(X)
                # est.fit(X)
                self.bg_ints_ = est.fitted_intensities_[0]

                warnings.warn(
                    "Assuming the `background` illuminant will be simulated "
                    "using the LEDs. Fitted background intensities: "
                    f"{self.bg_ints_.tolist()}."
                )
        elif self.background_ is None and self.bg_ints_ is not None:
            if not np.allclose(self.bg_ints_, 0):
                # will integrate with normalized spectra
                self.background_ = self._get_background_from_bg_ints(
                    self.bg_ints_, self.measured_spectra_
                )

        if (
            self.background_ is not None
            and np.allclose(self.background_.magnitude, 0)
        ):
            warnings.warn(
                "background array is all zero, "
                "setting `background` illuminant to None."
            )
            self.background_ = None

        if self.background_ is None and not self.ignore_capture_units:
            assert (
                self.measured_spectra_.units
                * self.photoreceptor_model_.sensitivity.units
                * self.photoreceptor_model_.wavelengths.units
            ).dimensionless, 'units not dimensionless'
        elif not self.ignore_capture_units:
            assert (
                self.measured_spectra_.units
                / self.background_.units
            ).dimensionless, 'units not dimensionless'

        # number of photoreceptors
        # if background is None (all zeros or None) then capture border is 0
        self.capture_border_ = float(self.background_ is not None)
        self.hard_sep_value_ = (
            self.capture_border_
            if self.hard_sep_value is None
            else self.hard_sep_value
        )

        # weighting for each photoreceptor
        if self.fit_weights is None:
            self.fit_weights_ = np.ones(self.photoreceptor_model_.pr_number)
        else:
            self.fit_weights_ = asarray(self.fit_weights)
            # assert len(fit_weights) == self.photoreceptor_model_.pr_number

        # do weighted initial fitting in linear domain
        self.weighted_init_fit_ = (
            not self.fit_to_transform
            and np.unique(self.fit_weights_).size > 1
        )

        self._set_A()

        # make sure requirements are only set once
        self._requirements_set = True

        return self

    def _set_A(self):
        if np.any(self.capture_noise_level_):
            self.noise_term_ = self.photoreceptor_model_.capture(
                np.zeros(self.normalized_spectra_.domain.size),
                wavelengths=self.normalized_spectra_.domain,
                background=self.background_, 
                return_units=False
            )[0]
            # eps / (qb+eps)
        else:
            self.noise_term_ = np.zeros(self.n_features_)

        # opsin x LED (taking transpose)
        self.A_ = self.photoreceptor_model_.capture(
            self.normalized_spectra_,
            background=self.background_,
            return_units=False
        ).T - self.noise_term_[:, None]

        if not self.is_measurement_background_:
            self.q_bg_ = self.photoreceptor_model_.capture(
                self.background_ - self.bg_ints_background_,
                background=self.background_,
                return_units=False
            )[0] - self.noise_term_ # length of opsin
        else:
            self.q_bg_ = 0

        return self

    def _fit(self, X):
        """
        Actual Fitting method. Allows easier subclassing
        """

        # overwrite this method when subclassing
        self.capture_X_, self.excite_X_ = self._process_X(X)

        # whether domain between spectra and photoreceptor model equal
        self._domain_equal_ = (
            self.normalized_spectra_.domain
            == self.photoreceptor_model_.sensitivity.domain
        )

        # separation of all negative or positive
        if is_listlike(self.hard_separation):
            sep_bound = optional_to(
                self.hard_separation,
                self.measured_spectra_.intensities.units
            )
            assert len(sep_bound) == self.n_leds_, (
                "hard_separation length must match length of "
                "measured spectra container."
            )
            self.sep_bound_ = sep_bound
            self.sep_result_ = None
        elif (
            self.hard_separation
            and np.all(self.hard_sep_value_ == self.capture_border_)
            and (self.bg_ints_ is not None)
        ):
            self.sep_bound_ = self.bg_ints_
            self.sep_result_ = None
        elif self.hard_separation:
            sep_value = (
                np.ones(self.photoreceptor_model_.pr_number)
                * self.hard_sep_value_
            )
            # separation bound using a fit to sep_value
            # capture and excite!
            sep_result = self._fit_sample(
                sep_value,
                self.photoreceptor_model_.excitefunc(sep_value),
            )
            self.sep_bound_ = sep_result.x  # just get weight
            self.sep_result_ = sep_result
        else:
            self.sep_bound_ = None
            self.sep_result_ = None

        # if only fit uniques used different iterator
        if self.fit_only_uniques:
            # get uniques
            _, xidcs, xinverse = np.unique(
                self.capture_X_, axis=0, return_index=True, return_inverse=True
            )
            self.container_ = OptimizeResultContainer(np.array([
                self._fit_sample(
                    capture_x, excite_x, self.sep_bound_
                )
                for capture_x, excite_x in
                zip(self.capture_X_[xidcs], self.excite_X_[xidcs])
            ])[xinverse])
        else:
            self.container_ = OptimizeResultContainer([
                self._fit_sample(
                    capture_x, excite_x, self.sep_bound_
                )
                for capture_x, excite_x in
                zip(self.capture_X_, self.excite_X_)
            ])

        if not np.all(self.container_.success):
            warnings.warn("Convergence was not accomplished "
                          "for all spectra in X; "
                          "increase the number of max iterations.")

        self.fitted_intensities_ = np.array(self.container_.x)
        # get fitted X
        self.fitted_excite_X_ = np.array([
            self._get_x_pred(w)
            for w in self.fitted_intensities_
        ])
        self.fitted_capture_X_ = self.photoreceptor_model_.inv_excitefunc(
            self.fitted_excite_X_
        )

        return self

    def fit(self, X, y=None):
        # set required objects
        self._set_required_objects(asarray(X).shape[1])
        # fit X
        self._fit(X)
        return self

    def inverse_transform(self, X):
        check_is_fitted(
            self, [
                'measured_spectra_',
                'normalized_spectra_',
                'photoreceptor_model_',
                'background_'
            ]
        )
        # X is samples x LEDs
        X = optional_to(X, self.output_units)
        X = check_array(X)

        assert X.shape[1] == self.n_leds_

        # got from output to intensity
        X = self.measured_spectra_.inverse_map(X, return_units=False)
        # get excitation given intensity
        X = np.array([self._get_x_pred(x) for x in X])
        return X

    def _fit_sample(self, capture_x, excite_x, sep_bound=None):
        # adjust bounds if necessary
        bounds = list(self.bounds_)
        if sep_bound is not None:
            if np.all(capture_x >= self.hard_sep_value_):
                bounds[0] = sep_bound
            elif np.all(capture_x <= self.hard_sep_value_):
                bounds[1] = sep_bound
        if self.bg_ints_ is not None:
            # if np.allclose(capture_x, 1)
            # if close to the background intensity just use those values
            if np.allclose(capture_x, self.capture_border_):
                return OptimizeResult(
                    x=self.bg_ints_,
                    cost=0.0,
                    fun=np.zeros(capture_x.size),
                    jac=np.zeros((capture_x.size, self.bg_ints_.size)),
                    grad=np.zeros(capture_x.size),
                    optimality=0.0,
                    active_mask=0,
                    nfev=1,
                    njev=None,
                    status=4,
                    message='Set result to background intensities.',
                    success=True
                )
        # find initial w0 using linear least squares
        w0 = self._init_sample(capture_x, bounds)
        # fitted result
        result = least_squares(
            self._objective,
            x0=w0,
            args=(excite_x,),
            bounds=tuple(bounds),
            max_nfev=self.max_iter,
            **({} if self.lsq_kwargs is None else self.lsq_kwargs)
        )
        return result

    def _init_sample(self, capture_x, bounds):
        # weighted fit is better for substitution types of fits
        if self.weighted_init_fit_:
            result = lsq_linear(
                self.fit_weights_[:, None] * self.A_,
                self.fit_weights_ * (capture_x - self.q_bg_),
                bounds=tuple(bounds),
                max_iter=self.max_iter
            )
        else:
            result = lsq_linear(
                self.A_, (capture_x - self.q_bg_),
                bounds=tuple(bounds),
                max_iter=self.max_iter
            )
        # return fitted intensity (w)
        return result.x

    def _objective(self, w, excite_x):
        x_pred = self._get_x_pred(w)
        return self.fit_weights_ * (excite_x - x_pred)

    def _get_x_capture(self, w):
        if self.photoreceptor_model_.filterfunc is None:
            # threshold by noise if necessary and apply nonlinearity
            x_pred = self.A_ @ w
            if self.capture_noise_level_:
                x_pred += self.noise_term_
        else:
            if self._domain_equal_:
                illuminant = self.measured_spectra_.ints_to_spectra(w)
            else:
                illuminant = self.measured_spectra_.ints_to_spectra(w)(
                    self.normalized_spectra_.domain
                )

            x_pred = self.photoreceptor_model_.capture(
                # normalized_spectrum has domain_axis=0
                # TODO write measured spectra function that interpolates
                # to the given spectrum
                illuminant,
                background=self.background_,
                return_units=False
            )
            # ensure vector form
            x_pred = np.atleast_1d(np.squeeze(x_pred))
        x_pred += self.q_bg_
        return x_pred

    @property
    def capture_noise_level_(self):
        return self.photoreceptor_model_.capture_noise_level

    def _correct_for_noise(self, q, w):
        pass

    def _get_x_pred(self, w):
        return self.photoreceptor_model_.excitefunc(self._get_x_capture(w))

    def _process_X(self, X):
        """
        Returns photon capture and excitation value.
        """
        X = optional_to(X, ureg(None).units)
        X = check_array(X)
        # check that input shape is correct
        if X.shape[1] != self.n_features_:
            raise ValueError("Shape of input is different from number"
                             "of measured spectra in container.")

        # use inverse of excitation function
        capture_X = self.photoreceptor_model_.inv_excitefunc(X)
        excite_X = X
        return capture_X, excite_X

    @property
    def input_units(self):
        return ureg(None).units

    @property
    def fitted_X_(self):
        return self.fitted_excite_X_

    @property
    def X_(self):
        return self.excite_X_


@inherit_docstrings
class TransformExcitationFit(IndependentExcitationFit):
    """
    Class to fit a linear transformation of
    (relative) photoreceptor excitations for each sample independently.

    Parameters
    ----------
    linear_transform : numpy.ndarray, optional
        A linear transformation of the photoreceptor excitation space
        (n_prs, m_space).
    inv_transform : numpy.ndarray, optional
        The inverse of `linear_transform`. If not given, the inverse
        will be estimated using `np.linalg.inv`.
    photoreceptor_model : dreye.Photoreceptor, optional
        A photoreceptor model that implements the `capture`, `excitation`,
        `excitefunc` and `inv_excitefunc` methods. If None,
        a fake photoreceptor model will be created with three different
        photoreceptor types.
    fit_weights : array-like, optional
        Weighting of the importance of each photoreceptor type in the model.
        If None, weighting will be equal between all photoreceptor types.
        Must be same length as the number of photoreceptor types
    background : dreye.Signal, optional
        The spectral distribution of the background illuminant.
    measured_spectra : dreye.MeasuredSpectraContainer, optional
        Container with all available LEDs and their measured spectra. If
        None, a fake LED measurement will be created with intensities
        ranging from 0 to 100 microphotonflux.
    max_iter : int, optional
        The number of maximum iterations. This is passed directly to
        `scipy.optimize.lsq_linear` and `scipy.optimize.least_squares`.
    hard_separation : bool or list-like, optional
        An array of LED intensities.
        If given and all capture values are below or above `hard_sep_value`,
        then do not allow the LED intensities to go above or below
        these intensities. If True, first estimate the optimal LED
        intensities that correspond to the relative capture
        of `hard_sep_value`. If True and `bg_ints` are given and
        all `hard_sep_value` are 1, then `hard_separation` is set to `bg_ints`.
    hard_sep_value : numeric or numpy.ndarray, optional
        The capture value for `hard_separation`. Defaults to 1, which
        corresponds to the relative capture when the illuminant equals
        the background.
    bg_ints : numpy.ndarray, optional
        The intensity values for each LED, when the relative capture of each
        photoreceptor equals one (i.e. background intensity).
        This will prevent fitting of the
        LED intensities if the background LED intensities
        are preset and the relative capture is 1.
    fit_only_uniques : bool, optional
        If True, use `numpy.unique` to select only the unique samples
        for fitting before transforming X back to the full array.
    ignore_bounds : bool, optional
        If True, ignore the bounds of the LED intensities. Howerver, a zero
        LED intensity bound will always exist.
    lsq_kwargs : dict, optional
        Keyword arguments passed directly to `scipy.optimize.least_squares`.

    Attributes
    ----------
    photoreceptor_model_ : dreye.Photoreceptor
        The photoreceptor model used for fitting. This will be the same if
        as `photoreceptor_model` if it is a `dreye.Photoreceptor` instance.
    measured_spectra_ : dreye.MeasuredSpectraContainer (n_leds)
        Measured spectrum container used for fitting. This will be the same
        if as `measured_spectra` if a `dreye.MeasuredSpectraContainer` instance
        was passed.
    bounds_ : numpy.ndarray (n_leds)
        The LED intensity bounds used for fitting.
    background_ : dreye.Spectrum
        The background used for calculating the relative photon capture.
    normalized_spectra_ : dreye.Spectra
        The normalizes LED spectra. Each spectrum integrates to 1.
    A_ : numpy.ndarray (n_prs, n_leds)
        The relative photon capture of each normalized LED spectrum.
    sep_bound_ : numpy.ndarray (n_leds)
        The LED intensities used to as new bounds, if `hard_separation`
        was set.
    sep_result_ : scipy.optimize.OptimizeResult
        The result if `hard_separation` was set to `True` for fitting.
    capture_X_ : numpy.ndarray (n_samples, n_prs)
        The current relative photon capture values used for fitting.
    excite_X_ : numpy.ndarray (n_samples, n_prs)
        The current photoreceptor excitation values used for fitting.
    transform_X_ : numpy.ndarray (n_samples, m_space)
        The current linear transformation of the photoreceptor excitations
        used for fitting.
    fitted_intensities_ : numpy.ndarray
        Intensities fit in units of `measured_spectra_.intensities.units`
    fitted_capture_X_ : numpy.ndarray (n_samples, n_prs)
        The recalculated relative photon capture values after fitting.
    fitted_excite_X_ : numpy.ndarray (n_samples, n_prs)
        The recalculated photoreceptor excitations after fitting.
    fitted_transform_X_ : numpy.ndarray (n_samples, n_prs)
        The recalculated linear transform of the photoreceptor excitations
        after fitting.
    """

    # same length as X but not X or fitted X
    _X_length = IndependentExcitationFit._X_length + [
        'excite_X_',
        'fitted_excite_X_'
    ]

    def __init__(
        self,
        *,
        linear_transform=None,  # array
        inv_transform=None,  # array
        photoreceptor_model=None,  # dict or Photoreceptor class
        fit_weights=None,
        background=None,  # dict or Spectrum instance or array-like
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        max_iter=None,
        hard_separation=False,  # bool or list-like (same length as number of LEDs)
        hard_sep_value=None,  # float in capture units (1 relative capture)
        bg_ints=None,
        fit_only_uniques=False,
        ignore_bounds=None,
        fit_to_transform=False,
        lsq_kwargs=None,
        ignore_capture_units=True,
        background_only_external=False, 
        intensity_bounds=None, 
        wavelengths=None
    ):
        super().__init__(
            photoreceptor_model=photoreceptor_model,
            measured_spectra=measured_spectra,
            background=background,
            max_iter=max_iter,
            hard_separation=hard_separation,
            hard_sep_value=hard_sep_value,
            fit_weights=fit_weights,
            fit_only_uniques=fit_only_uniques,
            lsq_kwargs=lsq_kwargs,
            ignore_bounds=ignore_bounds,
            bg_ints=bg_ints,
            ignore_capture_units=ignore_capture_units,
            background_only_external=background_only_external, 
            intensity_bounds=intensity_bounds, 
            wavelengths=wavelengths
        )
        self.linear_transform = linear_transform
        self.inv_transform = inv_transform
        self.fit_to_transform = fit_to_transform

    def fit(self, X, y=None):
        X = self._check_X(X)
        self.transform_X_ = X
        if self.linear_transform is None:
            self.W_ = np.eye(X.shape[1])
            self.inv_transform = self.W_
        else:
            self.W_ = asarray(self.linear_transform)
            assert self.W_.shape[0] == X.shape[1], (
                "Linear transform shape does not match X"
            )

        if self.inv_transform is None:
            self.Winv_ = np.linalg.inv(self.W_)
        else:
            self.Winv_ = asarray(self.inv_transform)
            assert self.Winv_.shape[1] == X.shape[1], (
                "Inverse transform shape does not match X"
            )
            assert self.Winv_.shape[0] == self.W_.shape[1], (
                "Inverse matrix row size must match matrix columns."
            )

        super().fit(X @ self.Winv_)
        # overwrite current X
        self.fitted_transform_X_ = self.fitted_excite_X_ @ self.W_
        return self

    def inverse_transform(self, X):
        excite_X = super().inverse_transform(X)
        return excite_X @ self.Winv_

    @property
    def fitted_X_(self):
        return self.fitted_transform_X_

    @property
    def X_(self):
        return self.transform_X_

    def _objective(self, w, excite_x):
        x_pred = self._get_x_pred(w)
        if self.fit_to_transform:
            excite_x = excite_x @ self.W_
            x_pred = x_pred @ self.W_
        return self.fit_weights_ * (excite_x - x_pred)


class NonlinearTransformExcitationFit(IndependentExcitationFit):
    """
    Class to fit a nonlinear transformation of
    (relative) photoreceptor excitations for each sample independently.

    Photoreceptor model and measured_spectra must produce dimensionless
    captures.
    """

    # same length as X but not X or fitted X
    _X_length = IndependentExcitationFit._X_length + [
        'excite_X_',
        'fitted_excite_X_'
    ]

    def __init__(
        self,
        *,
        transform_func=None,  # array
        inv_func=None,  # array
        photoreceptor_model=None,  # dict or Photoreceptor class
        fit_weights=None,
        background=None,  # dict or Spectrum instance or array-like
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        max_iter=None,
        hard_separation=False,  # bool or list-like (same length as number of LEDs)
        hard_sep_value=None,  # float in capture units (1 relative capture)
        bg_ints=None,
        fit_only_uniques=False,
        ignore_bounds=None,
        fit_to_transform=False,
        lsq_kwargs=None,
        ignore_capture_units=True,
        background_only_external=False, 
        intensity_bounds=None,
        wavelengths=None
    ):
        super().__init__(
            photoreceptor_model=photoreceptor_model,
            measured_spectra=measured_spectra,
            background=background,
            max_iter=max_iter,
            hard_separation=hard_separation,
            hard_sep_value=hard_sep_value,
            fit_weights=fit_weights,
            fit_only_uniques=fit_only_uniques,
            lsq_kwargs=lsq_kwargs,
            ignore_bounds=ignore_bounds,
            bg_ints=bg_ints,
            ignore_capture_units=ignore_capture_units,
            background_only_external=background_only_external, 
            intensity_bounds=intensity_bounds, 
            wavelengths=wavelengths
        )
        self.transform_func = transform_func
        self.inv_func = inv_func
        self.fit_to_transform = fit_to_transform

    def fit(self, X, y=None):
        X = self._check_X(X)
        self.transform_X_ = X
        if self.transform_func is None:
            def transform_func(X):
                return X
            self.transform_func_ = transform_func
        else:
            assert is_callable(self.transform_func), (
                "`transform_func` must be callable."
            )
            self.transform_func_ = self.transform_func

        if self.inv_func is None and self.transform_func is None:
            self.inv_func_ = transform_func
        elif self.inv_func is None:
            raise DreyeError("Supply `inv_func`, finding "
                             "inverse function not yet implemented.")
        else:
            assert is_callable(self.inv_func), (
                "`inv_func` must be callable."
            )
            self.inv_func_ = self.inv_func

        super().fit(self.inv_func_(X))
        # overwrite current X
        self.fitted_transform_X_ = self.transform_func_(self.fitted_excite_X_)
        return self

    def inverse_transform(self, X):
        excite_X = super().inverse_transform(X)
        return self.inv_func_(excite_X)

    @property
    def fitted_X_(self):
        return self.fitted_transform_X_

    @property
    def X_(self):
        return self.transform_X_

    def _objective(self, w, excite_x):
        x_pred = self._get_x_pred(w)
        if self.fit_to_transform:
            excite_x = self.transform_func_(excite_x)
            x_pred = self.transform_func_(x_pred)
        return self.fit_weights_ * (excite_x - x_pred)