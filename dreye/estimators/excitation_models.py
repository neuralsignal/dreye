"""
Excitation models
"""

import warnings

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import lsq_linear, least_squares
from sklearn.utils.validation import check_array, check_is_fitted

from dreye.utilities import (
    optional_to, asarray, is_listlike, is_callable
)
from dreye.constants import ureg
from dreye.core.spectrum import Spectra
from dreye.estimators.base import _SpectraModel, OptimizeResultContainer
from dreye.err import DreyeError
from dreye.utilities.abstract import inherit_docstrings


# TODO simple class that only requires A and bounds
# TODO create A from wls, sense, ill, back - capture simple

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
        The spectral distribution of the background illuminant.
    measured_spectra : dreye.MeasuredSpectraContainer, optional
        Container with all available LEDs and their measured spectra. If
        None, a fake LED measurement will be created with intensities
        ranging from 0 to 100 microphotonflux.
    smoothing_window : numeric, optional
        The smoothing window size to use to smooth over the measurements
        in the container.
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
    current_X_ : numpy.ndarray (n_samples, n_prs)
        Current photoreceptor excitation values used for fitting.
        This is the same as `excite_X_`
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
        "photoreceptor_fit_weights": "fit_weights",
        "q1_ints": "bg_ints"
    }
    fit_to_transform = False
    _skip_bg_ints = True

    def __init__(
        self,
        *,
        photoreceptor_model=None,  # dict or Photoreceptor class
        fit_weights=None,
        background=None,  # dict or Spectrum instance or array-like
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        smoothing_window=None,  # float
        max_iter=None,
        hard_separation=False,  # bool or list-like (same length as number of LEDs)
        hard_sep_value=1.0,  # float in capture units (1 relative capture)
        q1_ints=None,
        bg_ints=None,
        fit_only_uniques=False,
        ignore_bounds=False,
        lsq_kwargs=None,
        ignore_capture_units=False
    ):
        self.photoreceptor_model = photoreceptor_model
        self.measured_spectra = measured_spectra
        self.smoothing_window = smoothing_window
        self.background = background
        self.max_iter = max_iter
        self.hard_separation = hard_separation
        self.hard_sep_value = hard_sep_value
        self.fit_weights = fit_weights
        self.fit_only_uniques = fit_only_uniques
        self.lsq_kwargs = lsq_kwargs
        self.ignore_bounds = ignore_bounds
        self.q1_ints = q1_ints
        self.bg_ints = bg_ints
        self.ignore_capture_units = ignore_capture_units

    def _set_required_objects(self, size=None):
        # create photoreceptor model
        self.photoreceptor_model_ = self._check_photoreceptor_model(
            self.photoreceptor_model, size=size
        )

        # create measured_spectra_
        self.measured_spectra_ = self._check_measured_spectra(
            self.measured_spectra, self.smoothing_window,
            photoreceptor_model=self.photoreceptor_model_
        )
        # fit isotonic regression
        self.measured_spectra_.regressor

        # set background intensities if exist
        self.bg_ints_ = self._get_bg_ints(
            self.bg_ints, self.measured_spectra_,
            skip=self._skip_bg_ints  # skip if bg_ints is None
        )

        if self.q1_ints is not None and self.bg_ints_ is None:
            warnings.warn(
                "Use of `q1_ints` is deprecated, use `bg_ints` instead",
                warnings.DeprecationWarning
            )
            self.bg_ints_ = self._get_bg_ints(
                self.q1_ints, self.measured_spectra_,
                skip=self._skip_bg_ints  # skip if bg_ints is None
            )

        # measured_spectra attributes
        # intensity bounds as two-tuple
        if self.ignore_bounds:
            self.bounds_ = (
                np.zeros(len(self.measured_spectra_)),
                np.inf * np.ones(len(self.measured_spectra_))
            )
        else:
            self.bounds_ = self.measured_spectra_.intensity_bounds
        # normalized spectra
        self.normalized_spectra_ = self.measured_spectra_.normalized_spectra
        # sanity checks
        assert self.normalized_spectra_.domain_axis == 0

        # create background
        self.background_ = self._check_background(
            self.background, self.measured_spectra_
        )
        if self.background is None and self.bg_ints_ is not None:
            # will integrate with normalized spectra
            self.background_ = self._check_background(
                self.bg_ints_, self.measured_spectra_
            )

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
        self.n_features_ = self.photoreceptor_model_.pr_number
        self.channel_names_ = self.photoreceptor_model_.names

        return self

    def _set_A(self):
        # opsin x LED (taking transpose)
        self.A_ = self.photoreceptor_model_.capture(
            self.normalized_spectra_,
            background=self.background_,
            return_units=False,
            apply_noise_threshold=False
        ).T

    def _fit(self, X):
        """
        Actual Fitting method. Allows easier subclassing
        """

        # overwrite this method when subclassing
        self.capture_X_, self.excite_X_ = self._process_X(X)
        self.current_X_ = self.excite_X_

        self._set_A()

        # whether domain between spectra and photoreceptor model equal
        self._domain_equal_ = (
            self.normalized_spectra_.domain
            == self.photoreceptor_model_.sensitivity.domain
        )

        # weighting for each photoreceptor
        if self.fit_weights is None:
            fit_weights = np.ones(self.photoreceptor_model_.pr_number)
        else:
            fit_weights = asarray(self.fit_weights)
            # assert len(fit_weights) == self.photoreceptor_model_.pr_number

        # do weighted initial fitting in linear domain
        self.weighted_init_fit_ = (
            not self.fit_to_transform
            and np.unique(fit_weights).size > 1
        )

        # separation of all negative or positive
        if is_listlike(self.hard_separation):
            sep_bound = optional_to(
                self.hard_separation,
                self.measured_spectra_.intensities.units
            )
            assert len(sep_bound) == len(self.measured_spectra_), (
                "hard_separation length must match length of "
                "measured spectra container."
            )
            self.sep_bound_ = sep_bound
            self.sep_result_ = None
        elif (
            self.hard_separation
            and np.all(self.hard_sep_value == 1)
            and (self.bg_ints_ is not None)
        ):
            self.sep_bound_ = self.bg_ints_
            self.sep_result_ = None
        elif self.hard_separation:
            sep_value = (
                np.ones(self.photoreceptor_model_.pr_number)
                * self.hard_sep_value
            )
            # separation bound using a fit to sep_value
            # capture and excite!
            sep_result = self._fit_sample(
                sep_value,
                self.photoreceptor_model_.excitefunc(sep_value),
                fit_weights=fit_weights
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
                    capture_x, excite_x, fit_weights, self.sep_bound_
                )
                for capture_x, excite_x in
                zip(self.capture_X_[xidcs], self.excite_X_[xidcs])
            ])[xinverse])
        else:
            self.container_ = OptimizeResultContainer([
                self._fit_sample(
                    capture_x, excite_x, fit_weights, self.sep_bound_
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
            self._get_x_pred_noise(w)
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

        assert X.shape[1] == len(self.measured_spectra_)

        # got from output to intensity
        X = self.measured_spectra_.inverse_map(X, return_units=False)
        # get excitation given intensity
        X = np.array([self._get_x_pred(x) for x in X])
        return X

    def _fit_sample(self, capture_x, excite_x, fit_weights, sep_bound=None):
        # adjust bounds if necessary
        bounds = list(self.bounds_)
        if sep_bound is not None:
            if np.all(capture_x >= self.hard_sep_value):
                bounds[0] = sep_bound
            elif np.all(capture_x <= self.hard_sep_value):
                bounds[1] = sep_bound
        if self.bg_ints_ is not None:
            # if np.allclose(capture_x, 1)
            if np.allclose(capture_x, 1):
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
        w0 = self._init_sample(capture_x, bounds, fit_weights)
        # fitted result
        result = least_squares(
            self._objective,
            x0=w0,
            args=(excite_x, fit_weights),
            bounds=tuple(bounds),
            max_nfev=self.max_iter,
            **({} if self.lsq_kwargs is None else self.lsq_kwargs)
        )
        return result

    def _init_sample(self, capture_x, bounds, fit_weights):
        # weighted fit is better for substitution types of fits
        if self.weighted_init_fit_:
            result = lsq_linear(
                fit_weights[:, None] * self.A_, fit_weights * capture_x,
                bounds=tuple(bounds),
                max_iter=self.max_iter
            )
        else:
            result = lsq_linear(
                self.A_, capture_x,
                bounds=tuple(bounds),
                max_iter=self.max_iter
            )
        # return fitted intensity (w)
        return result.x

    def _objective(self, w, excite_x, fit_weights):
        x_pred = self._get_x_pred(w)
        return fit_weights * (excite_x - x_pred)

    def _get_x_capture(self, w):
        if self.photoreceptor_model_.filterfunc is None:
            # threshold by noise if necessary and apply nonlinearity
            x_pred = self.A_ @ w
        else:
            if self._domain_equal_:
                illuminant = self.normalized_spectra_.magnitude @ w
            else:
                illuminant = Spectra(
                    self.normalized_spectra_.magnitude @ w,
                    units=self.measured_spectra_.units,
                    domain=self.normalized_spectra_.domain
                )
            x_pred = self.photoreceptor_model_.capture(
                # normalized_spectrum has domain_axis=0
                # TODO write measured spectra function that interpolates
                # to the given spectrum
                illuminant,
                background=self.background_,
                return_units=False,
                apply_noise_level=False
            )
            # ensure vector form
            x_pred = np.atleast_1d(np.squeeze(x_pred))
        return x_pred

    def _get_x_pred_noise(self, w):
        return self.photoreceptor_model_.excitefunc(
            self.photoreceptor_model_.limit_q_by_noise_level(
                self._get_x_capture(w)
            )
        )

    def _get_x_pred(self, w):
        return self.photoreceptor_model_.excitefunc(
            self._get_x_capture(w)
        )

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
        capture_X = self.photoreceptor_model_.limit_q_by_noise_level(capture_X)
        excite_X = self.photoreceptor_model_.excitefunc(capture_X)
        return capture_X, excite_X

    @property
    def input_units(self):
        return ureg(None).units

    @property
    def fitted_X_(self):
        return self.fitted_excite_X_


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
    smoothing_window : numeric, optional
        The smoothing window size to use to smooth over the measurements
        in the container.
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
    current_X_ : numpy.ndarray (n_samples, m_space)
        Current linear transformatino of photoreceptor excitation values
        used for fitting. This is the same as `transform_X_`.
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
        smoothing_window=None,  # float
        max_iter=None,
        hard_separation=False,  # bool or list-like (same length as number of LEDs)
        hard_sep_value=1.0,  # float in capture units (1 relative capture)
        q1_ints=None,
        bg_ints=None,
        fit_only_uniques=False,
        ignore_bounds=False,
        fit_to_transform=False,
        lsq_kwargs=None,
        ignore_capture_units=False
    ):
        super().__init__(
            photoreceptor_model=photoreceptor_model,
            measured_spectra=measured_spectra,
            smoothing_window=smoothing_window,
            background=background,
            max_iter=max_iter,
            hard_separation=hard_separation,
            hard_sep_value=hard_sep_value,
            fit_weights=fit_weights,
            fit_only_uniques=fit_only_uniques,
            lsq_kwargs=lsq_kwargs,
            ignore_bounds=ignore_bounds,
            q1_ints=q1_ints,
            bg_ints=bg_ints,
            ignore_capture_units=ignore_capture_units
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
        self.current_X_ = self.transform_X_
        self.fitted_transform_X_ = self.fitted_excite_X_ @ self.W_
        return self

    def inverse_transform(self, X):
        excite_X = super().inverse_transform(X)
        return excite_X @ self.Winv_

    @property
    def fitted_X_(self):
        return self.fitted_transform_X_

    def _objective(self, w, excite_x, fit_weights):
        x_pred = self._get_x_pred(w)
        if self.fit_to_transform:
            excite_x = excite_x @ self.W_
            x_pred = x_pred @ self.W_
        return fit_weights * (excite_x - x_pred)


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
        smoothing_window=None,  # float
        max_iter=None,
        hard_separation=False,  # bool or list-like (same length as number of LEDs)
        hard_sep_value=1.0,  # float in capture units (1 relative capture)
        q1_ints=None,
        bg_ints=None,
        fit_only_uniques=False,
        ignore_bounds=False,
        fit_to_transform=False,
        lsq_kwargs=None,
        ignore_capture_units=False
    ):
        super().__init__(
            photoreceptor_model=photoreceptor_model,
            measured_spectra=measured_spectra,
            smoothing_window=smoothing_window,
            background=background,
            max_iter=max_iter,
            hard_separation=hard_separation,
            hard_sep_value=hard_sep_value,
            fit_weights=fit_weights,
            fit_only_uniques=fit_only_uniques,
            lsq_kwargs=lsq_kwargs,
            ignore_bounds=ignore_bounds,
            q1_ints=q1_ints,
            bg_ints=bg_ints,
            ignore_capture_units=ignore_capture_units
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
        self.current_X_ = self.transform_X_
        self.fitted_transform_X_ = self.func_(self.fitted_excite_X_)
        return self

    def inverse_transform(self, X):
        excite_X = super().inverse_transform(X)
        return excite_X @ self.Winv_

    @property
    def fitted_X_(self):
        return self.fitted_transform_X_

    def _objective(self, w, excite_x, fit_weights):
        x_pred = self._get_x_pred(w)
        if self.fit_to_transform:
            excite_x = self.func_(excite_x)
            x_pred = self.func_(x_pred)
        return fit_weights * (excite_x - x_pred)


@inherit_docstrings
class ReflectanceExcitationFit(IndependentExcitationFit):
    """
    Class to fit various reflectances given a photoreceptor model
    and LED system.

    Parameters
    ----------
    reflectances : dreye.Signals, optional
        A set of reflectances (usually max-normalized) used for fitting.
        X will be a multiples of the reflectances, before applying
        `add_background` and/or `filter_background`.
    add_background : bool, optional
        Add background to multiples of reflectances:
        :math:`X \odot reflectances + background`.
    filter_background : bool, optional
        Use reflectances as filters of the background.
        If `add_background` is False, then the following is used:
        :math:`X \odot reflectances \odot background`. If
        `add_background` is True, then the following is used:
        :math:`(1 + X \odot reflectances) \odot background`
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
    smoothing_window : numeric, optional
        The smoothing window size to use to smooth over the measurements
        in the container.
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
    wavelengths_ : numpy.ndarray
        The wavelength range considered for fitting in nanometers.
    spectra_used_for_fitting_ : dreye.Spectra
        The final spectra after background pre-processing used for fitting.
    reflectances_ : dreye.Spectra
        Reflectance spectra used for fitting.
    capture_X_ : numpy.ndarray (n_samples, n_prs)
        The current relative photon capture values used for fitting.
    excite_X_ : numpy.ndarray (n_samples, n_prs)
        The current photoreceptor excitation values used for fitting.
    current_X_ : numpy.ndarray (n_samples, n_prs)
        Current photoreceptor excitation values used for fitting.
        This is the same as `excite_X_`
    fitted_intensities_ : numpy.ndarray
        Intensities fit in units of `measured_spectra_.intensities.units`
    fitted_capture_X_ : numpy.ndarray (n_samples, n_prs)
        The recalculated relative photon capture values after fitting.
    fitted_excite_X_ : numpy.ndarray (n_samples, n_prs)
        The recalculated photoreceptor excitations after fitting.
    """

    # same length as X but not X or fitted X
    _X_length = IndependentExcitationFit._X_length + [
        'excite_X_',
        'fitted_excite_X_'
    ]

    def __init__(
        self,
        *,
        reflectances=None,
        photoreceptor_model=None,  # dict or Photoreceptor class
        fit_weights=None,
        background=None,  # dict or Spectrum instance or array-like
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        smoothing_window=None,  # float
        max_iter=None,
        hard_separation=False,  # bool or list-like (same length as number of LEDs)
        hard_sep_value=1.0,  # float in capture units (1 relative capture)
        q1_ints=None,
        bg_ints=None,
        fit_only_uniques=False,
        lsq_kwargs=None,
        ignore_bounds=False,
        add_background=True,
        filter_background=True,
        ignore_capture_units=False
    ):
        self.reflectances = reflectances
        self.photoreceptor_model = photoreceptor_model
        self.measured_spectra = measured_spectra
        self.smoothing_window = smoothing_window
        self.background = background
        self.max_iter = max_iter
        self.ignore_bounds = ignore_bounds
        self.hard_separation = hard_separation
        self.hard_sep_value = hard_sep_value
        self.q1_ints = q1_ints
        self.bg_ints = bg_ints
        self.fit_weights = fit_weights
        self.fit_only_uniques = fit_only_uniques
        self.lsq_kwargs = lsq_kwargs
        self.add_background = add_background
        self.filter_background = filter_background
        self.ignore_capture_units = ignore_capture_units

    def fit(self, X, y=None):
        self._set_required_objects()
        self.reflectances_ = self._check_reflectances(
            self.reflectances,
            self.measured_spectra_,
            self.photoreceptor_model_
        )
        # equalize illuminant and background
        if self.background_ is not None:
            self.reflectances_, self.background_ = \
                self.reflectances_.equalize_domains(self.background_)
        self.wavelengths_ = self.reflectances_.domain.magnitude

        # ignore units
        X = self._check_X(X)

        # check sizing (domain_axis=0)
        assert X.shape[1] == self.reflectances_.shape[1]

        spectra_units = self.reflectances_.units
        spectra = self.reflectances_.magnitude @ X.T
        if self.filter_background and self.background_ is not None:
            spectra_units = spectra_units * self.background_.units
            spectra = spectra * self.background_.magnitude[..., None]
        if self.add_background and self.background_ is not None:
            spectra = spectra + self.background_.magnitude[..., None]

        # check units
        if self.background_ is None:
            assert spectra_units.dimensionless, (
                f"Spectra units are not dimensionless: '{spectra_units}'."
            )
        else:
            assert (spectra_units / self.background_.units).dimensionless, (
                f"Spectra has wrong units {spectra_units}."
            )

        if np.any(spectra < 0):
            warnings.warn("Some spectra values are below zero, clipping...")
            spectra[spectra < 0] = 0

        spectra = Spectra(
            spectra,
            units=spectra_units,
            domain=self.wavelengths_
        )

        self.spectra_used_for_fitting_ = spectra

        excite_X = self.photoreceptor_model_.excitation(
            spectra,
            background=self.background_,
            return_units=False
        )

        self._fit(excite_X)
        return self
