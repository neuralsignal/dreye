"""
Excitation models
"""

import warnings

import numpy as np
from scipy.optimize import lsq_linear, least_squares
from sklearn.utils.validation import check_array, check_is_fitted

from dreye.utilities import (
    optional_to, asarray, is_listlike
)
from dreye.constants import ureg
from dreye.core.spectrum import Spectra
from dreye.estimators.base import _SpectraModel, OptimizeResultContainer


class IndependentExcitationFit(_SpectraModel):
    """
    Class to fit (relative) photoreceptor excitations for each sample
    independently.

    Photoreceptor model and measured_spectra Must produce dimensionless
    """

    # same length as X but not X or fitted X
    _X_length = [
        'capture_X_',
        'fitted_capture_X_',
        'fitted_intensities_'
    ]

    def __init__(
        self,
        *,
        photoreceptor_model=None,  # dict or Photoreceptor class
        photoreceptor_fit_weights=None,
        background=None,  # dict or Spectrum instance or array-like
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        smoothing_window=None,  # float
        max_iter=None,
        hard_separation=False,  # bool or list-like (same length as number of LEDs)
        hard_sep_value=1.0,  # float in capture units (1 relative capture)
        fit_only_uniques=False,
        lsq_kwargs=None
    ):
        self.photoreceptor_model = photoreceptor_model
        self.measured_spectra = measured_spectra
        self.smoothing_window = smoothing_window
        self.background = background
        self.max_iter = max_iter
        self.hard_separation = hard_separation
        self.hard_sep_value = hard_sep_value
        self.photoreceptor_fit_weights = photoreceptor_fit_weights
        self.fit_only_uniques = fit_only_uniques
        self.lsq_kwargs = lsq_kwargs

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

        # measured_spectra attributes
        # intensity bounds as two-tuple
        self.bounds_ = np.array(self.measured_spectra_.intensity_bounds).T
        # normalized spectra
        self.normalized_spectra_ = self.measured_spectra_.normalized_spectra
        # sanity checks
        assert self.normalized_spectra_.domain_axis == 0

        # create background
        self.background_ = self._check_background(
            self.background, self.measured_spectra_
        )

        if self.background_ is None:
            assert (
                self.measured_spectra_.units
                * self.photoreceptor_model_.sensitivity.units
                * self.photoreceptor_model_.wavelengths.units
            ).dimensionless, 'units not dimensionless'
        else:
            assert (
                self.measured_spectra_.units
                / self.background_.units
            ).dimensionless, 'units not dimensionless'

        # number of photoreceptors
        self.n_features_ = self.photoreceptor_model_.pr_number

        return self

    def _fit(self, X):
        """
        Actual Fitting method. Allows subclassing
        """

        # overwrite this method when subclassing
        self.capture_X_, self.excite_X_ = self._process_X(X)
        self.current_X_ = self.excite_X_

        # opsin x LED (taking transpose)
        self.A_ = self.photoreceptor_model_.capture(
            self.normalized_spectra_,
            background=self.background_,
            return_units=False
        ).T

        # weighting for each photoreceptor
        if self.photoreceptor_fit_weights is None:
            fit_weights = np.ones(self.photoreceptor_model_.pr_number)
        else:
            fit_weights = asarray(self.photoreceptor_fit_weights)
            assert len(fit_weights) == self.photoreceptor_model_.pr_number

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
            self._get_x_pred(w)
            for w in self.fitted_intensities_
        ])
        self.fitted_capture_X_ = self.photoreceptor_model_.inv_excitefunc(
            self.fitted_excite_X_
        )

        return self

    def fit(self, X, y=None):
        """
        Fit method.
        """
        # set required objects
        self._set_required_objects(asarray(X).shape[1])
        # fit X
        self._fit(X)
        return self

    def inverse_transform(self, X):
        """
        Transform output values to excitation values
        """
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
        bounds = self.bounds_.copy()
        if sep_bound is not None:
            if np.all(capture_x >= self.hard_sep_value):
                bounds[0] = sep_bound
            elif np.all(capture_x <= self.hard_sep_value):
                bounds[1] = sep_bound
        # find initial w0 using linear least squares
        w0 = self._init_sample(capture_x, bounds)
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

    def _init_sample(self, capture_x, bounds):
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

    def _get_x_pred(self, w):
        if self.photoreceptor_model_.filterfunc is None:
            x_pred = self.photoreceptor_model_.excitefunc(self.A_ @ w)
        else:
            # returns opsin vector
            # need to recalculate excitation if filterfunc defined
            x_pred = self.photoreceptor_model_.excitation(
                # normalized_spectrum has domain_axis=0
                Spectra(
                    self.normalized_spectra_.magnitude @ w,
                    units=self.measured_spectra_.units,
                    domain=self.normalized_spectra_.domain
                ),
                background=self.background_,
                return_units=False
            )
            # ensure vector form
            x_pred = np.squeeze(x_pred)
        return x_pred

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
        return self.photoreceptor_model_.inv_excitefunc(X), X

    @property
    def input_units(self):
        """units of X
        """
        return ureg(None).units

    @property
    def fitted_X(self):
        """X after fitting
        """
        return self.fitted_excite_X_


class TransformExcitationFit(IndependentExcitationFit):
    """
    Class to fit a linear transformation of
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
        linear_transform=None,  # array
        inverse_transform=None,  # array
        photoreceptor_model=None,  # dict or Photoreceptor class
        photoreceptor_fit_weights=None,
        background=None,  # dict or Spectrum instance or array-like
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        smoothing_window=None,  # float
        max_iter=None,
        hard_separation=False,  # bool or list-like (same length as number of LEDs)
        hard_sep_value=1.0,  # float in capture units (1 relative capture)
        fit_only_uniques=False,
        lsq_kwargs=None
    ):
        super().__init__(
            photoreceptor_model=photoreceptor_model,
            measured_spectra=measured_spectra,
            smoothing_window=smoothing_window,
            background=background,
            max_iter=max_iter,
            hard_separation=hard_separation,
            hard_sep_value=hard_sep_value,
            photoreceptor_fit_weights=photoreceptor_fit_weights,
            fit_only_uniques=fit_only_uniques,
            lsq_kwargs=lsq_kwargs
        )
        self.linear_transform = linear_transform
        self.inverse_transform = inverse_transform

    def fit(self, X, y=None):
        X = self._check_X(X)
        self.transform_X_ = X
        if self.linear_transform is None:
            self.W_ = np.eye(X.shape[1])
            self.inverse_transform = self.W_
        else:
            self.W_ = asarray(self.linear_transform)
            assert self.W_.shape[0] == X.shape[1], (
                "Linear transform shape does not match X"
            )

        if self.inverse_transform is None:
            self.Winv_ = np.linalg.inv(self.W_)
        else:
            self.Winv_ = asarray(self.inverse_transform)
            assert self.Winv_.shape[1] == X.shape[1], (
                "Inverse transform shape does not match X"
            )

        super().fit(X @ self.W_)
        # overwrite current X
        self.current_X_ = self.transform_X_
        self.fitted_transform_X_ = self.fitted_excite_X_ @ self.Winv_
        return self

    def inverse_transform(self, X):
        excite_X = super().inverse_transform(X)
        return excite_X @ self.Winv_

    @property
    def fitted_X(self):
        """X after fitting
        """
        return self.fitted_transform_X_


class ReflectanceExcitationFit(IndependentExcitationFit):

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
        photoreceptor_fit_weights=None,
        background=None,  # dict or Spectrum instance or array-like
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        smoothing_window=None,  # float
        max_iter=None,
        hard_separation=False,  # bool or list-like (same length as number of LEDs)
        hard_sep_value=1.0,  # float in capture units (1 relative capture)
        fit_only_uniques=False,
        lsq_kwargs=None,
        add_background=True,
        filter_background=True
    ):
        self.reflectances = reflectances
        self.photoreceptor_model = photoreceptor_model
        self.measured_spectra = measured_spectra
        self.smoothing_window = smoothing_window
        self.background = background
        self.max_iter = max_iter
        self.hard_separation = hard_separation
        self.hard_sep_value = hard_sep_value
        self.photoreceptor_fit_weights = photoreceptor_fit_weights
        self.fit_only_uniques = fit_only_uniques
        self.lsq_kwargs = lsq_kwargs
        self.add_background = add_background
        self.filter_background = filter_background

    def fit(self, X, y=None):
        """
        Fit method.
        """
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

        # TODO inverse not possible?
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
