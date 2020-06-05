"""
LED Estimators for intensities and spectra
"""

from abc import abstractmethod
import warnings

import numpy as np
from scipy.optimize import lsq_linear, least_squares, OptimizeResult
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from dreye.utilities import (
    is_dictlike, optional_to, asarray, is_listlike, is_string
)
from dreye.utilities.abstract import _AbstractContainer
from dreye.constants import ureg
from dreye.core.spectrum_utils import get_spectrum
from dreye.core.signal import _Signal2DMixin, Signal, Signals
from dreye.core.spectrum import Spectra, Spectrum
from dreye.core.spectral_measurement import MeasuredSpectraContainer
from dreye.core.measurement_utils import get_led_spectra_container
from dreye.core.photoreceptor import Photoreceptor, get_photoreceptor_model

# must contain fit, transform, and map

# by ducktyping measured_spectra?
# ducktyping photoreceptor -> capture, excitation
# TODO allow curve fit instead of isotonic regression? - SKlearn type class


class OptimizeResultContainer(_AbstractContainer):
    _allowed_instances = OptimizeResult


class _SpectraModel(BaseEstimator, TransformerMixin):
    """
    Abstract Spectra model used for various Dreye estimators.
    """

    @staticmethod
    def _check_measured_spectra(
        measured_spectra, smoothing_window,
        size=None, photoreceptor_model=None
    ):
        """
        check and create measured spectra container
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
            measured_spectra = get_led_spectra_container(
                **measured_spectra
            )
        elif measured_spectra is None:
            measured_spectra = get_led_spectra_container(size)
        else:
            raise ValueError("Measured Spectra must be Spectra "
                             "container or dict, but is type "
                             f"'{type(measured_spectra)}'.")

        if smoothing_window is not None:
            measured_spectra = measured_spectra.smooth(smoothing_window)

        # enforce photon flux for photoreceptor models
        if (
            photoreceptor_model is not None
            and (
                measured_spectra.units.dimensionality
                != ureg('uE').units.dimensionality
            )
        ):
            return measured_spectra.to('uE')

        return measured_spectra

    @staticmethod
    def _check_photoreceptor_model(photoreceptor_model, size=None):
        """
        check and create photoreceptor model
        """
        if isinstance(photoreceptor_model, Photoreceptor):
            pass
        elif is_dictlike(photoreceptor_model):
            photoreceptor_model['sensitivity'] = photoreceptor_model.get(
                'sensitivity', size
            )
            photoreceptor_model = get_photoreceptor_model(
                **photoreceptor_model
            )
        elif photoreceptor_model is None:
            photoreceptor_model = get_photoreceptor_model(size)
        else:
            raise ValueError("Photoreceptor model must be Photoreceptor "
                             "instance or dict, but is type "
                             f"'{type(photoreceptor_model)}'.")

        return photoreceptor_model

    @staticmethod
    def _check_background(
        background, measured_spectra, photoreceptor_model=None
    ):
        """
        check and create background
        """
        if is_dictlike(background):
            background['wavelengths'] = background.get(
                'wavelengths', measured_spectra.wavelengths
            )
            background['units'] = background.get(
                'units', measured_spectra.units
            )
            background = get_spectrum(**background)
        elif background is None:
            background = get_spectrum(
                wavelengths=measured_spectra.wavelengths,
                units=measured_spectra.units
            )
        elif is_string(background) and (background == 'null'):
            background = None
        elif isinstance(background, Signal):
            pass
        elif is_listlike(background):
            # normalized spectra are always on domain_axis=0
            assert measured_spectra.normalized_spectra.domain_axis == 0
            background = optional_to(
                background,
                measured_spectra.intensities.units
            )
            background = (
                background[None]
                * measured_spectra.normalized_spectra.magnitude
            ).sum(axis=-1)
            background = get_spectrum(
                intensities=background,
                wavelengths=measured_spectra.wavelengths,
                units=measured_spectra.units
            )
        else:
            raise ValueError(
                "Background must be Spectrum instance or dict-like, but"
                f"is of type {type(background)}."
            )
        return background

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _check_X(self, X):
        """units for X.

        e.g. in photonflux units.
        """
        X = optional_to(X, self.input_units)
        return check_array(X)

    @property
    def output_units(self):
        """units of transformed X
        """
        return self.measured_spectra_.labels_units

    @property
    @abstractmethod
    def input_units(self):
        """units of X
        """

    def to_dict(self):
        return self.get_params()

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


# TODO relativeIntensity model
class IntensityFit(_SpectraModel):
    """
    Fit intensity values to a given LED system.

    All models that fit chromatic stimuli have to accept a
    measured spectra object.
    """

    def __init__(
        self,
        *,
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        smoothing_window=None  # float
    ):
        self.measured_spectra = measured_spectra
        self.smoothing_window = smoothing_window

    def fit(self, X, y=None):
        """
        Fit method.
        """
        #
        self.measured_spectra_ = self._check_measured_spectra(
            self.measured_spectra, self.smoothing_window, asarray(X).shape[1]
        )
        # check X
        X = self._check_X(X)
        # call in order to fit isotonic regression
        self.measured_spectra_.regressor

        self.n_features_ = len(self.measured_spectra_)

        # check that input shape is correct
        if X.shape[1] != self.n_features_:
            raise ValueError("Shape of input is different from number"
                             "of measured spectra in container.")

        return self

    def transform(self, X):
        """
        Transform method
        """
        # check is fitted
        check_is_fitted(self, ['n_features_', 'measured_spectra_'])

        # input validation
        X = self._check_X(X)

        # check that input shape is correct
        if X.shape[1] != self.n_features_:
            raise ValueError("Shape of input is different from what was seen"
                             "in `fit`.")

        # apply mapping
        return self.measured_spectra_.map(X, return_units=False)

    def inverse_transform(self, X):
        """
        Inverse Transform method
        """
        # check is fitted
        check_is_fitted(self, ['n_features_', 'measured_spectra_'])

        # check X
        X = optional_to(X, self.output_units)
        X = check_array(X)

        # map output values to intensities
        return self.measured_spectra_.reverse_map(X, return_units=False)

    def score(self, X, y=None):
        """
        Score method
        """
        # input validation
        X = self._check_X(X)
        return self.measured_spectra_.score(X)

    @property
    def input_units(self):
        """units of X
        """
        return self.measured_spectra_.intensities.units


# TODO improve (not final version)
class IlluminantFit(_SpectraModel):
    """
    Fit illuminant spectra
    """

    def __init__(
        self,
        *,
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        smoothing_window=None,  # float
        max_iter=None
    ):
        self.measured_spectra = measured_spectra
        self.smoothing_window = smoothing_window
        self.max_iter = max_iter

    def fit(self, X, y=None):
        """
        Fit method.
        """
        # store x for checking
        self.stored_X_ = X
        # create measured_spectra_
        self.measured_spectra_ = self._check_measured_spectra(
            self.measured_spectra, self.smoothing_window
        )
        normalized_spectra = self.measured_spectra_.normalized_spectra.copy()
        assert normalized_spectra.domain_axis == 0
        # make 2D if necessary
        if isinstance(X, Signal):
            X = Signals(X)
        # move domain axis and equalize if necessary
        if isinstance(X, _Signal2DMixin):
            X = X.copy()
            # ensure domain axis is feature axis
            X.domain_axis = 1
            normalized_spectra, X = normalized_spectra.equalize_domains(X)
        # check X
        X = self._check_X(X)
        # also store checked X
        self.checked_X_ = X
        # spectra as array
        self.normalized_spectra_ = normalized_spectra
        self.wavelengths_ = self.normalized_spectra_.domain.magnitude
        self.bounds_ = np.array(self.measured_spectra_.intensity_bounds).T

        # creates regressor for mapping values
        self.measured_spectra_.regressor
        self.container_ = self._fit_samples(X)

        if not np.all(self.container_.success):
            warnings.warn("Convergence was not accomplished "
                          "for all spectra in X; "
                          "increase the number of max iterations.")

        self.n_features_ = X.shape[1]
        # samples x intensity
        self.fitted_intensities_ = np.array(
            self.container_.x
        ) * self.measured_spectra_.intensities.units
        # or self.input_units / self.normalized_spectra_.units

        return self

    def _fit_samples(self, X):
        """
        Fit individual samples
        """
        # TODO accuracy for wavelength range e.g. 10nm (in blocks)
        # TODO first integrate window filter?
        A = asarray(self.normalized_spectra_)
        container = OptimizeResultContainer()
        for x in X:
            container.append(
                lsq_linear(
                    A, x,
                    bounds=tuple(self.bounds_),
                    max_iter=self.max_iter
                )
            )
        return container

    def _equal_to_stored_X(self, X):
        # requires same spectrum
        if (
            X is not None
            and not np.all(self.stored_X_ == X)
            and not np.all(self.checked_X_ == X)
        ):
            raise ValueError("X to transform is not the same "
                             "as X used for fitting.")

    def transform(self, X=None):
        """
        Transform Method.
        """
        # check is fitted
        check_is_fitted(
            self, ['measured_spectra_', 'fitted_intensities_']
        )

        self._equal_to_stored_X(X)
        # map fitted_intensities
        return self.measured_spectra_.map(
            self.fitted_intensities_, return_units=False)

    def inverse_transform(self, X):
        """
        Transform output values to spectra
        """
        check_is_fitted(
            self, ['measured_spectra_', 'normalized_spectra_']
        )
        # X is samples x LEDs
        X = optional_to(X, self.output_units)
        X = check_array(X)

        # assert X.shape[0] == self.checked_X_.shape[0]
        assert X.shape[1] == len(self.measured_spectra_)

        # samples x LED
        X = self.measured_spectra_.inverse_map(X, return_units=False)
        return X @ self.normalized_spectra_.magnitude.T

    def residuals(self, X=None, y=None):
        """
        Residuals for each sample and wavelength
        """
        # apply transformation and compare to checked_X
        X_pred = self.inverse_transform(self.transform(X))
        X = self.checked_X_
        return X - X_pred

    def sample_scores(self, X=None, y=None):
        """
        R^2 for each sample.
        """
        # apply transformation and compare to checked_X
        X_pred = self.inverse_transform(self.transform(X))
        X = self.checked_X_

        # residual across wavelengths
        res = (X - X_pred)**2
        tot = (X - X.mean(axis=-1, keepdims=True))**2
        return 1 - res.sum(-1)/tot.sum(-1)

    def wavelength_scores(self, X=None, y=None):
        """
        R^2 for each wavelength
        """
        # apply transformation and compare to checked_X
        X_pred = self.inverse_transform(self.transform(X))
        X = self.checked_X_

        # residual across samples
        res = (X - X_pred)**2
        tot = (X - X.mean(axis=0, keepdims=True))**2
        return 1 - res.sum(0)/tot.sum(0)

    def score(self, X=None, y=None):
        """
        R^2 Score method.
        """
        r2 = self.wavelength_scores(X)
        return r2.mean()

    @property
    def input_units(self):
        """units of X
        """
        return self.measured_spectra_.units


class IndependentExcitationFit(_SpectraModel):
    """
    Class to fit (relative) photoreceptor excitations for each sample
    independently.

    Photoreceptor model and measured_spectra Must produce dimensionless
    """

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
        x_as_capture=False,  # if excitation or capture values are passed as X
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
        self.x_as_capture = x_as_capture
        self.lsq_kwargs = lsq_kwargs

    def fit(self, X, y=None):
        """
        Fit method.
        """
        # create photoreceptor model
        self.photoreceptor_model_ = self._check_photoreceptor_model(
            self.photoreceptor_model, size=asarray(X).shape[1]
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

        # overwrite this method when subclassing
        self.stored_X_ = X
        self.capture_X_, self.excite_X_ = self._process_X(X)

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
                for capture_x, excite_x in zip(self.capture_X_, self.excite_X_)
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

    def _equal_to_stored_X(self, X):
        # requires same spectrum
        if (
            X is not None
            and not np.all(self.stored_X_ == X)
            and not np.all(self.capture_X_ == X)
            and not np.all(self.excite_X_ == X)
        ):
            raise ValueError("X to transform is not the same "
                             "as X used for fitting.")

    def transform(self, X=None):
        """
        Transform method
        """
        # check is fitted
        check_is_fitted(
            self, ['measured_spectra_', 'fitted_intensities_']
        )

        self._equal_to_stored_X(X)
        # just need to map fitted_intensities
        return self.measured_spectra_.map(
            self.fitted_intensities_, return_units=False)

    def inverse_transform(self, X, as_capture=False):
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
        if as_capture:
            return self.photoreceptor_model_.inv_excitefunc(X)
        else:
            return X
        # get spectra given intensity
        # spectra = Spectra(
        #     self.normalized_spectra_.magnitude @ X.T,
        #     domain=self.normalized_spectra_.domain,
        #     units=self.measured_spectra_.units,
        # )
        # # get excitation given spectra
        # # samples x opsin
        # return self.photoreceptor_model_.excitation(
        #     spectra,
        #     background=self.background_,
        #     return_units=False
        # )

    def residuals(self, X=None, *, as_capture=False):
        """
        Residual for each photoreceptor and sample.
        """
        # apply transformation and compare to checked_X
        X_pred = self.inverse_transform(
            self.transform(X), as_capture=as_capture
        )
        X = (self.capture_X_ if as_capture else self.excite_X_)
        return X - X_pred

    def sample_scores(self, X=None, *, as_capture=False):
        """
        Sample scores
        """
        # apply transformation and compare to checked_X
        X_pred = self.inverse_transform(
            self.transform(X), as_capture=as_capture
        )
        X = (self.capture_X_ if as_capture else self.excite_X_)

        # residual across photoreceptors
        res = (X - X_pred)**2
        tot = (X - X.mean(axis=-1, keepdims=True))**2
        return 1 - res.sum(-1)/tot.sum(-1)

    def pr_scores(self, X=None, *, as_capture=False):
        """
        Scores for each photoreceptor
        """
        # apply transformation and compare to checked_X
        X_pred = self.inverse_transform(
            self.transform(X), as_capture=as_capture
        )
        X = (self.capture_X_ if as_capture else self.excite_X_)

        # residual across samples
        res = (X - X_pred)**2
        tot = (X - X.mean(axis=0, keepdims=True))**2
        return 1 - res.sum(0)/tot.sum(0)

    def score(self, X=None, y=None, *, as_capture=False):
        """
        Score method
        """
        return self.pr_scores(X, as_capture=as_capture).mean()

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
        # return fitted x
        return result.x

    def _objective(self, w, excite_x, fit_weights):
        # TODO consider resolution?
        # if self.measured_spectra_.resolution is not None:
        #     w = self.measured_spectra_.inverse_map(
        #         self.measured_spectra_.map(w, return_units=False),
        #         return_units=False
        #     )
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
                # TODO avoid initializing with spectrum (time-consuming)
                Spectra(
                    self.normalized_spectra_.magnitude @ w,
                    units=self.measured_spectra_.units,
                    domain=self.normalized_spectra_.domain
                ),
                background=self.background_,
                return_units=False
            )
            # ensure vector
            # (TODO - avoid having to do this by changing capture function)
            x_pred = np.squeeze(x_pred)
        return x_pred

    def _process_X(self, X):
        """
        Returns photon capture and excitation value.
        """
        X = self._check_X(X)
        # check that input shape is correct
        if X.shape[1] != self.n_features_:
            raise ValueError("Shape of input is different from number"
                             "of measured spectra in container.")

        if self.x_as_capture:
            # X is already in photon capture
            return X, self.photoreceptor_model_.excitefunc(X)
        else:
            # use inverse of excitation function
            return self.photoreceptor_model_.inv_excitefunc(X), X

    @property
    def input_units(self):
        """units of X
        """
        return ureg(None).units
