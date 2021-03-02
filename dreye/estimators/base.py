"""
LED Estimators for intensities and spectra
"""

from abc import abstractmethod

from scipy.optimize import OptimizeResult
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from dreye.utilities import (
    is_dictlike, optional_to, is_listlike, is_string,
    is_numeric
)
from dreye.utilities.abstract import _AbstractContainer
from dreye.constants import ureg
from dreye.core.spectrum_utils import (
    get_spectrum, get_max_normalized_gaussian_spectra
)
from dreye.core.signal import Signal, Signals
from dreye.core.spectral_measurement import MeasuredSpectraContainer
from dreye.core.measurement_utils import get_led_spectra_container
from dreye.core.photoreceptor import Photoreceptor, get_photoreceptor_model


class OptimizeResultContainer(_AbstractContainer):
    _allowed_instances = OptimizeResult


class _SpectraModel(BaseEstimator, TransformerMixin):
    """
    Abstract Spectra model used for various Dreye estimators.
    """

    # other attributes that are the length of X but not X
    # and are not the fitted signal
    _X_length = []

    @staticmethod
    def _check_measured_spectra(
        measured_spectra, smoothing_window,
        size=None, photoreceptor_model=None,
        change_dimensionality=True
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
            and change_dimensionality
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
        # enforce measured_spectra units
        if is_dictlike(background):
            background['wavelengths'] = background.get(
                'wavelengths', measured_spectra.wavelengths
            )
            background['units'] = measured_spectra.units
            background = get_spectrum(**background)
        elif background is None:
            return
        elif is_string(background) and (background == 'null'):
            return
        elif isinstance(background, Signal):
            background = background.to(measured_spectra.units)
        elif is_listlike(background):
            background = optional_to(background, measured_spectra.units)
            # check size requirements
            assert background.size == measured_spectra.normalized_spectra.shape[
                measured_spectra.normalized_spectra.domain_axis
            ]
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

        if np.allclose(background.magnitude, 0):
            # return None if background is just a bunch of zeros
            return

        return background

    @staticmethod
    def _get_background_from_bg_ints(
        bg_ints, measured_spectra, skip_zero=True
    ):
        """
        Get background from `bg_ints` attribute
        """
        if np.allclose(bg_ints, 0) and skip_zero:
            # if background intensities are zero return None
            return
        # sanity check
        assert measured_spectra.normalized_spectra.domain_axis == 0
        background = (
            bg_ints
            * measured_spectra.normalized_spectra.magnitude
        ).sum(axis=-1)
        background = get_spectrum(
            intensities=background,
            wavelengths=measured_spectra.wavelengths,
            units=measured_spectra.units
        )
        return background

    @staticmethod
    def _get_bg_ints(bg_ints, measured_spectra, skip=True, rtype=None):
        """
        Get background intensity values
        """
        if skip and bg_ints is None:
            return
        # set background intensities to default
        if bg_ints is None:
            if rtype in {'absolute', 'diff'}:
                bg_ints = np.zeros(len(measured_spectra))
            else:
                bg_ints = np.ones(len(measured_spectra))
        elif is_numeric(bg_ints):
            bg_ints = np.ones(
                len(measured_spectra)
            ) * optional_to(bg_ints, measured_spectra.intensities.units)
        else:
            if is_dictlike(bg_ints):
                names = measured_spectra.names
                bg_ints = [bg_ints.get(name, 1) for name in names]
            bg_ints = optional_to(
                bg_ints,
                measured_spectra.intensities.units
            )
            assert len(bg_ints) == len(measured_spectra)
            assert np.all(bg_ints >= 0)
        return bg_ints

    @staticmethod
    def _check_reflectances(
        reflectances, measured_spectra, photoreceptor_model=None
    ):
        """
        get max normalized reflectances
        """
        # enforce unitless
        if is_dictlike(reflectances):
            reflectances['wavelengths'] = reflectances.get(
                'wavelengths', measured_spectra.wavelengths
            )
            reflectances = get_max_normalized_gaussian_spectra(**reflectances)
        elif reflectances is None:
            reflectances = get_max_normalized_gaussian_spectra(
                wavelengths=measured_spectra.wavelengths,
                units=measured_spectra.units
            )
        elif isinstance(reflectances, Signals):
            pass
            # reflectances = reflectances.to(ureg(None).units)
        elif is_listlike(reflectances):
            reflectances = get_max_normalized_gaussian_spectra(
                intensities=reflectances,
                wavelengths=measured_spectra.wavelengths,
            )
        else:
            raise ValueError(
                "Illuminant must be Spectra instance or dict-like, but"
                f"is of type {type(reflectances)}."
            )

        # ensure domain axis on 0
        if reflectances.domain_axis != 0:
            reflectances = reflectances.copy()
            reflectances.domain_axis = 0
        return reflectances

    def fit_transform(self, X):
        """
        Fit X values and transform to output values
        of `MeasuredSpectraContainer`.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            X values passed to the `fit` method.

        Returns
        -------
        output : array-like (n_samples, n_leds)
            The transformed X values into LED output values.

        See Also
        --------
        fit
        transform
        """
        self.fit(X)
        return self.transform()

    def _check_X(self, X):
        """units for X.

        e.g. in photonflux units.
        """
        X = optional_to(X, self.input_units)
        return check_array(X)

    @property
    def output_units(self):
        """
        Units of transformed X.
        """
        return self.measured_spectra_.labels_units

    @property
    @abstractmethod
    def input_units(self):
        """
        Units of X.
        """

    @property
    @abstractmethod
    def fitted_X_(self):
        """
        X after fitting.
        """
        return self.current_X_

    def to_dict(self):
        """
        Get estimator initial dictionary.

        See Also
        --------
        from_dict
        """
        return self.get_params()

    @classmethod
    def from_dict(cls, data):
        """
        Create estimator from intial dictionary

        See Also
        --------
        to_dict
        """
        return cls(**data)

    def residuals(self, X=None):
        """
        Residual for each photoreceptor and sample.
        """
        # apply transformation and compare to checked_X
        if X is not None:
            self.fit(X)
        X_pred = self.fitted_X_
        X = self.current_X_
        return X - X_pred

    def relative_changes(self, X=None):
        """
        Deviations for each photoreceptor and sample.
        """
        # apply transformation and compare to checked_X
        if X is not None:
            self.fit(X)
        X_pred = self.fitted_X_
        X = self.current_X_
        return (X_pred - X) / np.abs(X)

    @staticmethod
    def _r2_scores(X, X_pred, axis=0):
        # residual across photoreceptors
        res = (X - X_pred)**2
        tot = (X - X.mean(axis=axis, keepdims=True))**2
        return 1 - res.sum(axis) / tot.sum(axis)

    @staticmethod
    def _abs_rel_changes_scores(X, X_pred, axis=0):
        # absolute deviations
        return np.abs((X_pred - X) / X).mean(axis=axis)

    @staticmethod
    def _rel_changes_scores(X, X_pred, axis=0):
        # deviations
        return ((X_pred - X) / np.abs(X)).mean(axis=axis)

    @staticmethod
    def _rel_changes_thresh_scores(X, X_pred, axis=0, thresh=0.01):
        # absolute deviations below a certain threshold value
        return (np.abs((X_pred - X) / X) < thresh).mean(axis=axis)

    @staticmethod
    def _corr_dist(X, X_pred, axis=0):
        # correlation distortion
        cX = np.corrcoef(X, rowvar=bool(axis % 2))
        cX_pred = np.corrcoef(X, rowvar=bool(axis % 2))
        return (cX_pred - cX) / np.abs(cX)

    def _mean_scores(self, X=None, axis=0, method='r2', **kwargs):
        # apply transformation and compare to checked_X
        if X is not None:
            self.fit(X)
        X_pred = self.fitted_X_
        X = self.current_X_

        # residual across photoreceptors
        if method == 'r2':
            return self._r2_scores(
                X, X_pred, axis=axis, **kwargs)
        elif method == 'rel':
            return self._rel_changes_scores(
                X, X_pred, axis=axis, **kwargs)
        elif method == 'absrel':
            return self._abs_rel_changes_scores(
                X, X_pred, axis=axis, **kwargs)
        elif method == 'threshrel':
            return self._rel_changes_thresh_scores(
                X, X_pred, axis=axis, **kwargs)
        elif method == 'corrdev':
            return self._corr_dist(
                X, X_pred, axis=axis, **kwargs
            )
        # raise
        raise NameError(f"Method `{method}` not recognized, use "
                        "`r2`, `rel`, `absrel`, `threshrel`, or `corrdev`.")

    def sample_scores(self, X=None, method='r2', **kwargs):
        """
        Sample scores.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            X values passed to the `fit` method.

        Returns
        -------
        r2 : numpy.ndarray (n_samples)
            Returns the r2-value for each sample individually.
        """
        return self._mean_scores(X=X, axis=1, method=method, **kwargs)

    def feature_scores(self, X=None, method='r2', **kwargs):
        """
        Feature scores.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            X values passed to the `fit` method.

        Returns
        -------
        r2 : numpy.ndarray (n_features)
            Returns the r2-value for each feature individually.
        """
        return self._mean_scores(X=X, axis=0, method=method, **kwargs)

    def score(self, X=None, y=None, method='r2', **kwargs):
        """
        Score method.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            X values passed to the `fit` method.

        Returns
        -------
        r2 : float
            Mean r2-value across features

        See Also
        --------
        feature_scores
        """
        return self.feature_scores(X, method=method, **kwargs).mean()

    def transform(self, X=None):
        """
        Transform X values to output values of `MeasuredSpectraContainer`.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            X values passed to the `fit` method.

        Returns
        -------
        output : array-like (n_samples, n_leds)
            The transformed X values into LED output values.

        See Also
        --------
        dreye.MeasuredSpectraContainer
        dreye.MeasuredSpectrum
        fit_transform
        """
        # check is fitted
        check_is_fitted(
            self, ['measured_spectra_', 'fitted_intensities_']
        )

        if X is not None:
            # refit if X is given
            self.fit(X)
        # map fitted_intensities
        return self.measured_spectra_.map(
            self.fitted_intensities_, return_units=False)

    @abstractmethod
    def inverse_transform(self, X):
        """
        Transform output values to values that can be used for scoring.

        Parameters
        ----------
        X : array-like (n_samples, n_leds)
            Output values in units of `output_units`.

        Returns
        -------
        X : array-like (n_samples, n_features)
            Input values in units of `input_units`.

        See Also
        --------
        dreye.MeasuredSpectraContainer
        dreye.MeasuredSpectrum
        fit_transform
        """
        pass

    @abstractmethod
    def fit(self, X):
        """
        Fit X values.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            X values in `input_units`.

        Returns
        -------
        self

        See Also
        --------
        dreye.MeasuredSpectraContainer
        dreye.MeasuredSpectrum
        fit_transform
        """
        pass


class _RelativeMixin:

    def _to_absolute_intensity(self, X, bg_ints=None):
        if bg_ints is None:
            bg_ints = self.bg_ints_
        if self.rtype == 'absolute':
            return X
        elif self.rtype == 'diff':
            return X + bg_ints
        # convert to intensity
        if self.rtype in {'fechner', 'log'}:
            X = np.exp(X)
        elif self.rtype not in {'weber', 'total_weber'}:
            assert np.all(X >= 0), 'If not log, X must be positive.'

        if self.rtype.startswith('total'):
            X = X * np.sum(self.bg_ints_)
        else:
            X = X * bg_ints
        if self.rtype in {'weber', 'total_weber'}:
            X = X + bg_ints
        return X

    def _to_relative_intensity(self, X, bg_ints=None):
        if bg_ints is None:
            bg_ints = self.bg_ints_
        if self.rtype == 'absolute':
            return X
        elif self.rtype == 'diff':
            return X - bg_ints
        # convert to relative intensity
        if self.rtype in {'weber', 'total_weber'}:
            X = X - bg_ints

        if self.rtype.startswith('total'):
            X = X / np.sum(self.bg_ints_)
        else:
            X = X / bg_ints

        if self.rtype in {'fechner', 'log'}:
            assert np.all(X > 0), 'If log, X cannot be zero or lower.'
            X = np.log(X)
        elif self.rtype not in {'weber', 'total_weber'}:
            assert np.all(X >= 0), 'If not log, X must be positive.'

        return X
