"""
LED Estimators for intensities and spectra
"""

from abc import abstractmethod

from scipy.optimize import OptimizeResult
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from dreye.utilities import (
    is_dictlike, optional_to, is_listlike, is_string
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
            background = get_spectrum(
                wavelengths=measured_spectra.wavelengths,
                units=measured_spectra.units
            )
        elif is_string(background) and (background == 'null'):
            background = None
        elif isinstance(background, Signal):
            background = background.to(measured_spectra.units)
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
        """units of transformed X
        """
        return self.measured_spectra_.labels_units

    @property
    @abstractmethod
    def input_units(self):
        """units of X
        """

    @property
    @abstractmethod
    def fitted_X(self):
        """X after fitting
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
        X_pred = self.inverse_transform(self.transform(X))
        X = self.current_X_
        return X - X_pred

    def sample_scores(self, X=None):
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
        # apply transformation and compare to checked_X
        X_pred = self.inverse_transform(self.transform(X))
        X = self.current_X_

        # residual across photoreceptors
        res = (X - X_pred)**2
        tot = (X - X.mean(axis=-1, keepdims=True))**2
        return 1 - res.sum(-1)/tot.sum(-1)

    def feature_scores(self, X=None):
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
        # apply transformation and compare to checked_X
        X_pred = self.inverse_transform(self.transform(X))
        X = self.current_X_

        # residual across samples
        res = (X - X_pred)**2
        tot = (X - X.mean(axis=0, keepdims=True))**2
        return 1 - res.sum(0)/tot.sum(0)

    def score(self, X=None, y=None):
        """
        Score method

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
        return self.feature_scores(X).mean()

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
