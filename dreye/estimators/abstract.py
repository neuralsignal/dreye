"""
Abstract Fit Model
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from dreye.utilities import is_dictlike, optional_to, asarray
from dreye.constants import ureg
from dreye.core.spectral_measurement import MeasuredSpectraContainer
from dreye.core.measurement_utils import get_led_spectra_container

# must contain fit, transform, and map

# by ducktyping measured_spectra?
# ducktyping photoreceptor -> capture, excitation


class IntensityFitModel(BaseEstimator, TransformerMixin):
    """
    Fit intensity values to a given LED system.

    All models that fit chromatic stimuli have to accept a
    measured spectra object.
    """

    def __init__(
        self,
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        smoothing_window=None  # float
    ):
        self.measured_spectra = measured_spectra
        self.smoothing_window = smoothing_window

    @staticmethod
    def check_measured_spectra(measured_spectra, smoothing_window, X):
        if isinstance(measured_spectra, MeasuredSpectraContainer):
            pass
        elif is_dictlike(measured_spectra):
            measured_spectra = get_led_spectra_container(
                **measured_spectra
            )
        elif measured_spectra is None:
            measured_spectra = get_led_spectra_container(X.shape[1])
        else:
            raise ValueError("Measured Spectra must be Spectra "
                             "container or dict, but is type "
                             f"'{type(measured_spectra)}'.")

        if smoothing_window is not None:
            measured_spectra = measured_spectra.smooth(smoothing_window)

        return measured_spectra

    def fit(self, X, y=None):
        """
        Fit method.
        """
        # input validation
        X = self.check_array(X)
        #
        self.measured_spectra_ = self.check_measured_spectra(
            self.measured_spectra, self.smoothing_window, X
        )
        # call in order to fit isotonic regression
        self.regressor

        self.n_features_ = len(self._measured_spectra)

        # check X
        X = self.check_X(X)

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
        X = self.check_X(X)

        # check that input shape is correct
        if X.shape[1] != self.n_features_:
            raise ValueError("Shape of input is different from what was seen"
                             "in `fit`.")

        # apply mapping
        return self.measured_spectra_.map(X, return_units=False)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """
        Inverse Transform method
        """
        # check is fitted
        check_is_fitted(self, ['n_features_', 'measured_spectra_'])

        # check X
        X = optional_to(X, self.measured_spectra_.labels_units)
        X = check_array(X)

        # map output values to intensities
        return self.measured_spectra_.reverse_map(X, return_units=False)

    def score(self, X, y=None):
        """
        Score method
        """
        # input validation
        X = self.check_X(X)
        return self.measured_spectra_.score(X)

    def check_X(self, X):
        """units for X.

        e.g. in photonflux units.
        """
        X = optional_to(X, self.measured_spectra_.intensities.units)
        return check_array(X)

    @property
    def output_units(self):
        """units of transformed X
        """
        return self.measured_spectra_.labels_units

    @property
    def input_units(self):
        """units of X
        """
        return self.measured_spectra_.intensities.units

    def to_dict(self):
        return self.get_params()

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class IlluminantFitModel(IntensityFitModel):
    """
    Fit illuminant spectra
    """

    def fit(self, X, y=None):
        """
        Fit method.
        """
        # X as array, unit-array - wavelength equal
        # X as domain_signal - equalize_labels
        # X as iterable of signals
        pass

    def transform(self, X):
        """
        """
        pass

    def inverse_transform(self, X):
        raise NotImplementedError(
            "inverse transform cannot be applied to "
        )

    @property
    def output_units(self):
        """units of transformed X
        """
        return self.measured_spectra_.intensities.units

    @property
    def input_units(self):
        """units of X
        """
        return self.measured_spectra_.units

#
# def fit(self, spectrum, return_res=False, return_fit=False, units=True):
#     """fit a single spectrum
#     """
#
#     # TODO move to stim_estimator
#
#     assert isinstance(spectrum, Spectrum)
#     assert spectrum.ndim == 1
#
#     spectrum = spectrum.copy()
#     spectrum.units = self.units
#
#     spectrum, normalized_sources = spectrum.equalize_domains(
#         self.normalized_spectrum, equalize_dimensions=False)
#
#     b = asarray(spectrum)
#     A = asarray(normalized_sources)
#
#     res = lsq_linear(A, b, bounds=self.bounds)
#
#     # Class which incorportates the following
#     # values=res.x, units=self.units, axis0_labels=self.labels
#
#     if units:
#         weights = res.x * self.units * ureg('nm')
#     else:
#         weights = res.x
#
#     fitted_spectrum = (
#         self.normalized_spectrum * weights[None, :]
#     ).sum(axis=1)
#
#     if return_res and return_fit:
#         return weights, res, fitted_spectrum
#     elif return_res:
#         return weights, res
#     elif return_fit:
#         return weights, fitted_spectrum
#     else:
#         return weights
#
# def fit_map(self, spectrum, **kwargs):
#     """
#     """
#
#     # TODO remove
#
#     values = self.fit(spectrum, **kwargs)
#
#     return self.map(values)
#
#
# def fit_background(
#     measured_spectra, background, return_fit=True, units=True,
#     **kwargs
# ):
#     """
#     Fit background spectrum to measurement of light sources.
#     """
#
#     assert isinstance(measured_spectra, MeasuredSpectraContainer)
#     assert isinstance(background, Spectrum)
#
#     return measured_spectra.fit(
#         background, return_fit=return_fit, units=units, **kwargs)
