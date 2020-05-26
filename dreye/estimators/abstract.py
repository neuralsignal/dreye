"""
Abstract Fit Model
"""

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator

from dreye.constants import ureg
from dreye.core.spectral_measurement import MeasuredSpectraContainer

# must contain fit, transform, and map


class AbstractFitModel(ABC, BaseEstimator):
    """
    Abstract Stimulus Fit Model.

    All models that fit chromatic stimuli have to accept a
    measured spectra object.
    """

    def __init__(self, measured_spectra, track_units=True):
        assert isinstance(measured_spectra, MeasuredSpectraContainer)
        self.measured_spectra = measured_spectra
        self.track_units = track_units

    def map(self, X, *args, **kwargs):
        """
        Alias for measured_spectra.map
        """
        # TODO make actual mapper
        return self.measured_spectra.map(X, *args, **kwargs)

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def fit_map(self, X):
        new_X = self.fit_transform(X)
        return self.map(new_X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @property
    def X_units(self):
        """units for X
        """
        return self.measured_spectra.units * ureg('nm').units

    @property
    def transformed_units(self):
        """units for mapping
        """
        return self.measured_spectra.units * ureg('nm').units

    @property
    def mapped_units(self):
        """mapped units
        """
        return self.measured_spectra.labels_units

    def to_dict(self):
        pass

    @classmethod
    def from_dict(cls, data):
        pass



    def fit(self, spectrum, return_res=False, return_fit=False, units=True):
        """fit a single spectrum
        """

        # TODO move to stim_estimator

        assert isinstance(spectrum, Spectrum)
        assert spectrum.ndim == 1

        spectrum = spectrum.copy()
        spectrum.units = self.units

        spectrum, normalized_sources = spectrum.equalize_domains(
            self.normalized_spectrum, equalize_dimensions=False)

        b = asarray(spectrum)
        A = asarray(normalized_sources)

        res = lsq_linear(A, b, bounds=self.bounds)

        # Class which incorportates the following
        # values=res.x, units=self.units, axis0_labels=self.labels

        if units:
            weights = res.x * self.units * ureg('nm')
        else:
            weights = res.x

        fitted_spectrum = (
            self.normalized_spectrum * weights[None, :]
        ).sum(axis=1)

        if return_res and return_fit:
            return weights, res, fitted_spectrum
        elif return_res:
            return weights, res
        elif return_fit:
            return weights, fitted_spectrum
        else:
            return weights

    def fit_map(self, spectrum, **kwargs):
        """
        """

        # TODO remove

        values = self.fit(spectrum, **kwargs)

        return self.map(values)


def fit_background(
    measured_spectra, background, return_fit=True, units=True,
    **kwargs
):
    """
    Fit background spectrum to measurement of light sources.
    """

    assert isinstance(measured_spectra, MeasuredSpectraContainer)
    assert isinstance(background, Spectrum)

    return measured_spectra.fit(
        background, return_fit=return_fit, units=units, **kwargs)
