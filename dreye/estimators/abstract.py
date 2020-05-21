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

    def __init__(self, measured_spectra, check_units=True):
        assert isinstance(measured_spectra, MeasuredSpectraContainer)
        self.measured_spectra = measured_spectra
        self.check_units = check_units

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
        return self.measured_spectra.label_units

    def to_dict(self):
        pass

    @classmethod
    def from_dict(cls, data):
        pass
