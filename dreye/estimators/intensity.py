"""
Stimulus Estimators
"""


from sklearn.base import BaseEstimator

from dreye.constants import ureg
from dreye.core.photoreceptor import AbstractPhotoreceptor
from dreye.core.spectral_measurement import MeasuredSpectraContainer


# Relative intensity model

class AbstractFitModel(BaseEstimator):
    """
    Abstract Chromatic Model.

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
        return self.measured_spectra.map(X, *args, **kwargs)

    def fit(self, X):
        """
        No fitting in the abstract model.
        """
        pass

    def transform(self, X):
        return X

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


class SpectrumFitModel(AbstractFitModel):

    def fit(self, X):
        """
        Fit a dreye.Spectrum object or numpy.array
        """
        pass

    def transform(self, X):
        """
        Go from spectrum to input values?
        """


class ExcitationFitModel(AbstractFitModel):

    def __init__(
        self,
        measured_spectra,
        photoreceptor_model=None,
        background=None,
        weights=None,
        units=True,
        # use_excitation=False,
        only_uniques=False,
        respect_zero=False,
    ):
        assert isinstance(photoreceptor_model, AbstractPhotoreceptor)
        assert isinstance(measured_spectra, MeasuredSpectraContainer)
        self.photoreceptor_model = photoreceptor_model
        self.measured_spectra = measured_spectra

    # fit and fit_transform and transform units property,
    # intensity units, output units, labels_units
    # X are the capture values! or excitation values?
    # map -> transform

    def fit(self, X):
        # spectrum object
        # if 3D assume that is is spectrum?
        # independent?
        # initialize with normal lsq_linear
        # the apply least_squares (reshape weights?)
        pass

    def transform(self, X):
        pass


# LinearTransformGLM
# IlluminantGLM
# IlluminantBgGLM
