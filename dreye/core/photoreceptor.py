"""Class to define photoreceptor/capture model
"""

import copy
from abc import ABC, abstractmethod

from scipy.stats import norm
import numpy as np

from dreye.core.signal import _SignalMixin, _Signal2DMixin, Signals
from dreye.core.spectral_sensitivity import Sensitivity
from dreye.err import DreyeError
from dreye.utilities import (
    is_callable, has_units, asarray, is_integer
)


def get_photoreceptor_model(
    sensitivity=None, wavelengths=None, filterfunc=None, labels=None,
    photoreceptor_type='linear', **kwargs
):
    """
    Create an arbitrary photoreceptor model

    Parameters
    ----------
    sensitivity : Sensitivity instance or array-like
    wavelengths : array-like
    filterfunc : callable
    labels : array-like
    photoreceptor_type : {'linear', 'log'}
    """
    if photoreceptor_type not in {'linear', 'log'}:
        raise DreyeError("Photoreceptor type must be 'linear' or 'log'.")
    if photoreceptor_type == 'linear':
        pr_class = LinearPhotoreceptor
    elif photoreceptor_type == 'log':
        pr_class = LogPhotoreceptor

    if sensitivity is None or is_integer(sensitivity):
        if wavelengths is None:
            wavelengths = np.arange(300, 700.1, 0.5)
        if sensitivity is None:
            centers = np.linspace(350, 550, 3)
        else:
            centers = np.linspace(350, 550, sensitivity)
        sensitivity = norm.pdf(wavelengths[:, None], centers[None, :], 40)
        sensitivity /= np.max(sensitivity, axis=0, keepdims=True)
    elif wavelengths is None and not isinstance(sensitivity, _SignalMixin):
        wavelengths = np.linspace(300, 700, len(sensitivity))

    return pr_class(
        sensitivity,
        wavelengths=wavelengths,
        filterfunc=filterfunc,
        labels=labels,
        **kwargs
    )


class Photoreceptor(ABC):
    """
    Photoreceptor model.

    Parameters
    ----------
    sensitivity : Sensitivity instance or array-like
    wavelengths : array-like, optional
    filterfunc : callable, optional
    labels : array-like, optional
    """

    def __init__(
        self, sensitivity, wavelengths=None, filterfunc=None,
        labels=None, **kwargs
    ):
        if not isinstance(sensitivity, Sensitivity):
            sensitivity = Sensitivity(
                sensitivity, domain=wavelengths, labels=labels,
                **kwargs
            )
        elif wavelengths is not None:
            sensitivity = sensitivity(wavelengths)
        if labels is not None:
            sensitivity.labels = labels

        # ensure domain axis = 0
        if sensitivity.domain_axis != 0:
            sensitivity = sensitivity.copy()
            sensitivity.domain_axis = 0

        assert is_callable(filterfunc) or filterfunc is None

        self._sensitivity = sensitivity
        self._filterfunc = filterfunc

    def to_dict(self):

        return {
            "sensitivity": self.sensitivity,
            "filterfunc": self.filterfunc
        }

    @classmethod
    def from_dict(cls, dictionary):

        return cls(**dictionary)

    def __copy__(self):
        return type(self)(
            **self.to_dict().copy()
        )

    def copy(self):
        return copy.copy(self)

    @abstractmethod
    def excitefunc(self, arr):
        """excitation function
        """

    @abstractmethod
    def inv_excitefunc(self, arr):
        """inverse of excitation function
        """

    @property
    def sensitivity(self):
        return self._sensitivity

    @property
    def pr_number(self):
        return self.sensitivity.shape[1]

    @property
    def wavelengths(self):
        return self.sensitivity.wavelengths

    @property
    def labels(self):
        return self.sensitivity.labels

    @property
    def filterfunc(self):
        return self._filterfunc

    def excitation(
        self,
        illuminant,
        reflectance=None,
        background=None,
        return_units=None,
        **kwargs
    ):
        """
        Calculate the photoreceptor excitation.
        """
        return self.excitefunc(
            self.capture(illuminant=illuminant,
                         reflectance=reflectance,
                         background=background,
                         return_units=return_units),
            **kwargs
        )

    def capture(
        self,
        illuminant,
        reflectance=None,
        background=None,
        return_units=None,
    ):
        """
        Calculate the photon capture.
        """
        if return_units is None:
            return_units = has_units(illuminant)

        if (
            not isinstance(illuminant, _Signal2DMixin)
            and asarray(illuminant).ndim <= 2
        ):
            # Not assuming any units
            illuminant = Signals(
                illuminant,
                domain=self.wavelengths
            )
            # set domain axis
            illuminant.domain_axis = 0
        elif asarray(illuminant).ndim > 2:
            raise ValueError("Illuminant must be 1- or 2-dimensional.")
        elif illuminant.domain_axis != 0:
            illuminant = illuminant.copy()
            illuminant.domain_axis = 0

        if reflectance is not None:
            illuminant = illuminant * reflectance

        sensitivity, illuminant = self.sensitivity.equalize_domains(
            illuminant
        )
        wls = sensitivity.domain

        new_units = illuminant.units * sensitivity.units * wls.units

        # TODO dealing with higher dimensional illuminant and 1-dimensional
        # TODO efficiency
        # added_dim = illuminant.ndim - 1
        # (slice(None, None, None), ) * 2 + (None, ) * added_dim
        sensitivity = sensitivity.magnitude[:, None, :]
        illuminant = illuminant.magnitude[:, :, None]
        wls = wls.magnitude[:, None, None]

        # illuminant can be filtered after equalizing domains
        # TODO how to handle background (also filter?)
        if self.filterfunc is not None:
            illuminant = self.filterfunc(wls, illuminant, sensitivity)

        # calculate capture
        # illuminant x opsin via integral
        q = np.trapz(sensitivity * illuminant, wls, axis=0)

        if return_units:
            q = q * new_units

        if background is None:
            return q
        else:
            q_bg = self.capture(background, return_units=return_units)
            # q_bg may have different units to q
            return q / q_bg


class LinearPhotoreceptor(Photoreceptor):
    """
    """

    @staticmethod
    def excitefunc(arr):
        """excitation function
        """

        return arr

    @staticmethod
    def inv_excitefunc(arr):
        """excitation function
        """

        return arr


class LogPhotoreceptor(Photoreceptor):
    """
    """

    @staticmethod
    def excitefunc(arr):
        """excitation function
        """

        return np.log(arr)

    @staticmethod
    def inv_excitefunc(arr):
        """excitation function
        """

        return np.exp(arr)


# TODO
# class SelfScreeningPhotoreceptor(LinearPhotoreceptor):
#     """
#     """
#
#     @staticmethod
#     def filterfunc(arr):
#         """
#         """
#
#         raise NotImplementedError('self screening photoreceptor class.')
