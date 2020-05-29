"""Class to define photoreceptor/capture model
"""

import numpy as np
import copy
from abc import ABC, abstractmethod

from dreye.core.signal import _Signal2DMixin, Signals
from dreye.core.spectral_sensitivity import Sensitivity
from dreye.utilities import is_callable, has_units, asarray


class Photoreceptor(ABC):
    """
    Photoreceptor model.

    Parameters
    ----------
    sensitivity : SpectralSensitivity instance or array-like
    wavelengths : array-like, optional
    filterfunc : callable, optional
    labels : array-like, optional
    """

    def __init__(
        self, sensitivity, wavelengths=None, filterfunc=None,
        labels=None
    ):
        if not isinstance(sensitivity, Sensitivity):
            sensitivity = Sensitivity(
                sensitivity, domain=wavelengths, labels=labels
            )
        elif wavelengths is not None:
            sensitivity = sensitivity(wavelengths)
        if labels is not None:
            sensitivity.labels = labels

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

        if reflectance is not None:
            illuminant = illuminant * reflectance
        # illuminant can be filtered after equalizing domains
        if self.filterfunc is not None:
            illuminant = self.filterfunc(illuminant)

        sensitivity, illuminant = self.sensitivity.equalize_domains(
            illuminant
        )
        wls = sensitivity.domain

        new_units = illuminant.units * sensitivity.units * wls.units

        # TODO dealing with higher dimensional illuminant
        # added_dim = illuminant.ndim - 1
        # (slice(None, None, None), ) * 2 + (None, ) * added_dim
        sensitivity = sensitivity.magnitude[:, :, None]
        illuminant = illuminant.magnitude[:, None, :]
        wls = wls.magnitude[:, None, None]

        # calculate capture
        # opsin x illuminant via integral
        q = np.trapz(sensitivity * illuminant, wls, axis=0)

        if return_units:
            q = q * new_units

        if background is None:
            return q
        else:
            q_bg = self.capture(background, return_units=False)
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
