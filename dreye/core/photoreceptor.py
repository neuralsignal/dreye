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
    Create an arbitrary photoreceptor model.

    Parameters
    ----------
    sensitivity : Sensitivity instance or array-like, optional
    wavelengths : array-like, optional
    filterfunc : callable, optional
    labels : array-like, optional
    photoreceptor_type : str ({'linear', 'log'}), optional
    kwargs : dict, optional

    Returns
    -------
    object : `Photoreceptor`
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
    kwargs : dict, optional
    """

    def __init__(
        self, sensitivity, wavelengths=None, filterfunc=None,
        labels=None, **kwargs
    ):
        if isinstance(sensitivity, Photoreceptor):
            sensitivity = sensitivity.sensitivity
            if filterfunc is None:
                filterfunc = sensitivity.filterfunc
        if not isinstance(sensitivity, Sensitivity):
            sensitivity = Sensitivity(
                sensitivity, domain=wavelengths, labels=labels,
                **kwargs
            )
        else:
            if wavelengths is not None:
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

    def __str__(self):
        return f"{type(self).__name__}{tuple(self.labels)}"

    def to_dict(self):
        """
        Returns `self` as dictionary containing the
        photoreceptor's sensitivity and the filter function.
        """
        return {
            "sensitivity": self.sensitivity,
            "filterfunc": self.filterfunc
        }

    @classmethod
    def from_dict(cls, dictionary):
        """
        Create Photoreceptor class given a dictionary containing
        the `sentivity` and `filterfunc` (optionally).
        """
        return cls(**dictionary)

    def __copy__(self):
        """
        Copy method for `Photoreceptor`.
        """
        return type(self)(
            **self.to_dict().copy()
        )

    def copy(self):
        """
        Copy `Photoreceptor` instance.
        """
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
        """
        `Sensitivity` instance containing all the spectral sensitivities.
        """
        return self._sensitivity

    @property
    def pr_number(self):
        """
        The number of photoreceptor types.
        """
        return self.sensitivity.shape[1]

    @property
    def wavelengths(self):
        """
        `Domain` instance of wavelength values.
        """
        return self.sensitivity.wavelengths

    @property
    def labels(self):
        """
        Labels of each photoreceptor type.

        See Also
        --------
        Photoreceptor.names
        """
        return self.sensitivity.labels

    @property
    def names(self):
        """
        Names of each photoreceptor type.

        Alias for `Photoreceptor.labels`.
        """
        return self.labels

    @property
    def filterfunc(self):
        """
        Filter function applied on light entering each photoreceptor.

        The filter function should accept three positional arguments
        corresponding to the wavelength `numpy.ndarray`, illuminant
        `numpy.ndarray`, and the sensitvity `numpy.ndarray` respectively.

        All three arrays are broadcasted already before being passed to
        `filterfunc`.
        """
        return self._filterfunc

    def excitation(
        self,
        illuminant,
        reflectance=None,
        background=None,
        return_units=None,
        wavelengths=None,
        **kwargs
    ):
        """
        Calculate photoreceptor excitation.

        Parameters
        ----------
        illuminant : `Signals` or array-like
            Illuminant spectrum. If array-like, the zeroth axis must have
            the same length as `Photoreceptor.sensitivity` (i.e. wavelength
            domain match) or `wavelengths` must be given.
        reflectance : `Signals` or array-like, optional
            If given, illuminant is multiplied by the reflectance.
        background : `Signal` or array-like, optional
            If given, the relative photon capture is calculated. By
            calculating the capture for the background.
        return_units : bool, optional
            If True, return a `pint.Quantity`. Otherwise, return a
            `numpy.ndarray`.
        wavelengths : array-like, optional
            If given and illuminant is not a `Signals` instance, this
            corresponds to its wavelength values.
        kwargs : dict
            Keyword arguments passed to the `Photoreceptor.excitefunc`
            function.

        Returns
        -------
        excitations : `numpy.ndarray` or `pint.Quantity`
            Array of row length equal to the number of illuminant and
            column length equal to the number of photoreceptor types.
        """
        return self.excitefunc(
            self.capture(illuminant=illuminant,
                         reflectance=reflectance,
                         background=background,
                         return_units=return_units,
                         wavelengths=wavelengths),
            **kwargs
        )

    def capture(
        self,
        illuminant,
        reflectance=None,
        background=None,
        return_units=None,
        wavelengths=None
    ):
        """
        Calculate photoreceptor capture.

        Parameters
        ----------
        illuminant : `Signals` or array-like
            Illuminant spectrum. If array-like, the zeroth axis must have
            the same length as `Photoreceptor.sensitivity` (i.e. wavelength
            domain match) or `wavelengths` must be given.
        reflectance : `Signals` or array-like, optional
            If given, illuminant is multiplied by the reflectance.
        background : `Signal` or array-like, optional
            If given, the relative photon capture is calculated. By
            calculating the capture for the background.
        return_units : bool, optional
            If True, return a `pint.Quantity`. Otherwise, return a
            `numpy.ndarray`.
        wavelengths : array-like, optional
            If given and illuminant is not a `Signals` instance, this
            corresponds to its wavelength values.

        Returns
        -------
        captures : `numpy.ndarray` or `pint.Quantity`
            Array of row length equal to the number of illuminant and
            column length equal to the number of photoreceptor types.
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
                domain=(
                    self.wavelengths
                    if wavelengths is None
                    else wavelengths
                )
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
            q_bg = self.capture(
                background, return_units=return_units, wavelengths=wavelengths
            )
            # q_bg may have different units to q
            return q / q_bg

    # TODO def decomp_sensitivity(self):
    #     self.sensitivity.magnitude


class LinearPhotoreceptor(Photoreceptor):

    @staticmethod
    def excitefunc(arr):
        """
        Returns `arr`.
        """
        return arr

    @staticmethod
    def inv_excitefunc(arr):
        """
        Returns `arr`.
        """
        return arr


class LogPhotoreceptor(Photoreceptor):

    @staticmethod
    def excitefunc(arr):
        """
        Returns the log of `arr`.
        """
        return np.log(arr)

    @staticmethod
    def inv_excitefunc(arr):
        """
        Returns the exp of `arr`.
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
