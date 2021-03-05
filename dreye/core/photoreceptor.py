"""Class to define photoreceptor/capture model
"""

import warnings
import copy
from abc import ABC, abstractmethod

from scipy.stats import norm
import numpy as np

from dreye.core.signal import _SignalMixin, _Signal2DMixin, Signals
from dreye.utilities.abstract import inherit_docstrings
from dreye.core.spectral_sensitivity import Sensitivity
from dreye.constants import ureg
from dreye.err import DreyeError
from dreye.utilities import (
    is_callable, has_units, asarray, is_integer
)


RELATIVE_SENSITIVITY_SIGNIFICANT = 1e-2


def get_photoreceptor_model(
    sensitivity=None, wavelengths=None, filterfunc=None, labels=None,
    photoreceptor_type='linear', **kwargs
):
    """
    Create an arbitrary photoreceptor model.

    Parameters
    ----------
    sensitivity : Sensitivity instance or array-like, optional
        An array that contains the sensitivity of different photoreceptor
        types across wavelengths (wavelengths x types).
    wavelengths : array-like, optional
        The wavelength values in nanometers. This must be the same size as
        the number of rows in the sensitivity array.
    filterfunc : callable, optional
        A function that accepts three positional arguments:
        wavelengths, illuminant, and sensitivity. All three arguments are
        `numpy.ndarray` objects that are broadcastable to each other.
        The function should return the illuminant after the wavelength-specific
        filter has been applied.
    labels : array-like, optional
        The labels for each photoreceptor. The length of labels must
        correspond to the length of the columns in `sensitivity`.
    photoreceptor_type : str ({'linear', 'log', 'hyperbolic'}), optional
        The photoreceptor model.
    kwargs : dict, optional
        A dictionary that is directly passed to the photoreceptor class
        instantiation.

    Returns
    -------
    object : `Photoreceptor` class
        A photoreceptor instance which has the `capture` and `excitation`
        method implemented.

    See Also
    --------
    dreye.Photoreceptor
    dreye.LinearPhotoreceptor
    dreye.LogPhotoreceptor
    dreye.HyperbolicPhotoreceptor

    Notes
    -----
    This is a convenience function to create arbitrary photoreceptor models.
    It is also possible to create a photoreceptor model directly using the
    `dreye.LogPhotoreceptor`, `dreye.LinearPhotoreceptor` and
    `dreye.HyperbolicPhotoreceptor` classes.

    It is usually not necessary to supply a `filterfunc` argument, unless the
    photoreceptor model contains a filter that varies with the intensity and
    wavelength of the illuminant (for example, the photoreceptor model
    by Stavenga et al, 2003).
    """
    if photoreceptor_type not in {'linear', 'log', 'hyperbolic'}:
        raise DreyeError("Photoreceptor type must be 'linear' or 'log'.")
    if photoreceptor_type == 'linear':
        pr_class = LinearPhotoreceptor
    elif photoreceptor_type == 'log':
        pr_class = LogPhotoreceptor
    elif photoreceptor_type == 'hyperbolic':
        pr_class = HyperbolicPhotoreceptor

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


# TODO convenience capture function


class Photoreceptor(ABC):
    """
    Abstract Photoreceptor model for subclassing.

    Parameters
    ----------
    sensitivity : Sensitivity instance or array-like, optional
        An array that contains the sensitivity of different photoreceptor
        types across wavelengths (wavelengths x types).
    wavelengths : array-like, optional
        The wavelength values in nanometers. This must be the same size as
        the number of rows in the sensitivity array.
    filterfunc : callable, optional
        A function that accepts three positional arguments:
        wavelengths, illuminant, and sensitivity. All three arguments are
        `numpy.ndarray` objects that are broadcastable to each other.
        The function should return the illuminant after the wavelength-specific
        filter has been applied.
    labels : array-like, optional
        The labels for each photoreceptor. The length of labels must
        correspond to the length of the columns in `sensitivity`.
    capture_noise_level : None or float, optional
        The relative capture noise level. This is used when calculating
        relative capture values.
    kwargs : dict, optional
        A dictionary that is directly passed to the instantiation of
        the `dreye.Sensitivity` class.

    See Also
    --------
    dreye.get_photoreceptor_model
    dreye.LinearPhotoreceptor
    dreye.LogPhotoreceptor
    dreye.HyperbolicPhotoreceptor

    Notes
    -----
    When subclassing this abstract class you need to implement two functions,
    that are `excitefunc` and `inv_excitefunc`. The `excitefunc` method
    accepts a an array-like object that converts photon capture values to
    photoreceptor excitations. The `inv_excitefunc` method accepts
    photoreceptor excitations and converts them back to photon capture values.

    It is usually not necessary to supply a `filterfunc` argument, unless the
    photoreceptor model contains a filter that varies with the intensity and
    wavelength of the illuminant (for example, the photoreceptor model
    by Stavenga et al, 2003).
    """

    def __init__(
        self, sensitivity, wavelengths=None, filterfunc=None,
        labels=None, capture_noise_level=None, **kwargs
    ):
        if isinstance(sensitivity, Photoreceptor):
            if filterfunc is None:
                filterfunc = sensitivity.filterfunc
            if capture_noise_level is None:
                capture_noise_level = sensitivity.capture_noise_level
            sensitivity = sensitivity.sensitivity
        if not isinstance(sensitivity, Sensitivity):
            sensitivity = Sensitivity(
                sensitivity, domain=wavelengths, labels=labels,
                **kwargs
            )
        else:
            if wavelengths is not None:
                sensitivity = sensitivity(wavelengths)
            if labels is not None:
                # TODO do we want to do this inplace?
                sensitivity.labels = labels

        # ensure domain axis = 0
        if sensitivity.domain_axis != 0:
            sensitivity = sensitivity.copy()
            sensitivity.domain_axis = 0

        assert is_callable(filterfunc) or filterfunc is None

        self._sensitivity = sensitivity
        self._filterfunc = filterfunc
        self._capture_noise_level = capture_noise_level

    @property
    def data(self):
        """
        `numpy.ndarray` of sensitivities.
        """
        return self.sensitivity.magnitude

    @property
    def wls(self):
        """
        `numpy.ndarray` of wavelengths
        """
        return self.sensitivity.domain.magnitude

    def compute_ratios(self, rtol=None, return_wls=False):
        """
        Compute ratios of the sensitivities for all significant wavelengths.
        """
        wl_range = self.wavelength_range(rtol)
        wls = np.arange(*wl_range)
        # NB: sensitivity in pr_model always has domain on zeroth axis
        s = self.sensitivity(
            wls, check_bounds=False, asarr=True
        )
        if np.any(s < 0):
            warnings.warn(
                "Zeros or smaller in sensitivities array!", RuntimeWarning
            )
            s[s < 0] = 0
        ratios = s / np.sum(np.abs(s), axis=1, keepdims=True)
        if return_wls:
            return wl_range, ratios
        else:
            return ratios

    def wavelength_range(self, rtol=None):
        """
        Range of wavelengths that the photoreceptor are sensitive to.
        Returns a tuple of the min and max wavelength value.
        """
        rtol = (RELATIVE_SENSITIVITY_SIGNIFICANT if rtol is None else rtol)
        tol = (
            (self.sensitivity.max() - self.sensitivity.min())
            * rtol
        )
        return self.sensitivity.nonzero_range(tol).boundaries

    def __str__(self):
        """
        String representation of photoreceptor model.
        """
        return f"{type(self).__name__}(\n\t{self.sensitivity.magnitude}\n)"

    def __repr__(self):
        """
        String representation of photoreceptor model.
        """
        return f"{type(self).__name__}{tuple(self.labels)}"

    def to_dict(self):
        """
        Returns `self` as dictionary containing the
        photoreceptor's sensitivity and the filter function.
        """
        return {
            "sensitivity": self.sensitivity,
            "filterfunc": self.filterfunc,
            "capture_noise_level": self.capture_noise_level
        }

    @classmethod
    def from_dict(cls, dictionary):
        """
        Create Photoreceptor class given a dictionary containing
        the `sentivity` and `filterfunc` (optional) keys.
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
        The `dreye.Sensitivity` instance containing
        all the spectral sensitivities.
        """
        return self._sensitivity

    @property
    def pr_number(self):
        """
        The number of photoreceptor types. This correspond to the number
        of spectral sensitivities of the `sensitivity` attribute.
        """
        return self.sensitivity.shape[1]

    @property
    def capture_noise_level(self):
        """
        Noise level for capture values
        """
        return self._capture_noise_level

    @property
    def wavelengths(self):
        """
        A `dreye.Domain` instance of the wavelength values.
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
        corresponding to the wavelength, illuminant, and the sensitvity.
        All three are `numpy.ndarray` objects that have been broadcast
        properly to allow for mathematical operations.
        """
        return self._filterfunc

    def excitation(
        self,
        illuminant,
        reflectance=None,
        background=None,
        return_units=None,
        wavelengths=None,
        apply_noise_threshold=True,
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
        apply_noise_threshold : bool, optional
            Whether to apply noise thresholding or not. Defaults to True.
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
                         wavelengths=wavelengths,
                         apply_noise_threshold=apply_noise_threshold),
            **kwargs
        )

    def capture(
        self,
        illuminant,
        reflectance=None,
        background=None,
        return_units=None,
        wavelengths=None,
        apply_noise_threshold=True
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
        apply_noise_threshold : bool, optional
            Whether to apply noise thresholding or not. Defaults to True.

        Returns
        -------
        captures : `numpy.ndarray` or `pint.Quantity`
            Array of row length equal to the number of illuminant and
            column length equal to the number of photoreceptor types.
        """
        if return_units is None:
            return_units = has_units(illuminant)

        if illuminant.ndim > 2:
            raise ValueError("Illuminant must be 1- or 2-dimensional.")

        if (
            hasattr(illuminant, 'equalize_domains')
            and hasattr(illuminant, 'domain_axis')
        ):
            if illuminant.ndim == 1:
                illuminant = Signals(illuminant)
                # set domain axis
                illuminant.domain_axis = 0

            if illuminant.domain_axis != 0:
                illuminant = illuminant.copy()
                illuminant.domain_axis = 0

            if reflectance is not None:
                illuminant = illuminant * reflectance

            sensitivity, illuminant = self.sensitivity.equalize_domains(
                illuminant
            )
        else:
            if reflectance is not None:
                illuminant = illuminant * reflectance

            if illuminant.ndim == 1:
                illuminant = illuminant[:, None]

            if not has_units(illuminant):
                illuminant = illuminant * ureg(None)

            sensitivity = self.sensitivity

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

        if np.any(q < 0):
            raise ValueError("Capture values calculated are below 0; "
                             "Make sure illuminant, reflectance, background "
                             "and sensitivities do not contain negative "
                             "values.")

        if return_units:
            q = q * new_units

        if background is None:
            return q

        # calculate relative capture
        q_bg = self.capture(
            background, return_units=return_units, wavelengths=wavelengths
        )
        # q_bg may have different units to q
        q = q / q_bg
        if apply_noise_threshold:
            return self.limit_q_by_noise_level(q)
        else:
            return q

    def limit_q_by_noise_level(self, q):
        """
        Return relative captures `q` after accounting for capture noise
        levels.
        """

        if (
            self.capture_noise_level is None
            or np.isnan(self.capture_noise_level)
        ):
            return q

        q = np.round(q / self.capture_noise_level, 0) * self.capture_noise_level
        q[q < self.capture_noise_level] = self.capture_noise_level
        return q

    # TODO def decomp_sensitivity(self):
    #     self.sensitivity.magnitude


@inherit_docstrings
class LinearPhotoreceptor(Photoreceptor):
    """
    A linear photoreceptor model

    Parameters
    ----------
    sensitivity : Sensitivity instance or array-like, optional
        An array that contains the sensitivity of different photoreceptor
        types across wavelengths (wavelengths x types).
    wavelengths : array-like, optional
        The wavelength values in nanometers. This must be the same size as
        the number of rows in the sensitivity array.
    filterfunc : callable, optional
        A function that accepts three positional arguments:
        wavelengths, illuminant, and sensitivity. All three arguments are
        `numpy.ndarray` objects that are broadcastable to each other.
        The function should return the illuminant after the wavelength-specific
        filter has been applied.
    labels : array-like, optional
        The labels for each photoreceptor. The length of labels must
        correspond to the length of the columns in `sensitivity`.
    capture_noise_level : None or float, optional
        The relative capture noise level. This is used when calculating
        relative capture values.
    kwargs : dict, optional
        A dictionary that is directly passed to the instantiation of
        the `dreye.Sensitivity` class.

    See Also
    --------
    dreye.get_photoreceptor_model
    dreye.LogPhotoreceptor
    dreye.HyperbolicPhotoreceptor

    Notes
    -----
    In the linear model, photon capture values correspond to photoreceptor
    excitations.

    It is usually not necessary to supply a `filterfunc` argument, unless the
    photoreceptor model contains a filter that varies with the intensity and
    wavelength of the illuminant (for example, the photoreceptor model
    by Stavenga et al, 2003).
    """

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


@inherit_docstrings
class LogPhotoreceptor(Photoreceptor):
    """
    A logarithmic photoreceptor model

    Parameters
    ----------
    sensitivity : Sensitivity instance or array-like, optional
        An array that contains the sensitivity of different photoreceptor
        types across wavelengths (wavelengths x types).
    wavelengths : array-like, optional
        The wavelength values in nanometers. This must be the same size as
        the number of rows in the sensitivity array.
    filterfunc : callable, optional
        A function that accepts three positional arguments:
        wavelengths, illuminant, and sensitivity. All three arguments are
        `numpy.ndarray` objects that are broadcastable to each other.
        The function should return the illuminant after the wavelength-specific
        filter has been applied.
    labels : array-like, optional
        The labels for each photoreceptor. The length of labels must
        correspond to the length of the columns in `sensitivity`.
    capture_noise_level : None or float, optional
        The relative capture noise level. This is used when calculating
        relative capture values.
    kwargs : dict, optional
        A dictionary that is directly passed to the instantiation of
        the `dreye.Sensitivity` class.

    See Also
    --------
    dreye.get_photoreceptor_model
    dreye.LinearPhotoreceptor
    dreye.HyperbolicPhotoreceptor

    Notes
    -----
    In the logarithmic photoreceptor model the photoreceptor excitations
    correspond to the log of the photon captures.

    It is usually not necessary to supply a `filterfunc` argument, unless the
    photoreceptor model contains a filter that varies with the intensity and
    wavelength of the illuminant (for example, the photoreceptor model
    by Stavenga et al, 2003).
    """

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


@inherit_docstrings
class HyperbolicPhotoreceptor(Photoreceptor):
    """
    A hyperbolic photoreceptor model.

    Parameters
    ----------
    sensitivity : Sensitivity instance or array-like, optional
        An array that contains the sensitivity of different photoreceptor
        types across wavelengths (wavelengths x types).
    wavelengths : array-like, optional
        The wavelength values in nanometers. This must be the same size as
        the number of rows in the sensitivity array.
    filterfunc : callable, optional
        A function that accepts three positional arguments:
        wavelengths, illuminant, and sensitivity. All three arguments are
        `numpy.ndarray` objects that are broadcastable to each other.
        The function should return the illuminant after the wavelength-specific
        filter has been applied.
    labels : array-like, optional
        The labels for each photoreceptor. The length of labels must
        correspond to the length of the columns in `sensitivity`.
    capture_noise_level : None or float, optional
        The relative capture noise level. This is used when calculating
        relative capture values.
    kwargs : dict, optional
        A dictionary that is directly passed to the instantiation of
        the `dreye.Sensitivity` class.

    See Also
    --------
    dreye.get_photoreceptor_model
    dreye.LinearPhotoreceptor
    dreye.LogPhotoreceptor

    Notes
    -----
    In the hyperbolic photoreceptor model, the photoreceptor excitations
    correspond to the hyperbolic transform of the photon captures:
    :math:`(q-1)/q`.

    It is usually not necessary to supply a `filterfunc` argument, unless the
    photoreceptor model contains a filter that varies with the intensity and
    wavelength of the illuminant (for example, the photoreceptor model
    by Stavenga et al, 2003).
    """

    @staticmethod
    def excitefunc(arr):
        """
        Returns the  `(arr - 1)/arr`.
        """
        return (arr - 1) / arr

    @staticmethod
    def inv_excitefunc(arr):
        """
        Returns the `1/(1-arr)`.
        """
        return 1 / (1-arr)


@inherit_docstrings
class LinearContrastPhotoreceptor(Photoreceptor):
    """
    A linear contrast photoreceptor model.

    Parameters
    ----------
    sensitivity : Sensitivity instance or array-like, optional
        An array that contains the sensitivity of different photoreceptor
        types across wavelengths (wavelengths x types).
    wavelengths : array-like, optional
        The wavelength values in nanometers. This must be the same size as
        the number of rows in the sensitivity array.
    filterfunc : callable, optional
        A function that accepts three positional arguments:
        wavelengths, illuminant, and sensitivity. All three arguments are
        `numpy.ndarray` objects that are broadcastable to each other.
        The function should return the illuminant after the wavelength-specific
        filter has been applied.
    labels : array-like, optional
        The labels for each photoreceptor. The length of labels must
        correspond to the length of the columns in `sensitivity`.
    capture_noise_level : None or float, optional
        The relative capture noise level. This is used when calculating
        relative capture values.
    kwargs : dict, optional
        A dictionary that is directly passed to the instantiation of
        the `dreye.Sensitivity` class.

    See Also
    --------
    dreye.get_photoreceptor_model
    dreye.LinearPhotoreceptor
    dreye.LogPhotoreceptor
    dreye.HyperbolicPhotoreceptor

    Notes
    -----
    In the linear photoreceptor model, the photoreceptor excitations
    correspond to the fractional difference of the photon captures:
    :math:`q-1`
    with :math:`q` corresponding to the relative capture.

    It is usually not necessary to supply a `filterfunc` argument, unless the
    photoreceptor model contains a filter that varies with the intensity and
    wavelength of the illuminant (for example, the photoreceptor model
    by Stavenga et al, 2003).
    """

    @staticmethod
    def excitefunc(arr):
        """
        Returns the  `arr`/(1+`arr`).
        """
        return arr - 1

    @staticmethod
    def inv_excitefunc(arr):
        """
        Returns the `arr`/(1 - `arr`).
        """
        return arr + 1


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
