"""Class to define photoreceptor/capture model
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import lsq_linear, least_squares

from dreye.core.spectrum import AbstractSpectrum
from dreye.core.spectral_sensitivity import AbstractSensitivity
from dreye.core.spectral_measurement import SpectrumMeasurement
from dreye.utilities import get_units, asarray


class AbstractPhotoreceptor(ABC):
    """

    Parameters
    ----------
    SpectralSensitivity instance

    Attributes
    ----------
    sensitivity

    Methods
    -------
    __init__
    capture
    fit
    excitation
    """

    def __init__(self, sensitivity):
        """
        """

        assert isinstance(sensitivity, AbstractSensitivity)

        self._sensitivity = sensitivity

    def to_dict(self):

        return {
            "sensitivity": self.sensitivity
        }

    @classmethod
    def from_dict(cls, dictionary):

        return cls(**dictionary)

    @abstractmethod
    def excitefunc(self, arr):
        """excitation function
        """

    @abstractmethod
    def inv_excitefunc(self, arr):
        """excitation function
        """

    def fit(
        self,
        spectrum_measurement,
        illuminant,
        reflectance=None,
        background=None,
        weights=None,
        units=True,
        return_res=False,
        **kwargs
    ):
        """
        Least-squares fitting of captures given available spectrum
        measurements.
        """

        # target captures opsin x illuminants
        targets, A = self.get_qs(
            spectrum_measurement,
            illuminant,
            reflectance=reflectance,
            background=background,
            units=units,
        )

        return self.fit_qs(
            targets, A,
            bounds=spectrum_measurement.bounds, units=units,
            weights=weights,
            return_res=return_res, inverse=False, **kwargs
        )

    def get_qs(
        self,
        spectrum_measurement,
        illuminant,
        reflectance=None,
        background=None,
        units=True,
        return_A=True,
    ):
        """
        """

        assert isinstance(spectrum_measurement, SpectrumMeasurement)

        # target captures illuminant x opsins
        targets = self.capture(
            illuminant,
            reflectance=reflectance,
            background=background,
            units=units
        ).T

        if return_A:
            # get A
            A = self.get_A(
                spectrum_measurement,
                background=background,
                units=units
            )

            return targets, A

        else:
            return targets

    def get_A(
        self,
        spectrum_measurement,
        background=None,
        units=True
    ):
        """
        """

        # TODO same units as background?
        normalized_spectrum = spectrum_measurement.normalized_spectrum

        # A is the normalized opsin x LED matrix
        A = self.capture(
            normalized_spectrum, background=background,
            units=units
        )

        return A

    def fit_qs(
        self, targets, A,
        bounds=None, units=True,
        weights=None,
        return_res=False, inverse=False,
        respect_zero=None,
        only_uniques=False,
        **kwargs
    ):
        """
        """

        if inverse:
            # revert back to normal capture before applyin nonlinearity later
            # necessary for initialization
            targets = self.inv_excitefunc(targets)

        if only_uniques:
            # requires 2D targets and will not keep track of units
            targets, idcs = np.unique(targets, axis=0, return_inverse=True)

        x0 = self._init_x0(
            A, targets, bounds, units, respect_zero=respect_zero
        )

        x = self._fit_targets_independently(
            self._lsq, A=A,
            targets=self.excitefunc(targets),
            x0=x0,
            bounds=bounds,
            units=units, return_res=return_res,
            weights=weights,
            respect_zero=respect_zero,
            **kwargs
        )

        if return_res:
            x, res = x

        if only_uniques:
            x = x[idcs]

        if return_res:
            return x, res
        else:
            return x

    def _lsq(self, A, target, x0, bounds, **kwargs):
        return least_squares(
            self._objective, x0=x0, args=(A, target),
            bounds=bounds, **kwargs
        )

    def _objective(self, x0, A, target):
        return self._weights * (
            self.fitted_qs(x0, A)
            - target
        )

    def fitted_qs(self, x, A):
        return self.excitefunc(np.dot(A, x))

    def _fit_targets_independently(
        self, method, A, targets,
        weights=None,
        x0=None,
        bounds=None, units=True,
        return_res=False,
        respect_zero=None,
        **kwargs
    ):
        """
        """

        # optional getting of units
        targets_units = get_units(targets)
        A_units = get_units(A)

        # for scalar target convert to numpy array
        A = asarray(A)
        targets = np.atleast_2d(asarray(targets))

        if x0 is not None:
            assert len(x0) == len(targets)
        if weights is None:
            self._weights = np.ones(targets.shape[1])
        else:
            assert len(weights) == targets.shape[1]
            self._weights = asarray(weights)

        # for targets, where all values are positive or negative
        # reset the min or max bounds respectively to respect_zero
        respect_truth = (respect_zero is not None) and (bounds is not None)
        if respect_truth:
            assert len(respect_zero) == len(bounds[0])
            assert len(respect_zero) == len(bounds[1])
            original_bounds = bounds

        x = np.zeros((targets.shape[0], A.shape[1]))
        res = []
        # iterate over each target individually
        for idx, target in enumerate(targets):

            if respect_truth:
                bounds = original_bounds.copy()
                if np.all(target >= 0):
                    bounds[0] = respect_zero
                elif np.all(target <= 0):
                    bounds[1] = respect_zero

            if x0 is None:
                res_ = method(
                    A,
                    target,
                    bounds=bounds,
                    **kwargs
                )

            else:
                res_ = method(
                    A,
                    target,
                    asarray(x0[idx]),
                    bounds=bounds,
                    **kwargs
                )

            x[idx] = res_.x
            res.append(res_)

        if units:
            x = x * targets_units/A_units

        # delete self._weights
        del(self._weights)

        if return_res:
            return x, asarray(res)
        else:
            return x

    def _init_x0(self, A, targets, bounds, units, **kwargs):
        """initialize x0 using linear regression
        """

        return self._fit_targets_independently(
            method=lsq_linear, A=A, targets=targets,
            bounds=bounds, units=units, **kwargs)

    @property
    def sensitivity(self):
        """
        """

        return self._sensitivity

    @staticmethod
    def filterfunc(arr):
        """
        """

        return arr

    def excitation(
        self,
        illuminant,
        reflectance=None,
        background=None,
        units=True,
        **kwargs
    ):
        """
        """

        # TODO separate units then apply function

        return self.excitefunc(
            self.capture(illuminant=illuminant,
                         reflectance=reflectance,
                         background=background,
                         units=units),
            **kwargs)

    def capture(
        self,
        illuminant,
        reflectance=None,
        background=None,
        units=True,
    ):
        """
        """

        assert isinstance(illuminant, AbstractSpectrum)
        assert isinstance(reflectance, AbstractSpectrum) or reflectance is None
        assert isinstance(background, AbstractSpectrum) or background is None

        # equalize domains
        if reflectance is None:
            sensitivity, illuminant = self.sensitivity.equalize_domains(
                illuminant)
        else:
            sensitivity, illuminant = self.sensitivity.equalize_domains(
                illuminant * reflectance)

        # wavelength differences
        # dlambda = sensitivity.domain.gradient
        wls = sensitivity.domain
        domain_axis = sensitivity.domain_axis

        # keep track of units
        # TODO filterfunc sensitivity units?
        new_units = illuminant.units * sensitivity.units * wls.units

        # reshape data and apply filter function
        if sensitivity.ndim == 2:
            sensitivity = np.moveaxis(
                self.filterfunc(sensitivity.magnitude),
                domain_axis, 0
            )[..., None]

            illuminant = np.moveaxis(
                illuminant.magnitude, domain_axis, 0
            )[:, None, :]

            wls = wls.magnitude[:, None, None]
        else:
            sensitivity = self.filterfunc(sensitivity.magnitude)
            illuminant = illuminant.magnitude
            wls = wls.magnitude

        # calculate capture
        # opsin x illuminant
        q = np.trapz(sensitivity * illuminant, wls, axis=0)

        if units:
            q = q * new_units

        # TODO pandas option

        if background is None:

            return q

        else:
            if background.ndim == 2:
                background = background.sum(axis=self.other_axis)

            q_bg = self.capture(background, units=units)

            return q / q_bg


class LinearPhotoreceptor(AbstractPhotoreceptor):
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


class LogPhotoreceptor(AbstractPhotoreceptor):
    """
    """

    def __init__(self, sensitivity):

        super().__init__(sensitivity)

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


class SelfScreeningPhotoreceptor(LinearPhotoreceptor):
    """
    """

    @staticmethod
    def filterfunc(arr):
        """
        """

        raise NotImplementedError('self screening photoreceptor class.')
