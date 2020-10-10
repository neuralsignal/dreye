"""Class to define Spectral Sensitivity function

TODO Govardoskii and Stavenga fitting of spectrum
"""

import numpy as np

from dreye.core.spectrum import Spectra
from dreye.constants import RELATIVE_ACCURACY
from dreye.err import DreyeError
from dreye.utilities.abstract import inherit_docstrings
from dreye.utilities import is_numeric, optional_to
from dreye.core.opsin_template import govardovskii2000_template


# TODO if numeric type then use Govardoskii fit method


@inherit_docstrings
class Sensitivity(Spectra):
    """
    Defines a set of spectral sensitivities.

    Besides the normal `Spectra` class, it ensures that `domain_min`
    and `domain_max` are set automatically to include the wavelength domain
    that is significantly above zero, if these arguments are not set
    explicitly.

    Parameters
    ----------
    values : array-like, str, signal-type
        Two-dimensional array that contains the value of your signal.
        If numeric, values is assumed to be the wavelength of max
        absorbance. To obtain the absorbance spectrum a template will be
        used.
    domain : `dreye.Domain` or array-like, optional
        The wavelength domain of the signal.
        This needs to be the same length as `values`.
    labels : array-like, optional
        A set of hashable objects that describe each individual signal.
        If None, ascending integer values are used as labels.
    units : str or `pint.Unit`, optional
        Units of the `values` array.
    domain_units : str or `pint.Unit`, optional
        Units of the `domain` array. Assumed to be nanometers.
    domain_axis : int, optional
        The axis that corresponds to the `domain` argument. Defaults to 0.
    domain_min : numeric, optional
        Defines the minimum value in your domain for the intpolation range.
    domain_max : numeric, optional
        Defines the minimum value in your domain for the intpolation range.
    signal_min : numeric or array-like, optional
        Will clip your signal to a minimum. Everything below this minimum will
        be set to the minumum.
    signal_max : numeric or array-like, optional
        Will clip your signal to a maximum. Everything above this maximum will
        be set to the maximum.
    attrs : dict, optoinal
        User-defined dictionary of objects that are associated with the
        signal, but that are not used for any particular computations.
    name : str, optional
        Name of the signal instance.
    interpolator : interpolate class, optional
        Callable function that allows you to interpolate between points. The
        callable should accept two positional arguments as `numpy.ndarray`
        objects and accept the keyword argument `axis`.
        Defaults to `scipy.interpolate.interp1d`.
    interpolator_kwargs : dict-like, optional
        Dictionary to specify other keyword arguments that are passed to
        the `interpolator`.
    smoothing_method : str, optional
        Smoothing method used when using the `smooth` method.
        Defaults to `savgol`.
    smoothing_window : numeric, optional
        Standard window size in units of the domain to smooth the signal.
    smoothing_args : dict, optional
        Keyword arguments passed to the `filter` method when smoothing.
    contexts : str or tuple, optoinal
        Contexts for unit conversion. See `pint` package.

    See Also
    --------
    Spectra
    """

    def __init__(
        self, values, domain=None, labels=None,
        template=govardovskii2000_template, **kwargs
    ):
        if is_numeric(values) and domain is not None:
            wavelengths = optional_to(domain, 'nm')
            values = govardovskii2000_template(wavelengths, values)

        super().__init__(values=values, domain=domain, labels=labels, **kwargs)
        self._set_sig_domain_bounds()

    def _set_sig_domain_bounds(self):

        maxima = np.nanmax(
            self.magnitude, axis=self.domain_axis, keepdims=True
        )
        # significant intensities
        sig = self.magnitude > (maxima * 10 ** -RELATIVE_ACCURACY)
        sig = np.any(sig, axis=self.labels_axis)

        # conditions for domain min
        # first value is significant
        if np.isnan(self.domain_min.magnitude):
            if sig[0]:
                self.domain_min = self.domain.start
            # no value is significant
            elif not np.any(sig):
                raise DreyeError(
                    "Couldn't find significant bound for spectral sensitivity"
                )
            # some other value is significant
            else:
                # get first significant value
                self.domain_min = self.domain.magnitude[
                    np.flatnonzero(sig)[0]
                ]

        # conditions for domain min
        # last value is significant
        if np.isnan(self.domain_max.magnitude):
            if sig[-1]:
                self.domain_max = self.domain.end
            # no value is significant
            elif not np.any(sig):
                raise DreyeError(
                    "Couldn't find significant bound for spectral sensitivity"
                )
            # some other value is significant
            else:
                # get last significant value
                self.domain_max = self.domain.magnitude[
                    np.flatnonzero(sig)[-1]
                ]
