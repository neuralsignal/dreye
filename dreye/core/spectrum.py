"""
Spectrum
========

Inherits signal class to build Spectrum class.
"""

from dreye.core.signal import (
    DomainSignal, Signals, Signal
)
from dreye.err import DreyeUnitError
from dreye.utilities import has_units, get_units
from dreye.utilities.abstract import inherit_docstrings



class _SpectrumMixin:

    _xlabel = r'$\lambda$ (nm)'

    def __init__(
        self,
        values,
        domain=None,
        *args,
        wavelengths=None,
        **kwargs
    ):

        kwargs['domain_units'] = 'nm'
        if domain is None and wavelengths is not None:
            domain = wavelengths
        # enforces nm units for domain and provides default flux context
        super().__init__(
            values, domain, *args,
            **kwargs
        )

    @property
    def wavelengths(self):
        """
        Alias for domain.
        """
        return self.domain


class _IntensityMixin:

    def __init__(
        self,
        values,
        domain=None,
        *args,
        wavelengths=None,
        units=None,
        **kwargs
    ):

        # default units
        if units is None and not has_units(values):
            units = 'spectralirradiance'

        super().__init__(
            values, domain,
            *args, units=units,
            wavelengths=wavelengths,
            **kwargs
        )

        # check units are correct dimensionality
        # i.e. irradiance or photon flux type
        truth = self._is_intensity_units(self.units)

        if not truth:
            raise DreyeUnitError(self.units, 'irradiance convertible units')

    @staticmethod
    def _is_intensity_units(units):
        return (
            (
                units.dimensionality
                == '[mass] / [length] / [time] ** 3'
            ) | (
                units.dimensionality
                == '[substance] / [length] ** 3 / [time]'
            )
        )


@inherit_docstrings
class Spectra(_SpectrumMixin, Signals):
    """
    Defines an arbitrary set of spectral distributions.

    Parameters
    ----------
    values : array-like, str, signal-type
        Two-dimensional array that contains the value of your signal.
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
    Signals
    IntensitySpectra
    """

    @property
    def _class_new_instance(self):
        return Spectra


@inherit_docstrings
class Spectrum(_SpectrumMixin, Signal):
    """
    Defines an arbitrary spectral distribution.

    Parameters
    ----------
    values : array-like, str, signal-type
        One-dimensional array that contains the values of the
        calibration measurement across wavelengths.
    domain/wavelengths : `dreye.Domain` or array-like, optional
        The wavelength domain of the signal.
        This needs to be the same length as `values`.
    units : str or `pint.Unit`, optional
        Units of the `values` array.
    domain_units : str or `pint.Unit`, optional
        Units of the `domain` array. Assumed to be nanometers.
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
    Signal
    IntensitySpectrum
    CalibrationSpectrum
    """

    @property
    def _class_new_instance(self):
        return Spectrum


@inherit_docstrings
class IntensitySpectra(_IntensityMixin, Spectra):
    """
    Defines an arbitrary set of spectral distributions with intensity units.

    Parameters
    ----------
    values : array-like, str, signal-type
        Two-dimensional array that contains the value of your signal.
    domain : `dreye.Domain` or array-like, optional
        The wavelength domain of the signal.
        This needs to be the same length as `values`.
    labels : array-like, optional
        A set of hashable objects that describe each individual signal.
        If None, ascending integer values are used as labels.
    units : str or `pint.Unit`, optional
        Units of the `values` array. Units must be convertible to
        photonflux or irradiance.
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
    Signals
    Spectra
    """

    def _class_new_instance(self, values, *args, units=None, **kwargs):
        if has_units(values):
            truth = self._is_intensity_units(values.units)
        else:
            truth = self._is_intensity_units(get_units(units))
        if truth:
            return type(self)(values, *args, units=units, **kwargs)
        else:
            return Spectra(values, *args, units=units, **kwargs)


@inherit_docstrings
class IntensitySpectrum(_IntensityMixin, Spectrum):
    """
    Defines an arbitrary spectral distribution with intensity units.

    Parameters
    ----------
    values : array-like, str, signal-type
        One-dimensional array that contains the values of the
        calibration measurement across wavelengths.
    domain/wavelengths : `dreye.Domain` or array-like, optional
        The wavelength domain of the signal.
        This needs to be the same length as `values`.
    units : str or `pint.Unit`, optional
        Units of the `values` array. Units must be convertible to
        photonflux or irradiance.
    domain_units : str or `pint.Unit`, optional
        Units of the `domain` array. Units are assumed to be nanometers.
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
    Signal
    Spectrum
    CalibrationSpectrum
    """

    def _class_new_instance(self, values, *args, units=None, **kwargs):
        if has_units(values):
            truth = self._is_intensity_units(values.units)
        else:
            truth = self._is_intensity_units(get_units(units))
        if truth:
            return type(self)(values, *args, units=units, **kwargs)
        else:
            return Spectrum(values, *args, units=units, **kwargs)


@inherit_docstrings
class DomainSpectrum(_SpectrumMixin, DomainSignal):
    """
    Two-dimensional signal with wavelength domain.

    Parameters
    ----------
    values : array-like, str, signal-type
        Two-dimensional array that contains the value of your signal.
    domain/wavelengths : `dreye.Domain` or array-like, optional
        The wavelength domain of the signal.
        This needs to be the same length as `values`.
    labels : `dreye.Domain` or array-like, optional
        The domain of the signal along the other axis.
        This needs to be the same length of
        the `values` array along the axis of the labels.
    units : str or `pint.Unit`, optional
        Units of the `values` array.
    domain_units : str or `pint.Unit`, optional
        Units of the `domain` array. Units are assumed to be nanometers.
    labels_units : str or `pint.Unit`, optional
        Units of the `labels` array.
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
    DomainSignal
    IntensityDomainSpectrum
    MeasuredSpectrum
    """

    @property
    def _class_new_instance(self):
        return DomainSpectrum

    @property
    def _switch_new_instance(self):
        return DomainSignal


@inherit_docstrings
class IntensityDomainSpectrum(_IntensityMixin, DomainSpectrum):
    """
    Two-dimensional intensity signal with wavelength domain.

    Parameters
    ----------
    values : array-like, str, signal-type
        Two-dimensional array that contains the value of your signal.
    domain/wavelengths : `dreye.Domain` or array-like, optional
        The wavelength domain of the signal.
        This needs to be the same length as `values`.
    labels : `dreye.Domain` or array-like, optional
        The domain of the signal along the other axis.
        This needs to be the same length of
        the `values` array along the axis of the labels.
    units : str or `pint.Unit`, optional
        Units of the `values` array. Units must be convertible to
        photonflux or irradiance.
    domain_units : str or `pint.Unit`, optional
        Units of the `domain` array. Units are assumed to be nanometers.
    labels_units : str or `pint.Unit`, optional
        Units of the `labels` array.
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
    DomainSignal
    DomainSpectrum
    MeasuredSpectrum
    """

    def _class_new_instance(self, values, *args, units=None, **kwargs):
        if has_units(values):
            truth = self._is_intensity_units(values.units)
        else:
            truth = self._is_intensity_units(get_units(units))
        if truth:
            return type(self)(values, *args, units=units, **kwargs)
        else:
            return DomainSpectrum(values, *args, units=units, **kwargs)
