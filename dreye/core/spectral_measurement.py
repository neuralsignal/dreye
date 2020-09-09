"""Class to define spectral measurement
"""

import numpy as np
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression

from dreye.utilities import (
    has_units, is_numeric,
    optional_to, is_listlike, has_units,
    get_units, get_value, digits_to_decimals
)
from dreye.utilities.abstract import inherit_docstrings
from dreye.constants import ureg
from dreye.core.signal import Signals, Signal
from dreye.core.spectrum import (
    Spectra, IntensityDomainSpectrum, Spectrum
)
from dreye.core.signal_container import DomainSignalContainer
from dreye.err import DreyeError


@inherit_docstrings
class CalibrationSpectrum(Spectrum):
    """
    Defines a calibration measurement.

    Parameters
    ----------
    values : array-like, str, signal-type
        One-dimensional array that contains the values of the
        calibration measurement across wavelengths.
    domain/wavelengths : `dreye.Domain` or array-like, optional
        The wavelength domain of the signal.
        This needs to be the same length as `values`.
    units : str or `pint.Unit`, optional
        Units of the `values` array. Defaults to microjoule.
    area : numeric or `pint.Quantity`, optional
        The area of the spectrophotometer used for collecting photons. If
        the units cannot be obtained, then it is assumed to be in units of
        :math:`cm^2`.
    domain_units : str or `pint.Unit`, optional
        Units of the `domain` array.
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
    IntensitySpectrum
    """

    def __init__(
        self,
        values,
        domain=None,
        *,
        area=None,
        units=None,
        **kwargs
    ):

        if area is None and not isinstance(values, CalibrationSpectrum):
            # assume area of 1 cm ** 2
            area = 1
        if units is None and not has_units(values):
            units = ureg('microjoule').units

        super().__init__(
            values=values, domain=domain, units=units,
            **kwargs
        )

        if area is None:
            area = self.attrs.get('area_', None)
        if not is_numeric(area):
            raise DreyeError("'area' argument must be a numeric value, "
                             f"but is of type '{type(area)}'.")

        if has_units(area):
            self.attrs['area_'] = area
        else:
            self.attrs['area_'] = area * ureg('cm ** 2')

    @property
    def area(self):
        return self.attrs['area_']


# TODO allow curve fit instead of isotonic regression? - SKlearn type class
@inherit_docstrings
class MeasuredSpectrum(IntensityDomainSpectrum):
    """
    Two-dimensional intensity signal of LED measurements
    with wavelength domain and output labels.

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
        the `values` array along the axis of the labels. The labels
        domain is assumed to be the output of the LED system.
        This can be volts or seconds in the case of pulse-width
        modulation.
    zero_intensity_bound : numeric, optional
        The output in labels units that correspond to zero intensity
        of the LED.
    max_intensity_bound : numeric, optional
        The output in labels units that correspond to maximum
        intensity that can be achieved.
    resolution : array-like, optional
        Array of individual steps that can be resolved by the hardware.
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
    IntensitySpectrum
    """

    # always in uE?

    def __init__(
        self, values, domain=None, labels=None, *,
        zero_intensity_bound=None,
        max_intensity_bound=None,
        resolution=None,
        **kwargs
    ):

        super().__init__(values=values, domain=domain, labels=labels, **kwargs)

        # getting correct values and converting units
        if zero_intensity_bound is None:
            zero_intensity_bound = self.attrs.get(
                'zero_intensity_bound_', None)
        if max_intensity_bound is None:
            max_intensity_bound = self.attrs.get(
                'max_intensity_bound_', None)
        if resolution is None:
            resolution = self.attrs.get('resolution_', None)

        self.attrs['zero_intensity_bound_'] = self._get_domain_bound(
            zero_intensity_bound, self.labels
        )
        self.attrs['max_intensity_bound_'] = self._get_domain_bound(
            max_intensity_bound, self.labels
        )
        # should be the minimum step that can be taken - or an array?
        if is_listlike(resolution):
            self.attrs['resolution_'] = optional_to(
                resolution, self.labels.units, *self.contexts
            ) * self.labels.units
        elif resolution is None:
            self.attrs['resolution_'] = None
        else:
            raise DreyeError("resolution must be list-like or None")

        self._intensity = None
        self._normalized_spectrum = None
        self._mapper = None
        self._regressor = None

        if self.name is None:
            idx = np.argmax(self.normalized_spectrum.magnitude)
            peak = self.domain.magnitude[idx]
            self.name = f"peak_at_{int(peak)}"

    @property
    def zero_is_lower(self):
        """
        Ascending or descending intensity values.

        This is inferred automatically.
        """
        if (
            np.isnan(self.zero_intensity_bound.magnitude)
            or np.isnan(self.max_intensity_bound.magnitude)
        ):
            # integral across wavelengths
            intensity = self.intensity.magnitude
            argsort = np.argsort(self.output.magnitude)
            return intensity[argsort][0] < intensity[argsort][-1]
        else:
            return (
                self.zero_intensity_bound.magnitude
                < self.max_intensity_bound.magnitude
            )

    @property
    def resolution(self):
        """
        Smallest possible label/output value differences.

        Includes units
        """
        if self.attrs['resolution_'] is None:
            return
        return self.attrs['resolution_'].to(self.labels.units)

    @property
    def zero_intensity_bound(self):
        """
        Label value corresponding to zero intensity across wavelengths.

        Includes units.
        """
        return self.attrs['zero_intensity_bound_'].to(self.labels.units)

    @property
    def max_intensity_bound(self):
        """
        Label/output value corresponding to max intensity across wavelengths.

        Includes units.
        """
        return self.attrs['max_intensity_bound_'].to(self.labels.units)

    @property
    def output_bounds(self):
        """
        Bounds of output, e.g. 0 and 5 volts.

        Does not include units.
        """

        output_bounds = list(self.output.boundaries)

        idx_zero = 1 - int(self.zero_is_lower)
        if not np.isnan(self.zero_intensity_bound.magnitude):
            output_bounds[idx_zero] = self.zero_intensity_bound.magnitude
        if not np.isnan(self.max_intensity_bound.magnitude):
            output_bounds[idx_zero - 1] = self.max_intensity_bound.magnitude

        return tuple(output_bounds)

    @property
    def output(self):
        """
        Alias for labels unless resolution is not None.

        If resolution is not None, label values will be mapped to closest
        resolution value.
        """
        if self.resolution is None:
            return self.labels
        return self.labels._class_new_instance(
            self._resolution_mapping(self.labels.magnitude),
            units=self.labels.units,
            **self.labels._init_kwargs
        )

    @property
    def intensity_bounds(self):
        """
        Bounds of intensity signal after wavelength integration.

        Does not include units.
        """

        integral = self.intensity.magnitude

        if not np.isnan(self.zero_intensity_bound.magnitude):
            lower = 0.0
            upper = np.max(integral)
        else:
            lower = np.max([np.min(integral), 0.0])
            upper = np.max(integral)

        return (lower, upper)

    @property
    def normalized_spectrum(self):
        """
        Spectrum normalized to integrate to one.

        Will take mean across intensities to remove noise.
        Returns a spectra instance.
        """

        if self._normalized_spectrum is None:
            values = self.mean(axis=self.labels_axis)
            units, values = get_units(values), get_value(values)
            # create spectrum
            spectrum = Spectrum(
                values=values,
                domain=self.domain,
                name=self.name,
                units=units,
                attrs=self.attrs,
                signal_min=0  # enforce zero as signal minimum
            )
            self._normalized_spectrum = spectrum.normalized_signal

        return self._normalized_spectrum

    @property
    def intensity(self):
        """
        The intensity of each measurement.

        This takes the integral across the wavelength domain and
        returns a `Signal` instance with the output domain.
        """

        if self._intensity is None:
            # calculate integral and make sure it's above zero
            integral = self.integral
            mag, units = integral.magnitude, integral.units
            mag[mag < 0] = 0.0
            self._intensity = Signal(
                mag,
                units=units,
                domain=self.output,
                name=self.name,
                attrs=self.attrs,
                contexts=self.contexts
            )
        return self._intensity

    def to_measured_spectra(self, units='uE'):
        """
        Convert to MeasuredSpectraContainer
        """
        return MeasuredSpectraContainer([self], units=units)

    def _resolution_mapping(self, values):
        """
        Map output values to given resolution.

        Parameters
        ----------
        values : np.ndarray or float
            Array that should already be in units of `labels.units`.
        """

        if self.resolution is None:
            return values

        numeric_type = is_numeric(values)
        new_values = np.atleast_1d(values)

        res_idx = np.argmin(
            np.abs(
                new_values[:, None]
                - self.resolution.magnitude[None, :]
            ),
            axis=1
        )

        new_values = self.resolution.magnitude[res_idx]
        if numeric_type:
            return np.squeeze(new_values)
        return new_values

    def map(self, values, return_units=True):
        """
        Map Intensity values to output values.

        Parameters
        ----------
        values : array-like
            samples in intensity-convertible units or no units.
        return_units : bool
            Whether to return mapped values with units.

        Returns
        -------
        output : `numpy.ndarray` or `pint.Quantity`
            Mapped output values.
        """

        values = optional_to(values, self.intensity.units, *self.contexts)
        assert values.ndim < 2, 'values must be 1 dimensional'

        # check intensity bound of values
        imin, imax = self.intensity_bounds
        truth = np.all(values >= imin) and np.all(values <= imax)
        assert truth, 'Some values to be mapped are out of bounds.'

        mapped_values = self._mapper_func(values)
        mapped_values = self._resolution_mapping(mapped_values)

        if return_units:
            return mapped_values * self.labels.units

        return mapped_values

    def inverse_map(self, values, return_units=True):
        """
        Go from output values to intensity values

        Parameters
        ----------
        values : array-like
            samples in output-convertible units or no units.
        return_units : bool
            Whether to return mapped values with units.

        Returns
        -------
        output : `numpy.ndarray` or `pint.Quantity`
            Mapped intensity values.
        """
        # this is going to be two dimensional, since it is a Signals instance
        values = optional_to(values, self.labels.units, *self.contexts)
        values = self._resolution_mapping(values)
        intensity = self.regressor.transform(values)

        if return_units:
            return intensity * self.intensity.units
        else:
            return intensity

    def get_residuals(self, values, return_units=True):
        """
        Get residuals between values and mapped values

        Parameters
        ----------
        values : array-like
            samples in intensity-convertible units or no units.
        return_units : bool
            Whether to return mapped values with units.

        Returns
        -------
        output : `numpy.ndarray` or `pint.Quantity`
            Mapped residual intensity values.
        """

        values = optional_to(values, units=self.intensity.units)
        mapped_values = self.map(values, return_units=False)

        # interpolate to new values given resolution
        res = self.inverse_map(mapped_values, return_units=False) - values

        if return_units:
            return res * self.intensity.units
        else:
            return res

    def score(self, values, **kwargs):
        """
        R^2 score for particular mapping.

        Parameters
        ----------
        values : array-like
            samples in intensity-convertible units or no units.
        return_units : bool
            Whether to return mapped values with units.

        Returns
        -------
        r2 : float
            R^2-score.
        """
        values = optional_to(values, self.intensity.units, *self.contexts)
        mapped_values = self.map(values, return_units=False)
        fit_values = self.inverse_map(mapped_values, return_units=False)
        res = (values - fit_values) ** 2
        tot = (values - values.mean()) ** 2
        return 1 - res.sum()/tot.sum()
        # return self.regressor.score(mapped_values, values, **kwargs)

    def _mapper_func(self, *args, **kwargs):
        """mapping using isotonic regression
        """

        if self._mapper is None:
            self._assign_mapper()

        return np.clip(
            self._mapper(*args, **kwargs),
            a_min=self.output_bounds[0],
            a_max=self.output_bounds[1]
        )

    @property
    def regressor(self):
        """
        Scikit-learn regressor instance used to fit output values
        to intensity values

        See Also
        --------
        sklearn.isotonic.IsotonicRegression
        """
        if self._regressor is None:
            self._assign_mapper()
        return self._regressor

    def _assign_mapper(self):
        # 1D signal
        y = self.intensity.magnitude  # integral across intensities
        x = self.output.magnitude
        # sort x, y (not necessary)
        argsort = np.argsort(x)
        x = x[argsort]
        y = y[argsort]
        # y_min and y_max
        y_min, y_max = self.intensity_bounds
        zero_is_lower = self.zero_is_lower
        zero_intensity_bound = self.zero_intensity_bound.magnitude

        # a little redundant but should ensure safety of method
        if zero_is_lower and zero_intensity_bound < np.min(x):
            x = np.concatenate([[zero_intensity_bound], x])
            y = np.concatenate([[0], y])
        # a little redundant but should ensure safety of method
        elif not zero_is_lower and zero_intensity_bound > np.max(x):
            x = np.concatenate([x, [zero_intensity_bound]])
            y = np.concatenate([y, [0]])

        # perform isotonic regression
        isoreg = IsotonicRegression(
            # lower and upper intensity values
            y_min=y_min,
            y_max=y_max,
            increasing=zero_is_lower
        )
        self._regressor = isoreg

        new_y = isoreg.fit_transform(x, y)

        # should throw bounds_error, since zero intensity bound
        # has been added
        # self._mapper = interp1d(new_y, x)
        # interpolation function
        if zero_is_lower:
            self._mapper = interp1d(
                new_y, x,
                bounds_error=False,
                fill_value=self.output_bounds
            )
        else:
            self._mapper = interp1d(
                new_y, x,
                bounds_error=False,
                fill_value=self.output_bounds[::-1]
            )


@inherit_docstrings
class MeasuredSpectraContainer(DomainSignalContainer):
    """
    A container that can hold multiple `dreye.MeasuredSpectrum` instances.

    The `map` methods are also accessible in the container.

    Parameters
    ----------
    container : list-like
        A list of `dreye.DomainSignal` instances.
    units : str or `ureg.Unit`, optional
        The units to convert the values to. If None,
        it will choose the units of the first signal in the list.
    domain_units : str or `ureg.Unit`, optional
        The units to convert the domain to. If None,
        it will choose the units of domain of the first signal in the list.
    labels_units : str or `ureg.Unit`, optional
        The units to convert the labels to. If None,
        it will choose the units of labels of the first signal in the list.

    See Also
    --------
    DomainSignalContainer
    """

    _xlabel = r'$\lambda$ (nm)'
    _cmap = 'viridis'
    _init_keys = [
        '_intensities',
        '_normalized_spectra',
        '_mapper'
    ]
    _allowed_instances = MeasuredSpectrum

    def map(self, values, return_units=True):
        """
        Map Intensity values to output values.

        Parameters
        ----------
        values : array-like
            2D samples in intensity-convertible units or no units.
        return_units : bool
            Whether to return mapped values with units.

        Returns
        -------
        output : `numpy.ndarray` or `pint.Quantity`
            Mapped output values.
        """

        values = optional_to(values, units=self.intensities.units)
        assert values.ndim < 3, 'values must be 1 or 2 dimensional'
        x = np.atleast_2d(values)

        y = np.empty(x.shape)
        for idx, measured_spectrum in enumerate(self):
            y[:, idx] = measured_spectrum.map(x[:, idx], return_units=False)

        if values.ndim == 1:
            y = y[0]

        if return_units:
            return y * self.labels_units

        return y

    def inverse_map(self, values, return_units=True):
        """
        Go from output values to intensity values

        Parameters
        ----------
        values : array-like
            2D samples in output-convertible units or no units.
        return_units : bool
            Whether to return mapped values with units.

        Returns
        -------
        output : `numpy.ndarray` or `pint.Quantity`
            Mapped intensity values.
        """
        values = optional_to(values, units=self.labels_units)
        assert values.ndim < 3, 'values must be 1 or 2 dimensional'
        x = np.atleast_2d(values)

        y = np.empty(x.shape)
        for idx, measured_spectrum in enumerate(self):
            y[:, idx] = measured_spectrum.inverse_map(
                x[:, idx], return_units=False
            )

        if values.ndim == 1:
            y = y[0]

        if return_units:
            return y * self.intensities.units

        return y

    def get_residuals(self, values, return_units=True):
        """
        Get residuals between values and mapped values

        Parameters
        ----------
        values : array-like
            2D samples in intensity-convertible units or no units.
        return_units : bool
            Whether to return mapped values with units.

        Returns
        -------
        output : `numpy.ndarray` or `pint.Quantity`
            Mapped residual intensity values.
        """

        values = optional_to(values, units=self.intensities.units)
        assert values.ndim < 3, 'values must be 1 or 2 dimensional'
        x = np.atleast_2d(values)

        y = np.empty(x.shape)
        for idx, measured_spectrum in enumerate(self):
            y[:, idx] = measured_spectrum.get_residuals(
                x[:, idx], return_units=False
            )

        if values.ndim == 1:
            y = y[0]

        if return_units:
            return y * self.intensities.units

        return y

    def score(self, values, average=True, **kwargs):
        """
        R^2 score for particular mapping.

        Parameters
        ----------
        values : array-like
            2D samples in intensity-convertible units or no units.
        return_units : bool
            Whether to return mapped values with units.

        Returns
        -------
        r2 : float
            R^2-score.
        """
        values = optional_to(values, units=self.intensities.units)
        assert values.ndim < 3, 'values must be 1 or 2 dimensional'
        x = np.atleast_2d(values)

        scores = np.array([
            measured_spectrum.score(x[:, idx], **kwargs)
            for idx, measured_spectrum in enumerate(self)
        ])

        if average:
            return np.mean(scores)

        return scores

    @property
    def resolution(self):
        """
        Resolution of devices
        """
        resolutions = self.__getattr__('resolution')
        nones = ([r is None for r in resolutions])
        if all(nones):
            return
        else:
            return resolutions

    @property
    def zero_is_lower(self):
        """
        Whether zero intensity output value is lower than the max intensity
        output value.
        """
        return np.array(self.__getattr__('zero_is_lower'))

    @property
    def zero_intensity_bound(self):
        """
        Boundary of output corresponding to zero intensity spectrum

        Contains output units.
        """
        return np.array([
            ele.magnitude
            for ele in self.__getattr__('zero_intensity_bound')
        ]) * self.labels_units

    @property
    def max_intensity_bound(self):
        """
        Boundary of output corresponding to maximum intensity spectrum.

        Contains output units.
        """
        return np.array([
            ele.magnitude
            for ele in self.__getattr__('max_intensity_bound')
        ]) * self.labels_units

    @property
    def intensity_bounds(self):
        """
        Intensity bounds for the Measured Spectra in two-tuple array form.

        Unit removed.
        """
        value = [ele.intensity_bounds for ele in self]
        return tuple(np.array(value).T)

    @property
    def output_bounds(self):
        """
        Output bounds for Measured Spectra in two-tuple array form.

        Unit removed
        """
        value = [ele.output_bounds for ele in self]
        return tuple(np.array(value).T)

    @property
    def normalized_spectra(self):
        """
        Normalized spectra for each LED measurement.

        Returns a `dreye.Spectra` instance in units of `1/nm`.

        See Also
        --------
        dreye.MeasuredSpectrum.normalized_spectrum
        """
        if self._normalized_spectra is None:
            for idx, ele in enumerate(self):
                if idx == 0:
                    spectra = Spectra(ele.normalized_spectrum)
                else:
                    spectra = spectra.labels_concat(
                        Spectra(ele.normalized_spectrum)
                    )

            self._normalized_spectra = spectra
        return self._normalized_spectra

    @property
    def intensities(self):
        """
        Intensities for each LED measurement across outputs.

        Returns a `dreye.Signals` instance in units of integrated
        intensity.

        See Also
        --------
        dreye.MeasuredSpectrum.intensity
        """
        if self._intensities is None:
            for idx, ele in enumerate(self):
                if idx == 0:
                    intensities = Signals(ele.intensity)
                else:
                    intensities = intensities.labels_concat(
                        Signals(ele.intensity)
                    )

            self._intensities = intensities
        return self._intensities

    @property
    def wavelengths(self):
        return self.normalized_spectra.domain

    @property
    def _ylabel(self):
        return self[0]._ylabel
