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
from dreye.constants import ureg
from dreye.core.signal import Signals, Signal
from dreye.core.spectrum import (
    Spectra, IntensityDomainSpectrum, Spectrum
)
from dreye.core.signal_container import DomainSignalContainer
from dreye.err import DreyeError


class CalibrationSpectrum(Spectrum):
    """
    Subclass of signal to define calibration spectrum.

    Units must be convertible to microjoule.
    """
    # TODO docstring and init

    def __init__(
        self,
        values,
        domain=None,
        labels=None,
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

        super.__init__(
            values=values, domain=domain, labels=labels, units=units,
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


class MeasuredSpectrum(IntensityDomainSpectrum):
    """
    Measured spectrum (e.g. LED)
    """

    # TODO docstring and init
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

        Calculated automatically.
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

        Does not include units
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
            **self.labels.init_kwargs
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
            # dealing with noise close to zero
            # assume negative intensity value are due to noise!
            # TODO
            # min_value = np.min(values)
            # if min_value < 0:
            #     decimals = int(digits_to_decimals(min_value, 0))
            #     values = np.round(values, decimals)
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
        Integral intensity
        """

        if self._intensity is None:
            self._intensity = Signal(
                self.integral,
                domain=self.output,
                name=self.name,
                attrs=self.attrs,
                contexts=self.contexts
            )
        return self._intensity

    def to_measured_spectra(self, units='uE'):
        return MeasuredSpectraContainer([self], units=units)

    def _resolution_mapping(self, values):
        """map output values to given resolution

        Parameters
        ----------
        values : np.ndarray or float
            Array that should already be in units of self.labels.units.
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
        R^2 score.
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
            y = np.concateante([y, [0]])

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


class MeasuredSpectraContainer(DomainSignalContainer):
    """Container for measured spectra

    Assumes each measured spectrum has a the same spectral
    distribution across intensities (e.g. LEDs).
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
        From intensity to output.

        Parameters
        ----------
        values : array-like
            samples x channels in intensity units set.
        return_units : bool
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
        From output to intensity
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
        Residuals given resolution.
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
        Score as mean of residuals given resolution
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

        return tuple(np.array(self._getattr__('intensity_bounds')).T)

    @property
    def output_bounds(self):
        """
        output bounds for Measured Spectra in two-tuple array form.

        Unit removed
        """

        return tuple(np.array(self._getattr__('output_bounds')).T)

    @property
    def normalized_spectra(self):
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
