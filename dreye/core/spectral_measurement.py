"""Class to define spectral measurement
"""

import numpy as np
from scipy.optimize import lsq_linear
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression

from dreye.utilities import (
    has_units, is_numeric, asarray,
    optional_to
)
from dreye.constants import ureg
from dreye.core.signal import Signals
from dreye.core.spectrum import Spectra, IntensityDomainSpectrum
from dreye.core.signal_container import DomainSignalContainer
from dreye.err import DreyeError


class CalibrationSpectrum(Spectra):
    """
    Subclass of signal to define calibration spectrum.

    Units must be convertible to microjoule.
    """
    # TODO docstring and init

    def __init__(
        self,
        values,
        *args,
        area=None,
        **kwargs
    ):

        if area is None and not isinstance(values, CalibrationSpectrum):
            raise DreyeError(
                f"Must provide 'area' argument to "
                f"initialize {type(self).__name__}."
            )

        super.__init__(values, *args, **kwargs)

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
    Contain Measured Spectrum
    """

    # TODO docstring and init

    def __init__(
        self, *args,
        zero_boundary=None,
        max_boundary=None,
        resolution=None,
        **kwargs
    ):

        super().__init__(
            *args,
            **kwargs
        )

        if self.name is None:
            raise DreyeError(f'name variable must be provided '
                             f'for {type(self).__name__} instance.')

        # getting correct values and converting units
        if zero_boundary is None:
            zero_boundary = self.attrs.get('zero_boundary_', None)
        if max_boundary is None:
            max_boundary = self.attrs.get('max_boundary_', None)
        if resolution is None:
            resolution = self.attrs.get('resolution_', None)

        self.attrs['zero_boundary_'] = self._get_domain_bound(
            zero_boundary, self.labels
        )
        self.attrs['max_boundary_'] = self._get_domain_bound(
            max_boundary, self.labels
        )
        # should be the minimum step that can be taken
        self.attrs['resolution_'] = self._get_domain_bound(
            resolution, self.labels
        )

        self._intensity = None
        self._normalized_spectrum = None

    @property
    def zero_is_lower(self):
        """
        Ascending or descending intensity values.

        Calculated automatically.
        """
        if (
            np.isnan(self.zero_boundary.magnitude)
            or np.isnan(self.max_boundary.magnitude)
        ):
            # integral across wavelengths
            total_intensity = self.intensity.magnitude
            return total_intensity[0] < total_intensity[-1]
        else:
            return self.zero_boundary.magnitude < self.max_boundary.magnitude

    @property
    def resolution(self):
        """
        Smallest possible label/input value differences.

        Includes units
        """
        return self.attrs['resolution_'].to(self.labels.units)

    @property
    def zero_boundary(self):
        """
        Label value corresponding to zero intensity across wavelengths.

        Includes units.
        """
        return self.attrs['zero_boundary_'].to(self.labels.units)

    @property
    def max_boundary(self):
        """
        Label/input value corresponding to max intensity across wavelengths.

        Includes units.
        """
        return self.attrs['max_boundary_'].to(self.labels.units)

    @property
    def lower_input_bound(self):
        """
        Lower boundary of input labels, e.g. 0 volts.

        Does not include units
        """
        if self.zero_is_lower:
            if np.isnan(self.zero_boundary.magnitude):
                return self.labels.start
            else:
                return self.zero_boundary.magnitude
        else:
            if np.isnan(self.zero_boundary.magnitude):
                return self.labels.end
            else:
                return self.zero_boundary.magnitude

    @property
    def upper_input_bound(self):
        """
        Upper boundary of input labels, e.g. 5 volts.

        Does not include units.
        """
        if self.zero_is_lower:
            if np.isnan(self.max_boundary.magnitude):
                return self.labels.end
            else:
                return self.max_boundary.magnitude
        else:
            if np.isnan(self.max_boundary.magnitude):
                return self.labels.start
            else:
                return self.max_boundary.magnitude

    @property
    def input_bounds(self):
        """
        Bounds of input, e.g. 0 adn 5 volts.

        Does not include units
        """

        input_bounds = list(self.input.boundaries)

        idx_zero = 1 - int(self.zero_is_lower)
        if not np.isnan(self.zero_boundary.magnitude):
            input_bounds[idx_zero] = self.zero_boundary.magnitude
        if not np.isnan(self.max_boundary.magnitude):
            input_bounds[idx_zero - 1] = self.max_boundary.magnitude

        return tuple(input_bounds)

    @property
    def input(self):
        """
        Alias for labels
        """
        return self.labels

    @property
    def intensity_bounds(self):
        """
        Bounds of intensity signal after wavelength integration.

        Does not include units.
        """

        integral = self.intensity.magnitude

        if not np.isnan(self.zero_boundary.magnitude):
            lower = 0.0
            upper = np.max(integral)
        else:
            lower = np.min(integral)
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
            values = self.mean(axis=1)
            spectra = Spectra(
                values=values,
                domain=self.domain,
                labels=self.name,
                attrs=self.attrs
            )
            self._normalized_spectrum = spectra.normalized_signal

        return self._normalized_spectrum

    @property
    def intensity(self):
        """
        Integral intensity
        """

        if self._intensity is None:
            self._intensity = Signals(
                self.integral,
                domain=self.labels,
                labels=self.name,
                attrs=self.attrs
            )
        return self._intensity

    def to_measured_spectra(self, units='uE'):
        return MeasuredSpectraContainer([self], units=units)

    def map(self, values, return_units=True):
        """
        Map Intensity values to input values.

        Parameters
        ----------
        values : array-like
            samples in intensity units or no units.
        return_units : bool
            Whether to return mapped values with units.
        """

        values = optional_to(values, units=self.intesity.units)
        assert values.ndim < 2, 'values must be 1 dimensional'

        # check intensity bound of values
        imin, imax = self.intensity_bounds
        truth = np.all(values >= imin) and np.all(values <= imax)
        assert truth, 'Some values to be mapped are out of bounds.'

        mapped_values = self._mapper_func(values)

        if return_units:
            return mapped_values * self.input.units

        return mapped_values

    @property
    def _mapper_func(self):
        """mapping using isotonic regression
        """

        # 1D signal
        y = self.intesity.magnitude  # integral across intensities
        x = self.labels.magnitude
        y_min, y_max = self.intensity_bounds
        zero_is_lower = self.zero_is_lower
        zero_boundary = self.zero_boundary.magnitude

        # a little redundant but should ensure safety of method
        if zero_is_lower and zero_boundary < np.min(x):
            x = np.concatenate([[zero_boundary], x])
            y = np.concatenate([[0], y])
        # a little redundant but should ensure safety of method
        elif not zero_is_lower and zero_boundary > np.max(x):
            x = np.concatenate([x, [zero_boundary]])
            y = np.concateante([y, [0]])

        # perform isotonic regression
        isoreg = IsotonicRegression(
            # lower and upper intensity values
            y_min=y_min,
            y_max=y_max,
            increasing=zero_is_lower
        )

        new_y = isoreg.fit_transform(x, y)
        interp_func = interp1d(
            new_y, x,
            bounds_error=False,
            # allow going beyond bounds
            # but fill values to lower and upper bounds
            # lower and upper input values
            fill_value=self.input_bounds
        )

        def clip_wrapper(*args, **kwargs):
            return np.clip(
                interp_func(*args, **kwargs),
                a_min=self.input_bounds[0],
                a_max=self.input_bounds[1]
            )

        return clip_wrapper


class MeasuredSpectraContainer(DomainSignalContainer):
    """Container for measured spectra

    Assumes each measured spectrum has a the same spectral
    distribution across intensities (e.g. LEDs).
    """

    _xlabel = 'wavelength (nm)'
    _cmap = 'viridis'
    _init_keys = [
        '_intensities',
        '_normalized_spectra',
        '_mapper'
    ]
    _allowed_instances = MeasuredSpectrum

    def map(self, values, return_units=True):
        """

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

    @property
    def zero_is_lower(self):
        """
        Whether zero intensity input value is lower than the max intensity
        input value.
        """
        return np.array(self.__getattr__('zero_is_lower'))

    @property
    def zero_boundary(self):
        """
        Boundary of input corresponding to zero intensity spectrum

        Contains input units.
        """
        return np.array([
            ele.magnitude
            for ele in self.__getattr__('zero_boundary')
        ]) * self.labels_units

    @property
    def max_boundary(self):
        """
        Boundary of input corresponding to maximum intensity spectrum.

        Contains input units.
        """
        return np.array([
            ele.magnitude
            for ele in self.__getattr__('max_boundary')
        ]) * self.labels_units

    @property
    def intensity_bounds(self):
        """
        Intensity bounds for the Measured Spectra in two-tuple array form.

        Unit removed.
        """

        return tuple(np.array(self._getattr__('intensity_bounds')).T)

    @property
    def input_bounds(self):
        """
        Input bounds for Measured Spectra in two-tuple array form.

        Unit removed
        """

        return tuple(np.array(self._getattr__('input_bounds')).T)

    @property
    def normalized_spectra(self):
        if self._normalized_spectra is None:
            for idx, ele in enumerate(self):
                if idx == 0:
                    spectra = ele.normalized_spectrum
                else:
                    spectra = spectra.labels_concat(ele.normalized_spectrum)

            self._normalized_spectra = spectra
        return self._normalized_spectra

    @property
    def intensities(self):
        if self._intensities is None:
            for idx, ele in enumerate(self):
                if idx == 0:
                    intensities = ele.intensity
                else:
                    intensities = intensities.labels_concat(ele.intensity)

            self._intensities = intensities
        return self._intensities

    @property
    def wavelengths(self):
        return self.normalized_spectra.domain

    @property
    def _ylabel(self):
        return self[0]._ylabel
