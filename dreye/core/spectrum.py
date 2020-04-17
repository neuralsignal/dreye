"""
Spectrum
========

Inherits signal class to build Spectrum class.
"""

from dreye.core.signal import Signal
from dreye.constants import DEFAULT_FLOAT_DTYPE, ureg
from dreye.err import DreyeUnitError
from dreye.utilities import has_units


class AbstractSpectrum(Signal):
    """abstract class for spectra
    """

    _xlabel = 'wavelength (nm)'

    @property
    def _class_new_instance(self):
        return AbstractSpectrum

    def __init__(
        self,
        values,
        domain=None,
        domain_axis=None,
        units=None,
        domain_units='nm',
        labels=None,
        dtype=DEFAULT_FLOAT_DTYPE,
        domain_dtype=DEFAULT_FLOAT_DTYPE,
        interpolator=None,
        interpolator_kwargs=None,
        contexts='flux',
        domain_kwargs=None,
        domain_min=None,
        domain_max=None,
        signal_min=None,
        signal_max=None,
        attrs=None,
        smoothing_method=None,
        smoothing_window=None,
        smoothing_args=None,
        name=None
    ):
        # enforces nm units for domain and provides default flux context

        super().__init__(
            values,
            units=units,
            domain_units=domain_units,
            dtype=dtype,
            domain_dtype=domain_dtype,
            domain=domain,
            domain_axis=domain_axis,
            labels=labels,
            interpolator=interpolator,
            interpolator_kwargs=interpolator_kwargs,
            contexts=contexts,
            domain_kwargs=domain_kwargs,
            attrs=attrs,
            domain_min=domain_min,
            domain_max=domain_max,
            signal_min=signal_min,
            signal_max=signal_max,
            name=name
        )

        if smoothing_method is not None:
            self.attrs.update({'_smoothing_method': smoothing_method})
        if smoothing_window is not None:
            self.attrs.update({'_smoothing_window': smoothing_window})
        if smoothing_args is not None:
            self.attrs.update({'_smoothing_args': smoothing_args})

    @property
    def smoothing_args(self):
        return self.attrs.get('_smoothing_args', {})

    @property
    def smoothing_window(self):
        return self.attrs.get('_smoothing_window', 1)

    @property
    def smoothing_method(self):
        return self.attrs.get('_smoothing_method', 'savgol')

    @property
    def smooth(self):
        """
        Performs smoothing on spectrum.
        """

        return self.window_filter(
            self.smoothing_window, self.smoothing_method,
            extrapolate=False,
            **self.smoothing_args
        )

    @property
    def wavelengths(self):
        """
        """

        return self.domain


class Spectrum(AbstractSpectrum):
    """
    Subclass of signal class to represent light spectra.
    Units must be irradiance or photon flux. wavelengths must be in nm.

    Methods
    -------
    smooth # smoothing spectrum
    conversion to photonflux
    """

    _unit_mappings = {
        'uE': 'microspectralphotonflux',
        'photonflux': 'spectralphotonflux',
        'irradiance': 'spectralirradiance'
    }

    def __init__(
        self,
        values,
        domain=None,
        domain_axis=None,
        units=None,
        **kwargs
    ):

        units = self._get_units(values, units)

        super().__init__(
            values=values,
            domain=domain,
            domain_axis=domain_axis,
            units=units,
            **kwargs
        )

    @property
    def normalized_spectrum(self):
        """
        """
        signal = self.normalized_signal
        signal._units = ureg(None).units
        return signal

    @classmethod
    def _get_units(cls, values, units):
        """
        """

        if units is None:
            if not has_units(values):
                units = 'spectral_irradiance'
            else:
                units = values.units
        elif units in cls._unit_mappings:
            units = cls._unit_mappings[units]
        cls._check_units(units)
        return units

    @classmethod
    def _check_units(cls, units):
        """
        """

        if not isinstance(units, str):
            units = str(units)

        truth_value = ureg(units).check(
            '[mass] / [length] / [time] ** 3')
        truth_value |= ureg(units).check(
            '[substance] / [length] ** 3 / [time]')

        if not truth_value:
            raise DreyeUnitError(units, 'irradiance convertible units')

    @property
    def photonflux(self):
        return self.to('spectralphotonflux')

    @property
    def uE(self):
        return self.to('microspectralphotonflux')

    @property
    def irradiance(self):
        return self.to('spectralirradiance')

    @property
    def _ylabel(self):
        prefix = str(self.units)[:str(self.units).find('spectral')]
        if (
            self.units.dimensionality
            == ureg('spectralphotonflux').dimensionality
        ):
            ylabel = (
                'photonflux'
                + ' ($\\frac{' + prefix
                + 'mol}{m^2\\cdot s \\cdot nm'
                + '}$)'
            )
            return ylabel
        else:
            ylabel = (
                'irradiance'
                + ' ($\\frac{' + prefix
                + 'W}{m^2 \\cdot nm'
                + '}$)'
            )
            return ylabel
