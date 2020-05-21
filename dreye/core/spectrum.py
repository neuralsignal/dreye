"""
Spectrum
========

Inherits signal class to build Spectrum class.
"""

import numpy as np

from dreye.core.signal import Signal, SignalContainer
from dreye.constants import ureg
from dreye.err import DreyeUnitError, DreyeError
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
            values=values,
            units=units,
            domain_units=domain_units,
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
        # applied window is added after smoothing
        if 'applied_window_' not in self.attrs:
            self.attrs['applied_window_'] = 'raw'

    @property
    def smoothing_args(self):
        return self.attrs.get('_smoothing_args', {})

    @property
    def smoothing_window(self):
        return self.attrs.get('_smoothing_window', 1)

    @property
    def smoothing_method(self):
        return self.attrs.get('_smoothing_method', 'savgol')

    def smooth(self, smoothing_window=None):
        """
        Performs smoothing on spectrum.
        """

        if smoothing_window is None:
            smoothing_window = self.smoothing_window

        spectrum = self.window_filter(
            smoothing_window, self.smoothing_method,
            extrapolate=False,
            **self.smoothing_args
        )
        spectrum.attrs['applied_window_'] = smoothing_window
        return spectrum

    @property
    def wavelengths(self):
        """Alias for domain attribute.
        """

        return self.domain

    def plotsmooth(
        self,
        min_window=None,
        max_window=None,
        steps=4,
        offset=0,
        **kwargs
    ):
        """plot spectrum with different smoothing parameters
        """

        if min_window is None and max_window is None:
            raise DreyeError('provide min_window or max_window')
        elif min_window is None:
            min_window = self.smoothing_window
        elif max_window is None:
            max_window = self.smoothing_window

        windows = np.linspace(min_window, max_window, steps)

        container = [self] + [self.smooth(window)+offset*(idx+1)
                              for idx, window in enumerate(windows)]

        # update default handler
        default_kws = dict(
            hue='applied_window_',
            col='labels',
            col_wrap=3,
        )
        default_kws.update(kwargs)

        return SignalContainer(container).relplot(**default_kws)


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
        domain_units='nm',
        labels=None,
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
        # default units
        if units is None and not has_units(values):
            units = 'spectralirradiance'

        super().__init__(
            values=values,
            units=units,
            domain_units=domain_units,
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
            smoothing_method=smoothing_method,
            smoothing_window=smoothing_window,
            smoothing_args=smoothing_args,
            name=name
        )

        # check units are correct dimensionality
        # i.e. irradiance or photon flux type
        truth = (
            (
                self.units.dimensionality
                == '[mass] / [length] / [time] ** 3'
            ) | (
                self.units.dimensionality
                == '[substance] / [length] ** 3 / [time]'
            )
        )

        if not truth:
            raise DreyeUnitError(self.units, 'irradiance convertible units')

    @property
    def normalized_spectrum(self):
        """
        """
        signal = self.normalized_signal
        signal._units = ureg(None).units
        return signal

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
        # TODO
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
