"""
Spectrum
========

Inherits signal class to build Spectrum class.
"""

from dreye.core.signal import (
    DomainSignal, Signals
)
from dreye.constants import ureg
from dreye.err import DreyeUnitError
from dreye.utilities import has_units


class _SpectrumMixin:

    _xlabel = 'wavelength (nm)'

    def __init__(self, *args, **kwargs):

        kwargs['domain_units'] = 'nm'
        kwargs['contexts'] = 'flux'
        # enforces nm units for domain and provides default flux context
        super().__init__(*args, **kwargs)

    @property
    def wavelengths(self):
        return self.domain


class _IntensityMixin:

    _unit_mappings = {
        'uE': 'microspectralphotonflux',
        'photonflux': 'spectralphotonflux',
        'irradiance': 'spectralirradiance'
    }

    def __init__(self, values, *args, units=None, **kwargs):

        # default units
        if units is None and not has_units(values):
            units = 'spectralirradiance'

        super().__init__(values, *args, units=units, **kwargs)

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
    def _ylabel(self):
        """
        Setting the ylabel for plotting
        """
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


# TODO inits


class Spectra(_SpectrumMixin, Signals):

    @property
    def _class_new_instance(self):
        return Spectra


class IntensitySpectra(_IntensityMixin, Spectra):
    pass


class DomainSpectrum(_SpectrumMixin, DomainSignal):
    pass


class IntensityDomainSpectrum(_IntensityMixin, DomainSpectrum):
    pass
