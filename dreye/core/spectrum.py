"""
Spectrum
========

Inherits signal class to build Spectrum class.
"""

from dreye.core.signal import (
    DomainSignal, Signals, Signal
)
from dreye.constants import ureg
from dreye.err import DreyeUnitError
from dreye.utilities import has_units


class _SpectrumMixin:

    _xlabel = r'$\lambda$ (nm)'

    def __init__(
        self,
        values,
        domain=None,
        *args,
        **kwargs
    ):

        kwargs['domain_units'] = 'nm'
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
        units=None,
        **kwargs
    ):

        # default units
        if units is None and not has_units(values):
            units = 'spectralirradiance'

        super().__init__(
            values, domain,
            *args, units=units,
            **kwargs
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


# TODO inits


class Spectra(_SpectrumMixin, Signals):

    @property
    def _class_new_instance(self):
        return Spectra


class Spectrum(_SpectrumMixin, Signal):

    @property
    def _class_new_instance(self):
        return Spectrum


class IntensitySpectra(_IntensityMixin, Spectra):
    pass


class IntensitySpectrum(_IntensityMixin, Spectrum):
    pass


class DomainSpectrum(_SpectrumMixin, DomainSignal):
    pass


class IntensityDomainSpectrum(_IntensityMixin, DomainSpectrum):
    pass
