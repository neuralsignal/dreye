"""
Spectrum Mixin classes
"""

from dreye.core.signal import (
    DomainSignal, Signals, Signal
)
from dreye.err import DreyeUnitError
from dreye.utilities import has_units, get_units


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


class _IntensitySpectraMixin:

    def _class_new_instance(self, values, *args, units=None, **kwargs):
        if has_units(values):
            truth = self._is_intensity_units(values.units)
        else:
            truth = self._is_intensity_units(get_units(units))
        if truth:
            return type(self)(values, *args, units=units, **kwargs)
        else:
            return Signals(values, *args, units=units, **kwargs)


class _IntensitySpectrumMixin:

    def _class_new_instance(self, values, *args, units=None, **kwargs):
        if has_units(values):
            truth = self._is_intensity_units(values.units)
        else:
            truth = self._is_intensity_units(get_units(units))
        if truth:
            return type(self)(values, *args, units=units, **kwargs)
        else:
            return Signal(values, *args, units=units, **kwargs)


class _IntensityDomainSpectrumMixin(_IntensityMixin, _SpectrumMixin):

    def _class_new_instance(self, values, *args, units=None, **kwargs):
        if has_units(values):
            truth = self._is_intensity_units(values.units)
        else:
            truth = self._is_intensity_units(get_units(units))
        if truth:
            return type(self)(values, *args, units=units, **kwargs)
        else:
            return DomainSignal(values, *args, units=units, **kwargs)