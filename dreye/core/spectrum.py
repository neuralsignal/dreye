"""
Spectrum
========

Inherits signal class to build Spectrum class.
"""

from dreye.core.signal import ClippedSignal, Signal
from dreye.core.mixin import IrradianceMixin
from dreye.constants import DEFAULT_FLOAT_DTYPE
from dreye.err import DreyeUnitError
from dreye.utilities import dissect_units


class AbstractSpectrum(ClippedSignal):
    """abstract class for spectra
    """

    convert_attributes = ()
    init_args = ClippedSignal.init_args + (
        'smoothing_window', 'smoothing_method', 'smoothing_args',)

    def __init__(
            self,
            values,
            domain=None,
            domain_axis=None,
            units=None,
            labels=None,
            interpolator=None,
            interpolator_kwargs=None,
            smoothing_window=None,
            smoothing_method=None,
            smoothing_args=None,
            **kwargs  # because of create instance
    ):

        # hacky way to take care of create instance method
        _kwargs = {}
        for key, value in kwargs.items():
            if key not in AbstractSpectrum.init_args:
                _kwargs[key] = value

        # ignore domain units
        _kwargs.pop('domain_units', None)

        super().__init__(
            values=values,
            domain=domain,
            domain_axis=domain_axis,
            units=units,
            domain_units='nm',
            labels=labels,
            interpolator=interpolator,
            interpolator_kwargs=interpolator_kwargs,
            dtype=DEFAULT_FLOAT_DTYPE,
            domain_dtype=DEFAULT_FLOAT_DTYPE,
            contexts='flux',
            # always the case this is why convert_attributes is empty
            signal_min=0,
            signal_max=None,
            smoothing_method=smoothing_method,
            smoothing_window=smoothing_window,
            smoothing_args=smoothing_args,
            **_kwargs
        )

    @property
    def smoothing_args(self):
        """
        """

        if self._smoothing_args is None:
            return {}
        else:
            return self._smoothing_args

    @property
    def smoothing_window(self):
        """
        """

        if self._smoothing_window is None:
            return 1
        else:
            return self._smoothing_window

    @property
    def smoothing_method(self):
        """
        """

        if self._smoothing_method is None:
            return 'boxcar'
        else:
            return self._smoothing_method

    def create_new_instance(self, values, **kwargs):
        """Any operation returns Signal instance and not Spectrum instance
        """

        try:
            return self.__class__(values, **{**self.init_kwargs, **kwargs})
        except DreyeUnitError:
            return Signal(values, **{**self.init_kwargs, **kwargs})

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


class Spectrum(AbstractSpectrum, IrradianceMixin):
    """
    Subclass of signal class to represent light spectra.
    Units must be irradiance or photon flux. wavelengths must be in nm.

    Methods
    -------
    smooth # smoothing spectrum
    conversion to photonflux
    """

    irradiance_integrated = False

    def __init__(self,
                 values,
                 domain=None,
                 domain_axis=None,
                 units=None,
                 labels=None,
                 interpolator=None,
                 interpolator_kwargs=None,
                 **kwargs):

        units = self.get_units(values, units)

        super().__init__(values=values,
                         domain=domain,
                         domain_axis=domain_axis,
                         units=units,
                         labels=labels,
                         interpolator=interpolator,
                         interpolator_kwargs=interpolator_kwargs,
                         **kwargs)

    @property
    def normalized_spectrum(self):
        """
        """

        return NormalizedSpectrum(self)


class NormalizedSpectrum(AbstractSpectrum):
    """Class to store normalized spectra (i.e. sum to one).
    units will be in 1/nm.
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        values, units = dissect_units(self.normalized_signal)
        self._values = values
        self._units = units

    def create_new_instance(self, values, **kwargs):
        """Any operation returns Signal instance and not Spectrum instance
        """

        try:
            return AbstractSpectrum(values, **{**self.init_kwargs, **kwargs})
        except DreyeUnitError:
            return Signal(values, **{**self.init_kwargs, **kwargs})
