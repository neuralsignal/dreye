"""Class to define spectral measurement
"""

# third party modules
import numpy as np
from scipy.optimize import lsq_linear

# dreye modules
from dreye.utilities import has_units, dissect_units, is_numeric
from dreye.constants import UREG
from dreye.core.spectrum import \
    AbstractSpectrum, Spectrum
from dreye.core.signal import ClippedSignal, Signal
from dreye.core.domain import Domain
from dreye.core.mixin import IrradianceMixin, MappingMixin
from dreye.err import DreyeUnitError


class CalibrationSpectrum(AbstractSpectrum):
    """Subclass of signal to define calibration spectrum.
    Units must be convertible to microjoule
    """

    init_args = AbstractSpectrum.init_args + ('area',)

    def __init__(
        self,
        values,
        domain=None,
        area=None,
        units='microjoule',
        labels=None,
        interpolator=None,
        interpolator_kwargs=None,
        area_units='cm ** 2',
        **kwargs
    ):

        # TODO unit checking
        if area is None:
            assert isinstance(values, CalibrationSpectrum)
            area = values.area
        else:
            assert is_numeric(area)

        if not has_units(area):
            area = area * UREG(area_units)
        else:
            area = area.to(area_units)

        super().__init__(values=values,
                         domain=domain,
                         units=units,
                         labels=labels,
                         interpolator=interpolator,
                         interpolator_kwargs=interpolator_kwargs,
                         area=area,
                         **kwargs)

        assert self.ndim == 1, \
            "calibration spectrum must always be one dimensional"

    def create_new_instance(self, values, **kwargs):
        """Any operation returns Signal instance and not Spectrum instance
        """

        if values.ndim == 2:

            kwargs.pop('area', None)
            init_kwargs = self.init_kwargs
            init_kwargs.pop('area', None)

            try:
                return AbstractSpectrum(values, **{**init_kwargs, **kwargs})
            except DreyeUnitError:
                return Signal(values, **{**init_kwargs, **kwargs})
        else:
            try:
                return CalibrationSpectrum(
                    values, **{**self.init_kwargs, **kwargs}
                )
            except DreyeUnitError:
                return Signal(values, **{**self.init_kwargs, **kwargs})

    @property
    def area(self):
        """
        """

        return self._area


class MeasuredSpectrum(Spectrum):
    """
    Subclass of Signal that also stores a spectrum associated to a
    spectrum measurement.

    Methods
    -------
    fit # fitting spectrum with a set of spectra
    conversion to photonflux

    Parameters
    ----------
    values : 2D
    """

    # TODO concat_labels method

    def __init__(
        self,
        values,
        domain=None,
        labels=None,
        units=None,
        domain_axis=None,
        interpolator=None,
        interpolator_kwargs=None,
        **kwargs
    ):
        # checks that labels is a domain
        if hasattr(values, 'labels') and labels is None:
            labels = values.labels

        assert isinstance(labels, Domain)

        super().__init__(values=values,
                         domain=domain,
                         units=units,
                         labels=labels,
                         domain_axis=domain_axis,
                         interpolator=interpolator,
                         interpolator_kwargs=interpolator_kwargs,
                         **kwargs)

        assert self.ndim == 2

    def create_new_instance(self, values, **kwargs):
        """Any operation returns Signal instance and not Spectrum instance
        """
        try:
            return MeasuredSpectrum(
                values, **{**self.init_kwargs, **kwargs}
            )
        except Exception:
            try:
                return Spectrum(values, **{**self.init_kwargs, **kwargs})
            except DreyeUnitError:
                return Signal(values, **{**self.init_kwargs, **kwargs})

    @property
    def smooth(self):
        """
        """

        signal = super().smooth
        return self.__class__(signal)

    @property
    def inputs(self):
        """
        """

        return self._labels

    def to_spectrum_measurement(self, name, units='uE', **kwargs):
        """
        """

        self = getattr(self, units)

        # Normalization options?
        labels = MeasuredNormalizedSpectrum(
            self.mean(axis=self.other_axis),
            labels=name
        )

        spm = SpectrumMeasurement(
            values=self.integral,
            domain=self.inputs,
            labels=labels,
            **kwargs
        )

        return getattr(spm, units)


class MeasuredNormalizedSpectrum(AbstractSpectrum):
    init_args = AbstractSpectrum.init_args + (
        'zero_boundary', 'max_boundary', 'zero_is_lower',
        'boundary_units'
    )

    def __init__(
        self,
        values,
        domain=None,
        labels=None,
        zero_boundary=None,
        max_boundary=None,
        zero_is_lower=None,
        boundary_units=None,
        units=None,
        **kwargs
    ):

        super().__init__(
            values=values,
            domain=domain,
            labels=labels,
            zero_boundary=zero_boundary,
            max_boundary=max_boundary,
            zero_is_lower=zero_is_lower,
            units=units,
            boundary_units=boundary_units,
            **kwargs
        )

        integral = np.array(self.integral)
        if integral.shape == () or integral.shape == (0, ):
            pass
        else:
            integral = np.expand_dims(integral, axis=self.domain_axis)

        self._values = (
            self.magnitude
            / integral
        )
        self._units = 1 / self.domain.units

    @property
    def boundary_units(self):
        return self._boundary_units

    @boundary_units.setter
    def boundary_units(self, value):
        if value is None:
            return
        if has_units(value):
            self._boundary_units = value.units
        else:
            self._boundary_units = UREG(str(value)).units

    @property
    def zero_is_lower(self):
        """
        """

        if self._zero_is_lower is None:
            truth = np.array([True]*self.other_len)
            notnan = ~(
                np.isnan(self.zero_boundary) | np.isnan(self.max_boundary)
            )
            truth[notnan] = (self.zero_boundary < self.max_boundary)[notnan]
            return truth
        elif self._zero_is_lower is True:
            return np.array([True]*self.other_len)
        elif self._zero_is_lower is False:
            return np.array([False]*self.other_len)
        else:
            zero_is_lower = np.array(self._zero_is_lower)
            assert self.other_len == zero_is_lower.size
            return zero_is_lower

    @zero_is_lower.setter
    def zero_is_lower(self, value):
        raise NotImplementedError('zero is lower setter')

    @property
    def zero_boundary(self):
        if self._zero_boundary is None:
            return np.nan * np.ones(self.other_len)
        else:
            zero_boundary = self._zero_boundary
            if has_units(zero_boundary):
                if self.boundary_units is None:
                    self._boundary_units = zero_boundary.units
                zero_boundary = dissect_units(
                    zero_boundary.to(self.boundary_units)
                )[0]
            if not isinstance(zero_boundary, np.ndarray):
                zero_boundary = np.array(zero_boundary)
            if zero_boundary.shape in ((), (0, )):
                zero_boundary = np.array([zero_boundary])
            assert self.other_len == zero_boundary.size
            return zero_boundary

    @zero_boundary.setter
    def zero_boundary(self, value):
        raise NotImplementedError('zero boundary setter')

    @property
    def max_boundary(self):
        if self._max_boundary is None:
            return np.nan * np.ones(self.other_len)
        else:
            max_boundary = self._max_boundary
            if has_units(max_boundary):
                if self.boundary_units is None:
                    self._boundary_units = max_boundary.units
                max_boundary = dissect_units(
                    max_boundary.to(self.boundary_units)
                )[0]
            if not isinstance(max_boundary, np.ndarray):
                max_boundary = np.array(max_boundary)
            if max_boundary.shape in ((), (0, )):
                max_boundary = np.array([max_boundary])
            assert self.other_len == max_boundary.size
            return max_boundary

    @max_boundary.setter
    def max_boundary(self, value):
        raise NotImplementedError('max boundary setter')

    def other_concat(self, signal, *args, left=False, **kwargs):
        assert isinstance(signal, MeasuredNormalizedSpectrum)
        assert signal.boundary_units == self.boundary_units
        new_signal = super().other_concat(
            signal, *args, left=left, **kwargs
        )

        # handle signals
        if left:
            signal1 = signal
            signal2 = self
        else:
            signal1 = self
            signal2 = signal
        new_signal._zero_boundary = np.concatenate([
            signal1.zero_boundary, signal2.zero_boundary
        ])
        new_signal._max_boundary = np.concatenate([
            signal1.max_boundary, signal2.max_boundary
        ])
        new_signal._zero_is_lower = np.concatenate([
            signal1.zero_is_lower, signal2.zero_is_lower
        ])

        return new_signal


class SpectrumMeasurement(ClippedSignal, IrradianceMixin, MappingMixin):
    """
    Subclass of Signal that also stores a spectrum associated to a
    spectrum measurement. Can also store multiple measured spectra.
    Assumes spectrum does not change with intensity.

    Methods
    -------
    fit # fitting spectrum with a set of spectra
    conversion to photonflux

    Parameters
    ----------
    values : 1D or 2D
    spectrum : 1D or 2D
    wavelengths : 1D
    """

    # TODO implement concat labels method

    init_args = ClippedSignal.init_args + (
        'mapping_method',
    )
    irradiance_integrated = True

    def __init__(
        self,
        values,
        domain=None,
        labels=None,
        label_names=None,
        mapping_method=None,
        zero_boundary=None,
        max_boundary=None,
        zero_is_lower=None,
        units=None,
        **kwargs
    ):
        """
        """

        units = self.get_units(values, units)

        if labels is None:
            assert isinstance(values, SpectrumMeasurement)
            labels = values.labels
        else:
            assert isinstance(labels, AbstractSpectrum)

            if labels.ndim == 2:
                labels = labels.moveaxis(labels.other_axis, 0)

        labels = MeasuredNormalizedSpectrum(
            labels, labels=label_names,
            zero_boundary=zero_boundary,
            max_boundary=max_boundary,
            zero_is_lower=zero_is_lower,
        )

        # TODO monotonicity warnings
        if 'interpolator' not in kwargs:
            # default is to not have a bounds error here
            interp_kwargs = kwargs.get('interpolator_kwargs', {})
            interp_kwargs['bounds_error'] = interp_kwargs.get(
                'bounds_error', False)
            interp_kwargs['fill_value'] = interp_kwargs.get(
                'fill_value', 'extrapolate'
            )

            kwargs['interpolator_kwargs'] = interp_kwargs

        # replaces signal min and max
        kwargs['signal_min'] = None
        kwargs['signal_max'] = None

        super().__init__(
            values=values,
            domain=domain,
            labels=labels,
            units=units,
            mapping_method=mapping_method,
            **kwargs
        )

        # set the correct units
        self.labels.boundary_units = self.domain.units

        signal_min = np.zeros(self.zero_boundary.shape)
        signal_min[np.isnan(self.zero_boundary)] = np.atleast_1d(np.min(
            self.magnitude, axis=self.domain_axis
        ))[np.isnan(self.zero_boundary)]
        self.signal_min = signal_min
        # max signal
        self.signal_max = np.max(self.magnitude, axis=self.domain_axis)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        truth = self.shape == other.shape
        if not truth:
            return False
        truth = np.allclose(
            self.magnitude, other.magnitude, equal_nan=True)
        if not truth:
            return False
        truth = self.labels.shape == other.labels.shape
        if not truth:
            return False
        truth = np.allclose(
            self.labels.magnitude, self.labels.magnitude, equal_nan=True)
        if not truth:
            return False
        truth = self.domain == other.domain
        if not truth:
            return False
        truth = np.all(self.label_names == other.label_names)
        if not truth:
            return False
        truth = np.all(self.signal_min == other.signal_min)
        if not truth:
            return False
        truth = np.all(self.signal_max == other.signal_max)
        if not truth:
            return False
        truth = np.all(self.zero_boundary == other.zero_boundary)
        if not truth:
            return False
        truth = np.all(self.max_boundary == other.max_boundary)
        if not truth:
            return False
        truth = self.units == other.units
        if not truth:
            return False
        truth = self.domain.units == other.domain.units
        if not truth:
            return False

        return truth

    @property
    def label_names(self):
        return self.labels.labels

    def _iso_increasing(self, idx=None):
        if idx is None:
            return bool(self.zero_is_lower)
        else:
            return self.zero_is_lower[idx]

    @property
    def zero_is_lower(self):
        """
        """

        return self.labels.zero_is_lower

    @property
    def zero_boundary(self):
        return self.labels.zero_boundary

    @property
    def max_boundary(self):
        return self.labels.max_boundary

    @property
    def mapping_method(self):
        """
        """

        return self._mapping_method

    def concat_labels(self, labels, left=False):
        """
        """

        assert self.ndim == 2
        assert isinstance(labels, MeasuredNormalizedSpectrum)
        assert self.labels.boundary_units == labels.boundary_units
        new_labels = self.labels.concat(labels, left=left)
        return new_labels

    def pre_mapping(self, values):
        """
        """

        min = np.atleast_2d(self.bounds[0])
        max = np.atleast_2d(self.bounds[1])
        if values.ndim == 1:
            if self.ndim == 2:
                v = values[:, None]
            else:
                v = values[None, :]
        else:
            v = values

        truth = np.all(v >= min)
        truth &= np.all(v <= max)
        assert truth, 'some values to be mapped are out of bounds.'

        return values

    def post_mapping(self, x, units=True, **kwargs):
        """
        """

        # TODO implement mapping_method
        if units:
            units = self.domain.units
        else:
            units = 1

        if self.mapping_method is None:
            # TODO test broadcasting
            return np.clip(
                x,
                a_min=self.lower_boundary[None, :],
                a_max=self.upper_boundary[None, :]
            ) * units

        else:
            raise NameError(
                f'mapping method {self.mapping_method} does not exit')

    @property
    def domain_bounds(self):
        """
        """

        if self.ndim == 2:
            return np.array([self.lower_boundary, self.upper_boundary]).T
        else:
            return np.squeeze([self.lower_boundary, self.upper_boundary])

    @property
    def lower_boundary(self):
        """
        """
        lower_boundary = np.zeros(self.zero_is_lower.shape)
        lower_boundary[self.zero_is_lower] = \
            self.zero_boundary[self.zero_is_lower]
        lower_boundary[~self.zero_is_lower] = \
            self.max_boundary[~self.zero_is_lower]
        lower_boundary[np.isnan(lower_boundary)] = self.domain.start
        return lower_boundary

    @property
    def upper_boundary(self):
        """
        """
        upper_boundary = np.zeros(self.zero_is_lower.shape)
        upper_boundary[~self.zero_is_lower] = \
            self.zero_boundary[~self.zero_is_lower]
        upper_boundary[self.zero_is_lower] = \
            self.max_boundary[self.zero_is_lower]
        upper_boundary[np.isnan(upper_boundary)] = self.domain.end
        return upper_boundary

    @property
    def spectrum(self):
        """
        """

        return self.labels

    @property
    def normalized_spectrum(self):
        """
        """

        # TODO any units?
        # wavelength x different sources
        return self.labels.moveaxis(0, -1)

    @property
    def bounds(self):
        """
        """

        bounds = list(self.boundaries.T)

        # TODO change
        if self.zero_boundary is not None:
            bounds[0] *= 0

        return bounds

    def fit(self, spectrum, return_res=False, return_fit=False, units=True):
        """
        """

        assert isinstance(spectrum, Spectrum)
        assert spectrum.ndim == 1

        # TODO integral = spectrum.integral
        # TODO unit checking of spectrum and self
        spectrum = spectrum.copy()
        spectrum.units = self.units / UREG('nm')

        spectrum, normalized_sources = spectrum.equalize_domains(
            self.normalized_spectrum, equalize_dimensions=False)

        b = np.array(spectrum)
        A = np.array(normalized_sources)

        res = lsq_linear(A, b, bounds=self.bounds)

        # Class which incorportates the following
        # values=res.x, units=self.units, axis0_labels=self.labels

        if units:
            weights = res.x * self.units
        else:
            weights = res.x

        fitted_spectrum = (
            self.normalized_spectrum * weights[None, :]
        ).sum(axis=1)

        if return_res and return_fit:
            return weights, res, fitted_spectrum
        elif return_res:
            return weights, res
        elif return_fit:
            return weights, fitted_spectrum
        else:
            return weights

    def fit_map(self, spectrum, independent=True, **kwargs):
        """
        """

        values = self.fit(spectrum, **kwargs)

        return self.map(values, independent=independent)
