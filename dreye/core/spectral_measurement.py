"""Class to define spectral measurement
"""

# third party modules
import numpy as np
from scipy.optimize import lsq_linear
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression

# dreye modules
from dreye.utilities import (
    has_units, is_numeric, asarray,
    _convert_get_val_opt
)
from dreye.constants import ureg
from dreye.core.spectrum import AbstractSpectrum, Spectrum
from dreye.core.signal import Signal
from dreye.core.domain import Domain
from dreye.err import DreyeError


class CalibrationSpectrum(AbstractSpectrum):
    """Subclass of signal to define calibration spectrum.
    Units must be convertible to microjoule
    """

    def __init__(
        self,
        values,
        domain=None,
        units='microjoule',
        labels=None,
        area=None,
        area_units='cm ** 2',
        **kwargs
    ):

        if area is None and not isinstance(values, CalibrationSpectrum):
            raise DreyeError(
                'Must provide Area for calibration spectrum.'
            )

        super().__init__(
            values=values,
            domain=domain,
            units=units,
            labels=labels,
            **kwargs
        )

        # set area key in attribute
        area = self.attrs.get('area', area)
        if not is_numeric(area):
            raise DreyeError('area must be a numeric value.')
        elif has_units(area):
            area = area.to(area_units)
        else:
            area = area * ureg(area_units)
        self.attrs['area'] = area

        if self.ndim != 1:
            raise DreyeError(
                "Calibration spectrum must always be one dimensional"
            )

    @property
    def area(self):
        """
        """

        return self.attrs['area']


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

    def __init__(
        self, *args,
        zero_boundary=None,
        max_boundary=None,
        zero_is_lower=None,
        label_units=None,
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        if not self.ndim == 2:
            raise DreyeError('MeasuredSpectrum must be two-dimensional.')
        if not isinstance(self.labels, Domain):
            try:
                self._labels = Domain(self.labels, units=label_units)
            except Exception:
                raise DreyeError(
                    'Labels must be domain or be able'
                    ' to be made into domain.'
                )
        elif label_units is not None:
            self._labels = self.labels.to(label_units)

        if self.name is None:
            raise DreyeError(
                'name variable must be provided for the MeasuredSpectrum class'
            )

        if zero_boundary is not None:
            self.attrs['zero_boundary'] = zero_boundary
        if max_boundary is not None:
            self.attrs['max_boundary'] = max_boundary
        if zero_is_lower is not None:
            self.attrs['zero_is_lower'] = zero_is_lower

        # convert to correct units, but only return value
        self.attrs['zero_boundary'] = _convert_get_val_opt(
            self.attrs['zero_boundary'], self.labels.units
        )
        self.attrs['max_boundary'] = _convert_get_val_opt(
            self.attrs['max_boundary'], self.labels.units
        )

        if (
            self.attrs['zero_is_lower'] is None
            and (
                self.attrs['max_boundary'] is None
                or self.attrs['zero_boundary'] is None
            )
        ):
            raise DreyeError(
                "Must provide zero_is_lower or max and zero boundary."
            )

        if self.attrs['zero_is_lower'] is None:
            self.attrs['zero_is_lower'] = (
                self.zero_boundary < self.max_boundary
            )

    @property
    def boundary_units(self):
        return self.labels.units

    @property
    def zero_is_lower(self):
        return self.attrs['zero_is_lower']

    @property
    def zero_boundary(self):
        return self.attrs['zero_boundary']

    @property
    def max_boundary(self):
        return self.attrs['max_boundary']

    @property
    def inputs(self):
        """
        """

        return self._labels

    def to_measured_spectra(self, units='uE'):
        return MeasuredSpectraContainer([self], units=units)


class MeasuredSpectraContainer:
    """Container for measured spectra
    """

    def __init__(self, measured_spectra, units=None):
        self._measured_spectra = measured_spectra
        self._check_list()
        self._init_attrs()
        # equalize units
        if units is None and len(set([ele.units for ele in self])) == 1:
            pass
        else:
            if units is None:
                units = 'uE'
            self._measured_spectra = [
                getattr(ele, units)
                for ele in self
            ]

        self._units = self._measured_spectra[0].units

    def __repr__(self):
        return f"{self.__class__.__name__}({self.names})"

    def to_dict(self):
        return self._measured_spectra

    @classmethod
    def from_dict(cls, data):
        # TODO unit saving
        return cls(data)

    @property
    def units(self):
        return self._units

    def _init_attrs(self):
        self._zero_is_lower = None
        self._zero_boundary = None
        self._max_boundary = None
        self._intensities_list = None
        self._intensities = None
        self._normalized_spectrum_list = None
        self._normalized_spectrum = None
        self._mapper = None

    def map(self, values, **kwargs):
        """

        Parameters
        ----------
        values : array-like
            samples x channels in intensity units set.
        """

        values = asarray(values)
        values = self.pre_mapping(values)
        assert values.ndim < 3

        x = self.mapper(np.atleast_2d(values))

        if values.ndim == 1:
            x = x[0]

        return self.post_mapping(x, **kwargs)

    @property
    def mapper(self):
        if self._mapper is None:
            mappers = []
            for idx, ele in enumerate(self):
                mappers.append(self._get_single_mapper(idx, ele))

            def mapper_func(x):
                y = np.zeros(x.shape)
                for idx in range(x.shape[1]):
                    y[:, idx] = mappers[idx](x[:, idx])
                return y

            self._mapper = mapper_func

        return self._mapper

    def _get_single_mapper(self, idx, ele):
        """mapping using isotonic regression
        """

        signal = self.intensities_list[idx]
        domain = signal.domain
        # TODO signal has bounds_error currently
        lower, upper = self.domain_bounds[idx]
        # append start and end
        # if lower < domain.start:
        #     domain = domain.append(lower, left=True)
        # if upper > domain.end:
        #     domain = domain.append(upper, left=False)

        y = signal.magnitude
        x = domain.magnitude

        isoreg = IsotonicRegression(
            y_min=self.bounds[0][idx],
            y_max=self.bounds[1][idx],
            increasing=ele.zero_is_lower
        )

        new_y = isoreg.fit_transform(x, y)
        return interp1d(
            new_y, x,
            bounds_error=False,
            fill_value=(lower, upper)
        )

    @property
    def domain_units(self):
        # TODO
        units = [ele.labels.units for ele in self]
        if len(set(units)) > 1:
            raise DreyeError('Input units do not match.')
        return units[0]

    def post_mapping(self, x, units=True, **kwargs):
        """
        """

        if units:
            units = self.domain_units
        else:
            units = 1

        return np.clip(
            x,
            a_min=self.lower_boundary[None, :],
            a_max=self.upper_boundary[None, :]
        ) * units

    def pre_mapping(self, values):
        min = np.atleast_2d(self.bounds[0])
        max = np.atleast_2d(self.bounds[1])

        truth = np.all(np.atleast_2d(values) >= min)
        truth &= np.all(np.atleast_2d(values) <= max)
        assert truth, 'some values to be mapped are out of bounds.'

        return values

    @property
    def intensities_list(self):
        if self._intensities_list is None:
            self._intensities_list = [
                Signal(
                    ele.integral,
                    domain=ele.labels,
                    labels=ele.name,
                )
                for ele in self
            ]

        return self._intensities_list

    @property
    def intensities(self):
        # will only work if domain and values have same units
        if self._intensities is None:
            signal = self.intensities_list[0]
            if len(self.intensities_list) == 1:
                signal = signal._expand_dims(1)
            else:
                for ele in self.intensities_list[1:]:
                    signal = signal.concat(ele)
            self._intensities = signal
        return self._intensities

    @property
    def zero_is_lower(self):
        if self._zero_is_lower is None:
            self._zero_is_lower = np.array([
                ele.zero_is_lower for ele in self
            ])
        return self._zero_is_lower

    @property
    def zero_boundary(self):
        if self._zero_boundary is None:
            self._zero_boundary = np.array([
                ele.zero_boundary
                if ele is not None
                else np.nan
                for ele in self
            ])

        return self._zero_boundary

    @property
    def max_boundary(self):
        if self._max_boundary is None:
            self._max_boundary = np.array([
                ele.max_boundary
                if ele is not None
                else np.nan
                for ele in self
            ])

        return self._max_boundary

    @property
    def starts(self):
        return np.array([
            ele.labels.start for ele in self
        ])

    @property
    def ends(self):
        return np.array([
            ele.labels.end for ele in self
        ])

    @property
    def bounds(self):

        bounds = np.array([
            [np.min(ele.magnitude), np.max(ele.magnitude)]
            for ele in self.intensities_list
        ])

        bounds[~np.isnan(self.zero_boundary), 0] = 0

        return tuple(bounds.T)

    @property
    def domain_bounds(self):
        return np.array([self.lower_boundary, self.upper_boundary]).T

    @property
    def lower_boundary(self):
        lower_boundary = np.zeros(self.zero_is_lower.shape)
        lower_boundary[self.zero_is_lower] = \
            self.zero_boundary[self.zero_is_lower]
        lower_boundary[~self.zero_is_lower] = \
            self.max_boundary[~self.zero_is_lower]
        lower_boundary[np.isnan(lower_boundary)] = self.starts[
            np.isnan(lower_boundary)]
        return lower_boundary

    @property
    def upper_boundary(self):
        upper_boundary = np.zeros(self.zero_is_lower.shape)
        upper_boundary[~self.zero_is_lower] = \
            self.zero_boundary[~self.zero_is_lower]
        upper_boundary[self.zero_is_lower] = \
            self.max_boundary[self.zero_is_lower]
        upper_boundary[np.isnan(upper_boundary)] = self.ends[
            np.isnan(upper_boundary)]
        return upper_boundary

    @property
    def normalized_spectrum_list(self):
        if self._normalized_spectrum_list is None:
            ele_list = []
            for ele in self:
                ele = ele.mean(axis=ele.other_axis)
                ele_list.append(
                    ele.normalized_signal
                )
            self._normalized_spectrum_list = ele_list
        return self._normalized_spectrum_list

    def __getitem__(self, key):
        return self._measured_spectra[key]

    @property
    def normalized_spectrum(self):
        if self._normalized_spectrum is None:
            signal = self.normalized_spectrum_list[0]
            if len(self.normalized_spectrum_list) == 1:
                signal = signal._expand_dims(1)
            else:
                for ele in self.normalized_spectrum_list[1:]:
                    signal = signal.concat(ele)
            self._normalized_spectrum = signal
        return self._normalized_spectrum

    @property
    def wavelengths(self):
        return self.normalized_spectrum.domain

    def __iter__(self):
        return iter(self._measured_spectra)

    def len(self):
        return len(self._measured_spectra)

    def append(self, value):
        self._measured_spectra.append(value)
        self._check_list()
        self._init_attrs()

    def extend(self, value):
        self._measured_spectra.extend(value)
        self._check_list()
        self._init_attrs()

    def pop(self, index):
        self._measured_spectra.pop(index)
        self._check_list()
        self._init_attrs()

    def popkey(self, key):
        index = self.names.index(key)
        self.pop(index)

    @property
    def names(self):
        return [ele.name for ele in self]

    def _check_list(self):
        if not isinstance(self._measured_spectra, list):
            raise DreyeError(
                'measured_spectra must be a list '
                'of MeasuredSpectrum instances.')
        if not all(
            isinstance(ele, MeasuredSpectrum)
            for ele in self._measured_spectra
        ):
            raise DreyeError(
                'not all elements of measured_spectra are '
                'of a MeasuredSpectrum instance.'
            )

    @property
    def uE(self):
        return self.__class__(
            [ele.uE for ele in self]
        )

    @property
    def irradiance(self):
        return self.__class__(
            [ele.irradiance for ele in self]
        )

    @property
    def photonflux(self):
        return self.__class__(
            [ele.photonflux for ele in self]
        )

    def fit(self, spectrum, return_res=False, return_fit=False, units=True):
        """
        """

        assert isinstance(spectrum, Spectrum)
        assert spectrum.ndim == 1

        spectrum = spectrum.copy()
        spectrum.units = self.units

        spectrum, normalized_sources = spectrum.equalize_domains(
            self.normalized_spectrum, equalize_dimensions=False)

        b = asarray(spectrum)
        A = asarray(normalized_sources)

        res = lsq_linear(A, b, bounds=self.bounds)

        # Class which incorportates the following
        # values=res.x, units=self.units, axis0_labels=self.labels

        if units:
            weights = res.x * self.units * ureg('nm')
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
