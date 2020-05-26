"""
Defines the class for implementing continuous signals
"""

import warnings
from abc import abstractmethod

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from dreye.utilities import (
    is_numeric, has_units,
    is_listlike, asarray, get_value,
    is_callable, is_dictlike, optional_to,
    is_hashable
)
from dreye.err import DreyeError
from dreye.constants import DEFAULT_FLOAT_DTYPE
from dreye.core.abstract import _UnitArray
from dreye.core.domain import Domain
from dreye.algebra.filtering import Filter1D
from dreye.core.plotting_mixin import _PlottingMixin
from dreye.core.numpy_mixin import _NumpyMixin


class _SignalMixin(_UnitArray, _PlottingMixin, _NumpyMixin):

    _init_args = (
        'domain',
        'interpolator', 'interpolator_kwargs',
        'contexts', 'attrs',
        'domain_min', 'domain_max',
        'name',
        'smoothing_method', 'smoothin_window', 'smoothing_args'
    )
    _domain_class = Domain
    # just accepts self
    _args_defaults = {
        'interpolator_kwargs': {},
        'interpolator': interp1d,
        'smoothing_args': {},
        'smoothing_window': 1,
        'smoothing_method': 'savgol'
    }
    # dictionary of attributes defaults if None
    _unit_conversion_params = {
        'domain': '_domain_values'
    }
    _convert_attributes = ('signal_min', 'signal_max')
    # attributes mapping passed to the "to" method of pint

    def __init__(
        self,
        values,
        domain=None,
        *,
        units=None,
        domain_units=None,
        interpolator=None,
        interpolator_kwargs=None,
        smoothing_method=None,
        smoothing_window=None,
        smoothing_args=None,
        contexts=None,
        domain_kwargs=None,
        domain_min=None,
        domain_max=None,
        attrs=None,
        name=None,
        **kwargs
    ):

        if isinstance(values, _SignalMixin) and domain is not None:
            if values.domain != domain:
                values = values(domain)

        super().__init__(
            values=values,
            units=units,
            domain=domain,
            domain_units=domain_units,
            interpolator=interpolator,
            interpolator_kwargs=interpolator_kwargs,
            smoothing_window=smoothing_window,
            smoothing_method=smoothing_method,
            smoothing_args=smoothing_args,
            contexts=contexts,
            domain_kwargs=domain_kwargs,
            domain_min=domain_min,
            domain_max=domain_max,
            attrs=attrs,
            name=name,
            **kwargs
        )

        # applied window is added after smoothing
        if 'applied_window_' not in self.attrs:
            self.attrs['applied_window_'] = 'raw'

    @abstractmethod
    def _preextract_attributes(self, values, kwargs):
        pass

    @abstractmethod
    def _preprocess_check_values(self, values):
        pass

    @abstractmethod
    def _process_other_attributes(self, values, kwargs):
        pass

    @abstractmethod
    def _get_signal_bound(self, values, bound):
        pass

    def _test_and_assign_values(self, values, **kwargs):
        """
        unpacking values
        """
        self._preextract_attributes(values, kwargs)

        if not is_listlike(values):
            raise DreyeError("Values must be array-like, but are "
                             f"of type '{type(values)}'.")

        values = asarray(values, DEFAULT_FLOAT_DTYPE)
        # check values dimensionality
        values = self._preprocess_check_values(values)

        # check other attributes
        assert is_callable(self.interpolator)
        assert is_dictlike(self.interpolator_kwargs)

        # check domain
        self._domain = self._get_domain(
            values, self._domain,
            kwargs.get('domain_units', None),
            kwargs.get('domain_kwargs', None),
            self.domain_axis,
            self._domain_class
        )
        # check domain and signal min and max
        self._domain_min = self._get_domain_bound(
            self._domain_min, self.domain)
        self._domain_max = self._get_domain_bound(
            self._domain_max, self.domain)
        self._check_domain_bounds(
            self.domain.magnitude,
            domain_min=self.domain_min.magnitude,
            domain_max=self.domain_max.magnitude
        )

        # check labels
        self._process_other_attributes(values, kwargs)
        self._signal_min = self._get_signal_bound(
            values, kwargs.get('signal_min', None))
        self._signal_max = self._get_signal_bound(
            values, kwargs.get('signal_max', None))

        # clip signal using signal min and max
        values = self._process_values(values.copy())
        self._values = values
        self._interpolate = None

    def _process_values(self, values):
        """
        clip values inplace using signal_min and signal_max
        """
        not_nan = ~np.isnan(self.signal_min)
        if np.any(not_nan):
            values[..., not_nan] = np.maximum(
                values[..., not_nan],
                self.signal_min[not_nan]
            )
        not_nan = ~np.isnan(self.signal_max)
        if np.any(not_nan):
            values[..., not_nan] = np.minimum(
                values[..., not_nan],
                self.signal_max[not_nan]
            )
        return values

    @staticmethod
    def _get_domain_bound(bound, domain):
        """
        used by _test_and_assign_values
        """
        if bound is None:
            value = np.nan * domain.units
        elif not is_numeric(bound):
            raise DreyeError(
                f"Domain bound variable must be numeric, but "
                f"is of type '{type(bound)}'."
            )
        elif has_units(bound):
            value = bound.to(domain.units)
        else:
            value = bound * domain.units

        return value

    @staticmethod
    def _get_domain(
        values, domain, domain_units,
        domain_kwargs, domain_axis, domain_class
    ):
        """
        used by _test_and_assign_values
        """

        if domain is None:
            domain = np.arange(values.shape[domain_axis])

        if not isinstance(domain, domain_class):
            domain = domain_class(
                domain,
                units=domain_units,
                **(
                    {} if domain_kwargs is None
                    else domain_kwargs
                )
            )

        assert len(domain) == values.shape[domain_axis], (
            "Domain must be same length as domain axis."
        )

        return domain

    @property
    def domain_min(self):
        """
        Returns the minimum value in domain.
        """
        return self._domain_min.to(self.domain.units)

    @domain_min.setter
    def domain_min(self, value):
        domain_min = self._get_domain_bound(value, self.domain)
        self._check_domain_bounds(
            self.domain.magnitude,
            domain_min=domain_min.magnitude,
            domain_max=self.domain_max.magnitude
        )
        self._domain_min = domain_min

    @property
    def domain_max(self):
        """
        Returns the maximum value in domain.
        """
        return self._domain_max.to(self.domain.units)

    @domain_max.setter
    def domain_max(self, value):
        domain_max = self._get_domain_bound(value, self.domain)
        self._check_domain_bounds(
            self.domain.magnitude,
            domain_max=domain_max.magnitude,
            domain_min=self.domain_min.magnitude
        )
        self._domain_max = domain_max

    @property
    def signal_min(self):
        """
        Returns the minimum value in signal, to which all lower values are
        clipped to.
        """
        return self._signal_min

    @signal_min.setter
    def signal_min(self, value):
        self._signal_min = self._get_signal_bound(self.magnitude, value)
        self._values = self._process_values(self.magnitude.copy())

    @property
    def signal_max(self):
        """
        Returns the maximum value in signal, to which all lower values are
        clipped to.
        """
        return self._signal_max

    @signal_max.setter
    def signal_max(self, value):
        self._signal_max = self._get_signal_bound(self.magnitude, value)
        self._values = self._process_values(self.magnitude.copy())

    @property
    def domain(self):
        """
        Domain object associated with signal.
        """

        return self._domain

    @property
    def domain_units(self):
        return self.domain.units

    @domain.setter
    def domain(self, value):
        """
        Setting a new domain
        """

        if isinstance(value, self._domain_class):
            self._domain = self._get_domain(
                self.magnitude,
                value,
                self.domain_axis,
                self._domain_class
            )
        else:
            self._domain = self._get_domain(
                self.magnitude,
                value,
                self.domain.units,
                self.domain.init_kwargs,
                self.domain_axis,
                self._domain_class
            )

    @property
    def _domain_values(self):
        """
        domain values for unit conversion properly broadcasted
        """
        return self.domain.values[..., None]

    @property
    def interpolator(self):
        """
        Returns the interpolator that was selected for use.
        """

        return self._interpolator

    @interpolator.setter
    def interpolator(self, value):
        """
        """

        if value is None:
            self._interpolator = interp1d
        elif is_callable(value):
            self._interpolator = value
            self._interpolate = None
        else:
            raise TypeError('interpolator needs to be a callable.')

    @property
    def interpolator_kwargs(self):
        """
        Returns the previously specified dictionary containing arguments to
        be passed to the interpolator function.
        """
        # always makes sure that integration occurs along the right axis
        self._interpolator_kwargs['axis'] = self.domain_axis
        return self._interpolator_kwargs

    @interpolator_kwargs.setter
    def interpolator_kwargs(self, value):
        """
        """

        if value is None:
            self._interpolator_kwargs = {}
        elif isinstance(value, dict):
            self._interpolator_kwargs = value
            self._interpolate = None
        else:
            raise TypeError('interpolator_kwargs must be dict.')

    @property
    def interpolate(self):
        """
        interpolate callable
        """

        if self._interpolate is None:

            interp = self.interpolator(
                self.domain.magnitude,
                self.magnitude,
                **self.interpolator_kwargs
            )

            def clip_wrapper(*args, **kwargs):
                interpolated = interp(*args, **kwargs)
                return self._process_values(interpolated)

            self._interpolate = clip_wrapper

        return self._interpolate

    @property
    def domain_axis(self):
        """
        Axis that correspond to the domain (i.e. 0).
        """
        return 0

    @property
    def boundaries(self):
        """
        Tuple the minimum and maximum values along zeroth axis.
        """

        return (
            np.min(self.magnitude, axis=self.domain_axis),
            np.max(self.magnitude, axis=self.domain_axis)
        )

    @property
    def span(self):
        """
        Returns the span along the zeroth axis.
        """
        return (
            np.max(self.magnitude, axis=self.domain_axis)
            - np.min(self.magnitude, axis=self.domain_axis)
        )

    def __call__(self, domain):
        """
        interpolate to new domain
        """

        if isinstance(domain, type(self.domain)):
            if domain == self.domain:
                return self.copy()

        domain_units = self.domain.units
        if has_units(domain):
            domain_units = domain.units
            domain_values = domain.to(domain_units).magnitude
        else:
            domain_values = asarray(domain)

        assert domain_values.ndim == 1

        # check domain min and max (must be bigger than this range)
        self._check_domain_bounds(
            domain_values,
            domain_min=self.domain_min.magnitude,
            domain_max=self.domain_max.magnitude
        )

        values = self.interpolate(domain_values)

        # for single value simply return quantity instance
        if values.ndim != self.ndim:
            return values * self.units
        else:
            new = self.copy()
            new._values = values
            new._domain = self._domain_class(
                domain, units=domain_units,
                # need to supply interval if size 1
                interval=(
                    self.domain.interval
                    if domain_values.size == 1
                    else None
                ),
                **self.domain.init_kwargs
            )
            return self

    @staticmethod
    def _check_domain_bounds(domain_values, domain_min, domain_max):
        # check domain min and max (must be bigger than this range)
        if np.min(domain_values) > domain_min:
            raise DreyeError("Interpolation domain above domain minimum.")
        if np.max(domain_values) < domain_max:
            raise DreyeError("Interpolation domain below domain maximum.")

    @property
    def integral(self):
        """
        Returns the integral for each signal.
        """
        return np.trapz(
            self.magnitude,
            self.domain.magnitude,
            axis=self.domain_axis
        ) * self.units * self.domain.units

    @property
    def normalized_signal(self):
        """
        Returns the signal divided by the integral. Integrates to 1.
        """
        return self / self.integral[None]

    @property
    def piecewise_integral(self):
        """
        Returns the calculated integral at each point using the trapezoidal
        area method.
        """
        return self * self.domain.gradient

    @property
    def piecewise_gradient(self):
        """
        Returns the instantanous gradient at each point in signal.
        """
        return self / self.domain.gradient

    @property
    def gradient(self):
        """
        Returns the gradient.
        """
        values = np.gradient(
            self.magnitude, self.domain.magnitude, axis=self.domain_axis
        )
        units = self.units / self.domain.units
        return self._class_new_instance(
            values=values, units=units, **self.init_kwargs
        )

    def enforce_uniformity(self, method=np.mean, on_gradient=True):
        """
        Returns the domain with a uniform interval, calculated from the
        average of all original interval values.
        """
        domain = self.domain.enforce_uniformity(
            method=method, on_gradient=on_gradient
        )
        return self(domain)

    def window_filter(
        self, domain_interval,
        method='savgol', extrapolate=False,
        **method_args
    ):
        """
        Filters signal instance using filter1d, which uses the savgol method.
        """

        assert self.domain.is_uniform, (
            "signal domain must be uniform for filtering"
        )

        M = domain_interval/self.domain.interval
        if M % 1 != 0:
            warnings.warn(
                "Chosen domain interval must be rounded down for filtering",
                RuntimeWarning
            )
        M = int(M)

        if method == 'savgol':

            method_args['polyorder'] = method_args.get('polyorder', 2)
            method_args['axis'] = self.domain_axis
            M = M + ((M+1) % 2)
            values = savgol_filter(self.magnitude, M, **method_args)

        elif extrapolate:
            # create filter instance
            filter1d = Filter1D(method, M, **method_args)
            # handle borders by interpolating
            start_idx, end_idx = int(np.floor((M-1)/2)), int(np.ceil((M-1)/2))
            # create new domain
            new_domain = self.domain.extend(
                start_idx, left=True
            ).extend(
                end_idx, left=False
            ).magnitude

            values = filter1d(
                self(new_domain).magnitude,
                axis=self.domain_axis,
                mode='valid'
            )

        else:
            # create filter instance
            filter1d = Filter1D(method, M, **method_args)
            # from function
            values = filter1d(
                self.magnitude,
                axis=self.domain_axis,
                mode='same'
            )

        new = self.copy()
        values = new._process_values(values)
        new._values = values
        return new

    @property
    def smoothing_args(self):
        return self._smoothing_args

    @property
    def smoothing_window(self):
        return self._smoothing_window

    @property
    def smoothing_method(self):
        return self._smoothing_method

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
    def _other_shape(self):
        return self._get_other_shape(self.shape)

    def _get_other_shape(self, shape):
        shape = list(shape)
        shape.pop(self.domain_axis)
        return shape

    def domain_concat(self, other, left=False):
        """
        Creates a new signal instance by appending two signals along the domain
        axis.
        """

        domain = self.domain
        self_values = self.magnitude

        if isinstance(other, _SignalMixin):
            # checks dimensionality, appends domain, converts units
            assert self._other_shape == other._other_shape

            # concatenate domain
            domain = domain.append(other.domain, left=left)
            # convert units
            other_values = other.to(self.units).magnitude

        elif is_listlike(other):
            # handles units, checks other shape, extends domain
            other_values = optional_to(
                other, self.units, *self.contexts,
                **self._unit_conversion_kws)

            assert (
                self._other_shape
                == self._get_other_shape(other_values.shape)
            ), 'shapes do not match for concatenation'

            domain = domain.extend(
                other_values.shape[self.domain_axis],
                left=left
            )

        else:
            raise DreyeError("Domain axis contenation "
                             f"with type: {type(other)}")

        if left:
            values = np.concatenate(
                [other_values, self_values],
                axis=self.domain_axis
            )

        else:
            values = np.concatenate(
                [self_values, other_values],
                axis=self.domain_axis
            )

        values = self._process_values(values)
        new = self.copy()
        new._values = values
        new._domain = domain
        return new

    def append(self, other, *args, **kwargs):
        """
        Append signals.
        """

        return self.domain_concat(other, *args, **kwargs)

    def equalize_domains(self, other):
        """
        equalize domains for both Signal instances
        """
        if self.domain != other.domain:
            domain = self.domain.equalize_domains(other.domain)
            return self(domain), other(domain)
        return self, other

    # TODO peak detection (min/max)
    # TODO peak summary - FWHM, HWHM-left, HWHM-right, domain value
    # TODO rolling window single D signal


class _Signal2DMixin(_SignalMixin):
    _init_args = _SignalMixin._init_args + ('labels',)

    def __init__(
        self,
        values,
        domain=None,
        labels=None,
        *,
        units=None,
        domain_units=None,
        interpolator=None,
        interpolator_kwargs=None,
        contexts=None,
        domain_kwargs=None,
        domain_min=None,
        domain_max=None,
        signal_min=None,
        signal_max=None,
        attrs=None,
        name=None,
        **kwargs
    ):
        super().__init__(
            values=values,
            units=units,
            domain=domain,
            domain_units=domain_units,
            labels=labels,
            interpolator=interpolator,
            interpolator_kwargs=interpolator_kwargs,
            contexts=contexts,
            domain_kwargs=domain_kwargs,
            domain_min=domain_min,
            domain_max=domain_max,
            signal_min=signal_min,
            signal_max=signal_max,
            attrs=attrs,
            name=name,
            **kwargs
        )

    @abstractmethod
    def _get_labels(self, values, labels, kwargs):
        pass

    @abstractmethod
    def labels_concat(self, other):
        pass

    def _preextract_attributes(self, values, kwargs):
        if isinstance(values, (pd.DataFrame, pd.Series)):
            self._extract_attr_from_pandas(values)

    def _extract_attr_from_pandas(self, values):
        """
        Extract various attributes from a pandas instance if passed
        attributes is None. Used by _test_and_assign_values
        """

        if self.domain is None:
            self._domain = values.index

        if self.labels is None and not isinstance(values, pd.Series):
            self._labels = values.columns

        if self.name is None:
            self.name = values.name

    def _preprocess_check_values(self, values):
        # must return values processed
        if values.ndim == 1:
            values = values[:, None]
        elif values.ndim != 2:
            raise DreyeError("Values must be a one- or two-dimensional array.")
        return values

    def _process_other_attributes(self, values, kwargs):
        # check labels
        self._labels = self._get_labels(values, self._labels, kwargs)

    @property
    def labels(self):
        """
        Returns signal labels, or the "name" for each signal.
        """
        return self._labels

    @labels.setter
    def labels(self, value):
        """
        Set new labels
        """
        self._labels = self._get_labels(self.magnitude, value)

    @property
    def labels_axis(self):
        """
        Axis that correspond to the label dimension (i.e. -1).
        """
        return -1

    @property
    def iterlabels(self):
        """
        iterate over labels
        """
        iter(np.moveaxis(self.magnitude, self.labels_axis, 0) * self.units)

    @property
    def nanless(self):
        """
        Returns signal with NaNs removed.
        """
        arange = self.domain.magnitude
        values = np.zeros(self.shape)

        for idx, ivalues in enumerate(self.iterlabels):
            iarr = ivalues.magnitude
            finites = np.isfinite(iarr)
            # interpolate nans
            ivalues = self.interpolator(
                arange[finites], iarr[finites],
                **self.interpolator_kwargs
            )(arange)
            ivalues[finites] = iarr[finites]
            values[..., idx] = ivalues

        new = self.copy()
        new._values = values
        return new

    def __str__(self):
        return (
            f"{type(self).__name__}"
            f"(\n\t name={self.name}, \n\t "
            f"labels={self.labels}, \n\t units={self.units}, \n\t "
            f"domain={self.domain} \n )"
        )

    def concat(self, other, *args, **kwargs):
        """
        Concatenate two signals.
        """

        return self.labels_concat(other, *args, **kwargs)

    def to_frame(self):
        """
        Convert signal class to dataframe/series.
        Units for the signal and domain will be lost.
        """
        df = pd.DataFrame(
            self.magnitude,
            index=self.domain.magnitude,
            columns=list(get_value(self.labels))
        )
        df.columns.name = 'labels'
        df.index.name = 'domain'
        return df

    def to_longframe(self):
        """
        Convert signal class to a long dataframe,
        which will also contain the attributes units,
        domain_units, name, domain_min, domain_max, signal_min,
        signal_max, and all the keys in the attrs dictionary.
        It also includes the dimensionality of the units.
        It will also preserve multiindex values for labels.
        """

        df = self.to_frame()
        while isinstance(df, pd.DataFrame):
            df = df.stack()

        df = df.reset_index()

        df.rename(columns={0: 'values'}, inplace=True)

        df['name'] = self.name
        df['units'] = str(self.units)
        df['units_dimensionality'] = str(self.units.dimensionality)
        df['domain_units'] = str(self.domain.units)
        df['domain_units_dimensionality'] = str(
            self.domain.units.dimensionality
        )
        df['domain_min'] = self.domain_min.magnitude
        df['domain_max'] = self.domain_max.magnitude
        # merge bounds
        df_bounds = pd.DataFrame(columns=['signal_min', 'signal_max'])
        df_bounds['labels'] = list(get_value(self.labels))
        df_bounds['signal_min'] = self.signal_min
        df_bounds['signal_max'] = self.signal_max
        df = df.merge(df_bounds, on='labels', how='left')

        # if labels are multiindex include them
        if isinstance(self.labels, pd.MultiIndex):
            df_labels = self.labels.to_frame(index=False)
            df_labels['labels'] = list(get_value(self.labels))
            df = df.merge(
                df_labels, on='labels', how='left',
                suffixes=('', '_label')
            )

        reserved_keys = df.columns

        for key, ele in self.attrs.items():
            if key in reserved_keys:
                warnings.warn(
                    f"Cannot use key '{key}' from "
                    "dictionary 'attrs' as column for long dataframe, "
                    f"as it is reserved; using '{key+'_attr'}'"
                )
                key = key + '_attr'
                if key in reserved_keys:
                    raise DreyeError(
                        "Cannot convert to long dataframe since "
                        f"dictionary 'attrs' contains key '{key}', which"
                        " is reserved."
                    )
            df[key] = get_value(ele)

        return df

    def _equalize(self, other):
        """
        Should just return equalized other_magnitude or NotImplemented
        """
        if isinstance(other, _UnitArray):
            if isinstance(other, Domain):
                return other.magnitude[..., None]
            elif isinstance(other, _Signal2DMixin):
                if (
                    self.shape[self.labels_axis] == 1
                    and other.shape[other.labels_axis] > 1
                ):
                    # do the reverse
                    return NotImplemented
                self, other = self.equalize_domains(other)
                return other.magnitude
            else:
                return NotImplemented
        elif is_numeric(other):
            return get_value(other)
        elif is_listlike(other):
            other = asarray(other)
            if other.ndim == 1:
                return other[:, None]
            elif other.ndim == 2:
                return other
            else:
                return NotImplemented
        else:
            return NotImplemented


class _SignalIndexLabels(_Signal2DMixin):

    def _get_labels(self, values, labels, kwargs={}):
        if (
            is_listlike(labels)
            and len(labels) == 1
            and values.shape[self.labels_axis] != 1
        ):
            warnings.warn("Lost labels during operation", RuntimeWarning)
            labels = None
        if isinstance(labels, pd.Index):
            pass
        elif labels is None:
            labels = pd.Index(np.arange(values.shape[self.labels_axis]))
        elif is_listlike(labels):
            labels = pd.Index(labels)
        elif is_hashable(labels):
            labels = pd.Index([labels]*values.shape[self.labels_axis])
        else:
            raise DreyeError(f"Labels wrong type: '{type(labels)}'")

        assert len(labels) == values.shape[self.labels_axis], (
            "Labels must be same length as label axis"
        )

        return labels

    def _get_signal_bound(self, values, bound):

        if bound is None:
            bound = np.nan * np.ones(values.shape[self.labels_axis])
        else:
            bound = optional_to(
                bound, self.units,
                *self.contexts, **self._unit_conversion_kws
            )
            if is_numeric(bound):
                bound = np.ones(values.shape[self.labels_axis]) * bound

        assert len(bound) == values.shape(self.labels_axis), (
            "Bounds size must match label axis."
        )

        return bound

    def _concat_labels(self, labels, left=False):
        """
        Concatenate labels of two signal instances.
        """
        if left:
            return labels.append(self.labels)
        else:
            return self.labels.append(labels)

    def _concat_signal_bounds(self, signal_min, signal_max, left=False):
        """
        Concatenate signal bound
        """
        if left:
            return (
                np.concatenate([signal_min, self.signal_min]),
                np.concatenate([signal_max, self.signal_max])
            )
        else:
            return (
                np.concatenate([self.signal_min, signal_min]),
                np.concatenate([self.signal_max, signal_max])
            )

    def labels_concat(
        self, other, labels=None,
        signal_min=None, signal_max=None,
        left=False
    ):
        """
        Create a new signal instance by concatenating two existing signal
        instances. If domains are not equivalent, interpolate if possible and
        enforce the same domain range by using the equalize_domains function.
        """

        if isinstance(other, _SignalIndexLabels):
            # equalizing domains
            self, signal = self.equalize_domains(other)
            self_values = self.magnitude
            # convert units - will also convert signal bounds
            other = other.to(self.units)
            other_values = other.magnitude
            # labels
            if labels is None:
                labels = other.labels
            if signal_min is None:
                signal_min = other.signal_min
            if signal_max is None:
                signal_max = other.signal_max

        elif is_listlike(signal):
            # self numpy array
            self_values = self.magnitude
            # check if it has units
            other_values = optional_to(
                other, self.units, *self.contexts,
                **self._unit_conversion_kws)
            # handle labels
            labels = self._get_labels(other_values, labels)
            signal_min = self._get_signal_bound(other_values, signal_min)
            signal_max = self._get_signal_bound(other_values, signal_max)

        else:
            raise DreyeError(
                f"other axis contenation with type: {type(signal)}.")

        # handle labels and bounds
        labels = self._concat_labels(labels, left)
        signal_min, signal_max = self._concat_signal_bounds(
            signal_min, signal_max, left
        )

        if left:
            values = np.concatenate(
                [other_values, self_values],
                axis=self.labels_axis
            )

        else:
            values = np.concatenate(
                [self_values, other_values],
                axis=self.labels_axis
            )

        new = self.copy()
        new._values = values
        # values dependent
        new.labels = labels
        new.signal_min = signal_min
        new.signal_max = signal_max
        return new


class _SignalDomainLabels(_Signal2DMixin):
    _init_args = _Signal2DMixin._init_args + ('labels_min', 'labels_max')
    _domain_labels_class = Domain

    def __init__(
        self,
        values,
        domain=None,
        labels=None,
        *,
        units=None,
        domain_units=None,
        labels_units=None,
        interpolator=None,
        interpolator_kwargs=None,
        contexts=None,
        domain_kwargs=None,
        labels_kwargs=None,
        domain_min=None,
        domain_max=None,
        labels_min=None,
        labels_max=None,
        signal_min=None,
        signal_max=None,
        attrs=None,
        name=None,
        **kwargs
    ):

        if isinstance(values, _SignalDomainLabels) and labels is not None:
            if values.labels != labels:
                values = values.T(labels).T

        super().__init__(
            values=values,
            units=units,
            domain=domain,
            domain_units=domain_units,
            labels=labels,
            labels_units=labels_units,
            interpolator=interpolator,
            interpolator_kwargs=interpolator_kwargs,
            contexts=contexts,
            domain_kwargs=domain_kwargs,
            labels_kwargs=labels_kwargs,
            domain_min=domain_min,
            labels_min=labels_min,
            domain_max=domain_max,
            labels_max=labels_max,
            signal_min=signal_min,
            signal_max=signal_max,
            attrs=attrs,
            name=name,
            **kwargs
        )

    def equalize_domains(self, other):
        """
        equalize domains for both Signal instances
        """
        self, other = super().equalize_domains(other)
        if isinstance(other, _SignalDomainLabels):
            if self.labels != other.labels:
                labels = self.labels.equalize_domains(other.labels)
                return self.T(labels).T, other.T(labels).T
        return self, other

    @property
    def labels_units(self):
        return self.labels.units

    @property
    def T(self):
        """
        Transpose Array.
        """
        return self._class_new_instance(
            values=self.magnitude.T,
            units=self.units,
            **{
                **self.init_kwargs,
                **dict(
                    labels=self.domain,
                    domain=self.labels,
                    domain_min=self.labels_min,
                    domain_max=self.labels_max,
                    labels_min=self.domain_min,
                    labels_max=self.domain_max
                )
            }
        )

    def labels_concat(self, other, left=False):
        """
        Concatenate Labels
        """
        return self.T.domain_concat(other, left=left).T

    def _get_signal_bound(self, values, bound):

        if bound is None:
            bound = np.array(np.nan)
        else:
            bound = optional_to(
                bound, self.units,
                *self.contexts, **self._unit_conversion_kws
            )
            if not is_numeric(bound):
                if len(set(bound)) != 1:
                    raise DreyeError("Bound must be numeric, "
                                     f"but is of type {type(bound)}.")
                bound = bound[0]
            bound = np.array(DEFAULT_FLOAT_DTYPE(bound))

        return bound

    def _get_labels(self, values, labels, kwargs):
        labels_units = kwargs.get('labels_units', None)
        labels_kwargs = kwargs.get('labels_kwargs', None)
        if labels_units is None and hasattr(labels, 'units'):
            labels_units = labels.units
        if labels_kwargs is None and hasattr(labels, 'init_kwargs'):
            labels_kwargs = labels.init_kwargs
        return self._get_domain(
            values,
            labels,
            labels_units,
            labels_kwargs,
            self.labels_axis,
            self._domain_labels_class
        )

    def _process_other_attributes(self, values, kwargs):
        # check labels
        super()._process_other_attributes(values, kwargs)
        self._labels_min = self._get_domain_bound(
            self._labels_min, self.labels)
        self._labels_max = self._get_domain_bound(
            self._labels_max, self.labels)
        self._check_domain_bounds(
            self.labels.magnitude,
            domain_min=self.labels_min.magnitude,
            domain_max=self.labels_max.magnitude
        )
        self._labels_interpolate = None

    @property
    def labels_min(self):
        """
        Returns the minimum value in domain.
        """
        return self._labels_min.to(self.labels.units)

    @labels_min.setter
    def labels_min(self, value):
        labels_min = self._get_domain_bound(value, self.labels)
        self._check_domain_bounds(
            self.labels.magnitude,
            domain_min=labels_min.magnitude,
            domain_max=self.labels_max.magnitude
        )
        self._labels_min = labels_min

    @property
    def labels_max(self):
        """
        Returns the maximum value in labels.
        """
        return self._labels_max.to(self.labels.units)

    @labels_max.setter
    def labels_max(self, value):
        labels_max = self._get_domain_bound(value, self.labels)
        self._check_domain_bounds(
            self.labels.magnitude,
            domain_max=labels_max.magnitude,
            domain_min=self.labels_min.magnitude
        )
        self._labels_max = labels_max


class Signals(_SignalIndexLabels):
    """
    Defines the base class for continuous signals (unit-aware).

    Parameters
    ----------
    values : array-like or str
        1 or 2 dimensional array that contains the value of your signal.
    domain : domain, tuple, dict, or array-like, optional
        Can either be a domain instance- a list of numbers for domain that has
        to be equal to same length as the axis length of your values, tuple
        with start end or interval, or dictionary which can be passed directly
        to the domain class.
    labels : list-like, optional
        A list-like parameter that must be the same length as the number of
        signals. This serves as the "name" for each signal. Defaults to a
        numeric list (e.g. [1, 2, 3]) if signal is a 2D, and to none if the
        signal is a 1D array.
    units : str, optional
        Units you define. can define domain units as an extra value. if domain
        is arrray like, list of a bunch of values, haven't assigned units.
    interpolator : interpolate class, optional
        Callable function that allows you to interpolate between points. It
        accepts x and y (domain and values). As a key word arguement it has to
        include axis. Defaults to scipy.interpolate.interp1d.
    interpolator_kwargs : dict-like, optional
        Dictionary in which you can specify there are other arguments that you
        want to pass to your interpolator function.
    contexts :
        Contexts for unit conversion.
    domain_kwargs : dict-like, optional
        Dictionary that will be passed to instantiate your domain. Uses the
        previous domain class and passes it when you intialize. Only
        optional when you pass signal values. Defaults to 0.
    domain_min : int, optional
        Defines the minimum value in your domain for the intpolation range.
        Defaults to None.
    domain_max : int, optional
        Defines the minimum value in your domain for the intpolation range.
        Defaults to None.
    signal_min : int, optional
        Will clip your signal to a minimum. Everything below this minimum will
        be set to the minumum.
    signal_max : int, optional
        Will clip your signal to a maximum. Everything above this maximum will
        be set to the maximum.
    attrs :
        User defined dictionary that will keep track of anything needed for
        performing operations on the signal.
    name : str, optional
        Name of the signal instance.
    """

    # TODO work on docstring

    @property
    def _class_new_instance(self):
        return Signals

    def __init__(
        self,
        values,
        domain=None,
        labels=None,
        *,
        units=None,
        domain_units=None,
        interpolator=None,
        interpolator_kwargs=None,
        contexts=None,
        domain_kwargs=None,
        domain_min=None,
        domain_max=None,
        signal_min=None,
        signal_max=None,
        attrs=None,
        name=None,
    ):
        super().__init__(
            values=values,
            units=units,
            domain=domain,
            domain_units=domain_units,
            labels=labels,
            interpolator=interpolator,
            interpolator_kwargs=interpolator_kwargs,
            contexts=contexts,
            domain_kwargs=domain_kwargs,
            domain_min=domain_min,
            domain_max=domain_max,
            signal_min=signal_min,
            signal_max=signal_max,
            attrs=attrs,
            name=name,
        )


class DomainSignal(_SignalDomainLabels):
    """
    """
    # TODO docstring

    def __init__(
        self,
        values,
        domain=None,
        labels=None,
        *,
        units=None,
        domain_units=None,
        labels_units=None,
        interpolator=None,
        interpolator_kwargs=None,
        contexts=None,
        domain_kwargs=None,
        labels_kwargs=None,
        domain_min=None,
        domain_max=None,
        labels_min=None,
        labels_max=None,
        signal_min=None,
        signal_max=None,
        attrs=None,
        name=None,
    ):

        super().__init__(
            values=values,
            units=units,
            domain=domain,
            domain_units=domain_units,
            labels=labels,
            labels_units=labels_units,
            interpolator=interpolator,
            interpolator_kwargs=interpolator_kwargs,
            contexts=contexts,
            domain_kwargs=domain_kwargs,
            labels_kwargs=labels_kwargs,
            domain_min=domain_min,
            labels_min=labels_min,
            domain_max=domain_max,
            labels_max=labels_max,
            signal_min=signal_min,
            signal_max=signal_max,
            attrs=attrs,
            name=name,
        )

    @property
    def _class_new_instance(self):
        return DomainSignal
