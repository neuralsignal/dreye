"""
Defines the class for implementing continuous signals
"""

import warnings

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import cloudpickle
import pickle

from dreye.utilities import (
    is_numeric, has_units, is_arraylike, convert_units,
    asarray, array_equal, is_listlike, get_values, is_hashable
)
from dreye.utilities.abstract import AbstractContainer
from dreye.constants import ureg
from dreye.err import DreyeError
from dreye.io import read_json, write_json
from dreye.constants import DEFAULT_FLOAT_DTYPE
from dreye.core.abstract import AbstractDomain, AbstractSignal
from dreye.core.unpack_mixin import UnpackSignalMixin
from dreye.core.estimator_mixin import EstimatorMixin
from dreye.core.domain import Domain
from dreye.algebra.filtering import Filter1D
from dreye.core.plotting import SignalPlottingMixin


class Signal(
    AbstractSignal, UnpackSignalMixin, SignalPlottingMixin, EstimatorMixin
):
    """
    Defines the base class for continuous signals (unit-aware).

    TODO: Description

    Parameters
    ----------
    values : array-like or str
        1 or 2 dimensional array that contains the value of your signal.
    domain : domain, tuple, dict, or array-like, optional
        Can either be a domain instance- a list of numbers for domain that has
        to be equal to same length as the axis length of your values, tuple with
        start end or interval, or dictionary which can be passed directly to the
        domain class.
    domain_axis : int, optional
        Axis for which the domain is aligned. If 2D, it can be 0 or 1. If 0,
        this is the axis along which you have your values data aligned with your
        domain.
    units : str, optional
        Units you define. can define domain units as an extra value. if domain
        is arrray like, list of a bunch of values, haven't assigned units.
    labels : list-like, optional
        A list-like parameter that must be the same length as the number of
        signals. This serves as the "name" for each signal. Defaults to a
        numeric list (e.g. [1, 2, 3]) if signal is a 2D, and to none if the
        signal is a 1D array.
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

    _init_args = (
        'domain', 'interpolator', 'interpolator_kwargs', 'dtype',
        'domain_axis', 'labels', 'contexts', 'attrs',
        'domain_min', 'domain_max', 'signal_min', 'signal_max',
        'name'
    )
    domain_class = Domain
    _label_class = pd.Index
    _force_labels = True
    _convert_attributes = ('signal_min', 'signal_max')

    @property
    def _class_new_instance(self):
        """
        """
        return Signal

    def __init__(
        self,
        values,
        domain=None,
        domain_axis=None,
        units=None,
        domain_units=None,
        labels=None,
        dtype=DEFAULT_FLOAT_DTYPE,
        domain_dtype=DEFAULT_FLOAT_DTYPE,
        interpolator=None,
        interpolator_kwargs=None,
        contexts=None,
        domain_kwargs=None,
        domain_min=None,
        domain_max=None,
        signal_min=None,
        signal_max=None,
        attrs=None,
        name=None
    ):

        values, dtype, container = self._unpack(
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
        self._dtype = dtype
        self._values = values
        self._units = container['units']

        for key, value in container.items():
            if key in self._init_args:
                setattr(self, '_' + key, value)

        # set interpolate to None
        self._interpolate = None
        # check signal min and max values
        self._check_clip_value(self._signal_min)
        self._check_clip_value(self._signal_max)

    @property
    def domain_min(self):
    """
    Returns the minimum value in domain.
    """
        return self._domain_min

    @property
    def domain_max(self):
    """
    Returns the maximum value in domain.
    """
        return self._domain_max

    @property
    def signal_min(self):
    """
    Returns the minimum value in signal, to which all lower values are clipped
    to.
    """
        return self._signal_min

    @property
    def signal_max(self):
    """
    Returns the maximum value in signal, to which all lower values are clipped
    to.
    """
        return self._signal_max

    @property
    def attrs(self):
    """
    Returns the previously defined dictionary created for performing more
    specific operations on the signal.
    """
        if self._attrs is None:
            self._attrs = {}
        return self._attrs

    @property
    def name(self):
    """
    Returns the name of the signal instance.
    """
        return self._name

    @name.setter
    def name(self, value):
        if not is_hashable(value):
            raise DreyeError(
                f'New name value of type {type(value)} is not hashable.')
        self._name = value

    def to_dict(self, add_pickled_class=True):
        dictionary = {
            'values': self.magnitude,
            'units': self.units,
            **self.init_kwargs
        }
        if add_pickled_class:
            dictionary['pickled_class'] = str(
                cloudpickle.dumps(self.__class__)
            )
        return dictionary

    def to_frame(self):
        """
        Convert signal class to dataframe/series.
        Units for the signal and domain will be lost.
        """
        if self.ndim == 1:
            df = pd.Series(
                self.magnitude,
                index=self.domain.magnitude,
            )
            df.index.name = 'domain'
        elif self.domain_axis in (0, -2):
            df = pd.DataFrame(
                self.magnitude,
                index=self.domain.magnitude,
                columns=list(self.labels)
            )
            df.columns.name = 'labels'
            df.index.name = 'domain'
        else:
            df = pd.DataFrame(
                self.magnitude,
                columns=self.domain.magnitude,
                index=list(self.labels)
            )
            df.index.name = 'labels'
            df.columns.name = 'domain'
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
        if isinstance(df, pd.Series):
            df = df.reset_index()
            df['labels'] = list(self.labels)
        else:
            df = df.stack().reset_index()

        df.rename(columns={0: 'values'}, inplace=True)

        df['name'] = self.name
        df['units'] = str(self.units)
        df['units_dimensionality'] = str(self.units.dimensionality)
        df['domain_units'] = str(self.domain.units)
        df['domain_units_dimensionality'] = str(
            self.domain.units.dimensionality
        )
        if self.domain_min is None:
            df['domain_min'] = np.nan
        else:
            df['domain_min'] = self.domain_min.to(self.domain.units).magnitude
        if self.domain_max is None:
            df['domain_max'] = np.nan
        else:
            df['domain_max'] = self.domain_max.to(self.domain.units).magnitude
        if self.signal_min is None:
            df['signal_min'] = np.nan
        else:
            df['signal_min'] = self.signal_min.magnitude
        if self.signal_max is None:
            df['signal_max'] = np.nan
        else:
            df['signal_max'] = self.signal_max.magnitude

        # if labels are multiindex include them
        if isinstance(self.labels, pd.MultiIndex):
            df_labels = self.labels.to_frame(index=False)
            df_labels['labels'] = list(self.labels)
            df = df.merge(
                df_labels, on='labels', how='left',
                suffixes=('', '_label')
            )

        reserved_keys = df.columns

        for key, ele in self.attrs.items():
            if key in reserved_keys:
                raise DreyeError(
                    "Cannot convert to long dataframe since "
                    f"dictionary 'attrs' contains key '{key}', which"
                    " is reserved."
                )
            df[key] = ele

        return df

    @classmethod
    def from_dict(cls, data):
        """
        Create a class from dictionary.
        """

        try:
            # see if you can load original class
            cls = pickle.loads(eval(data.pop('pickled_class')))
        except Exception:
            pass

        return cls(**data)

    @classmethod
    def load(cls, filename):
        data = read_json(filename)
        return cls.from_dict(data)

    def save(self, filename):
        return write_json(filename, self.to_dict())

    @property
    def dtype(self):
        """
        Returns the data type.
        """

        return self._dtype

    @dtype.setter
    def dtype(self, value):
        """
        Set the data type.
        """

        if value is None:
            pass

        elif isinstance(value, str):
            value = np.dtype(value).type

        elif not hasattr(value, '__call__'):
            raise AttributeError('dtype attribute: must be callable.')

        self._dtype = value
        self.domain.dtype = value
        self._values = value(self._values)

    @property
    def domain(self):
        """
        Returns domain as previously defined. Can be a domain instance, list of
        numbers, tuple, or dictionary.
        """

        return self._domain

    @property
    def values(self):
        """
        Returns the array containing the value of signal.
        """

        return self._values * self.units

    @values.setter
    def values(self, value):
        if has_units(value):
            value = value.to(self.units)
        value = asarray(value)
        if not value.shape == self.shape:
            raise DreyeError('Array for values assignment must be same shape.')
        self._values = value

    @property
    def boundaries(self):
        """
        Returns the minimum and maximum value of each signal.
        """

        return asarray([
            np.min(self.magnitude, axis=self.domain_axis),
            np.max(self.magnitude, axis=self.domain_axis)
        ]).T

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
            pass
        elif hasattr(value, '__call__'):
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
            pass
        elif isinstance(value, dict):
            self._interpolator_kwargs = value
            self._interpolate = None
        else:
            raise TypeError('interpolator_kwargs must be dict.')

    @property
    def interpolate(self):
        """

        """

        if self._interpolate is None:

            interp = self.interpolator(
                self.domain.magnitude,
                self.magnitude,
                **self.interpolator_kwargs
            )

            def clip_wrapper(*args, **kwargs):
                return self._clip_values(
                    interp(*args, **kwargs),
                    get_values(self.signal_min),
                    get_values(self.signal_max)
                )

            self._interpolate = clip_wrapper

        return self._interpolate

    @staticmethod
    def _clip_values(values, a_min, a_max):
        """
        Define the minumum and maximum values in a signal and clip any values
        outside of these boundaries.

        Parameters
        ----------
        values :
            Values of the signal to perform the operation on.
        a_min :
            Minimum value which a signal shall be clipped to.
        a_max :
            Maximum value which a signal shall be clipped to.


        """
        if a_min is None and a_max is None:
            return values
        else:
            return np.clip(
                values,
                a_min=a_min,
                a_max=a_max
            )

    @property
    def labels(self):
        """
        Returns signal labels, or the "name" for each signal.
        """

        return self._labels

    @property
    def domain_axis(self):
        """
        Signal axis to which the domain is aligned.
        """

        return self._domain_axis

    @property
    def domain_units(self):
        return self.domain.units

    @property
    def domain_magnitude(self):
        return self.domain.magnitude

    @property
    def other_axis(self):
        """
        Signal axis to which the domain is not aligned.
        """

        if self.ndim == 1:
            return None  # self.domain_axis TODO: behavior
        else:
            return (self.domain_axis + 1) % 2

    @property
    def other_len(self):
        """
        Returns the length of the signal axis to which domain is not aligned.
        """

        if self.ndim == 1:
            return 1
        else:
            return self.shape[self.other_axis]

    @property
    def domain_len(self):
        """
        Returns the length of the signal axis to which domain is aligned.
        """

        return self.shape[self.domain_axis]

    @property
    def T(self):
        """
        Returns the transpose of signal.
        """

        if self.ndim == 1:
            return self.copy()
        else:
            self = self.copy()
            self._values = self.magnitude.T
            self._flip_axes_assignment()
            return self

    def moveaxis(self, source, destination, copy=True):
        """
        Swap the order of the axes in signal.
        """

        assert self.ndim == 2

        values = np.moveaxis(self.magnitude, source, destination)
        if copy:
            self = self.copy()
        self._values = values

        if int(source % 2) != int(destination % 2):
            self._flip_axes_assignment()

        return self

    def __call__(self, domain):
        """

        """

        domain_units = self.domain.units

        if isinstance(domain, AbstractDomain):
            domain_units = domain.units
            domain = domain.to(self.domain.units)
        else:
            domain = convert_units(domain, domain_units, True)

        # check domain min and max (must be bigger than this range)
        if self.domain_min is not None:
            domain_min = self.domain_min.to(domain_units).magnitude
            if np.min(asarray(domain)) > domain_min:
                raise DreyeError(
                    'Interpolation domain above domain minimum.'
                )
        if self.domain_max is not None:
            domain_max = self.domain_max.to(domain_units).magnitude
            if np.max(asarray(domain)) < domain_max:
                raise DreyeError(
                    'Interpolation domain below domain maximum.'
                )

        values = self.interpolate(asarray(domain))

        # for single value simply return quantity instance
        if (values.ndim < self.ndim) or values.shape[self.domain_axis] == 1:
            return values * self.units
        else:
            self = self.copy()
            self._values = values
            self._domain = self.domain_class(domain, units=domain_units)
            return self

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

        return self / self._broadcast(self.integral, self.other_axis)

    @property
    def piecewise_integral(self):
        """
        Returns the calculated integral at each point using the trapezoidal area
        method.
        """

        if self.ndim == 1:
            values = asarray(self) * asarray(self.domain.gradient)
        else:
            values = (asarray(self) * np.expand_dims(
                asarray(self.domain.gradient), self.other_axis))

        return self._create_new_instance(
            values, units=self.units * self.domain.units)

    @property
    def piecewise_gradient(self):
        """
        Returns the instantanous gradient at each point in signal.
        """

        if self.ndim == 1:
            values = asarray(self) / asarray(self.domain.gradient)
        else:
            values = (asarray(self) / np.expand_dims(
                asarray(self.domain.gradient), self.other_axis))

        return self._create_new_instance(
            values, units=self.units / self.domain.units)

    @property
    def gradient(self):
        """
        Returns the overall gradient.
        """

        return self._create_new_instance(
            np.gradient(self.magnitude,
                        self.domain.magnitude,
                        axis=self.domain_axis),
            units=self.units / self.domain.units,
        )

    @property
    def nanless(self, copy=True):
        """
        Returns signal with NaNs removed.
        """

        arr = self.magnitude
        arange = self.domain.magnitude

        if self.ndim == 1:
            finites = np.isfinite(arr)
            # interpolate nans
            values = self.interpolator(
                arange[finites], arr[finites],
                **self.interpolator_kwargs
            )(arange)
            values[finites] = arr[finites]

        else:
            arr = np.moveaxis(arr, self.other_axis, 0)
            values = np.zeros(arr.shape)

            for idx, iarr in enumerate(arr):
                finites = np.isfinite(iarr)
                # interpolate nans
                ivalues = self.interpolator(
                    arange[finites], iarr[finites],
                    **self.interpolator_kwargs
                )(arange)
                ivalues[finites] = iarr[finites]

                values[idx] = ivalues

            values = np.moveaxis(values, 0, self.other_axis)

        if copy:
            self = self.copy()

        self._values = values
        return self

    def enforce_uniformity(self, method=np.mean, on_gradient=True):
        """
        Returns the domain with a uniform interval, calculated from the average
        of all original interval values.
        """

        domain = self.domain.enforce_uniformity(
            method=method, on_gradient=on_gradient
        )

        return self(domain)

    def window_filter(
        self, domain_interval,
        method='savgol', extrapolate=False, copy=True, **method_args
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
                "chosen domain interval must be rounded down for filtering",
                RuntimeWarning
            )

        M = int(M)

        if method == 'savgol':

            method_args['polyorder'] = method_args.get('polyorder', 2)
            method_args['axis'] = method_args.get('axis', self.domain_axis)

            M = M + ((M+1) % 2)

            values = savgol_filter(self.magnitude, M, **method_args)

        elif extrapolate:
            # create filter instance
            filter1d = Filter1D(method, M, **method_args)

            # handle borders by interpolating
            start_idx, end_idx = int(np.floor((M-1)/2)), int(np.ceil((M-1)/2))
            new_domain = self.domain.extend(
                start_idx, left=True
            ).extend(
                end_idx, left=False
            ).asarray()

            values = filter1d(
                self(new_domain).magnitude,
                axis=self.domain_axis,
                mode='valid'
            )

        else:
            # create filter instance
            filter1d = Filter1D(method, M, **method_args)

            values = filter1d(
                self.magnitude,
                axis=self.domain_axis,
                mode='same'
            )

        if copy:
            self = self.copy()

        self._values = values
        return self

    def dot(self, other, pandas=False, units=True, sample_axis=None):
        """
        Returns the dot product of two signal instances. The dot product is
        computed along the domain.
        """

        if not isinstance(other, AbstractSignal):
            raise NotImplementedError('other must also be from signal class: '
                                      f'{type(other)}.')

        if sample_axis is None:
            sample_axis = self.domain_axis

        if int(sample_axis % 2) == int(self.domain_axis % 2):
            # equalize domains if necessary
            self, other = self.equalize_domains(other)
            self_labels = self.labels
            other_labels = other.labels
        elif self.other_len != other.other_len:
            raise DreyeError(
                'When sample axis is label axis then the '
                'label lengths of the two signals must match for '
                'dot product operation.'
            )
        else:
            self_labels = self.domain.magnitude
            other_labels = other.domain.magnitude

        self_values = np.moveaxis(self.magnitude, sample_axis, -1)
        other_values = np.moveaxis(other.magnitude, sample_axis, 0)

        new_units = self.units * other.units

        dot_array = np.dot(self_values, other_values)

        if units and not pandas:
            dot_array = dot_array * new_units

        if pandas:

            if self.ndim == 2 and other.ndim == 2:
                return pd.DataFrame(dot_array,
                                    index=self_labels,
                                    columns=other_labels)

            elif self.ndim == 1 and other.ndim == 1:
                return dot_array

            elif self.ndim == 1:
                return pd.Series(dot_array,
                                 index=other_labels,
                                 name=self_labels)

            elif other.ndim == 1:
                return pd.Series(dot_array,
                                 index=self_labels,
                                 name=other_labels)

        else:
            return dot_array

    def cov(
        self, pandas=False, units=True, mean_center=True, sample_axis=None
    ):
        """
        Calculate covariance matrix of signal.
        """

        assert self.ndim == 2

        if sample_axis is None:
            sample_axis = self.domain_axis

        if mean_center:
            self = self - self.mean(axis=(sample_axis + 1) % 2, keepdims=True)

        return self.dot(
            self, pandas=pandas, units=units, sample_axis=sample_axis)

    def corr(
        self, pandas=False, units=True, mean_center=True, sample_axis=None
    ):
        """
        Calculate pearson's correlation matrix containing correlation
        coefficients (variance/variance squared).

        Parameters
        ----------
        pandas: bool, optional
            If set to True, will return a Pandas dataframe.
        """

        cov = self.cov(
            pandas=False, units=units, mean_center=mean_center,
            sample_axis=sample_axis
        )

        if pandas:
            var = np.diag(cov.values)
            units = 1
        elif units:
            cov = cov.magnitude
            var = np.diag(cov)
            units = ureg(None)
        else:
            var = np.diag(cov)
            units = 1
        corr = cov / np.sqrt(np.outer(var, var))
        return corr * units

    def numpy_estimator(
        self,
        func,
        axis=None,
        weight=None,
        **kwargs
    ):
        """
        General method for using mean, sum, etc.
        """

        if weight is not None:
            # TODO _broadcasting and label handling
            self, weight, labels = self._instance_handler(weight)

            self = self * weight

        values = func(self.magnitude, axis=axis, **kwargs)

        if (axis == self.other_axis) and (self.ndim == 2):
            if values.ndim == 2:
                return self._create_new_instance(
                    values,
                    units=self.units,
                    labels=self.name)
            else:
                return self._create_new_instance(
                    values,
                    units=self.units,
                    labels=self.name,
                    domain_axis=0)
        else:
            return values * self.units

    def mean(self, *args, **kwargs):
        """
        Compute the arithmetic mean along the specified axis.
        """

        return self.numpy_estimator(np.mean, *args, **kwargs)

    def nanmean(self, *args, **kwargs):
        """
        Compute the arithmetic mean along the specified axis, ignoring NaNs.
        """

        return self.numpy_estimator(np.nanmean, *args, **kwargs)

    def sum(self, *args, **kwargs):
        """
        Sum of array elements over a given axis.
        """

        return self.numpy_estimator(np.sum, *args, **kwargs)

    def nansum(self, *args, **kwargs):
        """
        Return the sum of array elements over a given axis treating Not a
        Numbers (NaNs) as zero.

        """

        return self.numpy_estimator(np.nansum, *args, **kwargs)

    def std(self, *args, **kwargs):
        """
        Compute the standard deviation along the specified axis.
        """

        return self.numpy_estimator(np.std, *args, **kwargs)

    def nanstd(self, *args, **kwargs):
        """
        Compute the standard deviation along the specified axis, while ignoring
        NaNs.
        """

        return self.numpy_estimator(np.nanstd, *args, **kwargs)

    def min(self, *args, **kwargs):
        """
        Return the minimum along a given axis.
        """

        return self.numpy_estimator(np.min, *args, **kwargs)

    def nanmin(self, *args, **kwargs):
        """
        Return minimum of an array or minimum along an axis, ignoring any NaNs.
        When all-NaN slices are encountered a RuntimeWarning is raised and Nan
        is returned for that slice.
        """

        return self.numpy_estimator(np.nanmin, *args, **kwargs)

    def max(self, *args, **kwargs):
        """
        Element-wise maximum of array elements.
        """

        return self.numpy_estimator(np.max, *args, **kwargs)

    def nanmax(self, *args, **kwargs):
        """
        Return the maximum of an array or maximum along an axis, ignoring any
        NaNs. When all-NaN slices are encountered a RuntimeWarning is raised and
        NaN is returned for that slice.
        """

        return self.numpy_estimator(np.nanmax, *args, **kwargs)

    def domain_concat(self, signal, left=False, copy=True):
        """
        Creates a new signal instance by appending two signals along the domain
        axis.
        """

        domain = self.domain
        self_values = self.magnitude

        if isinstance(signal, AbstractSignal):
            # checks dimensionality, appends domain, converts units
            assert self.ndim == signal.ndim

            domain = domain.append(signal.domain, left=left)

            if self.domain_axis != signal.domain_axis:
                signal = signal.T

            assert self.other_len == signal.other_len
            # check labels, handling of different labels?

            other_values = asarray(signal.to(self.units))

        elif is_arraylike(signal):
            # handles units, checks other shape, extends domain
            other_values = asarray(convert_units(signal, self.units))

            if self.ndim == 2:
                assert self.other_len == other_values.shape[self.other_axis]

            domain = domain.extend(
                other_values.shape[self.domain_axis],
                left=left
            )

        else:
            raise DreyeError(
                f"domain axis contenation with type: {type(signal)}")

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

        if copy:
            self = self.copy()

        self._values = values
        self._domain = domain
        return self

    def concat_labels(self, labels, left=False, label_class=None):
        """
        Concatenate labels of two signal instances.
        """

        assert self.ndim == 2

        if label_class is None:
            label_class = self._label_class

        if left:
            new_labels = label_class(labels)
            _new_labels = new_labels.append(self.labels)
            if _new_labels is None:  # assumes inplace
                return new_labels
            else:
                return _new_labels
        else:
            new_labels = label_class(self.labels)
            _new_labels = new_labels.append(labels)
            if _new_labels is None:  # assumes inplace
                return new_labels
            else:
                return _new_labels

    def other_concat(
        self, signal, labels=None, left=False,
        copy=True, label_class=None
    ):
        """
        Create a new signal instance by concatenating two existing signal
        instances. If domains are not equivalent, interpolate if possible and
        enforce the same domain range by using the equalize_domains function.
        """

        if copy:
            self = self.copy()

        if self.ndim == 1:
            self = self._expand_dims(1, copy=False)

        if isinstance(signal, AbstractSignal):
            # equalizing domains
            self, signal = self.equalize_domains(signal)
            self_values = self.magnitude
            # check units
            other_values = asarray(signal.to(self.units))
            # labels
            if labels is None:
                labels = signal.labels
            # handle labels
            labels = self.concat_labels(labels, left, label_class=label_class)

        elif is_arraylike(signal):
            # self numpy array
            self_values = self.magnitude
            # check if it has units
            other_values = asarray(convert_units(signal, self.units))
            # handle labels
            if labels is None:
                labels = [
                    str(i)
                    for i in range(other_values.shape[self.other_axis])
                ]
            labels = self.concat_labels(labels, left, label_class=label_class)

        else:
            raise DreyeError(
                f"other axis contenation with type: {type(signal)}")

        if left:

            values = np.concatenate(
                [other_values, self_values],
                axis=self.other_axis
            )

        else:

            values = np.concatenate(
                [self_values, other_values],
                axis=self.other_axis
            )

        self._values = values
        self._labels = labels
        return self

    def concat(self, signal, *args, **kwargs):
        """
        Concatenate two signals.
        """

        return self.other_concat(signal, *args, **kwargs)

    def append(self, signal, *args, **kwargs):
        """
        Append signals.
        """

        return self.domain_concat(signal, *args, **kwargs)

    def __eq__(self, other):

        if self.__class__ != other.__class__:
            return False

        return (
            (self.units == other.units)
            and (self.domain == other.domain)
            and array_equal(asarray(self), asarray(other))
        )

    def __str__(self):

        return (
            f"{self.__class__.__name__}"
            f"(\n\t name={self.name}, \n\t "
            f"labels={self.labels}, \n\t units={self.units}, \n\t "
            f"shape={self.shape}, \n\t "
            f"domain={self.domain}, \n\t domain_axis={self.domain_axis} \n )"
        )

    def _check_clip_value(self, value):
        """
        """

        if is_listlike(value):
            if self.ndim == 2:
                assert len(value) == self.other_len
            else:
                if len(value) != self.other_len:
                    raise ValueError('signal is one-dimensional '
                                     'but clipping is list-like.')


class SignalContainer(AbstractContainer, SignalPlottingMixin, EstimatorMixin):
    """A class that contains multiple signal instances in a list.
    All attributes and methods of a Signal instance can be used directly.
    Domain units and signal units must have the same dimensionality.
    """
    _allowed_instances = Signal
    _init_keys = [
        '_signals',
    ]

    def __init__(self, container, units=None, domain_units=None):
        super().__init__(container)

        if units is None:
            self._units = self.container[0].units
        else:
            # TODO to work with uE
            self._units = ureg(str(units))
        if domain_units is None:
            self._domain_units = self.container[0].domain.units
        else:
            self._domain_units = ureg(str(domain_units))

        # convert to correct units
        for idx, ele in enumerate(self.container):
            ele.units = self.units
            ele.domain.units = self.domain_units

    def to_longframe(self):
        return pd.concat(
            self.__getattr__('to_longframe')(),
            ignore_index=True,
        )

    @property
    def ndim(self):
        return np.min([np.sum(self.__getattr__('ndim')), 2])

    # def plot(self, *args, **kwargs):
    #     raise NotImplementedError(
    #         f'plot method is not implemented for {self.__class__}; '
    #         'use the relplot method instead.'
    #     )

    def relplot(self, **kwargs):
        # default behavior is that each signal gets own
        # column according to name
        # and that columns are wrapped in threes
        if not {'row', 'col'} & set(kwargs):
            kwargs['col'] = 'name'
            kwargs['col_wrap'] = 3
        super().relplot(**kwargs)

    def __getitem__(self, key):
        if is_numeric(key):
            return super().__getitem__(key)
        elif key in self.names:
            key = self.names.index(key)
        elif key in self.container:
            key = self.container.index(key)
        return super().__getitem__(key)

    @property
    def domain_units(self):
        return self._domain_units

    @property
    def units(self):
        return self._units

    @property
    def names(self):
        return [ele.name for ele in self]

    def popkey(self, key):
        index = self.names.index(key)
        self.pop(index)

    @property
    def signals(self):
        if self._signals is None:
            # enforce Signal class
            signals = Signal(self[0])
            # create flat index (pd.Index method)
            if signals.ndim == 2:
                signals._labels = signals.labels.to_flat_index()
            # concat signals
            for _signal in self[1:]:
                _signal = Signal(_signal)
                # create flat index (pd.Index method)
                if _signal.ndim == 2:
                    _signal._labels = _signal.labels.to_flat_index()
                signals.other_concat(_signal, copy=False, label_class=list)
            # assign attribute
            self._signals = signals
        return self._signals

    @property
    def magnitude(self):
        return self.signals.magnitude

    @property
    def domain_magnitude(self):
        return self.signals.domain.magnitude

    @property
    def labels(self):
        return self.signals.labels

    @property
    def other_len(self):
        return np.sum(self.__getattr__('other_len'))

    @property
    def other_axis(self):
        return self.signals.other_axis

    @property
    def domain_axis(self):
        return self.signals.domain_axis

    @property
    def domain_len(self):
        return self.signals.domain_len
