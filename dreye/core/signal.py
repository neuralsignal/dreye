"""
Signal
======

Defines the class for implementing continuous signals:

-   :class:`dreye.core.Signal`
"""

import warnings

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from pint import DimensionalityError
import cloudpickle
import pickle

from dreye.utilities import (
    dissect_units, is_numeric, has_units, is_arraylike, convert_units,
)
from dreye.io import read_json, write_json
from dreye.constants import DEFAULT_FLOAT_DTYPE
from dreye.core.abstract import AbstractDomain, AbstractSignal
from dreye.core.mixin import UnpackSignalMixin, CheckClippingValueMixin
from dreye.core.domain import Domain
from dreye.algebra.filtering import Filter1D


class Signal(AbstractSignal, UnpackSignalMixin):
    """
    Defines the base class for continuous signals (unit-aware).

    TODO: Description

    Parameters
    ----------
    values : array-like or str
    domain : Domain, tuple, dict, or array-like, optional
    axis : int, optional
        Axis of the domain.
    units : str, optional
    labels : str or tuple, optional
    interpolator : interpolate class, optional
    interpolator_kwargs : dict-like, optional


    Attributes
    ----------
    values
    data
    domain # setter
    labels # setter
    ndim # from data
    shape # from data
    size # from data
    interpolator # setter

    Methods
    -------
    __call__
    [arithmetical operations]
    to_frame
    to_array
    to_dict
    load
    save
    unpack
    plot
    dot # only along domain axis
    """

    init_args = ('domain', 'interpolator', 'interpolator_kwargs', 'dtype',
                 'domain_axis', 'labels', 'contexts')
    domain_class = Domain

    def __init__(self,
                 values,
                 domain=None,
                 domain_axis=None,
                 units=None,
                 domain_units=None,
                 labels=None,
                 dtype=DEFAULT_FLOAT_DTYPE,
                 domain_dtype=None,
                 interpolator=None,
                 interpolator_kwargs=None,
                 contexts=None,
                 domain_kwargs=None,
                 **kwargs):

        if isinstance(dtype, str):
            dtype = np.dtype(dtype).type

        values, units, kwargs = self.unpack(
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
            **kwargs)

        self._values = values
        self._units = units

        for key, value in kwargs.items():
            if key in self.convert_attributes:
                if value is None:
                    pass
                elif has_units(value):
                    value = value.to(self.units)
                else:
                    value = value * self.units
            setattr(self, '_' + key, value)

        self._interpolate = None

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

    @classmethod
    def from_dict(cls, data):
        """create class from dictionary
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
        """

        return self._dtype

    @dtype.setter
    def dtype(self, value):
        """
        """

        if value is None:
            pass

        elif isinstance(value, str):
            value = np.dtype(value).type

        elif not hasattr(value, '__call__'):
            raise AttributeError('dtype attribute: must be callable.')

        self._dtype = value
        self.domain.dtype = value
        self.values = value(self.values)

    @property
    def domain(self):
        """
        """

        return self._domain

    @domain.setter
    def domain(self, value):
        """
        """

        # same as interpolating
        self(value)

    @property
    def values(self):
        """
        """

        return self._values * self.units

    @values.setter
    def values(self, value):
        """
        """

        values, units = dissect_units(value)
        values = np.array(values)

        if units is not None and units != self.units:
            raise DimensionalityError(units, self.units)

        values, labels = self.check_values(values, self.domain,
                                           self.domain_axis, self.labels)

        self._values = values
        self._labels = labels
        self._interpolate = None

    @property
    def boundaries(self):
        """
        """

        return np.array([
            np.min(self.magnitude, axis=self.domain_axis),
            np.max(self.magnitude, axis=self.domain_axis)
        ]).T

    @property
    def interpolator(self):
        """
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

            self._interpolate = self.interpolator(self.domain.asarray(),
                                                  self.magnitude,
                                                  **self.interpolator_kwargs)

        return self._interpolate

    @property
    def labels(self):
        """
        """

        return self._labels

    @property
    def domain_axis(self):
        """
        """

        return self._domain_axis

    def __call__(self, domain):
        """
        """

        # TODO more automated way of keeping track of units?

        domain_units = self.domain.units

        if isinstance(domain, AbstractDomain):
            if self.domain == domain.convert_to(self.domain.units):
                self = self.copy()
                self.domain.units = domain.units
                return self

            domain_units = domain.units
            domain = self.domain.convert_units(domain)
            # OR domain = domain.convert_to(self.domain.units)
        else:
            domain = convert_units(domain, domain_units, True)

        values = self.interpolate(np.array(domain))

        return self.create_new_instance(
            values,
            domain=domain,
            domain_units=domain_units,
            units=self.units,
        )

    @property
    def integral(self):
        """
        """

        return np.trapz(
            self.magnitude,
            self.domain.magnitude,
            axis=self.domain_axis
        ) * self.units * self.domain.units

    @property
    def normalized_signal(self):
        """
        """

        return self / self.broadcast(self.integral, self.other_axis)

    @property
    def piecewise_integral(self):
        """
        """

        if self.ndim == 1:
            values = np.array(self) * np.array(self.domain.gradient)
        else:
            values = (np.array(self) * np.expand_dims(
                np.array(self.domain.gradient), self.other_axis))

        return self.create_new_instance(
            values, units=self.units * self.domain.units)

    @property
    def piecewise_gradient(self):
        """
        """

        if self.ndim == 1:
            values = np.array(self) / np.array(self.domain.gradient)
        else:
            values = (np.array(self) / np.expand_dims(
                np.array(self.domain.gradient), self.other_axis))

        return self.create_new_instance(
            values, units=self.units / self.domain.units)

    @property
    def gradient(self):
        """
        """

        return self.create_new_instance(
            np.gradient(self.magnitude,
                        self.domain.magnitude,
                        axis=self.domain_axis),
            units=self.units / self.domain.units,
        )

    @property
    def nanless(self):
        """
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

        return self.create_new_instance(
            values,
            units=self.units,
        )

    def enforce_uniformity(self, method=np.mean, on_gradient=True):
        """enforce uniform domain (interpolate)
        """

        domain = self.domain.enforce_uniformity(
            method=method, on_gradient=on_gradient
        )

        return self(domain)

    def window_filter(
        self, domain_interval, method, extrapolate=False, **method_args
    ):
        """Filter Signal instance using filter1d
        """

        # TODO implement savgol filter

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

            M = M + ((M+1) % 2)

            values = savgol_filter(self.magnitude, M, **method_args)

        elif extrapolate:
            # TODO smooth interpolation? - small savgol?
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

        return self.create_new_instance(values, units=self.units)

    def dot(self, other, pandas=False, units=True):
        """Return dot product of two signal instances.
        Dot product is always computed along the domain.
        """

        if not isinstance(other, AbstractSignal):
            raise NotImplementedError('other must also be from signal class: '
                                      f'{type(other)}.')

        self, other = self.equalize_domains(other)

        self_values = np.moveaxis(self.magnitude, self.domain_axis, -1)
        other_values = np.moveaxis(other.magnitude, other.domain_axis, 0)

        new_units = self.units * other.units

        dot_array = np.dot(self_values, other_values)

        if units:
            dot_array = dot_array * new_units

        if pandas:

            if self.ndim == 2 and other.ndim == 2:
                return pd.DataFrame(dot_array,
                                    index=self.labels,
                                    columns=other.labels)

            elif self.ndim == 1 and other.ndim == 1:
                return dot_array

            elif self.ndim == 1:
                return pd.Series(dot_array,
                                 index=other.labels,
                                 name=self.labels)

            elif other.ndim == 1:
                return pd.Series(dot_array,
                                 index=self.labels,
                                 name=other.labels)

        else:
            return dot_array

    def cov(self, pandas=False, units=True, mean_center=True):
        """calculate covariance matrix
        """

        if mean_center:
            self = self - self.mean(axis=self.other_axis, keepdims=True)

        return self.dot(self, pandas=pandas, units=units)

    def corr(self, pandas=False, units=True, mean_center=True):
        """calculate pearson's correlation matrix
        """

        cov = self.cov(pandas=False, units=units, mean_center=mean_center)

        if pandas:
            raise NotImplementedError('correlation matrix with pandas')

        if is_numeric(cov):
            return 1

        # covariance is two dimensional
        # TODO check if it keeps up with the units
        var = np.diag(cov)
        corr = cov / np.sqrt(var * var)
        return corr

    def numpy_estimator(self,
                        func,
                        axis=None,
                        weight=None,
                        keepdims=False,
                        **kwargs):
        """General method for using mean, sum, etc.
        """

        if weight is not None:
            # TODO broadcasting and label handling
            self, weight, labels = self.instance_handler(weight)

            self = self * weight

        values = func(self.magnitude, axis=axis, keepdims=keepdims, **kwargs)

        if (axis == self.other_axis) and (self.ndim == 2):
            if keepdims:
                return self.create_new_instance(values,
                                                units=self.units,
                                                labels=None)
            else:
                return self.create_new_instance(values,
                                                units=self.units,
                                                labels=None,
                                                domain_axis=0)
        else:
            return values * self.units

    def mean(self, *args, **kwargs):
        """
        """

        return self.numpy_estimator(np.mean, *args, **kwargs)

    def nanmean(self, *args, **kwargs):
        """
        """

        return self.numpy_estimator(np.nanmean, *args, **kwargs)

    def sum(self, *args, **kwargs):
        """
        """

        return self.numpy_estimator(np.sum, *args, **kwargs)

    def nansum(self, *args, **kwargs):
        """
        """

        return self.numpy_estimator(np.nansum, *args, **kwargs)

    def std(self, *args, **kwargs):
        """
        """

        return self.numpy_estimator(np.std, *args, **kwargs)

    def nanstd(self, *args, **kwargs):
        """
        """

        return self.numpy_estimator(np.nanstd, *args, **kwargs)

    def domain_concat(self, signal, left=False, copy=True):
        """appends along domain axis
        """

        # needs to do domain checking or first append domain
        # dealing with interpolator?
        if not copy:
            raise NotImplementedError('inplace concatenation')

        domain = self.domain
        self_values = np.array(self)  # self.magnitude?

        if isinstance(signal, AbstractSignal):
            # checks dimensionality, appends domain, converts units
            assert self.ndim == signal.ndim

            domain = domain.append(signal.domain, left=left, copy=copy)

            if self.domain_axis != signal.domain_axis:
                signal = signal.T

            assert self.other_len == signal.other_len
            # check labels, handling of different labels?

            other_values = np.array(signal.convert_to(self.units))

        elif is_arraylike(signal):
            # handles units, checks other shape, extends domain
            other_values = np.array(convert_units(signal, self.units))

            if self.ndim == 2:
                assert self.other_len == other_values.shape[self.other_axis]

            domain = domain.extend(
                other_values.shape[self.domain_axis],
                left=left, copy=copy
            )

        else:
            raise TypeError(
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

        return self.create_new_instance(
            values, units=self.units,
            domain=domain
        )

    def concat_labels(self, labels, left=False):
        """
        """

        assert self.ndim == 2

        if left:
            return list(labels) + list(self.labels)
        else:
            return list(self.labels) + list(labels)

    def other_concat(self, signal, labels=None, left=False, copy=True):
        """
        """

        if not copy:
            raise NotImplementedError('inplace concatenation')

        if self.ndim == 1:
            self = self.expand_dims(1)

        self_values = np.array(self)

        if isinstance(signal, AbstractSignal):
            # equalizing domains
            self, signal = self.equalize_domains(signal)
            # check units
            other_values = np.array(signal.convert_to(self.units))
            # handle labels
            labels = self.concat_labels(signal.labels, left)

        elif is_arraylike(signal):
            # check if it has units
            other_values = np.array(convert_units(signal, self.units))
            # handle labels
            assert labels is not None, "must provide labels"
            labels = self.concat_labels(labels, left)

        else:
            raise TypeError(
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

        return self.create_new_instance(
            values, units=self.units,
            labels=labels
        )

    def concat(self, signal, *args, **kwargs):
        """concatenates two signals
        """

        return self.other_concat(signal, *args, **kwargs)

    def append(self, signal, *args, **kwargs):
        """append signals
        """

        return self.domain_concat(signal, *args, **kwargs)

    # TODO def __eq__(self, other)


class ClippedSignal(Signal, CheckClippingValueMixin):
    """Same as signal, but signal will be clipped when interpolating.
    For example, for a spectral distribution the values can never be above
    or below zero.
    """

    init_args = Signal.init_args + ('signal_min', 'signal_max')
    convert_attributes = ('signal_min', 'signal_max')

    def __init__(
        self,
        values,
        domain=None,
        domain_axis=None,
        units=None,
        domain_units=None,
        labels=None,
        dtype=DEFAULT_FLOAT_DTYPE,
        domain_dtype=None,
        interpolator=None,
        interpolator_kwargs=None,
        contexts=None,
        signal_min=None,
        signal_max=None,
        **kwargs
    ):

        super().__init__(
            values=values,
            domain=domain,
            domain_axis=domain_axis,
            units=units,
            domain_units=domain_units,
            labels=labels,
            dtype=dtype,
            domain_dtype=domain_dtype,
            interpolator=interpolator,
            interpolator_kwargs=interpolator_kwargs,
            contexts=contexts,
            signal_min=signal_min,
            signal_max=signal_max,
            **kwargs
        )

        if self.signal_min is None and self.signal_max is None:
            self._signal_min = 0 * self.units

        self._check_clip_value(self.signal_min)
        self._check_clip_value(self.signal_max)

    @property
    def signal_min(self):
        """
        """

        return self._signal_min

    @property
    def signal_max(self):
        """
        """

        return self._signal_max

    @signal_min.setter
    def signal_min(self, value):
        """
        """

        if value is None:
            pass
        else:
            value = convert_units(value, self.units)
            if has_units(value):
                self._signal_min = value
            else:
                self._signal_min = value * self.units

    @signal_max.setter
    def signal_max(self, value):
        """
        """

        if value is None:
            pass
        else:
            value = convert_units(value, self.units)
            if has_units(value):
                self._signal_max = value
            else:
                self._signal_max = value * self.units

    @property
    def interpolate(self):
        """
        """

        interp = super().interpolate

        def clip_wrapper(*args, **kwargs):
            return np.clip(interp(*args, **kwargs),
                           a_min=dissect_units(self.signal_min)[0],
                           a_max=dissect_units(self.signal_max)[0])

        return clip_wrapper
