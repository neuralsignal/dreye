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
    is_numeric, has_units, is_listlike, asarray, get_value,
    is_callable, is_dictlike, optional_to,
    is_hashable, is_integer, is_broadcastable
)
from dreye.utilities.abstract import inherit_docstrings
from dreye.err import DreyeError
from dreye.constants import DEFAULT_FLOAT_DTYPE
from dreye.core.abstract import _UnitArray
from dreye.core.domain import Domain
from dreye.utilities import Filter1D
from dreye.core.plotting_mixin import _PlottingMixin
from dreye.core.numpy_mixin import _NumpyMixin


def labels_concat(objs, left=False):
    """
    Concatenate multiple Signal objects together along the labels dimension.

    Parameters
    ----------
    objs : list of signal-type
        A list of signal-type instances to concatenate
    left : bool, optional
        Wheter to do a left side concatenation or not

    Returns
    -------
    obj : signal-type
        A concatenated signal-type instance.
    """
    assert is_listlike(objs)
    assert len(objs) > 0
    obj = objs[0]
    for obj_ in objs[1:]:
        obj = obj.labels_concat(obj_, left=left)
    return obj


def domain_concat(objs, left=False):
    """
    Concatenate multiple Signal objects together along the domain dimension.

    Parameters
    ----------
    objs : list of signal-type
        A list of signal-type instances to concatenate
    left : bool, optional
        Wheter to do a left side concatenation or not

    Returns
    -------
    obj : signal-type
        A concatenated signal-type instance.
    """
    assert is_listlike(objs)
    assert len(objs) > 0
    obj = objs[0]
    for obj_ in objs[1:]:
        obj = obj.domain_concat(obj_, left=left)
    return obj


@inherit_docstrings
class _SignalMixin(_UnitArray, _PlottingMixin, _NumpyMixin):

    _init_args = (
        'domain',
        'interpolator', 'interpolator_kwargs',
        'contexts', 'attrs',
        'domain_min', 'domain_max',
        'name',
        'smoothing_method', 'smoothing_window', 'smoothing_args',
        'domain_axis'
    )
    _domain_class = Domain
    # just accepts self
    _args_defaults = {
        'interpolator_kwargs': {},
        'interpolator': interp1d,
        'smoothing_args': {},
        'smoothing_window': 1.0,
        'smoothing_method': 'savgol',
        'domain_axis': 0
    }
    # dictionary of attributes defaults if None
    _unit_conversion_params = {
        'domain': '_domain_values'
    }
    _convert_attributes = ('signal_min', 'signal_max')
    # attributes mapping passed to the "to" method of pint

    @property
    def _init_aligned_attrs(self):
        """
        What attributes are aligned to which axis in the signal values.
        """
        return {
            self.domain_axis: ('domain',),
            # tuple allow for broadcastable arrays
            tuple(range(self.ndim)): ('signal_min', 'signal_max',)
        }

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
        domain_axis=None,
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
            domain_axis=domain_axis,
            **kwargs
        )

        # applied window is added after smoothing
        if 'applied_window_' not in self.attrs:
            self.attrs['applied_window_'] = 'raw'

    # --- these methods need to be overwritten for multi-D signals --- #

    def _preextract_attributes(self, values, kwargs):
        """
        Extract various attributes from a pandas instance if passed
        attributes is None. Used by _test_and_assign_values
        """
        if isinstance(values, pd.Series):
            if self.domain is None:
                self._domain = values.index
            if self.name is None:
                self.name = values.name

    def _preprocess_check_values(self, values):
        if values.ndim != 1:
            raise DreyeError("Values must be a one- or two-dimensional array.")
        return values

    def _process_other_attributes(self, values, kwargs):
        pass

    def to_frame(self):
        """
        Convert instance to `pandas.Series` or `pandas.DataFrame`.

        Returns
        -------
        obj : `pandas.Series` or `pandas.DataFrame`
            If signal-type is one dimensional this method returns
            a `pandas.Series`, otherwise it is a `pandas.DataFrame`

        Notes
        -----
        The index of the `pandas` object will correspond to the values of
        `domain` attribute. In the case of two-dimensional signal
        objects the columns in the `pandas.DataFrame` will correspond
        to the values in the `labels` object.
        """

        series = pd.Series(
            self.magnitude,
            index=pd.MultiIndex.from_arrays(
                [self.domain.magnitude, self.signal_min, self.signal_max],
                names=['domain', 'signal_min', 'signal_max']
            ),
        )
        return series

    @property
    def _iter_values(self):
        """
        Iterate over individual domain axis.

        zip(slices, values)
        """

        return zip([self._none_slices_except_domain_axis], [self.magnitude])

    @property
    def ndim(self):
        return 1

    # --- methods that should not have to be
    # --- overwritted for multi-dimensional signal class

    def _get_domain_axis(self, domain_axis):
        assert is_integer(domain_axis)
        domain_axis = int(domain_axis % self.ndim)
        assert (domain_axis >= 0)
        return domain_axis

    def _test_and_assign_values(self, values, kwargs):
        # get values as array
        old_values = values
        if not is_listlike(values):
            raise DreyeError("Values must be array-like, but are "
                             f"of type '{type(values)}'.")
        values = asarray(values, DEFAULT_FLOAT_DTYPE)

        self._domain_axis = self._get_domain_axis(self.domain_axis)

        # extract attributes
        self._preextract_attributes(old_values, kwargs)
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
        values = self._process_values(
            values.copy(), self.signal_min, self.signal_max
        )
        self._values = values
        self._interpolate = None

    @staticmethod
    def _process_values(values, smin, smax):
        """
        clip values inplace using signal_min and signal_max.

        Before assigning the `values` property,
        but after domain_axis determined.

        Parameters
        ----------
        values : np.ndarray
        """
        # signal min and max are same dimensionality as values
        not_nan = ~np.isnan(smin)
        if np.any(not_nan):
            # inplace
            values[not_nan] = np.where(
                values[not_nan] < smin[not_nan],
                smin[not_nan],
                values[not_nan]
            )

        not_nan = ~np.isnan(smax)
        if np.any(not_nan):
            # inplace
            values[not_nan] = np.where(
                values[not_nan] > smax[not_nan],
                smax[not_nan],
                values[not_nan]
            )
        return values

    def _get_signal_bound(self, values, bound):
        """
        Standard signal_min and max are numeric or array-like types.
        """

        if bound is None:
            return np.broadcast_to(np.nan, values.shape)
        # optional_to takes care of non-listlike values
        bound = optional_to(
            bound, self.units,
            *self.contexts, **self._unit_conversion_kws
        )
        bound = np.array(bound, dtype=DEFAULT_FLOAT_DTYPE, ndmin=self.ndim)
        if bound.ndim > self.ndim:
            raise DreyeError(
                "Bound dimensionality bigger "
                "than self dimensionality."
            )
        if not is_broadcastable(bound.shape, values.shape):
            # try alligning to domain_axis - 1
            # assumes (domain_axis, ax3, ax2, ax1) (0, -3, -2, -1)
            assumed_order = self._get_axes_order(0)
            # get correct order
            correct_order = self._axes_order  # uses same function
            bound = np.moveaxis(bound, assumed_order, correct_order)
            # try rolling backwards until it fits
            n = 1
            while (
                not is_broadcastable(bound.shape, values.shape)
                and n < self.ndim
            ):
                bound = np.moveaxis(
                    bound,
                    np.arange(bound.ndim),
                    np.roll(np.arange(bound.ndim), -n)
                )
                n += 1
            if not is_broadcastable(bound.shape, values.shape):
                raise DreyeError(
                    "Bound is not broadcastable "
                    "to values."
                )
        # signal_min and max will have same shape as values
        return np.broadcast_to(bound, values.shape)

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

        if (
            not isinstance(domain, domain_class)
            or domain_units is not None
            or domain_kwargs is not None
        ):
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

        This property is used during interpolation. If all values for
        interpolation are bigger than `domain_min`, interpolation will
        result in an error.
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

        This property is used during interpolation. If all values for
        interpolation are smaller than `domain_max`, interpolation will
        result in an error.
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
        clipped to after filtering or interpolation.
        """
        return self._signal_min

    @signal_min.setter
    def signal_min(self, value):
        self._signal_min = self._get_signal_bound(self.magnitude, value)
        self._values = self._process_values(
            self.magnitude.copy(), self.signal_min, self.signal_max
        )

    @property
    def signal_max(self):
        """
        Returns the maximum value in signal, to which all lower values are
        clipped to after filtering or interpolation.
        """
        return self._signal_max

    @signal_max.setter
    def signal_max(self, value):
        self._signal_max = self._get_signal_bound(self.magnitude, value)
        self._values = self._process_values(
            self.magnitude.copy(), self.signal_min, self.signal_max
        )

    @property
    def domain(self):
        """
        `dreye.Domain` instance associated with signal-type.
        """
        return self._domain

    @property
    def domain_units(self):
        """
        Units of the `dreye.Domain` instance.

        See Also
        --------
        dreye.Domain.units
        """
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
                None,
                None,
                self.domain_axis,
                self._domain_class
            )
        else:
            self._domain = self._get_domain(
                self.magnitude,
                value,
                self.domain.units,
                self.domain._init_kwargs,
                self.domain_axis,
                self._domain_class
            )

    @property
    def _domain_values(self):
        """
        domain values for unit conversion properly broadcasted.
        """
        return self.domain.values[self._none_slices_except_domain_axis]

    @property
    def _none_slices_except_domain_axis(self):
        """
        None for all axes except domain axis.
        """
        slices = self.ndim * [None]
        slices[self.domain_axis] = slice(None, None, None)
        return tuple(slices)

    @property
    def _slices_except_none_domain_axis(self):
        """
        slices for all axes except domain axis.
        """
        slices = self.ndim * [slice(None, None, None)]
        slices[self.domain_axis] = None
        return tuple(slices)

    @property
    def interpolator(self):
        """
        Returns the interpolator.

        The default interpolator is `scipy.interpolate.interp1d`. The
        interpolator can be set during initialization.

        See Also
        --------
        scipy.interpolate.interp1d
        """
        return self._interpolator

    @interpolator.setter
    def interpolator(self, value):
        """
        Setting a new interpolator
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
        A dictionary directly passed to the `interpolator`.
        """
        # always makes sure that integration occurs along the right axis
        self._interpolator_kwargs['axis'] = self.domain_axis
        return self._interpolator_kwargs

    @interpolator_kwargs.setter
    def interpolator_kwargs(self, value):
        """
        Set keyword arguments for interpolator
        """
        if value is None:
            self._interpolator_kwargs = {}
        elif isinstance(value, dict):
            self._interpolator_kwargs = value
            self._interpolate = None
        else:
            raise TypeError('interpolator_kwargs must be dict.')

    def _get_interp_function(
        self, domain, mag, kwargs, smin, smax
    ):
        """
        get the interpolate function given the domain, magnitude,
        keyword argument dictionary, and signal min and max array.
        """
        interp = self.interpolator(
            domain,
            mag,
            **kwargs
        )
        min_interp = self.interpolator(
            domain,
            smin,
            **kwargs
        )
        max_interp = self.interpolator(
            domain,
            smax,
            **kwargs
        )

        def clip_wrapper(*args, **kwargs):
            interpolated = interp(*args, **kwargs)
            min_interpolated = min_interp(*args, **kwargs)
            max_interpolated = max_interp(*args, **kwargs)
            return self._process_values(
                interpolated, min_interpolated, max_interpolated
            ), min_interpolated, max_interpolated

        return clip_wrapper

    @property
    def _interp_function(self):
        """
        interpolate callable.

        See Also
        --------
        _get_interp_function
        """
        if self._interpolate is None:

            self._interpolate = self._get_interp_function(
                self.domain.magnitude,
                self.magnitude,
                self.interpolator_kwargs,
                self.signal_min,
                self.signal_max
            )

        return self._interpolate

    @property
    def domain_axis(self):
        """
        The axis that corresponds to the domain.

        This axis will always be a positive number.
        """
        return self._domain_axis

    @domain_axis.setter
    def domain_axis(self, value):
        """
        Every other axis should depend on domain_axis.

        Changing domain_axis should automatically change the others.
        """
        domain_axis = self._get_domain_axis(value)

        if domain_axis == self.domain_axis:
            return

        values = np.moveaxis(
            self.magnitude,
            self._axes_order,
            self._get_axes_order(domain_axis)
        )
        signal_min = np.moveaxis(
            self.signal_min,
            self._axes_order,
            self._get_axes_order(domain_axis)
        )
        signal_max = np.moveaxis(
            self.signal_max,
            self._axes_order,
            self._get_axes_order(domain_axis)
        )

        self._values = values
        self._signal_min = signal_min
        self._signal_max = signal_max
        self._domain_axis = domain_axis

    @property
    def boundaries(self):
        """
        Tuple of the minimum and maximum values along domain axis.
        """
        return (
            np.min(self.magnitude, axis=self.domain_axis),
            np.max(self.magnitude, axis=self.domain_axis)
        )

    @property
    def span(self):
        """
        Returns the span along the domain axis.
        """
        return (
            np.max(self.magnitude, axis=self.domain_axis)
            - np.min(self.magnitude, axis=self.domain_axis)
        )

    def _abstract_call(
        self, interp_function, self_domain, domain,
        domain_min, domain_max,
        domain_class, assign_to_name,
        check_bounds=True
    ):
        """
        Method used for `domain_interp`.
        """
        if isinstance(domain, type(self_domain)):
            if domain == self_domain:
                return self.copy()

        domain_units = self_domain.units
        if has_units(domain):
            domain_units = domain.units
            domain_values = domain.to(domain_units).magnitude
        else:
            domain_values = asarray(domain)

        # assert domain_values.ndim <= 1
        # check domain min and max (must be bigger than this range)
        # staticmethod so can be used as is
        if check_bounds:
            self._check_domain_bounds(
                domain_values,
                domain_min=domain_min.magnitude,
                domain_max=domain_max.magnitude
            )

        # usually self._interp_function
        values, smin, smax = interp_function(domain_values)

        # for single value simply return quantity instance
        if not hasattr(values, 'ndim') or (values.ndim != self.ndim):
            return values * self.units
        else:
            new = self.copy()
            new._values = values
            new._signal_min = smin
            new._signal_max = smax
            # assign to _domain usually or _labels
            setattr(
                new, assign_to_name,
                domain_class(
                    domain, units=domain_units,
                    **self_domain._init_kwargs
                )
            )
            return new

    def __call__(self, domain, check_bounds=True):
        """
        Interpolate to signal to new domain values.

        Alias for `domain_interp` method.

        Parameters
        ----------
        domain : dreye.Domain or array-like
            The new domain values to interpolate to. If `domain` has
            not units, it is assumed that the units are the same as for the
            `domain_units` attribute.

        See Also
        --------
        domain_interp
        """

        return self.domain_interp(domain, check_bounds=check_bounds)

    def domain_interp(self, domain, check_bounds=True):
        """
        Interpolate to signal to new domain values.

        Parameters
        ----------
        domain : dreye.Domain or array-like
            The new domain values to interpolate to. If `domain` has
            not units, it is assumed that the units are the same as for the
            `domain_units` attribute.

        See Also
        --------
        __call__
        """

        return self._abstract_call(
            self._interp_function, self.domain, domain,
            self.domain_min, self.domain_max,
            self._domain_class, '_domain',
            check_bounds=check_bounds
        )

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
        Returns the integral along the domain axis as a `pint.Quantity`.

        See Also
        --------
        numpy.trapz

        Notes
        -----
        The integral is calculated using the trapezoidal method using the
        signal and domain magnitude. The units of the integral will correspond
        to `self.units * self.domain_units`.
        """
        return np.trapz(
            self.magnitude,
            self.domain.magnitude,
            axis=self.domain_axis
        ) * self.units * self.domain.units

    @property
    def normalized_signal(self):
        """
        Returns the signal divided by the integral.

        The normalized signal will have units of the inverse of the
        `domain_units`.
        """
        return self / self.integral[self._slices_except_none_domain_axis]

    @property
    def max_normalized(self):
        """
        Returns max normalized signal (across non-domain axes).

        The max-normalized signal will have dimensionless units.
        """
        return self / self.max(axis=self.domain_axis, keepdims=True)

    @property
    def piecewise_integral(self):
        """
        Returns the signal multiplied by the gradient of the domain.

        See Also
        --------
        dreye.Domain.gradient
        """
        return self * self.domain.gradient

    @property
    def piecewise_gradient(self):
        """
        Returns the signal divided by the gradient of the domain.

        See Also
        --------
        dreye.Domain.gradient
        """
        return self / self.domain.gradient

    @property
    def gradient(self):
        """
        Returns the gradient of the signal.

        See Also
        --------
        numpy.gradient
        """
        values = np.gradient(
            self.magnitude, self.domain.magnitude, axis=self.domain_axis
        )
        units = self.units / self.domain.units
        return self._class_new_instance(
            values=values, units=units, **self._init_kwargs
        )

    def enforce_uniformity(self):
        """
        Returns a new signal instance with a uniform-interval domain.
        """
        domain = self.domain.enforce_uniformity()
        return self(domain)

    def filter(
        self, domain_interval,
        method='savgol', extrapolate=False,
        **method_args
    ):
        """
        Filter signal using `scipy.signal.windows` function or
        the `savgol` method.

        Parameters
        ----------
        domain_interval : numeric, optional
            The domain interval window to use for filtering. This should
            be in units of `domain_units` or be convertible to these
            units using `pint`'s `to` method.
        method : str, optional
            The method used for filtering the signal. Defaults to 'savgol'.
            See `scipy.signal.windows` for more options.
        extrapolate : bool, optional
            Whether to extrapolate when applying a window filter from
            `scipy.signal.windows`, in order to deal with edge cases.
        method_args : dict, optional
            Arguments passed to the filter method as is.

        Returns
        -------
        object : signal-type
            Filtered version of `self`.

        See Also
        --------
        dreye.utilities.Filter1D
        """

        assert self.domain.is_uniform, (
            "signal domain must be uniform for filtering"
        )

        domain_interval = optional_to(
            domain_interval, self.domain.units,
            *self.contexts
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

        # filtering is shape-preserving
        # -> no need to reassign signal_min and max
        # sanity check
        assert values.shape == self.shape, "Shape mismatch for filtering."
        new = self.copy()
        values = new._process_values(values, self.signal_min, self.signal_max)
        new._values = values
        return new

    @property
    def smoothing_args(self):
        """
        Keyword arguments used for smoothing the signal.
        """
        return self._smoothing_args

    @property
    def smoothing_window(self):
        """
        The standard size of the smoothing window.
        """
        return self._smoothing_window

    @property
    def smoothing_method(self):
        """
        The standard method used for smoothing.
        """
        return self._smoothing_method

    def smooth(self, smoothing_window=None):
        """
        Performs smoothing on signal type.

        Parameters
        ----------
        smoothing_window : numeric, optional
            Size of smoothing window in domain units.

        Returns
        -------
        object : signal-type
            Smoothed signal type.

        See Also
        --------
        filter
        """

        if smoothing_window is None:
            smoothing_window = self.smoothing_window

        spectrum = self.filter(
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
        Creates a new signal-type instance by appending two signals
        along the domain axis.

        Parameters
        ----------
        other : signal-type or array-like
            A signal type instance or array-like.
        left : bool, optional
            Append other on the left side.

        Returns
        -------
        object : signal-type
            A new domain-concatenated signal instance.

        See Also
        --------
        append
        numpy.concatenate
        """

        domain = self.domain

        if isinstance(other, _SignalMixin):
            # checks dimensionality, appends domain, converts units
            assert self._other_shape == other._other_shape, \
                "Shapes are not compatible."

            # concatenate domain
            domain = domain.append(other.domain, left=left)
            # convert units
            other = other.to(self.units)
            other_mag = other.magnitude
            other_min = other.signal_min
            other_max = other.signal_max

        elif is_listlike(other):
            # handles units, checks other shape, extends domain
            other_mag = optional_to(
                other, self.units, *self.contexts,
                **self._unit_conversion_kws)

            assert (
                self._other_shape
                == self._get_other_shape(other_mag.shape)
            ), "shapes do not match for concatenation"

            domain = domain.extend(
                other_mag.shape[self.domain_axis],
                left=left
            )
            other_min = self._get_signal_bound(other_mag, None)
            other_max = self._get_signal_bound(other_mag, None)

        else:
            raise DreyeError("Domain axis contenation "
                             f"with type: {type(other)}")

        if left:
            values = np.concatenate(
                [other_mag, self.magnitude],
                axis=self.domain_axis
            )
            smin = np.concatenate(
                [other_min, self.signal_min],
                axis=self.domain_axis
            )
            smax = np.concatenate(
                [other_max, self.signal_max],
                axis=self.domain_axis
            )

        else:
            values = np.concatenate(
                [self.magnitude, other_mag],
                axis=self.domain_axis
            )
            smin = np.concatenate(
                [self.signal_min, other_min],
                axis=self.domain_axis
            )
            smax = np.concatenate(
                [self.signal_max, other_max],
                axis=self.domain_axis
            )

        # concatenation is not shape preserving,
        # but should not require creating from anew
        new = self.copy()
        new._values = values
        new._signal_min = smin
        new._signal_max = smax
        new._domain = domain
        return new

    def append(self, other, *args, **kwargs):
        """
        Creates a new signal-type instance by appending two signals
        along the domain axis.

        Parameters
        ----------
        other : signal-type or array-like
            A signal type instance or array-like.
        left : bool, optional
            Append other on the left side.

        Returns
        -------
        object : signal-type
            A new domain-concatenated signal instance.

        See Also
        --------
        domain_concat
        numpy.concatenate
        """
        return self.domain_concat(other, *args, **kwargs)

    def equalize_domains(self, other):
        """
        Equalize domains for two signal-type instances.

        Parameters
        ----------
        other : signal-type
            A signal-type instance that have equalizable domains.

        Returns
        -------
        self, other : signal-type
            Tuple containing self and other. This will be a copy of self and
            other, if they had to be equalized.
        """
        if self.domain != other.domain:
            domain = self.domain.equalize_domains(other.domain)
            return self(domain), other(domain)
        return self, other

    def _slices_ndim(self, ndim):
        """
        Given a dimensionality, return domain-aligned slices.

        (domain_axis, a3, a2, a1) and ndim 2
        -> (slice, None, None, slice)
        (a1, domain_axis, a3, a2) and ndim 2
        -> (slice, slice, None, None)

        Parameters
        ----------
        ndim : int
            Assumes ndim < self.ndim.
        """
        slices = [None] * self.ndim
        for i in range(ndim):
            slices[self.domain_axis - i] = slice(None, None, None)
        return tuple(slices)

    @property
    def _axes_order(self):
        """
        Order of axis with domain being zero.

        (domain_axis, a3, a2, a1)
        -> (0, -3, -2, -1)
        (a3, a2, a1, domain_axis)
        -> (-3, -2, -1, 0)

        Returns
        -------
        axes_order : np.ndarray
        """
        return self._get_axes_order(self.domain_axis)

    def _get_axes_order(self, domain_axis):
        """
        Get axis order
        """
        order = -np.arange(self.ndim)[::-1]
        return np.roll(order, shift=domain_axis+1)

    def _equalize(self, other):
        """
        Should just return equalized other_magnitude or NotImplemented
        """
        # TODO error messages!
        if isinstance(other, _UnitArray):
            if isinstance(other, Domain):
                return other.magnitude[
                    self._none_slices_except_domain_axis
                ], self
            elif isinstance(other, _SignalMixin):
                # Do reverse if self ndim is smaller
                if self.ndim < other.ndim:
                    return NotImplemented, self
                # equalize domains
                self, other = self.equalize_domains(other)
                if sum(self._other_shape) < sum(other._other_shape):
                    # switch roles
                    self, other = other, self
                other_magnitude = other.magnitude
                if self.ndim > other.ndim:
                    # all possible dim before domain_axis is assumed the same
                    # and missing dimensions are added (prepending)
                    # if other (domain, a1) -> (a1, domain):
                    # self (domain, a2, a1) -> other (domain, None, a1)
                    # self (a2, a1, domain) -> other (None, a1, domain)
                    # if other (a1, domain): -> other (a1, domain)
                    # self (domain, a2, a1) -> other (domain, None, a1)
                    # self (a2, a1, domain) -> other (None, a1, domain)

                    # move domain position and add missing dimension
                    # (None, None, ..., a3, a2, a1, domain_axis)
                    other_magnitude = np.moveaxis(
                        other_magnitude,
                        other._axes_order,
                        # domain axis make last
                        -np.arange(other.ndim)[::-1]
                    )[(None,) * (self.ndim - other.ndim)]
                    # move domain to same position as in self
                    other_magnitude = np.moveaxis(
                        other_magnitude,
                        # domain axis is last
                        -np.arange(self.ndim)[::-1],
                        self._axes_order
                    )

                elif self.domain_axis != other.domain_axis:
                    # self (domain, a2, a1) and other (a2, a1, domain)
                    # -> other (domain, a2, a1)
                    other_magnitude = np.moveaxis(
                        other_magnitude,
                        other._axes_order,
                        self._axes_order
                    )
                return other_magnitude, self
            else:
                return NotImplemented, self
        elif is_numeric(other):
            return get_value(other), self
        elif is_listlike(other):
            # TODO error message for larger dimensions,
            # instead of notimplemented
            other = asarray(other)
            if other.ndim == self.ndim:
                return other, self
            elif other.ndim > self.ndim:
                return NotImplemented, self
            else:
                slices = self._slices_ndim(other.ndim)
                return other[slices], self
        else:
            return NotImplemented, self

    @property
    def iterdomain(self):
        """
        Iterate over domain values.

        Yields
        ------
        value : `pint.Quantity`
        """
        iter(np.moveaxis(self.magnitude, self.domain_axis, 0) * self.units)

    @property
    def nanless(self):
        """
        Returns signal-type with NaNs removed.

        See Also
        --------
        domain_interp
        __call__
        """
        arange = self.domain.magnitude
        values = np.zeros(self.shape)
        interpolator_kwargs = self.interpolator_kwargs.copy()
        interpolator_kwargs.pop('axis', None)

        for slice, iarr in self._iter_values:
            finites = np.isfinite(iarr)
            # interpolate nans
            ivalues = self.interpolator(
                arange[finites], iarr[finites],
                **interpolator_kwargs
            )(arange)
            ivalues[finites] = iarr[finites]
            values[slice] = ivalues

        # shape-preserving
        # sanity check
        assert values.shape == self.shape, "Nanless didn't preserve shape!"
        new = self.copy()
        new._values = values
        return new

    def _assign_further_longdf_values(self, df):
        """
        Assign other values to a long dataframe given self.
        """
        return df

    def to_longframe(self, ignore_signal_bounds=True):
        """
        Convert signal class to a long dataframe.

        The long dataframe contains the `name`, `domain`,
        `domain_min`, `domain_max`, `attrs` (expanded),
        `signal_min` (optional), `signal_max` (optional),
        and the dimensionality of units.

        Returns
        -------
        object : `pandas.DataFrame`
            A long-format dataframe.
        """
        self._ignore_signal_bounds = ignore_signal_bounds

        df = self.to_frame()
        while isinstance(df, pd.DataFrame):
            df = df.stack()

        df.name = 'values'
        df = df.reset_index()

        df['name'] = self.name
        df['units'] = str(self.units)
        df['units_dimensionality'] = str(self.units.dimensionality)
        df['domain_units'] = str(self.domain.units)
        df['domain_units_dimensionality'] = str(
            self.domain.units.dimensionality
        )
        df['domain_min'] = self.domain_min.magnitude
        df['domain_max'] = self.domain_max.magnitude

        df = self._assign_further_longdf_values(df)

        reserved_keys = df.columns

        for key, ele in self.attrs.items():
            if not is_hashable(ele):
                continue
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

    def __str__(self):
        """
        Standard string representation for single dimensional signal classes
        """
        return (
            f"{type(self).__name__}"
            f"(\n\t name={self.name}, \n\t "
            f"units={self.units}, \n\t "
            f"domain={self.domain} \n )"
        )

    # TODO peak detection (min/max)
    # TODO peak summary - FWHM, HWHM-left, HWHM-right, domain value
    # TODO rolling window single D signal


@inherit_docstrings
class _Signal2DMixin(_SignalMixin):
    _init_args = _SignalMixin._init_args + ('labels',)

    @property
    def _init_aligned_attrs(self):
        return {
            self.domain_axis: ('domain',),
            self.labels_axis: ('labels',)
        }

    def __init__(
        self,
        values,
        domain=None,
        labels=None,
        *,
        domain_axis=None,
        **kwargs
    ):
        """
        Added labels to init.
        """
        # change domain axis if necessary
        if isinstance(values, _Signal2DMixin) and domain_axis is not None:
            if domain_axis != values.domain_axis:
                values = values.copy()
                values.domain_axis = domain_axis
        super().__init__(
            values=values,
            domain=domain,
            labels=labels,
            domain_axis=domain_axis,
            **kwargs
        )

    @abstractmethod
    def _get_labels(self, values, labels, kwargs):
        pass

    @abstractmethod
    def labels_concat(self, other):
        """
        Concatenate to two-dimensional signal-type instances along
        the labels axis.

        Parameters
        ----------
        other : signal-type
            A two-dimensional signal type
        left : bool, optional
            If other is concatenated from the left side or not.

        Returns
        -------
        object : signal-type
            A new concatenated signal type

        See Also
        --------
        concat
        """
        pass

    def _preextract_attributes(self, values, kwargs):
        """
        Extract various attributes from a pandas instance if passed
        attributes is None. Used by _test_and_assign_values
        """
        if isinstance(values, pd.DataFrame):
            if self.domain is None:
                if self.domain_axis:
                    self._domain = values.columns
                else:
                    self._domain = values.index
            if self.labels is None:
                if self.domain_axis:
                    self._labels = values.index
                else:
                    self._labels = values.columns

        if isinstance(values, pd.Series):
            if self.domain is None:
                self._domain = values.index

        if isinstance(values, (pd.DataFrame, pd.Series)):
            if self.name is None:
                self.name = values.name

    def _preprocess_check_values(self, values):
        # must return values processed
        if values.ndim != 2:
            raise DreyeError("Values must be a one- or two-dimensional array.")
        return values

    def _process_other_attributes(self, values, kwargs):
        # check labels
        self._labels = self._get_labels(values, self._labels, kwargs)

    @property
    def labels(self):
        """
        Labels for the two-dimensional signal-type instance.
        """
        return self._labels

    @labels.setter
    def labels(self, value):
        """
        Set new labels
        """
        self._labels = self._get_labels(self.magnitude, value, {})

    @property
    def labels_axis(self):
        """
        Axis that correspond to the label dimension.

        The labels axis is always a positive integer.
        """
        # always behind domain axis
        return (self.domain_axis - 1) % self.ndim

    @property
    def ndim(self):
        return 2

    @property
    def iterlabels(self):
        """
        Iterate over the labels axis.

        Yields
        ------
        slice : `pint.Quantity`
        """
        return iter(
            np.moveaxis(self.magnitude, self.labels_axis, 0)
            * self.units
        )

    @property
    def _iter_values(self):
        """
        Iterate over numpy arrays along labels.

        zip(slices, values)

        slice: (index_label, slice(None, None, None)) for domain_axis = 1
        slice: (slice(None, None, None), index_label) for domain_axis = 0
        """
        _slice = list(self._none_slices_except_domain_axis)
        slices = []
        for index in range(self.shape[self.labels_axis]):
            _slice[self.labels_axis] = index
            slices.append(tuple(_slice))
        return zip(
            slices,
            np.moveaxis(self.magnitude, self.labels_axis, 0)
        )

    def __str__(self):
        return (
            f"{type(self).__name__}"
            f"(\n\t name={self.name}, \n\t units={self.units}, \n\t "
            f"domain={self.domain}, \n\t labels={self.labels} \n )"
        )

    def concat(self, other, *args, **kwargs):
        """
        Concatenate to two-dimensional signal-type instances along
        the labels axis.

        Parameters
        ----------
        other : signal-type
            A two-dimensional signal type
        left : bool, optional
            If other is concatenated from the left side or not.

        Returns
        -------
        object : signal-type
            A new concatenated signal type

        See Also
        --------
        labels_concat
        """

        return self.labels_concat(other, *args, **kwargs)

    def to_frame(self, data='magnitude'):
        if self.domain_axis:
            index, index_name = list(get_value(self.labels)), 'labels'
            columns, columns_name = self.domain.magnitude, 'domain'
        else:
            columns, columns_name = list(get_value(self.labels)), 'labels'
            index, index_name = self.domain.magnitude, 'domain'

        df = pd.DataFrame(
            getattr(self, data),
            index=index,
            columns=columns
        )
        df.columns.name = columns_name
        df.index.name = index_name
        return df

    def _assign_further_longdf_values(self, df):
        # stack all
        if not self._ignore_signal_bounds:
            signal_min = self.to_frame('signal_min').stack()
            signal_min.name = 'signal_min'
            signal_min = signal_min.reset_index()

            df = df.merge(signal_min, on=['labels', 'domain'], how='left')

            signal_max = self.to_frame('signal_max').stack()
            signal_max.name = 'signal_max'
            signal_max = signal_max.reset_index()

            df = df.merge(signal_max, on=['labels', 'domain'], how='left')

        return df

    @property
    def T(self):
        """
        Transpose signal-type instance.
        """
        new = self.copy()
        new.domain_axis = self.domain_axis - 1
        return new


@inherit_docstrings
class _SignalIndexLabels(_Signal2DMixin):

    @property
    def _init_aligned_attrs(self):
        return {
            self.labels_axis: ('labels',),
            **(super()._init_aligned_attrs)
        }

    def __init__(
        self,
        values,
        domain=None,
        labels=None,
        **kwargs
    ):
        if (
            isinstance(values, _SignalMixin)
            and (values.ndim == 1)
            and labels is None
            and not hasattr(values, 'labels')
        ):
            # set labels to name if not existing labels
            labels = values.name

        super().__init__(
            values=values,
            domain=domain,
            labels=labels,
            **kwargs
        )

    def _preprocess_check_values(self, values):
        # must return values processed
        if values.ndim == 1:
            values = values[self._none_slices_except_domain_axis]
        if values.ndim != 2:
            raise DreyeError("Values must be a one- or two-dimensional array.")
        return values

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
            labels = pd.Index([labels] * values.shape[self.labels_axis])
        else:
            raise DreyeError(f"Labels wrong type: '{type(labels)}'")

        assert len(labels) == values.shape[self.labels_axis], (
            "Labels must be same length as label axis"
        )

        return labels

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
                np.concatenate(
                    [signal_min, self.signal_min], axis=self.labels_axis),
                np.concatenate(
                    [signal_max, self.signal_max], axis=self.labels_axis)
            )
        else:
            return (
                np.concatenate(
                    [self.signal_min, signal_min], axis=self.labels_axis),
                np.concatenate(
                    [self.signal_max, signal_max], axis=self.labels_axis)
            )

    def labels_concat(
        self, other, labels=None,
        signal_min=None, signal_max=None,
        left=False
    ):

        # docstring is copied over
        # domain_axis must be aligned
        if (
            isinstance(other, _SignalMixin)
            and not isinstance(other, _SignalIndexLabels)
            and other.ndim <= 2
        ):
            other = self._class_new_instance(other)
            other.domain_axis = self.domain_axis

        if isinstance(other, _SignalIndexLabels):
            # equalizing domains
            self, other = self.equalize_domains(other)
            self_values = self.magnitude
            # convert units - will also convert signal bounds
            other = other.to(self.units)
            other_values = other.magnitude
            # labels
            labels = other.labels
            signal_min = other.signal_min
            signal_max = other.signal_max

        elif is_listlike(other):
            # self numpy array
            self_values = self.magnitude
            # check if it has units
            other_values = optional_to(
                other, self.units, *self.contexts,
                **self._unit_conversion_kws)
            # handle labels and bounds
            labels = self._get_labels(other_values, labels)
            signal_min = self._get_signal_bound(other_values, signal_min)
            signal_max = self._get_signal_bound(other_values, signal_max)

        else:
            raise DreyeError("other axis contenation "
                             f"with type: {type(other)}.")

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

        # labels_concat is not shape preserving
        new = self.copy()
        new._labels = labels
        new._signal_min = signal_min
        new._signal_max = signal_max
        new._values = new._process_values(values, signal_min, signal_max)
        return new

    def _assign_further_longdf_values(self, df):
        # merge bounds
        df = super()._assign_further_longdf_values(df)

        # if labels are multiindex include them
        if isinstance(self.labels, pd.MultiIndex):
            df_labels = self.labels.to_frame(index=False)
            df_labels['labels'] = list(get_value(self.labels))
            df = df.merge(
                df_labels, on='labels', how='left',
                suffixes=('', '_label')
            )
        return df


@inherit_docstrings
class _SignalDomainLabels(_Signal2DMixin):
    _init_args = _Signal2DMixin._init_args + ('labels_min', 'labels_max')
    _domain_labels_class = Domain

    @property
    def _switch_new_instance(self):
        return self._class_new_instance

    def __init__(
        self,
        values,
        domain=None,
        labels=None,
        **kwargs
    ):

        if isinstance(values, _SignalDomainLabels) and labels is not None:
            if values.labels != labels:
                values = values.labels_interp(labels)

        super().__init__(
            values=values,
            domain=domain,
            labels=labels,
            **kwargs
        )

    def equalize_domains(self, other):
        self, other = super().equalize_domains(other)
        if isinstance(other, _SignalDomainLabels):
            if self.labels != other.labels:
                labels = self.labels.equalize_domains(other.labels)
                return self.labels_interp(labels), other.labels_interp(labels)
        return self, other

    @property
    def labels_units(self):
        """
        Units assigned to `dreye.Domain` instance of `labels` attribute.
        """
        return self.labels.units

    @property
    def switch(self):
        """
        Returns domain-signal-type with switched domain and labels.

        `self.labels` become `self.domain`, and vice versa.

        Notes
        -----
        Using `T` is different from using `switch`. `T` only transposes the
        matrix but does not switch what is referred to as the `domain` and
        `labels`. This affects mathematical operations and interpolation.
        """
        return self._switch_new_instance(
            values=self.magnitude,
            units=self.units,
            **{
                **self._init_kwargs,
                **dict(
                    labels=self.domain,
                    domain=self.labels,
                    domain_min=self.labels_min,
                    domain_max=self.labels_max,
                    labels_min=self.domain_min,
                    labels_max=self.domain_max,
                    signal_min=self.signal_min,
                    signal_max=self.signal_max,
                    domain_axis=self.domain_axis - 1,
                )
            }
        )

    @property
    def _labels_interp_function(self):
        """
        interpolate callable
        """
        if self._labels_interpolate is None:

            interpolator_kwargs = self.interpolator_kwargs
            interpolator_kwargs['axis'] = self.labels_axis

            self._labels_interpolate = self._get_interp_function(
                self.labels.magnitude,
                self.magnitude,
                interpolator_kwargs,
                self.signal_min,
                self.signal_max
            )

        return self._labels_interpolate

    def labels_interp(self, domain, check_bounds=True):
        """
        Interpolate to new labels.

        Parameters
        ----------
        domain : dreye.Domain or array-like
            The new domain values to interpolate to along the `labels`
            attribute. If `domain` does not have
            units, it is assumed that the units are the same as for the
            `domain_units` attribute.

        See Also
        --------
        domain_interp
        """

        values = self._abstract_call(
            self._labels_interp_function, self.labels, domain,
            self.labels_min, self.labels_max,
            self._domain_labels_class, '_labels',
            check_bounds=check_bounds
        )
        if values.ndim == 1:
            values = self._1d_signal_class(
                values,
                domain=self.domain,
                name=domain
            )
        return values

    # TODO without switch (i.e. no reinstantiation)
    def labels_concat(self, other, left=False):
        return self.switch.domain_concat(other, left=left).switch

    def _get_labels(self, values, labels, kwargs={}):
        labels_units = kwargs.get('labels_units', None)
        labels_kwargs = kwargs.get('labels_kwargs', None)
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
        Returns the minimum value for `labels` attribute.

        This property is used during labels interpolation. If all values for
        interpolation are bigger than `labels_min`, interpolation will
        result in an error.
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
        Returns the maximum value for `labels` attribute.

        This property is used during labels interpolation. If all values for
        interpolation are smaller than `labels_max`, interpolation will
        result in an error.
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


@inherit_docstrings
class Signal(_SignalMixin):
    """
    Defines the base class for a continuous
    one-dimensional signal (unit-aware).

    Parameters
    ----------
    values : array-like, str, signal-type
        One-dimensional array that contains the value of your signal.
    domain : `dreye.Domain` or array-like, optional
        The domain of the signal. This needs to be the same length as
        `values`.
    units : str or `pint.Unit`, optional
        Units of the `values` array.
    domain_units : str or `pint.Unit`, optional
        Units of the `domain` array.
    domain_min : numeric, optional
        Defines the minimum value in your domain for the intpolation range.
    domain_max : numeric, optional
        Defines the minimum value in your domain for the intpolation range.
    signal_min : numeric or array-like, optional
        Will clip your signal to a minimum. Everything below this minimum will
        be set to the minumum.
    signal_max : numeric or array-like, optional
        Will clip your signal to a maximum. Everything above this maximum will
        be set to the maximum.
    attrs : dict, optoinal
        User-defined dictionary of objects that are associated with the
        signal, but that are not used for any particular computations.
    name : str, optional
        Name of the signal instance.
    interpolator : interpolate class, optional
        Callable function that allows you to interpolate between points. The
        callable should accept two positional arguments as `numpy.ndarray`
        objects and accept the keyword argument `axis`.
        Defaults to `scipy.interpolate.interp1d`.
    interpolator_kwargs : dict-like, optional
        Dictionary to specify other keyword arguments that are passed to
        the `interpolator`.
    smoothing_method : str, optional
        Smoothing method used when using the `smooth` method.
        Defaults to `savgol`.
    smoothing_window : numeric, optional
        Standard window size in units of the domain to smooth the signal.
    smoothing_args : dict, optional
        Keyword arguments passed to the `filter` method when smoothing.
    contexts : str or tuple, optoinal
        Contexts for unit conversion. See `pint` package.

    See Also
    --------
    Signals
    DomainSignal
    Spectrum
    IntensitySpectrum
    CalibrationSpectrum
    """

    @property
    def _class_new_instance(self):
        return Signal

    def __init__(
        self,
        values,
        domain=None,
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
        smoothing_method=None,
        smoothing_window=None,
        smoothing_args=None,
        attrs=None,
        name=None,
        domain_axis=None
    ):
        super().__init__(
            values=values,
            units=units,
            domain=domain,
            domain_units=domain_units,
            interpolator=interpolator,
            interpolator_kwargs=interpolator_kwargs,
            contexts=contexts,
            domain_kwargs=domain_kwargs,
            domain_min=domain_min,
            domain_max=domain_max,
            signal_min=signal_min,
            signal_max=signal_max,
            smoothing_method=smoothing_method,
            smoothing_window=smoothing_window,
            smoothing_args=smoothing_args,
            attrs=attrs,
            name=name,
            domain_axis=domain_axis
        )


@inherit_docstrings
class Signals(_SignalIndexLabels):
    """
    Defines the base class for a set of continuous
    one-dimensional signals (unit-aware).

    Parameters
    ----------
    values : array-like, str, signal-type
        Two-dimensional array that contains the value of your signal.
    domain : `dreye.Domain` or array-like, optional
        The domain of the signal. This needs to be the same length of
        the `values` array along the axis of the domain.
    labels : array-like, optional
        A set of hashable objects that describe each individual signal.
        If None, ascending integer values are used as labels.
    units : str or `pint.Unit`, optional
        Units of the `values` array.
    domain_units : str or `pint.Unit`, optional
        Units of the `domain` array.
    domain_axis : int, optional
        The axis that corresponds to the `domain` argument. Defaults to 0.
    domain_min : numeric, optional
        Defines the minimum value in your domain for the intpolation range.
    domain_max : numeric, optional
        Defines the minimum value in your domain for the intpolation range.
    signal_min : numeric or array-like, optional
        Will clip your signal to a minimum. Everything below this minimum will
        be set to the minumum.
    signal_max : numeric or array-like, optional
        Will clip your signal to a maximum. Everything above this maximum will
        be set to the maximum.
    attrs : dict, optoinal
        User-defined dictionary of objects that are associated with the
        signal, but that are not used for any particular computations.
    name : str, optional
        Name of the signal instance.
    interpolator : interpolate class, optional
        Callable function that allows you to interpolate between points. The
        callable should accept two positional arguments as `numpy.ndarray`
        objects and accept the keyword argument `axis`.
        Defaults to `scipy.interpolate.interp1d`.
    interpolator_kwargs : dict-like, optional
        Dictionary to specify other keyword arguments that are passed to
        the `interpolator`.
    smoothing_method : str, optional
        Smoothing method used when using the `smooth` method.
        Defaults to `savgol`.
    smoothing_window : numeric, optional
        Standard window size in units of the domain to smooth the signal.
    smoothing_args : dict, optional
        Keyword arguments passed to the `filter` method when smoothing.
    contexts : str or tuple, optoinal
        Contexts for unit conversion. See `pint` package.

    See Also
    --------
    Signal
    DomainSignal
    Spectra
    IntensitySpectra
    """

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
        smoothing_method=None,
        smoothing_window=None,
        smoothing_args=None,
        attrs=None,
        name=None,
        domain_axis=None
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
            smoothing_method=smoothing_method,
            smoothing_window=smoothing_window,
            smoothing_args=smoothing_args,
            attrs=attrs,
            name=name,
            domain_axis=domain_axis
        )


@inherit_docstrings
class DomainSignal(_SignalDomainLabels):
    """
    Defines the base class for a two-dimensional signal (unit-aware).

    Parameters
    ----------
    values : array-like, str, signal-type
        Two-dimensional array that contains the value of your signal.
    domain : `dreye.Domain` or array-like, optional
        The domain of the signal. This needs to be the same length of
        the `values` array along the axis of the domain.
    labels : `dreye.Domain` or array-like, optional
        The domain of the signal along the other axis.
        This needs to be the same length of
        the `values` array along the axis of the labels.
    units : str or `pint.Unit`, optional
        Units of the `values` array.
    domain_units : str or `pint.Unit`, optional
        Units of the `domain` array.
    labels_units : str or `pint.Unit`, optional
        Units of the `labels` array.
    domain_axis : int, optional
        The axis that corresponds to the `domain` argument. Defaults to 0.
    domain_min : numeric, optional
        Defines the minimum value in your domain for the intpolation range.
    domain_max : numeric, optional
        Defines the minimum value in your domain for the intpolation range.
    signal_min : numeric or array-like, optional
        Will clip your signal to a minimum. Everything below this minimum will
        be set to the minumum.
    signal_max : numeric or array-like, optional
        Will clip your signal to a maximum. Everything above this maximum will
        be set to the maximum.
    attrs : dict, optoinal
        User-defined dictionary of objects that are associated with the
        signal, but that are not used for any particular computations.
    name : str, optional
        Name of the signal instance.
    interpolator : interpolate class, optional
        Callable function that allows you to interpolate between points. The
        callable should accept two positional arguments as `numpy.ndarray`
        objects and accept the keyword argument `axis`.
        Defaults to `scipy.interpolate.interp1d`.
    interpolator_kwargs : dict-like, optional
        Dictionary to specify other keyword arguments that are passed to
        the `interpolator`.
    smoothing_method : str, optional
        Smoothing method used when using the `smooth` method.
        Defaults to `savgol`.
    smoothing_window : numeric, optional
        Standard window size in units of the domain to smooth the signal.
    smoothing_args : dict, optional
        Keyword arguments passed to the `filter` method when smoothing.
    contexts : str or tuple, optoinal
        Contexts for unit conversion. See `pint` package.

    See Also
    --------
    Signal
    Signals
    DomainSpectrum
    IntensityDomainSpectrum
    MeasuredSpectrum
    """

    @property
    def _class_new_instance(self):
        return DomainSignal

    @property
    def _1d_signal_class(self):
        return Signal

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
        smoothing_method=None,
        smoothing_window=None,
        smoothing_args=None,
        attrs=None,
        name=None,
        domain_axis=None,
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
            smoothing_method=smoothing_method,
            smoothing_window=smoothing_window,
            smoothing_args=smoothing_args,
            attrs=attrs,
            name=name,
            domain_axis=domain_axis,
        )
