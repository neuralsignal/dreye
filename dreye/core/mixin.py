"""Mixin class for signal and domain
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.optimize import root, least_squares
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression

from dreye.err import DreyeUnitError, DreyeError
from dreye.utilities import (
    convert_units, diag_chunks, is_listlike, arange,
    is_numeric, is_uniform, array_domain, has_units,
    asarray
)
from dreye.constants import UREG, ABSOLUTE_ACCURACY, DEFAULT_FLOAT_DTYPE
from dreye.core.abstract import AbstractSignal, AbstractDomain


class UnpackDomainMixin(ABC):

    @classmethod
    def unpack(cls, values=None, **kwargs):
        """
        Returns correct attributes for domain.

        Parameters
        ----------

        Returns
        -------
        """

        for key in kwargs:
            assert key in cls._required

        if not is_numeric(kwargs['start']) and values is None:

            values = kwargs['start']
            kwargs['start'] = None

        if isinstance(values, str):

            values = cls.load(values, dtype=kwargs['dtype'])

        if values is None:
            pass

        elif isinstance(values, AbstractDomain):

            values.units = kwargs['units']

            # run through all keys
            for key, item in kwargs.items():
                if item is None and hasattr(values, key):
                    kwargs[key] = getattr(values, key)

        else:

            if has_units(values):
                # assign units or convert values
                if kwargs['units'] is None:
                    kwargs['units'] = values.units
                else:
                    values = values.to(kwargs['units'])

            values = asarray(values)

            start, end, interval = array_domain(values,
                                                uniform=is_uniform(values))

            if kwargs['start'] is None:
                kwargs['start'] = start
            if kwargs['end'] is None:
                kwargs['end'] = end
            if kwargs['interval'] is None:
                kwargs['interval'] = interval

        cls._extract_check_units(kwargs)

        dtype = kwargs['dtype']

        kwargs['start'], kwargs['end'] = \
            dtype(kwargs['start']), dtype(kwargs['end'])

        if (
            (kwargs['start'] is None)
            or (kwargs['end'] is None)
            or (kwargs['units'] is None)
            or (kwargs['interval'] is None)
        ):

            raise TypeError(
                f"Unable to create Domain; None types present: {kwargs}")

        values, kwargs['interval'] = cls.create_values(kwargs['start'],
                                                       kwargs['end'],
                                                       kwargs['interval'],
                                                       dtype)

        return values, kwargs

    @staticmethod
    def _extract_check_units(kwargs):
        """function used by unpack to return units
        """

        if isinstance(kwargs['units'], str):
            kwargs['units'] = UREG(kwargs['units']).units

        for key, item in kwargs.items():
            if has_units(item):
                if kwargs['units'] is None:
                    kwargs['units'] = item.units
                else:
                    kwargs[key] = item.to(kwargs['units'])

        if kwargs['units'] is None:
            kwargs['units'] = UREG(None).units

    @staticmethod
    def create_values(start, end, interval, dtype=DEFAULT_FLOAT_DTYPE):
        """
        """

        if is_listlike(interval):
            # TODO: dealing with tolerance
            interval = asarray(interval)
            interval_diff = (end - start) - np.sum(interval)
            if interval_diff > 0:
                # TODO: raise warning
                # raise ValueError('Intervals given smaller than range')
                interval = np.append(interval, interval_diff)
            elif interval_diff < 0:
                raise DreyeError('Intervals given bigger than range.')

            values = np.append(start, start + np.cumsum(interval))
            values = values.astype(dtype)
            interval = interval.astype(dtype)

            if values.size != np.unique(values).size:
                raise DreyeError('values are non-unique: {0}'.format(values))

        elif is_numeric(interval):
            if interval > end - start:
                raise DreyeError(('interval attribute: value '
                                  '{0} bigger than range {1}').format(
                                      interval, end - start))

            values, interval = arange(start, end, interval, dtype=dtype)
            interval = dtype(interval)

        else:
            raise DreyeError(('interval not type numeric or '
                             'array-like: {0}').format(type(interval)))

        return values, interval


class UnpackSignalMixin(ABC):

    @classmethod
    def unpack(cls,
               values,
               units=None,
               domain_units=None,
               dtype=DEFAULT_FLOAT_DTYPE,
               domain_dtype=None,
               domain_kwargs=None,
               **kwargs):
        """
        """

        if domain_kwargs is None:
            domain_kwargs = {}

        for key in kwargs:
            assert key in cls.init_args, (
                f'{key} not in init_args for {cls.__name__}.'
            )

        if isinstance(values, str) or isinstance(values, AbstractSignal):

            if isinstance(values, str):
                values = cls.load(values, dtype=dtype)

            units, kwargs = \
                cls._extract_attr_from_signal_instance(
                    values, units, **kwargs
                )

            kwargs['domain'] = cls.get_domain(kwargs['domain'], domain_units,
                                              domain_dtype, **domain_kwargs)

            values = cls.convert_values(
                values.values,
                units,
                kwargs['contexts'],
                domain=kwargs['domain'].values,
                axis=values.other_axis
            )

            values = asarray(values).astype(dtype)

        elif isinstance(values, (pd.DataFrame, pd.Series)):

            kwargs.update(
                cls._extract_attr_from_pandas(values, kwargs['domain'],
                                              kwargs['domain_axis'],
                                              kwargs['labels']))

            values = asarray(values).astype(dtype)
            kwargs['domain'] = cls.get_domain(kwargs['domain'], domain_units,
                                              domain_dtype, **domain_kwargs)

        elif is_listlike(values):
            # TODO what about a list of signal? - concatenation?

            kwargs['domain'] = cls.get_domain(kwargs['domain'], domain_units,
                                              domain_dtype, **domain_kwargs)

            if hasattr(values, 'units') and units is None:
                units = values.units
            elif hasattr(values, 'units'):
                # TODO check contexts etc.
                values = cls.convert_values(
                    values,
                    units,
                    kwargs['contexts'],
                    domain=kwargs['domain'].values,
                    axis=(
                        1 if kwargs['domain_axis'] is None
                        else (kwargs['domain_axis'] + 1) % 2
                    )
                )

            values = asarray(values).astype(dtype)

        else:
            raise TypeError('values must be string or array-like.')

        assert values.ndim < 3, "signal array must be 1D or 2D."

        if kwargs['domain_axis'] is None:
            kwargs['domain_axis'] = 0
        assert kwargs['domain_axis'] in [0, 1, -1, -2], \
            'domain axis must be in 0 or 1'

        if kwargs['interpolator'] is None:
            kwargs['interpolator'] = interp1d
        assert hasattr(kwargs['interpolator'], '__call__')

        if kwargs['interpolator_kwargs'] is None:
            kwargs['interpolator_kwargs'] = {}
        assert isinstance(kwargs['interpolator_kwargs'], dict)

        if isinstance(units, str) or units is None:
            units = UREG(units).units

        values, kwargs['labels'] = cls.check_values(values, kwargs['domain'],
                                                    kwargs['domain_axis'],
                                                    kwargs['labels'])

        return values, units, kwargs

    @classmethod
    def get_domain(cls, domain, domain_units, domain_dtype, **kwargs):
        """
        """

        if domain is None:
            raise Exception('must provide domain.')

        # TODO does not work atm - not hierarchy of domains
        # if isinstance(domain, AbstractDomain):
        #     domain_class = domain.__class__
        # else:
        #     domain_class = cls.domain_class
        if domain_dtype is not None:
            kwargs['dtype'] = domain_dtype
        if domain_units is not None:
            kwargs['units'] = domain_units

        return cls.domain_class(domain, **kwargs)

    @staticmethod
    def _extract_attr_from_pandas(values, domain, domain_axis, labels):
        """Extract various attributes from a pandas instance if passed
        attributes is None. This function is used by unpack.
        """

        if domain is None:
            if domain_axis is None or domain_axis == 0:
                domain = asarray(values.index)
            else:
                assert isinstance(values, pd.DataFrame)
                domain = asarray(values.columns)

        if labels is None:
            if isinstance(values, pd.Series):
                labels = values.name
            else:
                if domain_axis is None or domain_axis == 0:
                    labels = asarray(values.columns)
                else:
                    labels = asarray(values.index)

        return {'domain': domain, 'domain_axis': domain_axis, 'labels': labels}

    @staticmethod
    def _extract_attr_from_signal_instance(signal, units, **kwargs):
        """Extract various attributes from a signal instance if passed
        attributes is None. This function is used by unpack.
        """

        for key, value in kwargs.items():
            if value is None and hasattr(signal, key):
                kwargs[key] = getattr(signal, key)

        if units is None:
            units = signal.units

        return units, kwargs

    @staticmethod
    def check_values(values, domain, domain_axis, labels):
        """
        """

        assert len(domain) == values.shape[domain_axis], (
            f"domain axis {values.shape[domain_axis]} "
            f"must equal length of domain {len(domain)}."
        )

        if not is_listlike(labels) and values.ndim == 2:
            if labels is None:
                labels = np.arange(values.shape[((domain_axis + 1) % 2)])
            else:
                assert values.shape[(domain_axis + 1) % 2] == 1
                labels = (labels, )

        # TODO can have listlike as single labels, some other constraint?
        # elif is_listlike(labels) and values.ndim == 1:
        #     # assert len(labels) == 1, 'labels and values do not match'
        #     if domain_axis == 1:
        #         values = values[None, :]
        #     else:
        #         values = values[:, None]

        elif is_listlike(labels) and values.ndim == 2:
            assert len(labels) == values.shape[(domain_axis + 1) % 2], (
                f"labels are of length {len(labels)}, "
                f"but other axis is of length "
                f"{values.shape[(domain_axis + 1) % 2]}."
            )

        if values.ndim == 1 and ((domain_axis > 0) or (domain_axis < -1)):
            raise ValueError(
                'If values is 1D, domain_axis must be smaller than 1')

        return values, labels


class MappingMixin(ABC):
    """mapping mixin for signal class
    """

    @abstractmethod
    def domain_bounds(self):
        pass

    @abstractmethod
    def bounds(self):
        pass

    def map(
        self, values, independent=True, top=None,
        method='isotonic',
        **kwargs
    ):
        """

        Parameters
        ----------
        values : array-like
            samples x channels
        """

        independent = independent | (self.ndim == 1)

        values = asarray(convert_units(values, self.units))
        values = self.pre_mapping(values)
        assert values.ndim < 3

        if values.ndim == 2:
            # values samples along axis -1
            values = values.T
            transposed = True
        else:
            transposed = False

        if method == 'isotonic':

            if self.ndim == 2:

                x = np.zeros(values.shape)

                for idx, bounds in enumerate(zip(*self.bounds)):

                    X, y = self._get_Xy_isotonic(idx=idx)
                    # take correct axis
                    iy = y.take(idx, axis=self.other_axis)
                    ivalues = values[idx]

                    x[idx] = self._iso_mapper(
                        X, iy, ivalues, bounds, idx=idx
                    )

            else:
                X, y = self._get_Xy_isotonic()
                x = self._iso_mapper(X, y, values, self.bounds)

        else:
            # values samples along axis -1
            x = self.get_mapping_x0(
                values, independent=independent, top=top
            )
            # keep track of shape
            shape = x.shape
            assert shape == values.shape, (
                f"shape mismatch: {shape} != {values.shape}"
            )

            x = np.ravel(x)
            values = np.ravel(values)

            # find root
            # TODO test if needed, or can we just stay with x0
            # TODO test best method
            if method == 'root':
                x = root(
                    self._root_mapper(values), x
                ).x.reshape(shape)

            # TODO dealing with domain bounds
            elif method == 'ls':
                x = least_squares(
                    self._ls_mapper(values), x,
                    bounds=tuple(self.domain_bounds.T),
                ).x.reshape(shape)

            elif method is None:
                x = x.reshape(x)

            else:
                raise NameError(
                    f'method {method} does not exist: '
                    'must be in ["root", "ls", None].'
                )

        if transposed:
            x = x.T

        # TODO check things depending on mapping_method
        x = self.post_mapping(x, **kwargs)

        return x

    def _iso_increasing(self, idx):
        return 'auto'

    def _iso_mapper(self, x, y, values, bounds, idx=None):
        """
        """

        isoreg = IsotonicRegression(
            y_min=bounds[0],
            y_max=bounds[1],
            increasing=self._iso_increasing(idx)
        )

        new_y = isoreg.fit_transform(
            x, y
        )

        # indexing domain bounds
        if idx is None:
            fill_value = tuple(self.domain_bounds)
        else:
            fill_value = tuple(self.domain_bounds[idx])

        x = interp1d(
            new_y, x,
            bounds_error=False,
            fill_value=fill_value
        )(values)

        return x

    def _get_Xy_isotonic(self, idx=None):
        """
        """
        domain = self.domain
        # index domain bounds in 2D array
        if idx is None:
            domain_bounds = self.domain_bounds
        else:
            domain_bounds = self.domain_bounds[idx]
        # append end and start
        if domain_bounds[0] not in domain:
            domain = domain.append(
                domain_bounds[0], left=True
            )
        if domain_bounds[1] not in domain:
            domain = domain.append(
                domain_bounds[1], left=False
            )

        X = asarray(domain)
        y = asarray(self(domain))

        return X, y

    def _index_helper(self, a):
        if a.ndim == 1:
            return a
        else:
            a = np.moveaxis(a, self.domain_axis, -1)
            return diag_chunks(a)

    def _root_mapper(self, values):
        """
        """

        return lambda x: np.ravel(
            (self._index_helper(self.interpolate(x)) - values)
        )

    def _ls_mapper(self, values):
        """
        """

        return lambda x: np.ravel(
            (self._index_helper(self.interpolate(x)) - values) ** 2
        )

    @abstractmethod
    def post_mapping(self, x):
        pass

    @abstractmethod
    def pre_mapping(self, values):
        pass

    def get_mapping_x0(self, values, independent=True, top=None):
        """
        Initial guess for domain values for root method.

        Select closest index in available samples.

        Selects by using the expected index/argument from a probability mass
        function, where the the probability of each index/argument is
        determined by the distance between the desired values and available
        values.

        """

        if not independent:
            raise NotImplementedError('dependent mapping')

        values = np.expand_dims(values, axis=self.domain_axis)
        if self.ndim == 1:
            mag = self.magnitude[:, None]
        else:
            mag = self.magnitude

        if values.ndim == 3:
            distances = np.abs(mag[..., None] - values)
        else:
            distances = np.abs(mag - values)

        if top is None:
            return np.argmin(distances, axis=self.domain_axis)

        else:

            distances = np.moveaxis(distances, self.domain_axis, 0)

            top_argsort = np.argsort(distances, axis=0)[:top]

            distances = np.take_along_axis(distances, top_argsort, axis=0)

            inv_distances = 1 / np.clip(distances, ABSOLUTE_ACCURACY, None)

            # inverse distance probability mass function
            inv_distances /= np.sum(inv_distances, axis=0, keepdims=True)

            idcs = np.arange(self.domain_len)

            if self.ndim == 2:
                idcs = np.expand_dims(idcs, axis=1)
            if inv_distances.ndim == 3:
                idcs = idcs[..., None]

            idcs = np.take_along_axis(idcs, top_argsort, axis=0)

            expected_idcs = np.sum(inv_distances * idcs, axis=0)
            upper = np.ceil(expected_idcs).astype(int)
            lower = np.floor(expected_idcs).astype(int)
            left = expected_idcs - lower

            upper_values = asarray(self.domain)[upper]
            lower_values = asarray(self.domain)[lower]

            x0 = left * upper_values + (1 - left) * lower_values

            return x0


class CheckClippingValueMixin(ABC):
    """Mixin class that provides method _check_clip_value
    for ClippedSignal and ClippedDomain
    """

    def _check_clip_value(self, value):
        """
        """

        if is_listlike(value):
            if self.ndim == 2:
                assert len(value) == self.other_len
            else:
                if len(value) != self.other_len:
                    raise ValueError(
                        'signal is one-dimensional but clipping'
                        ' is list-like.')


class IrradianceMixin:
    """Mixin class for checking irradiance units
    """

    def __getattr__(self, name):
        """
        """

        # necessary for properties
        return super().__getattribute__(name)

    @classmethod
    def get_units(cls, values, units):
        """
        """

        if units is None:
            if not hasattr(values, 'units'):
                if cls.irradiance_integrated:
                    units = 'irradiance'
                else:
                    units = 'spectral_irradiance'
            else:
                units = values.units

        cls.check_units(units)

        return units

    @classmethod
    def check_units(cls, units):
        """
        """

        if not isinstance(units, str):
            units = str(units)

        truth_value = UREG(units).check('[mass] / [length] / [time] ** 3')
        truth_value |= UREG(units).check('[mass] / [time] ** 3')
        truth_value |= UREG(units).check('[mass] * [length] / [time] ** 3')
        truth_value |= UREG(units).check(
            '[substance] / [length] ** 2 / [time]')
        truth_value |= UREG(units).check(
            '[substance] / [length] ** 3 / [time]')

        if not truth_value:
            raise DreyeUnitError('No irradiance convertible units.')

    @property
    def photonflux(self):
        """
        """

        if self.irradiance_integrated:
            return self.convert_to('photonflux')
        else:
            return self.convert_to('spectralphotonflux')

    @property
    def uE(self):
        """
        """

        if self.irradiance_integrated:
            return self.convert_to('microphotonflux')
        else:
            return self.convert_to('microspectralphotonflux')

    @property
    def irradiance(self):
        """
        """

        if self.irradiance_integrated:
            return self.convert_to('irradiance')
        else:
            return self.convert_to('spectralirradiance')
