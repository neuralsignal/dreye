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
    asarray, get_values, is_hashable
)
from dreye.constants import ureg, ABSOLUTE_ACCURACY, DEFAULT_FLOAT_DTYPE
from dreye.core.abstract import AbstractSignal, AbstractDomain


class UnpackDomainMixin(ABC):

    @classmethod
    def _unpack(
        cls,
        values=None,
        start=None,
        end=None,
        interval=None,
        dtype=None,
        contexts=None,
        units=None,
    ):
        """
        Returns correct attributes for domain.

        Parameters
        ----------

        Returns
        -------
        """

        if not is_numeric(start) and values is None:

            values = start
            start = None

        if isinstance(values, str):

            values = cls.load(values, dtype=dtype)

        if values is None:
            pass

        elif isinstance(values, AbstractDomain):

            values.units = units
            if start is None:
                start = values.start
            if end is None:
                end = values.end
            if interval is None:
                interval = values.interval
            if dtype is None:
                dtype = values.dtype
            if contexts is None:
                contexts = values.contexts
            if units is None:
                units = values.units

        else:

            if has_units(values):
                # assign units or convert values
                if units is None:
                    units = values.units
                else:
                    values = values.to(units)

            values = asarray(values)

            _start, _end, _interval = array_domain(
                values, uniform=is_uniform(values))

            if start is None:
                start = _start
            if end is None:
                end = _end
            if interval is None:
                interval = _interval

        if isinstance(units, str) or units is None:
            units = ureg(units).units

        start = get_values(convert_units(start, units))
        end = get_values(convert_units(end, units))
        interval = get_values(convert_units(interval, units))

        start, end = dtype(start), dtype(end)

        if (
            (start is None)
            or (end is None)
            or (units is None)
            or (interval is None)
        ):

            raise TypeError(
                f"Unable to create Domain; None types present.")

        values, interval = cls._create_values(
            start,
            end,
            interval,
            dtype)

        return values, start, end, interval, contexts, dtype, units

    @staticmethod
    def _create_values(start, end, interval, dtype=DEFAULT_FLOAT_DTYPE):
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
            raise DreyeError(
                'interval not type numeric'
                ' or array-like: {0}').format(type(interval))

        return values, interval


class UnpackSignalMixin(ABC):

    @classmethod
    def _unpack(
        cls,
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
        domain_min=None,
        domain_max=None,
        signal_min=None,
        signal_max=None,
        attrs=None,
        name=None
    ):
        """_unpack signal instance
        """

        container = dict(
            units=units,
            domain_units=domain_units,
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

        if isinstance(dtype, str):
            dtype = np.dtype(dtype).type
        if domain_kwargs is None:
            container['domain_kwargs'] = {}

        if isinstance(values, str) or isinstance(values, AbstractSignal):

            if isinstance(values, str):
                values = cls.load(values, dtype=dtype)

            cls._extract_attr_from_signal_instance(
                values, container
            )
            cls._get_domain(container)

            prev_domain = values.domain
            if units is not None:
                values = values._convert_values(
                    values.values,
                    units,
                    contexts=(
                        values.contexts if contexts is None else contexts
                    ),
                    domain=values.domain,
                    axis=values.other_axis
                )

            if prev_domain == container['domain']:
                values = values.magnitude
            else:
                values = values(container['domain']).magnitude

        elif isinstance(values, (pd.DataFrame, pd.Series)):

            cls._extract_attr_from_pandas(values, container)

            values = asarray(values).astype(dtype)
            cls._get_domain(container)

        elif is_listlike(values):
            # TODO what about a list of signal? - concatenation?
            cls._get_domain(container)
            if has_units(values) and container['units'] is None:
                container['units'] = values.units
            elif has_units(values):
                # TODO check contexts etc.
                values = cls._convert_values(
                    values,
                    container['units'],
                    container['contexts'],
                    domain=container['domain'].values,
                    axis=(
                        1 if container['domain_axis'] is None
                        else (container['domain_axis'] + 1) % 2
                    )
                )
            values = asarray(values).astype(dtype)

        else:
            raise TypeError('values must be string or array-like.')

        assert values.ndim < 3, "signal array must be 1D or 2D."

        if container['domain_axis'] is None:
            container['domain_axis'] = 0
        assert container['domain_axis'] in [0, 1, -1, -2], \
            'domain axis must be in 0 or 1'

        if container['interpolator'] is None:
            container['interpolator'] = interp1d
            if container['interpolator_kwargs'] is None:
                container['interpolator_kwargs'] = {'bounds_error': False}
        assert hasattr(container['interpolator'], '__call__')

        if container['interpolator_kwargs'] is None:
            container['interpolator_kwargs'] = {}
        assert isinstance(container['interpolator_kwargs'], dict)

        if isinstance(container['units'], str) or container['units'] is None:
            container['units'] = ureg(container['units']).units

        values, container['labels'] = cls._check_values(
            values, container['domain'],
            container['domain_axis'],
            container['labels']
        )

        cls._update_domain_bound(container, 'domain_min')
        cls._update_domain_bound(container, 'domain_max')
        cls._update_signal_bound(container, 'signal_min')
        cls._update_signal_bound(container, 'signal_max')

        values = cls._clip_values(
            values,
            get_values(container['signal_min']),
            get_values(container['signal_max'])
        )

        return values, dtype, container

    @staticmethod
    def _update_domain_bound(container, key):

        bound = container[key]

        if bound is None:
            pass
        elif not is_numeric(bound):
            raise DreyeError(
                f"domain bound variable {key} must be numeric, but "
                f"is of type '{type(bound)}'."
            )
        elif has_units(bound):
            container[key] = bound.to(container['domain'].units)
        else:
            container[key] = bound * container['domain'].units

    @staticmethod
    def _update_signal_bound(container, key):

        bound = container[key]

        if bound is None:
            pass
        elif has_units(bound):
            container[key] = bound.to(container['units'])
        else:
            container[key] = bound * container['units']

    @classmethod
    def _get_domain(cls, container):
        """
        """

        if container['domain'] is None:
            raise DreyeError('must provide domain.')

        container['domain'] = cls.domain_class(
            container['domain'],
            units=container['domain_units'],
            dtype=container['domain_dtype'],
            **container['domain_kwargs']
        )

    @staticmethod
    def _extract_attr_from_pandas(values, container):
        """Extract various attributes from a pandas instance if passed
        attributes is None. This function is used by _unpack.
        """

        if container['domain'] is None:
            if (
                container['domain_axis'] is None
                or container['domain_axis'] == 0
            ):
                container['domain'] = values.index
            else:
                assert isinstance(values, pd.DataFrame)
                container['domain'] = values.columns

        if container['labels'] is None:
            if isinstance(values, pd.Series):
                container['labels'] = values.name
            else:
                if (
                    container['domain_axis'] is None
                    or container['domain_axis'] == 0
                ):
                    container['labels'] = values.columns
                else:
                    container['labels'] = values.index

        if container['name'] is None and isinstance(values, pd.DataFrame):
            container['name'] = values.name

    @staticmethod
    def _extract_attr_from_signal_instance(signal, container):
        """Extract various attributes from a signal instance if passed
        attributes is None. This function is used by _unpack.
        """

        for key, value in container.items():
            if value is None and hasattr(signal, key):
                container[key] = getattr(signal, key)

    @classmethod
    def _check_values(cls, values, domain, domain_axis, labels):
        """
        """

        assert len(domain) == values.shape[domain_axis], (
            f"domain axis {values.shape[domain_axis]} "
            f"must equal length of domain {len(domain)}."
        )

        if not is_listlike(labels) and values.ndim == 2:
            if not is_hashable(labels):
                raise DreyeError(
                    f'label of type {type(labels)} is not hashable'
                )
            if labels is None:
                labels = cls._label_class(
                    np.arange(values.shape[((domain_axis + 1) % 2)])
                )
            else:
                if not values.shape[(domain_axis + 1) % 2] == 1:
                    raise DreyeError(
                        'When providing single label and values is a 2D array'
                        ' there can only be a single signal.'
                    )
                labels = cls._label_class([labels])

        elif values.ndim == 1 and not is_hashable(labels):
            raise DreyeError(
                f'label of type {type(labels)} is not hashable'
            )

        elif values.ndim == 2:
            if not len(labels) == values.shape[(domain_axis + 1) % 2]:
                raise DreyeError(
                    f"labels are of length {len(labels)}, "
                    f"but number of signals is "
                    f"{values.shape[(domain_axis + 1) % 2]}."
                )
            if not all(is_hashable(label) for label in labels):
                raise DreyeError('not all labels are hashable')
            if not isinstance(labels, cls._label_class) and cls._force_labels:
                labels = cls._label_class(labels)

        if values.ndim == 1 and ((domain_axis > 0) or (domain_axis < -1)):
            raise DreyeError('If values is 1D, domain_axis '
                             'must be smaller than 1')

        return values, labels
