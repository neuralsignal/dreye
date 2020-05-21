"""Mixin class for signal and domain
"""

import copy

import numpy as np
import pandas as pd
from scipy.optimize import root, least_squares
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression

from dreye.err import DreyeError
from dreye.utilities import (
    convert_units, diag_chunks, is_listlike, arange,
    is_numeric, is_uniform, array_domain, has_units,
    asarray, get_values, is_hashable
)
from dreye.constants import ureg, ABSOLUTE_ACCURACY, DEFAULT_FLOAT_DTYPE
from dreye.core.abstract import _UnitArray

# TODO make unpacking more general

class _UnpackDomain:
    """
    Mixin class for unpacking domain
    """

    def _unpack(
        self,
        values=None,
        start=None,
        end=None,
        interval=None,
        contexts=None,
        units=None,
        name=None,
        attrs=None
    ):

        # get mapping of units if exists
        units = self._unit_mappings.get(units, units)

        if not is_numeric(start) and values is None:
            values = start
            start = None

        if isinstance(values, str):
            values = self.load(values)

        if values is None:
            pass

        elif isinstance(values, _UnitArray):
            # TODO handling when signal is passed
            values.units = units
            if start is None:
                start = values.start
            if end is None:
                end = values.end
            if interval is None:
                interval = values.interval
            if contexts is None:
                contexts = values.contexts
            if units is None:
                units = values.units
            if name is None:
                name = values.name
            if attrs is None:
                attrs = values.attrs

        else:
            if has_units(values):
                # assign units or convert values
                if units is None:
                    units = values.units
                else:
                    values = values.to(units)

            values = asarray(values).astype(DEFAULT_FLOAT_DTYPE)

            # check if domain is sorted
            if not np.all(np.sort(values) == values):
                raise DreyeError(f'Values for domain initialization '
                                 f'must be sorted: {values}.')

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

        start, end = DEFAULT_FLOAT_DTYPE(start), DEFAULT_FLOAT_DTYPE(end)

        if (
            (start is None)
            or (end is None)
            or (units is None)
            or (interval is None)
        ):

            raise TypeError(
                f"Unable to create Domain; None types present.")

        values, interval = self._create_values(
            start, end, interval)

        return dict(
            values=values,
            start=start,
            end=end,
            interval=interval,
            contexts=contexts,
            units=units,
            name=name,
            attrs=attrs
        )

    @staticmethod
    def _create_values(start, end, interval):
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
            values = values.astype(DEFAULT_FLOAT_DTYPE)
            interval = interval.astype(DEFAULT_FLOAT_DTYPE)

            if values.size != np.unique(values).size:
                raise DreyeError('values are non-unique: {0}'.format(values))

        elif is_numeric(interval):
            if interval > end - start:
                raise DreyeError(('interval attribute: value '
                                  '{0} bigger than range {1}').format(
                                      interval, end - start))

            values, interval = arange(
                start, end, interval, dtype=DEFAULT_FLOAT_DTYPE)
            interval = DEFAULT_FLOAT_DTYPE(interval)

        else:
            raise DreyeError(
                'interval not type numeric'
                ' or array-like: {0}').format(type(interval))

        return values, interval


class _UnpackSignal:
    """
    Mixin class for unpacking our signal
    """

    def _unpack(
        self,
        values,
        domain=None,
        domain_axis=None,
        units=None,
        domain_units=None,
        labels=None,
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
        """
        unpack signal instance
        """

        # get mapping of units if exists
        units = self._unit_mappings.get(units, units)

        container = dict(
            units=units,
            domain_units=domain_units,
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

        if domain_kwargs is None:
            container['domain_kwargs'] = {}

        if isinstance(values, str) or isinstance(values, _UnitArray):

            if isinstance(values, str):
                values = self.load(values)

            # convert units if possible
            if units is not None:
                values = values.to(units)

            # copies over all necessary attributes
            self._extract_attr_from_signal_instance(values, container)
            # will throw error if domain not present
            self._get_domain(container)

            # not the case if domain instance was passed
            if hasattr(values, 'domain'):
                if values.domain == container['domain']:
                    values = values.magnitude
                else:
                    # interpolate to new domain
                    values = values(container['domain']).magnitude
            else:
                values = values.magnitude

        elif isinstance(values, (pd.DataFrame, pd.Series)):
            # extract labels and such
            self._extract_attr_from_pandas(values, container)
            # convert values dtype
            values = asarray(values).astype(DEFAULT_FLOAT_DTYPE)
            self._get_domain(container)

        elif is_listlike(values):
            # get container class
            self._get_domain(container)
            # if has units
            if has_units(values):
                if container['units'] is None:
                    container['units'] = values.units
                else:
                    raise DreyeError(
                        f"Strip units from {type(values)}, when also "
                        "supplying units in the initialization."
                    )
            # get array of values
            values = asarray(values).astype(DEFAULT_FLOAT_DTYPE)

        else:
            raise TypeError('Values must be string or array-like.')

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

        values, container['labels'] = self._check_values(
            values, container['domain'],
            container['domain_axis'],
            container['labels']
        )

        self._update_domain_bound(container, 'domain_min')
        self._update_domain_bound(container, 'domain_max')
        self._update_signal_bound(container, 'signal_min')
        self._update_signal_bound(container, 'signal_max')

        values = self._clip_values(
            values,
            get_values(container['signal_min']),
            get_values(container['signal_max'])
        )

        return values, container

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

    def _get_domain(self, container):
        """
        """

        if container['domain'] is None:
            raise DreyeError('must provide domain.')

        container['domain'] = self.domain_class(
            container['domain'],
            units=container['domain_units'],
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
                container[key] = copy.copy(getattr(signal, key))

    def _check_values(self, values, domain, domain_axis, labels):
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
                labels = self._label_class(
                    np.arange(values.shape[((domain_axis + 1) % 2)])
                )
            else:
                if not values.shape[(domain_axis + 1) % 2] == 1:
                    raise DreyeError(
                        'When providing single label and values is a 2D array'
                        ' there can only be a single signal.'
                    )
                labels = self._label_class([labels])

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
            if (
                not isinstance(labels, self._label_class)
                and self._force_labels
            ):
                labels = self._label_class(labels)

        if values.ndim == 1 and ((domain_axis > 0) or (domain_axis < -1)):
            raise DreyeError('If values is 1D, domain_axis '
                             'must be smaller than 1')

        return values, labels
