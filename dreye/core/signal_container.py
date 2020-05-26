"""
Signal container classes
"""

import pandas as pd
import numpy as np

from dreye.utilities.abstract import _AbstractContainer
from dreye.utilities import is_numeric
from dreye.core.signal import (
    Signals, _SignalDomainLabels, _SignalIndexLabels
)
from dreye.core.plotting_mixin import _PlottingMixin


class _SignalContainer(_AbstractContainer, _PlottingMixin,):
    """
    A class that contains multiple signal instances in a list.
    All attributes and methods of a Signal instance can be used directly.
    Domain units and signal units must have the same dimensionality.
    """
    _unit_mappings = {}
    _init_keys = [
        '_equalized_domain',
    ]

    def to_longframe(self):
        return pd.concat(
            self.__getattr__('to_longframe')(),
            ignore_index=True, sort=True
        )

    def plotsmooth(self):
        raise NotImplementedError('smooth plotting with container.')

    def plot(self, **kwargs):
        # default behavior is that each signal gets own
        # column according to name
        # and that columns are wrapped in threes
        if not {'row', 'col'} & set(kwargs):
            kwargs['col'] = 'name'
            kwargs['col_wrap'] = 3
        return super().plot(**kwargs)

    def __getitem__(self, key):
        if is_numeric(key):
            pass
        elif key in self.names:
            key = self.names.index(key)
        elif key in self.container:
            key = self.container.index(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if is_numeric(key):
            pass
        elif key in self.names:
            key = self.names.index(key)
        elif key in self.container:
            key = self.container.index(key)
        return super().__setitem__(key, value)

    @property
    def equalized_domain(self):
        if self._equalized_domain is None:
            domain = self[0].domain
            for signal in self[1:]:
                domain = domain.equalize_domains(signal.domain)
            self._equalized_domain = domain.copy()
        return self._equalized_domain

    @property
    def units(self):
        return self[0].units

    @property
    def domain_units(self):
        return self[0].domain.units

    @property
    def names(self):
        return [ele.name for ele in self]

    def popname(self, name):
        index = self.names.index(name)
        return self.pop(index)

    @property
    def ndim(self):
        return 2


class SignalsContainer(_SignalContainer):
    _allowed_instances = (_SignalIndexLabels, _SignalDomainLabels)
    _enforce_instance = Signals
    _init_keys = ['_signals']

    def __init__(self, container, units=None, domain_units=None):
        super().__init__(container)

        # units
        units = self._unit_mappings.get(units, units)
        if units is None:
            units = self[0].units
        # domain units
        domain_units = self._unit_mappings.get(domain_units, domain_units)
        if domain_units is None:
            domain_units = self[0].domain.units

        # convert to correct units
        for idx, ele in enumerate(self):
            ele = ele.to(self.units)
            domain = ele.domain.to(domain_units)
            ele._domain = domain
            self[idx] = ele

    @property
    def signals(self):
        if self._signals is None:
            # concat signals
            for idx, signal in enumerate(self):
                signal = self._enforce_instance(signal)
                if idx == 0:
                    signals = signal
                else:
                    signals = signals.labels_concat(signal)
            # assign attribute
            self._signals = signals
        return self._signals

    @property
    def magnitude(self):
        return self.signals.magnitude

    @property
    def domain(self):
        return self.signals.domain

    @property
    def labels(self):
        return self.signals.labels

    @property
    def labels_axis(self):
        return self.signals.labels_axis

    @property
    def domain_axis(self):
        return self.signals.domain_axis

    @property
    def shape(self):
        return self.signals.shape


class DomainSignalContainer(_SignalContainer):
    _allowed_instances = _SignalDomainLabels
    _init_keys = [
        '_stacked_values',
        '_equalized_domain',
        '_equalized_labels'
    ]

    def __init__(
        self, container,
        units=None, domain_units=None, labels_units=None
    ):
        super().__init__(container)

        # units
        units = self._unit_mappings.get(units, units)
        if units is None:
            units = self[0].units
        # domain units
        domain_units = self._unit_mappings.get(domain_units, domain_units)
        if domain_units is None:
            domain_units = self[0].domain.units
        # labels units
        labels_units = self._unit_mappings.get(labels_units, labels_units)
        if labels_units is None:
            labels_units = self[0].labels.units

        # convert to correct units
        for idx, ele in enumerate(self):
            ele = ele.to(self.units)
            ele.domain = ele.domain.to(domain_units)
            ele.labels = ele.labels.to(labels_units)
            self[idx] = ele

    @property
    def labels_units(self):
        return self[0].labels.units

    @property
    def equalized_labels(self):
        if self._equalized_labels is None:
            labels = self[0].labels
            for signal in self[1:]:
                labels = labels.equalize_labelss(signal.labels)
            self._equalized_labels = labels.copy()
        return self._equalized_labels

    @property
    def stacked_values(self):
        if self._stacked_values is None:
            values = np.empty((
                len(self),
                len(self.equalized_domain),
                len(self.equalized_labels)
            ))
            for idx, signal in enumerate(self):
                values[idx] = signal(
                    self.equalized_domain
                ).T(
                    self.equalized_labels
                ).T.magnitude

            self._stack_values = values * self.units
        return self._stacked_values
