"""
Signal container classes
"""

import pandas as pd
import numpy as np

from dreye.utilities.abstract import _AbstractContainer, inherit_docstrings
from dreye.utilities import is_numeric, is_listlike
from dreye.core.signal import (
    Signals, DomainSignal, Signal
)
from dreye.plotting.plotting_mixin import _PlottingMixin


@inherit_docstrings
class _SignalContainer(_AbstractContainer, _PlottingMixin,):
    """
    A class that contains multiple signal instances in a list.

    All attributes and methods of a Signal instance can be used directly.

    Domain units and Signal units must have the same dimensionality.
    """
    _unit_mappings = {}
    _init_keys = [
        '_equalized_domain',
    ]

    def to_longframe(self, *args, **kwargs):
        """
        Convert the signal-type container into a long dataframe.

        This method uses the `to_longframe` method of the
        signal-type objects that are contained within and then
        concatenates all
        """
        return pd.concat(
            self.__getattr__('to_longframe')(*args, **kwargs),
            ignore_index=True, sort=True
        )

    def plotsmooth(self):
        """
        Warnings
        --------
        This method is not implemented for signal container-type objects.
        """
        raise NotImplementedError('smooth plotting with container.')

    def plot(self, **kwargs):
        # default behavior is that each signal gets own
        # column according to name
        # and that columns are wrapped in threes
        if not {'row', 'col'} & set(kwargs):
            kwargs['col'] = 'name'
            kwargs['col_wrap'] = kwargs.get('col_wrap', 3)
        return super().plot(**kwargs)

    def __getitem__(self, key):
        if isinstance(key, np.ndarray):
            key = key.tolist()

        if is_numeric(key):
            pass
        elif key in self.names:
            key = self.names.index(key)
        elif key in self.container:
            key = self.container.index(key)
        elif is_listlike(key):
            if np.asarray(key).dtype == np.bool:
                key = np.arange(len(self))[np.asarray(key)].tolist()
            return type(self)([self.__getitem__(ikey) for ikey in key])
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if is_numeric(key):
            pass
        elif key in self.names:
            key = self.names.index(key)
        elif key in self.container:
            key = self.container.index(key)
        elif is_listlike(key) and is_listlike(value):
            for ikey, ivalue in zip(key, value):
                self.__setitem__(ikey, ivalue)
            return
        return super().__setitem__(key, value)

    @property
    def equalized_domain(self):
        """
        The domain equalized across all signal-type instances
        in the container.
        """
        if self._equalized_domain is None:
            domain = self[0].domain
            for signal in self[1:]:
                domain = domain.equalize_domains(signal.domain)
            self._equalized_domain = domain.copy()
        return self._equalized_domain

    @property
    def units(self):
        """
        The units of the `values` arrays in the container.
        """
        return self[0].units

    @units.setter
    def units(self, value):
        """
        Setting the units of the `values` arrays in the container.
        """
        for idx, ele in enumerate(self):
            ele = ele.to(value)
            self[idx] = ele

    @property
    def domain_units(self):
        """
        The units of the `dreye.Domain` instance.

        As required, the domain units are the same across all signal-type
        objects in the container.
        """
        return self[0].domain.units

    @domain_units.setter
    def domain_units(self, value):
        """
        Setting the domain units.
        """
        for idx, ele in enumerate(self):
            ele.domain = ele.domain.to(value)

    @property
    def names(self):
        """
        List of the names of each signal-type object.
        """
        return [ele.name for ele in self]

    def popname(self, name):
        """
        Remove a signal-type object from the container given its
        name.

        Returns
        -------
        obj : signal-type
            Returns the signal-type object that was removed
        """
        index = self.names.index(name)
        return self.pop(index)

    @property
    def ndim(self):
        """
        The dimensionality of the container.

        The dimensionality is set to 2.
        """
        return 2


@inherit_docstrings
class SignalsContainer(_SignalContainer):
    """
    A container that can hold multiple two-dimensional signal-type
    instances

    Parameters
    ----------
    container : list-like
        A list of signal-type instances
    units : str or `ureg.Unit`, optional
        The units to convert the values to. If None,
        it will choose the units of the first signal in the list.
    domain_units : str or `ureg.Unit`, optional
        The units to convert the domain to. If None,
        it will choose the units of domain of the first signal in the list.
    """
    _allowed_instances = (Signals, DomainSignal, Signal)
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
            ele.domain = ele.domain.to(domain_units)
            self[idx] = ele

    @property
    def signals(self):
        """
        Returns concatenated `Signals` class using all signals in container.

        See Also
        --------
        dreye.Signals.concat
        """
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
        """
        The magnitude of the `signals` attribute.
        """
        return self.signals.magnitude

    @property
    def domain(self):
        """
        The domain of the `signals` attribute.
        """
        return self.signals.domain

    @property
    def labels(self):
        """
        The labels of the `signals` attribute.
        """
        return self.signals.labels

    @property
    def labels_axis(self):
        """
        The labels axis of the `signals` attribute.
        """
        return self.signals.labels_axis

    @property
    def domain_axis(self):
        """
        The domain axis of the `signals` attribute.
        """
        return self.signals.domain_axis

    @property
    def shape(self):
        """
        The shape of the `signals` attribute
        """
        return self.signals.shape


@inherit_docstrings
class DomainSignalContainer(_SignalContainer):
    """
    A container that can hold multiple `dreye.DomainSignal` instances

    Parameters
    ----------
    container : list-like
        A list of `dreye.DomainSignal` instances.
    units : str or `ureg.Unit`, optional
        The units to convert the values to. If None,
        it will choose the units of the first signal in the list.
    domain_units : str or `ureg.Unit`, optional
        The units to convert the domain to. If None,
        it will choose the units of domain of the first signal in the list.
    labels_units : str or `ureg.Unit`, optional
        The units to convert the labels to. If None,
        it will choose the units of labels of the first signal in the list.

    See Also
    --------
    MeasuredSpectraContainer
    """
    _allowed_instances = DomainSignal
    _enforce_instance = DomainSignal
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
        """
        The units of the `dreye.Domain` instance corresponding to the labels.

        As required, the domain units are the same across all signal-type
        objects in the container.
        """
        return self[0].labels.units

    @labels_units.setter
    def labels_units(self, value):
        """
        Setting the labels units.
        """
        for idx, ele in enumerate(self):
            ele.labels = ele.labels.to(value)

    @property
    def equalized_labels(self):
        """
        The labels equalized across all signal-type instances
        in the container.
        """
        if self._equalized_labels is None:
            labels = self[0].labels
            for signal in self[1:]:
                labels = labels.equalize_domains(signal.labels)
            self._equalized_labels = labels.copy()
        return self._equalized_labels

    @property
    def stacked_values(self):
        """
        Stacks all `DomainSignal` instance on top of each other to
        create a three-dimensional `pint.Quantity` instances
        """
        if self._stacked_values is None:
            values = np.empty((
                len(self),
                len(self.equalized_domain),
                len(self.equalized_labels)
            ))
            for idx, signal in enumerate(self):
                values[idx] = self._enforce_instance(signal)(
                    self.equalized_domain
                ).labels_interp(
                    self.equalized_labels
                ).magnitude

            self._stacked_values = values * self.units
        return self._stacked_values
