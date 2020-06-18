"""base classes for output and system
"""

from abc import abstractmethod, ABC
import numpy as np
import pandas as pd

from dreye.constants import ureg
from dreye.io import write_json, read_json
from dreye.utilities import is_numeric, is_listlike
from dreye.core.spectral_measurement import (
    MeasuredSpectraContainer, MeasuredSpectrum
)
from dreye.err import DreyeError


class AbstractSender(ABC):

    @abstractmethod
    def send_value(self, value):
        """sending a single value(s) (float or 1D)
        Usually requires output to be open already.
        """
        pass

    @abstractmethod
    def send(
        self, values, rate=None, return_value=None, trigger=None,
        **trigger_kwargs
    ):
        """sending multiple values (array-like) (2D or 1D).
        Usually opens and closes output.
        """
        pass

    def map_value(self, value):
        return self.map(value)

    def map_send_value(self, value):
        value = self.map_value(value)
        self.send_value(value)

    def map(self, values):
        if self.measured_spectra is None:
            raise DreyeError("Need to assign a 'measured_spectra'.")

        return self.measured_spectra.map(values)

    def map_send(self, values, rate=None):
        values = self.map(values, rate)
        self.send(values)

    @property
    @abstractmethod
    def measured_spectra(self):
        pass

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @staticmethod
    def _create_trigger_array(length, rate, trigger_rate, on=1, off=0):
        # get length and points of trigger
        trigger_values = np.ones(length) * off
        trigger_length = int((rate / trigger_rate) // 2)
        trigger_points = np.arange(
            0, length, trigger_length
        ).astype(int)

        for n, idx in enumerate(trigger_points):
            # set trigger to value
            if (n % 2) == 0:
                trigger_values[idx:idx+trigger_length] = on

        # make sure trigger always ends at 0
        trigger_values[-1] = 0

        return trigger_values


class AbstractOutput(AbstractSender):

    def __init__(
        self,
        object_name,
        name,
        max_intensity_bound,
        zero_intensity_bound,
        units,
        measured_spectrum=None
    ):
        assert isinstance(object_name, str)
        assert isinstance(name, str)
        assert is_numeric(max_intensity_bound)
        assert is_numeric(zero_intensity_bound)
        assert isinstance(units, str) or units is None
        self._name = name
        self._object_name = object_name
        self._max_intensity_bound = max_intensity_bound
        self._zero_intensity_bound = zero_intensity_bound
        self._units = units
        self._measured_spectrum = measured_spectrum

    def channel_exists(self):
        return False

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"object={self.object_name}, name={self.name}, "
            f"max={self.max_intensity_bound}, "
            f"zero={self.zero_intensity_bound})"
        )

    @property
    def name(self):
        return self._name

    @property
    def object_name(self):
        return self._object_name

    @property
    def object(self):
        return self._object_name

    @property
    def max_intensity_bound(self):
        return self._max_intensity_bound * self.units

    @property
    def zero_intensity_bound(self):
        return self._zero_intensity_bound * self.units

    @property
    def min_val(self):
        return np.min([self._max_intensity_bound, self._zero_intensity_bound])

    @property
    def max_val(self):
        return np.max([self._max_intensity_bound, self._zero_intensity_bound])

    @property
    def units(self):
        return ureg(self._units).units

    @property
    def measured_spectra(self):
        return self.measured_spectrum

    @property
    def measured_spectrum(self):
        return self._measured_spectrum

    @measured_spectrum.setter
    def measured_spectrum(self, value):
        assert isinstance(value, MeasuredSpectrum)
        self._measured_spectrum = value

    def assign_measured_spectrum(
        self, values, wavelengths, output, units=None
    ):
        self._measured_spectrum = MeasuredSpectrum(
            values=values,
            units=units,
            domain=wavelengths,
            labels=output,
            zero_intensity_bound=self.zero_intensity_bound,
            max_intensity_bound=self.max_intensity_bound,
            name=self.name,
            labels_units=self.units
        )
        return self

    def _zero(self):
        self.send_value(self._zero_intensity_bound)

    def _max(self):
        self.send_value(self._max_intensity_bound)

    def steps(self, n_steps):
        return np.linspace(
            self._zero_intensity_bound,
            self._max_intensity_bound,
            n_steps
        )

    def to_dict(self):
        dictionary = {
            "name": self.name,
            "max_intensity_bound": self._max_intensity_bound,
            "zero_intensity_bound": self._zero_intensity_bound,
            "units": self._units,
            "measured_spectrum": self._measured_spectrum,
            "object_name": self._object_name
        }
        return dictionary

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def save(self, filename):
        write_json(filename, self)

    @classmethod
    def load(cls, filename):
        return read_json(filename)

    def __eq__(self, other):

        truth = isinstance(other, type(self))
        if not truth:
            return False

        return self.object_name == other.object_name


class AbstractSystem(AbstractSender):

    _output_class = AbstractOutput

    def __init__(self, outputs):
        if isinstance(outputs, AbstractSystem):
            self._outputs = outputs.outputs.copy()
        else:
            assert is_listlike(outputs)
            outputs = list(outputs)
            assert all(
                isinstance(output, self._output_class) for output in outputs
            )
            self._outputs = outputs

        # attributes calculated on the go
        self._measured_spectra = None

    def __repr__(self):
        return (
            f"{type(self).__name__} contains:\n" +
            "\n".join(f"{output}" for output in self)
        )

    def __getitem__(self, key):
        output = self.output_series[key]
        if isinstance(output, self._output_class):
            return output
        else:
            return type(self)(list(output))

    def channels_exists(self):
        return all(output.channel_exists() for output in self)

    def check_channels(self):
        if not self.channels_exists():
            raise DreyeError(
                "Some channels in System cannot be attached or do not exist!"
            )

    @property
    def output_series(self):
        return pd.Series(self.output_dict)

    @property
    def output_dict(self):
        return {
            output.name: output
            for output in self
        }

    @property
    def names(self):
        return [output.name for output in self]

    @property
    def outputs(self):
        return self._outputs

    def __len__(self):
        return len(self.outputs)

    def __iter__(self):
        return iter(self.outputs)

    def to_dict(self):
        return self.outputs

    @property
    def measured_spectra(self):
        if self._measured_spectra is None:
            container = [output.measured_spectrum for output in self]
            if all([ele is not None for ele in container]):
                self._measured_spectra = MeasuredSpectraContainer(
                    container
                )
        return self._measured_spectra

    @classmethod
    def from_dict(cls, data):
        return cls(data)

    def save(self, filename):
        write_json(filename, self)

    @classmethod
    def load(cls, filename):
        return read_json(filename)
