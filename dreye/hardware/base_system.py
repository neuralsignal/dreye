"""base classes for output and system
"""

from abc import abstractmethod, ABC
import numpy as np
import pandas as pd

from dreye.constants import UREG
from dreye.io.json import write_json, read_json
from dreye.utilities import is_numeric, is_listlike
from dreye.core.spectral_measurement import (
    SpectrumMeasurement, MeasuredSpectrum
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
        if self.spm is None:
            raise DreyeError('Need to assign a SpectrumMeasurement.')

        return self.spm.map(values)

    def map_send(self, values, rate=None):
        values = self.map(values, rate)
        self.send(values)

    @property
    @abstractmethod
    def spm(self):
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
        self, object_name, name, max_boundary, zero_boundary, units,
        spm=None, mspectrum=None
    ):
        assert isinstance(object_name, str)
        assert isinstance(name, str)
        assert is_numeric(max_boundary)
        assert is_numeric(zero_boundary)
        assert isinstance(units, str)
        assert isinstance(spm, SpectrumMeasurement) or spm is None
        self._name = name
        self._object_name = object_name
        self._max_boundary = max_boundary
        self._zero_boundary = zero_boundary
        self._units = units
        self._spm = spm
        self._mspectrum = mspectrum

    def channel_exists(self):
        return False

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"object={self.object_name}, name={self.name}, "
            f"max={self.max_boundary}, zero={self.zero_boundary})"
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
    def max_boundary(self):
        return self._max_boundary * self.units

    @property
    def zero_boundary(self):
        return self._zero_boundary * self.units

    @property
    def min_val(self):
        return np.min([self._max_boundary, self._zero_boundary])

    @property
    def max_val(self):
        return np.max([self._max_boundary, self._zero_boundary])

    @property
    def units(self):
        return UREG(self._units).units

    @property
    def spm(self):
        if self._spm is None:
            self._assign_new_measurement()
        return self._spm

    @spm.setter
    def spm(self, value):
        self._assign_new_measurement(value)

    @property
    def mspectrum(self):
        return self._mspectrum

    @mspectrum.setter
    def mspectrum(self, value):
        assert isinstance(value, MeasuredSpectrum)
        self._mspectrum = value
        self._spm = None

    def _zero(self):
        self.send_value(self._zero_boundary)

    def steps(self, n_steps):
        return np.linspace(
            self._zero_boundary,
            self._max_boundary,
            n_steps
        )

    def _assign_new_measurement(self, spm=None, units='uE'):
        if self.mspectrum is not None and spm is None:
            self._spm = self.mspectrum.to_spectrum_measurement(
                name=self.name,
                zero_boundary=self.zero_boundary,
                max_boundary=self.max_boundary,
                units=units
            )
        elif spm is not None:
            assert isinstance(spm, SpectrumMeasurement)
            assert spm.ndim == 1
            assert float(self.zero_boundary) == float(spm.zero_boundary)
            assert float(self.max_boundary) == float(spm.max_boundary)
            assert str(self.name) == str(spm.label_names)
            self._spm = spm

            if self.mspectrum is not None:
                spm = self.mspectrum.to_spectrum_measurement(
                    name=self.name,
                    zero_boundary=self.zero_boundary,
                    max_boundary=self.max_boundary,
                    units=units
                )

                assert spm == self._spm

    def to_dict(self):
        dictionary = {
            "name": self.name,
            "max_boundary": self._max_boundary,
            "zero_boundary": self._zero_boundary,
            "units": self._units,
            "spm": self._spm,
            "mspectrum": self._mspectrum,
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

        truth = isinstance(other, self.__class__)
        if not truth:
            return False

        return self.object_name == other.object_name


class AbstractSystem(AbstractSender):

    def __init__(self, outputs):
        if isinstance(outputs, AbstractSystem):
            self._outputs = outputs.outputs
        else:
            assert is_listlike(outputs)
            assert all(
                isinstance(output, AbstractOutput) for output in outputs
            )
            self._outputs = outputs

        # attributes calculated on the go
        self._spms = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__} contains:\n" +
            "\n".join(f"{output}" for output in self)
        )

    def __getitem__(self, key):
        output = self.output_series[key]
        if isinstance(output, AbstractOutput):
            return output
        else:
            return self.__class__(list(output))

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

    def __iter__(self):
        return iter(self.outputs)

    def to_dict(self):
        return self.outputs

    @property
    def spms(self):
        if self._spms is None:
            for n, output in enumerate(self):
                if output.spm is None:
                    break
                if n == 0:
                    spms = output.spm
                else:
                    spms = spms.concat(output.spm)
            else:
                self._spms = spms
        return self._spms

    @property
    def spm(self):
        return self.spms

    @classmethod
    def from_dict(cls, data):
        return cls(data)

    def save(self, filename):
        write_json(filename, self)

    @classmethod
    def load(cls, filename):
        return read_json(filename)
