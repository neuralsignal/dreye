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
from dreye.utilities.abstract import inherit_docstrings


class AbstractSender(ABC):
    """
    Abstract Sender class.
    """

    @abstractmethod
    def send_value(self, value):
        """
        Send a single value to an open hardware output or system.

        Parameters
        ----------
        value : float or array
            The value sent to the open hardware.

        Notes
        -----
        If the object is an `AbstractOutput` value should always be a
        float.

        If the object is an `AbstractSystem` value should always be a
        one-dimensional array.

        This method is mainly used to run measurements with the
        `dreye.hardware.MeasurementRunner`. For stimulus display,
        the `send` method should be used.

        See Also
        --------
        send
        """
        pass

    @abstractmethod
    def send(
        self, values, rate=None, return_value=None, trigger=None,
        **trigger_kwargs
    ):
        """
        Send multiple values at a particular rate to a hardware
        output or system.

        Parameters
        ----------
        values : array-like
            An array-like object, where the first dimension
            corresponds to each frame.
        rate : numeric
            The rate in Hz of the `values`.
        return_value : float or array-like
            If given, this value is passed to `send_value` before
            closing the hardware after all `values` have been sent.
        trigger : object
            A trigger device that can be used to send a trigger to
            some other hardware. Implementation is entirely user-defined.
        trigger_kwargs : dict
            Keyword arguments used by the trigger device.

        Notes
        -----
        This method should open all hardware, send the list of values
        at the specified rate and close all hardware.

        See Also
        --------
        open
        close
        send_value
        """
        pass

    @property
    @abstractmethod
    def measured_spectra(self):
        """
        A `dreye.MeasuredSpectrum` or a `dreye.MeasuredSpectraContainer`
        object.
        """
        pass

    @abstractmethod
    def open(self):
        """
        Open hardware to be able to send values.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close hardware to be able to send values.
        """
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

        # make sure trigger always ends at off
        trigger_values[-1] = off

        return trigger_values


@inherit_docstrings
class AbstractOutput(AbstractSender):
    """
    Abstract output class for sending values to one particular hardware
    or output.

    Parameters
    ----------
    object_name : str
        A unique object name used for finding the correct hardware.
    name : str
        A user-defined name.
    max_intensity_bound : numeric
        The output value of the hardware that corresponds to
        the maximum intensity of the LED.
    zero_intensity_bound : numeric
        The output value of the hardware that corresponds to
        the zero intensity of the LED.
    units : str or None
        Units of the outputs (e.g. volts).
    measured_spectrum : dreye.MeasuredSpectrum, optional
        A already measured spectrum.

    Notes
    -----
    The following method need to be implemented for subclassing:

    * `open` method
    * `close` method
    * `send` method
    * `send_value` method
    """

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
        """
        Does the channel exist.
        """
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
        """
        Name of output hardware.
        """
        return self._name

    @property
    def object_name(self):
        """
        Unique object name of output hardware.
        """
        return self._object_name

    @property
    def object(self):
        """
        Alias for `object_name`.
        """
        return self._object_name

    @property
    def max_intensity_bound(self):
        """
        The output value of the hardware that corresponds to
        the maximum intensity of the LED (with units).
        """
        return self._max_intensity_bound * self.units

    @property
    def zero_intensity_bound(self):
        """
        The output value of the hardware that corresponds to
        the zero intensity of the LED (with units).
        """
        return self._zero_intensity_bound * self.units

    @property
    def min_val(self):
        """
        The minimum output value (no units).
        """
        return np.min([self._max_intensity_bound, self._zero_intensity_bound])

    @property
    def max_val(self):
        """
        The maximum output value (no units).
        """
        return np.max([self._max_intensity_bound, self._zero_intensity_bound])

    @property
    def units(self):
        """
        Units of the output device.
        """
        return ureg(self._units).units

    @property
    def measured_spectra(self):
        """
        The associated `dreye.MeasuredSpectrum` instance.
        """
        return self.measured_spectrum

    @property
    def measured_spectrum(self):
        """
        The associated `dreye.MeasuredSpectrum` instance.
        """
        return self._measured_spectrum

    @measured_spectrum.setter
    def measured_spectrum(self, value):
        assert isinstance(value, MeasuredSpectrum)
        self._measured_spectrum = value

    def assign_measured_spectrum(
        self, values, wavelengths, output, units=None
    ):
        """
        Assign a `dreye.MeasuredSpectrum` instance.
        """
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
        """
        set intensity to zero
        """
        self.send_value(self._zero_intensity_bound)

    def _max(self):
        """
        set to max intensity
        """
        self.send_value(self._max_intensity_bound)

    def steps(self, n_steps):
        """
        Create linearly spaced array between zero and max intensity bound.

        See Also
        --------
        numpy.linspace
        """
        return np.linspace(
            self._zero_intensity_bound,
            self._max_intensity_bound,
            n_steps
        )

    def to_dict(self):
        """
        Convert output hardware to dictionary.
        """
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
        """
        Create output hardware object from dictionary.
        """
        return cls(**data)

    def save(self, filename):
        """
        Save output hardware as JSON file.
        """
        write_json(filename, self)

    @classmethod
    def load(cls, filename):
        """
        Load output hardware from JSON file.
        """
        return read_json(filename)

    def __eq__(self, other):

        truth = isinstance(other, type(self))
        if not truth:
            return False

        return self.object_name == other.object_name


@inherit_docstrings
class AbstractSystem(AbstractSender):
    """
    Abstract class for implementing a complete system of output
    hardware (a list of `AbstractOutput` objects).

    Parameters
    ----------
    outputs : list-like or AbstractSystem
        A list of AbstractOutput objects.

    Notes
    -----
    The following method need to be implemented for subclassing:

    * `open` method
    * `close` method
    * `send` method
    * `send_value` method
    """

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
        """
        Check if all output channels exists.

        See Also
        --------
        AbstractOutput.channel_exists
        """
        return all(output.channel_exists() for output in self)

    def check_channels(self):
        """
        Raise error if not all channels exists.

        See Also
        --------
        channels_exists
        """
        if not self.channels_exists():
            raise DreyeError(
                "Some channels in System cannot be attached or do not exist!"
            )

    @property
    def output_series(self):
        """
        Return output hardware as `pandas.Series`.
        """
        return pd.Series(self.output_dict)

    @property
    def output_dict(self):
        """
        Return output hardware as dictionary.
        """
        return {
            output.name: output
            for output in self
        }

    @property
    def names(self):
        """
        List of name for each output hardware.
        """
        return [output.name for output in self]

    @property
    def outputs(self):
        """
        List of `AbstractOutput` objects.
        """
        return self._outputs

    def __len__(self):
        return len(self.outputs)

    def __iter__(self):
        return iter(self.outputs)

    def to_dict(self):
        """
        Returns list of `AbstractOutput` objects.
        """
        return self.outputs

    @property
    def measured_spectra(self):
        """
        The associated `dreye.MeasuredSpectraContainer`
        object.
        """
        if self._measured_spectra is None:
            container = [output.measured_spectrum for output in self]
            if all([ele is not None for ele in container]):
                self._measured_spectra = MeasuredSpectraContainer(
                    container
                )
        return self._measured_spectra

    @classmethod
    def from_dict(cls, data):
        """
        Create system object from dictionary.
        """
        return cls(data)

    def save(self, filename):
        """
        Save system as JSON.
        """
        write_json(filename, self)

    @classmethod
    def load(cls, filename):
        """
        Load system as JSON.
        """
        return read_json(filename)
