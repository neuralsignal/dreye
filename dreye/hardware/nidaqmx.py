"""
National instruments hardware
"""

import time
import sys

import numpy as np
import pandas as pd

from dreye.err import DreyeError
from dreye.hardware.base_system import AbstractOutput, AbstractSystem
from dreye.utilities import optional_to, asarray
from dreye.utilities.abstract import inherit_docstrings

# HARDWARE API IMPORTS
try:
    import nidaqmx
    import nidaqmx.system
    from nidaqmx.stream_writers import \
        AnalogMultiChannelWriter, AnalogSingleChannelWriter
    NIDAQMX = True
except ImportError:
    NIDAQMX = False


def _check_install():
    if not NIDAQMX:
        raise DreyeError(f"You need to install nidaqmx.")


def get_channels():
    """get all channels from all NI DAQ devices.
    """
    _check_install()

    system = nidaqmx.system.System.local()
    return [
        chan.name
        for device in system.devices
        for chan in device.ao_physical_chans
    ]


def get_channel_mappings():
    """get all NI DAQ devices
    """
    _check_install()

    system = nidaqmx.system.System.local()
    return {
        chan.name: (device, chan)
        for device in system.devices
        for chan in device.ao_physical_chans
    }


@inherit_docstrings
class NiDaqMxOutput(AbstractOutput):
    """
    Output hardware object for single analog output of a
    National Instruments DAQ.

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
    This class uses the `nidaqmx` python module. Values
    are sent by creating a task and opening the analog output channel.
    """
    # writer to send values
    task = None
    """
    Task used when opening hardware
    """
    writer = None
    """
    Writer used when opening hardware
    """

    def __init__(self, *args, **kwargs):
        _check_install()
        super().__init__(*args, **kwargs)

    def open(self):
        # create task
        # create writer
        if not self.channel_exists():
            raise DreyeError(
                f'Channel "{self.object_name}" does not exist'
            )

        if self.task is not None:
            self.close()

        self.task = nidaqmx.Task()
        self.task.ao_channels.add_ao_voltage_chan(
            self.object_name,
            max_val=self.max_val,
            min_val=self.min_val
        )
        self.writer = AnalogSingleChannelWriter(
            self.task.out_stream
        )

    def close(self):
        # close task
        self.task.close()
        self.task = None
        self.writer = None

    def send(self, values, rate=None, return_value=None):
        dt = float(optional_to(1 / rate, 's'))
        values = asarray(values).astype(np.float64)
        self.open()
        for value in values:
            now = time.clock()
            # assumes sending does not take a lot of time
            self.send_value(value)
            while time.clock() - now < dt:
                pass
        if return_value is not None:
            self.send_value(return_value)
        self.close()
        return values

    def send_value(self, value):
        self.writer.write_one_sample(value)
        return value

    def channel_exists(self):
        return self.object_name in get_channels()

    @property
    def channel(self):
        """
        Channel of output
        """
        return get_channel_mappings().get(self.object_name, None)[1]

    @property
    def device(self):
        """
        Device name of output.
        """
        return get_channel_mappings().get(self.object_name, None)[0]


@inherit_docstrings
class NiDaqMxSystem(AbstractSystem):
    """
    A set of analog output National Instruments outputs.

    Parameters
    ----------
    outputs : list-like or NiDaqMxSystem
        A list of `NiDaqMxOutput` objects.

    Notes
    -----
    This class uses the `nidaqmx` python module. Values
    are sent by creating a task for each unique device and
    opening the analog output channels.
    """
    writers = None
    tasks = None
    trigger = None

    _output_class = NiDaqMxOutput

    def __init__(self, *args, **kwargs):
        _check_install()
        super().__init__(*args, **kwargs)

    @property
    def unique_devices(self):
        """
        A `numpy.ndarray` of unique devices.
        """
        return np.unique(self.device_order)

    @property
    def device_order(self):
        """
        Order of devices as a `numpy.ndarray` object.
        """
        devices = [
            output.device.name
            for output in self
        ]
        if self.trigger is None:
            return asarray(devices)
        else:
            return asarray(devices + [self.trigger.device.name])

    @property
    def object_order(self):
        """
        Order of each object as a `numpy.ndarray` object.
        """
        objects = [
            output.object_name
            for output in self
        ]
        if self.trigger is None:
            return asarray(objects)
        else:
            return asarray(objects + [self.trigger.object_name])

    def open(self):
        # create task
        # create writer
        if self.tasks is not None:
            self.close()

        self.tasks = pd.Series()

        for device in self.unique_devices:
            self.tasks[device] = nidaqmx.Task(
                device
            )

        for output in self:
            self.tasks[
                output.device.name
            ].ao_channels.add_ao_voltage_chan(
                output.object_name,
                max_val=output.max_val,
                min_val=output.min_val
            )

        if self.trigger is not None:
            self.tasks[
                self.trigger.device.name
            ].ao_channels.add_ao_voltage_chan(
                self.trigger.object_name,
                max_val=self.trigger.max_val,
                min_val=self.trigger.min_val
            )

        self.writers = pd.Series()

        for device_name, task in self.tasks.items():
            self.writers[device_name] = (
                AnalogMultiChannelWriter(task.out_stream)
            )

    def close(self):
        # close task
        for task in self.tasks:
            task.close()
        self.writers = None
        self.tasks = None
        self.trigger = None

    def send(
        self, values, rate=None, return_value=None,
        trigger=None, trigger_rate=1, verbose=False
    ):

        if trigger is not None:
            if isinstance(trigger, NiDaqMxOutput):
                self.trigger = trigger
            else:
                self.trigger = NiDaqMxOutput(
                    trigger, 'trigger',
                    max_intensity_bound=5,
                    zero_intensity_bound=0, units='V',
                )
            self.trigger.rate = trigger_rate

        self.open()

        dt = float(optional_to(1 / rate, 's'))
        values = asarray(values, dtype=np.float64)
        grouped_values, values = self._group_values(values, rate)
        length = len(grouped_values)
        percentile = length // 100
        if not percentile:
            percentile = 1

        if verbose:
            sys.stdout.write('[')

        for idx, grouped_value in enumerate(grouped_values):
            now = time.clock()
            # assumes sending does not take a lot of time
            self._send_value(grouped_value)
            if verbose and not ((idx+1) % percentile):
                sys.stdout.write('*')
            while time.clock() - now < dt:
                pass

        if verbose:
            sys.stdout.write(']')

        if return_value is not None:
            # TODO append trigger if does not exist
            self.send_value(return_value)

        self.close()
        return values

    def _group_values(self, values, rate):

        # add trigger trace
        if values.shape[1] == (self.device_order.size - 1):
            assert self.trigger is not None
            trigger_values = self._create_trigger_array(
                values.shape[0], rate, self.trigger.rate,
                on=self.trigger.max_intensity_bound.magnitude,
                off=self.trigger.zero_intensity_bound.magnitude
            )
            values = np.hstack([
                values, trigger_values[:, None]
            ])
        # assert correct length
        assert values.shape[1] == self.device_order.size

        grouped_values = pd.DataFrame()

        for n, device in enumerate(self.unique_devices):
            _v = values[:, self.device_order == device]
            _v = [v.astype(np.float64) for v in _v]
            grouped_values[n] = _v

        return asarray(grouped_values), values

    def _group_value(self, value):
        grouped_value = []
        for n, device in enumerate(self.unique_devices):
            _v = value[self.device_order == device]
            grouped_value.append(_v)
        return asarray(grouped_value)

    def _send_value(self, grouped_value):
        for writer, value in zip(self.writers, grouped_value):
            writer.write_one_sample(value)

    def send_value(self, value):
        value = asarray(value, dtype=np.float64)
        grouped_value = self._group_value(value)
        self._send_value(grouped_value)
        return value
