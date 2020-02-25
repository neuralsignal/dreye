"""
"""

import time
import os
import datetime

import numpy as np
import pandas as pd

from dreye.err import DreyeError
from dreye.hardware.base_system import AbstractOutput, AbstractSystem
from dreye.utilities import convert_units

# HARDWARE API IMPORTS
try:
    import nidaqmx
    import nidaqmx.system
    from nidaqmx.stream_writers import \
        AnalogMultiChannelWriter, AnalogSingleChannelWriter
except ImportError as e:
    raise DreyeError(f"You need to install nidaqmx: {e}")


def get_channels():
    """get all channels from all NI DAQ devices.
    """

    system = nidaqmx.system.System.local()
    return [
        chan.name
        for device in system.devices
        for chan in device.ao_physical_chans
    ]


def get_channel_mappings():
    """get all NI DAQ devices
    """
    system = nidaqmx.system.System.local()
    return {
        chan.name: (device, chan)
        for device in system.devices
        for chan in device.ao_physical_chans
    }


class NiDaqMxOutput(AbstractOutput):
    # writer to send values
    task = None
    writer = None

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
        dt = float(convert_units(1 / rate, 's'))
        self.open()
        for value in values.astype(np.float64):
            # assumes sending does not take a lot of time
            self.send_value(value)
            time.sleep(dt)
        if return_value is not None:
            self.send_value(return_value)
        self.close()

    def send_value(self, value):
        self.writer.write_one_sample(value)

    def channel_exists(self):
        return self.object_name in get_channels()

    @property
    def channel(self):
        return get_channel_mappings().get(self.object_name, None)[1]

    @property
    def device(self):
        return get_channel_mappings().get(self.object_name, None)[0]


class NiDaqMxSystem(AbstractSystem):
    writers = None
    tasks = None
    trigger = None

    @property
    def unique_devices(self):
        return np.unique(self.device_order)

    @property
    def device_order(self):
        devices = [
            output.device
            for output in self
        ]
        if self.trigger is None:
            return np.array(devices)
        else:
            return np.array(devices + [self.trigger.device])

    @property
    def object_order(self):
        objects = [
            output.object_name
            for output in self
        ]
        if self.trigger is None:
            return np.array(objects)
        else:
            return np.array(objects + [self.trigger.object_name])

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
                output.device
            ].ao_channels.add_ao_voltage_chan(
                output.object_name,
                max_val=output.max_val,
                min_val=output.min_val
            )

        if self.trigger is not None:
            self.tasks[
                self.trigger.device
            ].ao_channels.add_ao_voltage_chan(
                self.trigger.object_name,
                max_val=self.trigger.max_val,
                min_val=self.trigger.min_val
            )

        self.writers = pd.Series()

        for device, task in self.tasks.items():
            self.writers[device] = (
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
        trigger=None, trigger_rate=1,
    ):

        if trigger is not None:
            if isinstance(trigger, NiDaqMxOutput):
                self.trigger = trigger
            else:
                self.trigger = NiDaqMxOutput(
                    trigger, 'trigger',
                    max_boundary=5,
                    zero_boundary=0, units='V',
                )
            self.trigger.rate = trigger_rate

        self.open()

        dt = float(convert_units(1 / rate, 's'))
        grouped_values = self._group_values(values, rate)

        for grouped_value in grouped_values:
            # assumes sending does not take a lot of time
            self._send_value(grouped_value)
            time.sleep(dt)

        if return_value is not None:
            self.send_value(return_value)

        self.close()

    def _group_values(self, values, rate):

        # add trigger trace
        if values.shape[1] == (self.device_order.size - 1):
            assert self.trigger is not None
            trigger_values = self._create_trigger_array(
                values.shape[0], rate, self.trigger.rate,
                on=self.trigger.max_boundary,
                off=self.trigger.zero_boundary
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

        return np.array(grouped_values)

    def _group_value(self, value):
        grouped_value = []
        for n, device in enumerate(self.unique_devices):
            _v = value[self.object_order == device]
            grouped_value.append(_v)
        return np.array(grouped_value)

    def _send_value(self, grouped_value):
        for writer, value in zip(self.writers, grouped_value):
            writer.write_one_sample(value)

    def send_value(self, value):
        grouped_value = self._group_value(value)
        self._send_value(grouped_value)
