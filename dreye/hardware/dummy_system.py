"""Dummy system
"""

from dreye.hardware.base_system import AbstractOutput, AbstractSystem
from dreye.utilities.abstract import inherit_docstrings


@inherit_docstrings
class DummyOutput(AbstractOutput):
    """
    Dummy output hardware.

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
    """
    # writer to send values
    _current_value = None
    _open = False

    def open(self):
        self._open = True

    def close(self):
        self._open = False

    def send(self, values, rate=None, return_value=None):
        pass

    def send_value(self, value):
        self._current_value = value

    def channel_exists(self):
        return True


@inherit_docstrings
class DummySystem(AbstractSystem):
    """
    Dummy output system.

    Parameters
    ----------
    outputs : list-like or AbstractSystem
        A list of AbstractOutput objects.
    """
    _current_value = None
    _open = False

    def open(self):
        self._open = True

    def close(self):
        self._open = False

    def send(
        self, values, rate=None, return_value=None,
        trigger=None, trigger_rate=1, verbose=False
    ):
        pass

    def send_value(self, value):
        self._current_value = value
