"""Dummy system
"""

import time
import os
import sys
import datetime

import numpy as np
import pandas as pd

from dreye.err import DreyeError
from dreye.hardware.base_system import AbstractOutput, AbstractSystem
from dreye.utilities import convert_units


class DummyOutput(AbstractOutput):
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


class DummySystem(AbstractSystem):
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
