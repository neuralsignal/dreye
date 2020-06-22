"""
Hardware API and abstract classes to create own hardware API
"""

from dreye.hardware.base_spectrometer import AbstractSpectrometer
from dreye.hardware.base_system import AbstractOutput, AbstractSystem
from dreye.hardware.dummy_spectrometer import DummySpectrometer
from dreye.hardware.dummy_system import DummyOutput, DummySystem
from dreye.hardware.measurement_runner import MeasurementRunner
from dreye.hardware.nidaqmx import NiDaqMxSystem, NiDaqMxOutput
from dreye.hardware.seabreeze import OceanSpectrometer


__all__ = [
    'MeasurementRunner',
    'NiDaqMxSystem',
    'NiDaqMxOutput',
    'OceanSpectrometer',
    'AbstractSpectrometer',
    'AbstractSystem',
    'AbstractOutput',
    'DummySpectrometer',
    'DummyOutput',
    'DummySystem',
]
