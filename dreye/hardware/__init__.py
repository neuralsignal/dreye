"""
======================
Hardware API Reference
======================

.. currentmodule:: dreye.hardware


Abstract base classes
=====================

.. autosummary::
   :toctree: api/

   AbstractSpectrometer
   AbstractOutput
   AbstractSystem


Available Spectrometers
=======================

.. autosummary::
   :toctree: api/

   DummySpectrometer
   OceanSpectrometer


Available Output Devices
========================

.. autosummary::
   :toctree: api/

   DummyOutput
   DummySystem
   NiDaqMxOutput
   NiDaqMxSystem

Misc functions
==============

.. autosummary::
   :toctree: api/

   read_calibration_file
"""

from dreye.hardware.base_spectrometer import AbstractSpectrometer
from dreye.hardware.base_system import AbstractOutput, AbstractSystem
from dreye.hardware.dummy_spectrometer import DummySpectrometer
from dreye.hardware.dummy_system import DummyOutput, DummySystem
from dreye.hardware.measurement_runner import MeasurementRunner
from dreye.hardware.nidaqmx import NiDaqMxSystem, NiDaqMxOutput
from dreye.hardware.seabreeze import OceanSpectrometer, read_calibration_file


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
    'read_calibration_file'
]
