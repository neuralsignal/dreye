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
