=============
API Reference
=============

.. currentmodule:: dreye

Signal-type classes
===================

One-dimensional `Signal` classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   Signal
   Spectrum
   IntensitySpectrum
   CalibrationSpectrum

Two-dimensional labelled `Signals` classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   Signals
   Spectra
   IntensitySpectra


Two-dimensional `DomainSignal` classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   DomainSignal
   DomainSpectrum
   IntensityDomainSpectrum
   MeasuredSpectrum


Signal-type container classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   SignalsContainer
   DomainSignalContainer
   MeasuredSpectraContainer


Convenience functions to create signal-type classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   create_gaussian_spectrum
   create_measured_spectrum
   get_led_spectra_container


Photoreceptor models
====================


.. autosummary::
   :toctree: api/

   Photoreceptor
   LinearPhotoreceptor
   LogPhotoreceptor
   HyperbolicPhotoreceptor


Scikit-learn type estimators
============================

.. autosummary::
   :toctree: api/

   IntensityFit
   RelativeIntensityFit
   IndependentExcitationFit
   TransformExcitationFit
   ReflectanceExcitationFit
