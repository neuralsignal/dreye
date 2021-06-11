==================
Core API Reference
==================

.. currentmodule:: dreye


Domain class
============

.. autosummary::
   :toctree: api/

   Domain


Signal-type classes
===================

.. autosummary::
   :toctree: api/

   Signal
   Signals
   DomainSignal


Signal-type container classes
=============================

.. autosummary::
   :toctree: api/

   SignalsContainer
   DomainSignalContainer


Signal-type classes for handling spectral measurements
======================================================

.. autosummary::
   :toctree: api/

   CalibrationSpectrum
   MeasuredSpectrum
   MeasuredSpectraContainer


Convenience functions to create signal-type classes
===================================================

.. autosummary::
   :toctree: api/

   create_gaussian_spectrum
   create_measured_spectrum
   create_measured_spectra_container


Photoreceptor models
====================

.. autosummary::
   :toctree: api/

   Photoreceptor
   LinearPhotoreceptor
   LinearContrastPhotoreceptor
   LogPhotoreceptor
   HyperbolicPhotoreceptor
   create_photoreceptor_model


Scikit-learn type estimators
============================

.. autosummary::
   :toctree: api/

   RelativeIntensityFit
   IndependentExcitationFit
   TransformExcitationFit
   NonlinearTransformExcitationFit
   BestSubstitutionFit
   LedSubstitutionFit


Miscellaneous functions and classes
===================================

.. autosummary::
  :toctree: api/

  ureg
  read_json
  write_json
  read_pickle
  write_pickle
  irr2flux
  flux2irr
  stavenga1993_template
  govardovskii2000_template
