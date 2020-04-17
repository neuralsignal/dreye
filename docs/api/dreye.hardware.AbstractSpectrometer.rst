AbstractSpectrometer
====================

.. currentmodule:: dreye.hardware

.. autoclass:: AbstractSpectrometer
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~AbstractSpectrometer.cal
      ~AbstractSpectrometer.calibration
      ~AbstractSpectrometer.current_it
      ~AbstractSpectrometer.ideal_lower_bound
      ~AbstractSpectrometer.ideal_mid_point
      ~AbstractSpectrometer.ideal_upper_bound
      ~AbstractSpectrometer.int
      ~AbstractSpectrometer.integration_time
      ~AbstractSpectrometer.intensities
      ~AbstractSpectrometer.intensity
      ~AbstractSpectrometer.ints
      ~AbstractSpectrometer.max_it
      ~AbstractSpectrometer.max_photon_count
      ~AbstractSpectrometer.maxi
      ~AbstractSpectrometer.min_it
      ~AbstractSpectrometer.original_it
      ~AbstractSpectrometer.signal
      ~AbstractSpectrometer.size
      ~AbstractSpectrometer.wavelengths
      ~AbstractSpectrometer.wls

   .. rubric:: Methods Summary

   .. autosummary::

      ~AbstractSpectrometer.avg_ints
      ~AbstractSpectrometer.close
      ~AbstractSpectrometer.find_best_it
      ~AbstractSpectrometer.maxi_within_bounds
      ~AbstractSpectrometer.open
      ~AbstractSpectrometer.perform_measurement
      ~AbstractSpectrometer.set_it

   .. rubric:: Attributes Documentation

   .. autoattribute:: cal
   .. autoattribute:: calibration
   .. autoattribute:: current_it
   .. autoattribute:: ideal_lower_bound
   .. autoattribute:: ideal_mid_point
   .. autoattribute:: ideal_upper_bound
   .. autoattribute:: int
   .. autoattribute:: integration_time
   .. autoattribute:: intensities
   .. autoattribute:: intensity
   .. autoattribute:: ints
   .. autoattribute:: max_it
   .. autoattribute:: max_photon_count
   .. autoattribute:: maxi
   .. autoattribute:: min_it
   .. autoattribute:: original_it
   .. autoattribute:: signal
   .. autoattribute:: size
   .. autoattribute:: wavelengths
   .. autoattribute:: wls

   .. rubric:: Methods Documentation

   .. automethod:: avg_ints
   .. automethod:: close
   .. automethod:: find_best_it
   .. automethod:: maxi_within_bounds
   .. automethod:: open
   .. automethod:: perform_measurement
   .. automethod:: set_it
