AbstractSpectrum
================

.. currentmodule:: dreye

.. autoclass:: AbstractSpectrum
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~AbstractSpectrum.T
      ~AbstractSpectrum.attrs
      ~AbstractSpectrum.boundaries
      ~AbstractSpectrum.contexts
      ~AbstractSpectrum._convert_attributes
      ~AbstractSpectrum.domain
      ~AbstractSpectrum.domain_axis
      ~AbstractSpectrum.domain_len
      ~AbstractSpectrum.domain_max
      ~AbstractSpectrum.domain_min
      ~AbstractSpectrum.dtype
      ~AbstractSpectrum.gradient
      ~AbstractSpectrum._init_args
      ~AbstractSpectrum.init_kwargs
      ~AbstractSpectrum.integral
      ~AbstractSpectrum.interpolate
      ~AbstractSpectrum.interpolator
      ~AbstractSpectrum.interpolator_kwargs
      ~AbstractSpectrum.labels
      ~AbstractSpectrum.magnitude
      ~AbstractSpectrum.name
      ~AbstractSpectrum.nanless
      ~AbstractSpectrum.ndim
      ~AbstractSpectrum.normalized_signal
      ~AbstractSpectrum.other_axis
      ~AbstractSpectrum.other_len
      ~AbstractSpectrum.piecewise_gradient
      ~AbstractSpectrum.piecewise_integral
      ~AbstractSpectrum.shape
      ~AbstractSpectrum.signal_max
      ~AbstractSpectrum.signal_min
      ~AbstractSpectrum.size
      ~AbstractSpectrum.smooth
      ~AbstractSpectrum.smoothing_args
      ~AbstractSpectrum.smoothing_method
      ~AbstractSpectrum.smoothing_window
      ~AbstractSpectrum.units
      ~AbstractSpectrum.values
      ~AbstractSpectrum.wavelengths

   .. rubric:: Methods Summary

   .. autosummary::

      ~AbstractSpectrum.__call__
      ~AbstractSpectrum.append
      ~AbstractSpectrum.asarray
      ~AbstractSpectrum.concat
      ~AbstractSpectrum.concat_labels
      ~AbstractSpectrum.copy
      ~AbstractSpectrum.corr
      ~AbstractSpectrum.cov
      ~AbstractSpectrum.domain_concat
      ~AbstractSpectrum.dot
      ~AbstractSpectrum.enforce_uniformity
      ~AbstractSpectrum.equalize_domains
      ~AbstractSpectrum.from_dict
      ~AbstractSpectrum.load
      ~AbstractSpectrum.max
      ~AbstractSpectrum.mean
      ~AbstractSpectrum.min
      ~AbstractSpectrum.moveaxis
      ~AbstractSpectrum.nanmax
      ~AbstractSpectrum.nanmean
      ~AbstractSpectrum.nanmin
      ~AbstractSpectrum.nanstd
      ~AbstractSpectrum.nansum
      ~AbstractSpectrum.numpy_estimator
      ~AbstractSpectrum.other_concat
      ~AbstractSpectrum.plot
      ~AbstractSpectrum.save
      ~AbstractSpectrum.std
      ~AbstractSpectrum.sum
      ~AbstractSpectrum.to
      ~AbstractSpectrum.to_dict
      ~AbstractSpectrum.window_filter

   .. rubric:: Attributes Documentation

   .. autoattribute:: T
   .. autoattribute:: attrs
   .. autoattribute:: boundaries
   .. autoattribute:: contexts
   .. autoattribute:: _convert_attributes
   .. autoattribute:: domain
   .. autoattribute:: domain_axis
   .. autoattribute:: domain_len
   .. autoattribute:: domain_max
   .. autoattribute:: domain_min
   .. autoattribute:: dtype
   .. autoattribute:: gradient
   .. autoattribute:: _init_args
   .. autoattribute:: init_kwargs
   .. autoattribute:: integral
   .. autoattribute:: interpolate
   .. autoattribute:: interpolator
   .. autoattribute:: interpolator_kwargs
   .. autoattribute:: labels
   .. autoattribute:: magnitude
   .. autoattribute:: name
   .. autoattribute:: nanless
   .. autoattribute:: ndim
   .. autoattribute:: normalized_signal
   .. autoattribute:: other_axis
   .. autoattribute:: other_len
   .. autoattribute:: piecewise_gradient
   .. autoattribute:: piecewise_integral
   .. autoattribute:: shape
   .. autoattribute:: signal_max
   .. autoattribute:: signal_min
   .. autoattribute:: size
   .. autoattribute:: smooth
   .. autoattribute:: smoothing_args
   .. autoattribute:: smoothing_method
   .. autoattribute:: smoothing_window
   .. autoattribute:: units
   .. autoattribute:: values
   .. autoattribute:: wavelengths

   .. rubric:: Methods Documentation

   .. automethod:: __call__
   .. automethod:: append
   .. automethod:: asarray
   .. automethod:: concat
   .. automethod:: concat_labels
   .. automethod:: copy
   .. automethod:: corr
   .. automethod:: cov
   .. automethod:: domain_concat
   .. automethod:: dot
   .. automethod:: enforce_uniformity
   .. automethod:: equalize_domains
   .. automethod:: from_dict
   .. automethod:: load
   .. automethod:: max
   .. automethod:: mean
   .. automethod:: min
   .. automethod:: moveaxis
   .. automethod:: nanmax
   .. automethod:: nanmean
   .. automethod:: nanmin
   .. automethod:: nanstd
   .. automethod:: nansum
   .. automethod:: numpy_estimator
   .. automethod:: other_concat
   .. automethod:: plot
   .. automethod:: save
   .. automethod:: std
   .. automethod:: sum
   .. automethod:: to
   .. automethod:: to_dict
   .. automethod:: window_filter
