Signal
======

.. currentmodule:: dreye

.. autoclass:: Signal
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~Signal.T
      ~Signal.attrs
      ~Signal.boundaries
      ~Signal.convert_attributes
      ~Signal.domain
      ~Signal.domain_axis
      ~Signal.domain_len
      ~Signal.domain_max
      ~Signal.domain_min
      ~Signal.dtype
      ~Signal.gradient
      ~Signal.init_args
      ~Signal.integral
      ~Signal.interpolate
      ~Signal.interpolator
      ~Signal.interpolator_kwargs
      ~Signal.labels
      ~Signal.name
      ~Signal.nanless
      ~Signal.normalized_signal
      ~Signal.other_axis
      ~Signal.other_len
      ~Signal.piecewise_gradient
      ~Signal.piecewise_integral
      ~Signal.signal_max
      ~Signal.signal_min
      ~Signal.values

   .. rubric:: Methods Summary

   .. autosummary::

      ~Signal.__call__
      ~Signal.append
      ~Signal.concat
      ~Signal.concat_labels
      ~Signal.corr
      ~Signal.cov
      ~Signal.domain_concat
      ~Signal.dot
      ~Signal.enforce_uniformity
      ~Signal.from_dict
      ~Signal.load
      ~Signal.max
      ~Signal.mean
      ~Signal.min
      ~Signal.moveaxis
      ~Signal.nanmax
      ~Signal.nanmean
      ~Signal.nanmin
      ~Signal.nanstd
      ~Signal.nansum
      ~Signal.numpy_estimator
      ~Signal.other_concat
      ~Signal.save
      ~Signal.std
      ~Signal.sum
      ~Signal.to_dict
      ~Signal.window_filter

   .. rubric:: Attributes Documentation

   .. autoattribute:: T
   .. autoattribute:: attrs
   .. autoattribute:: boundaries
   .. autoattribute:: convert_attributes
   .. autoattribute:: domain
   .. autoattribute:: domain_axis
   .. autoattribute:: domain_len
   .. autoattribute:: domain_max
   .. autoattribute:: domain_min
   .. autoattribute:: dtype
   .. autoattribute:: gradient
   .. autoattribute:: init_args
   .. autoattribute:: integral
   .. autoattribute:: interpolate
   .. autoattribute:: interpolator
   .. autoattribute:: interpolator_kwargs
   .. autoattribute:: labels
   .. autoattribute:: name
   .. autoattribute:: nanless
   .. autoattribute:: normalized_signal
   .. autoattribute:: other_axis
   .. autoattribute:: other_len
   .. autoattribute:: piecewise_gradient
   .. autoattribute:: piecewise_integral
   .. autoattribute:: signal_max
   .. autoattribute:: signal_min
   .. autoattribute:: values

   .. rubric:: Methods Documentation

   .. automethod:: __call__
   .. automethod:: append
   .. automethod:: concat
   .. automethod:: concat_labels
   .. automethod:: corr
   .. automethod:: cov
   .. automethod:: domain_concat
   .. automethod:: dot
   .. automethod:: enforce_uniformity
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
   .. automethod:: save
   .. automethod:: std
   .. automethod:: sum
   .. automethod:: to_dict
   .. automethod:: window_filter
