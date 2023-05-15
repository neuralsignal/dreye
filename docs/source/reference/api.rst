==================
Core API Reference
==================

.. currentmodule:: dreye

ReceptorEstimator class
=======================

.. autosummary::
   :toctree: api/

   ReceptorEstimator

Template functions
==================

.. autosummary::
  :toctree: api/

  gaussian_template
  govardovskii2000_template
  stavenga1993_template

Conversion functions
====================

.. autosummary::
  :toctree: api/

  irr2flux
  flux2irr
  optional_to
  has_units

Utility functions
=================

.. autosummary::
  :toctree: api/

  l1norm
  l2norm
  integral
  calculate_capture
  round_to_precision
  round_to_significant_digits
  arange_with_interval

Barycentric functions
=====================

.. autosummary::
  :toctree: api/

  barycentric_to_cartesian
  cartesian_to_barycentric

Convex Hull functions
=====================

.. autosummary::
  :toctree: api/

  in_hull
  range_of_solutions

Domain functions
================

.. autosummary::
  :toctree: api/

  equalize_domains

Metrics functions
=================

.. autosummary::
  :toctree: api/

  compute_jensen_shannon_divergence
  compute_jensen_shannon_similarity
  compute_mean_width
  compute_mean_correlation
  compute_mean_mutual_info
  compute_volume
  compute_gamut

Projection functions
====================

.. autosummary::
  :toctree: api/

  proj_P_for_hull
  proj_P_to_simplex
  line_to_simplex
  proj_B_to_hull
  alpha_for_B_with_P
  B_with_P

Sampling functions
==================

.. autosummary::
  :toctree: api/

  sample_in_hull
  d_equally_spaced

Spherical functions
===================

.. autosummary::
  :toctree: api/

  spherical_to_cartesian
  cartesian_to_spherical
