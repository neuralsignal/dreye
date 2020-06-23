"""
=======================
Utilities API Reference
=======================

.. currentmodule:: dreye.utilities


Unit-handling
=============

.. autosummary::
   :toctree: api/

   has_units
   optional_to
   get_value
   get_units


Array-handling
==============

.. autosummary::
   :toctree: api/

   asarray
   around
   digits_to_decimals
   round_to_significant
   array_equal
   unique_significant
   spacing
   asarray
   is_uniform
   array_domain
   arange
   is_broadcastable


Instance-checking
=================

.. autosummary::
   :toctree: api/

   is_hashable
   is_string
   is_listlike
   is_dictlike
   is_integer
   is_numeric
   is_callable


Other
=====

.. autosummary::
   :toctree: api/

   irr2flux
   flux2irr
   Filter1D
   CallableList
   convert_truncnorm_clip
"""

from dreye.utilities.common import (
    has_units, optional_to,
    is_hashable, is_string,
    is_listlike, is_dictlike,
    get_value, get_units,
    is_integer, is_numeric,
    is_callable,
    irr2flux,
    flux2irr,
)
from dreye.utilities.array import (
    around, digits_to_decimals,
    round_to_significant, array_equal,
    unique_significant, spacing, asarray,
    is_uniform, array_domain, arange,
    is_broadcastable
)
from dreye.utilities.stats import (
    convert_truncnorm_clip
)
from dreye.utilities.filter1d import Filter1D
from dreye.utilities.abstract import CallableList


__all__ = [
    'CallableList',
    'Filter1D',
    # array
    'array_equal',
    'unique_significant',
    'spacing',
    'is_uniform',
    'array_domain',
    'arange',
    'asarray',
    'digits_to_decimals',
    'round_to_significant',
    'around',
    'is_broadcastable',
    # common
    'has_units',
    'optional_to',
    'is_numeric',
    'is_integer',
    'is_string',
    'is_listlike',
    'is_dictlike',
    'is_hashable',
    'get_units',
    'get_value',
    'is_callable',
    # stats
    'convert_truncnorm_clip',
    'irr2flux',
    'flux2irr'
]
