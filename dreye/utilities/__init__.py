"""
utilities
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
    is_signallike, is_signalslike
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
from dreye.utilities.barycentric import (
    barycentric_to_cartesian,
    barycentric_dim_reduction,
    barycentric_to_cartesian_transformer
)
from dreye.utilities.metrics import (
    compute_mean_width,
    compute_jensen_shannon_similarity,
    compute_jensen_shannon_divergence

)


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
    'is_signallike', 
    'is_signalslike',
    # stats
    'convert_truncnorm_clip',
    'irr2flux',
    'flux2irr',
    # barycentric
    'barycentric_to_cartesian',
    'barycentric_dim_reduction',
    'barycentric_to_cartesian_transformer',
    'compute_mean_width',
    'compute_jensen_shannon_similarity',
    'compute_jensen_shannon_divergence',
]
