"""
"""

import numpy as np

DEFAULT_FLOAT_DTYPE = np.float64
DEFAULT_INT_DTYPE = np.int64

RELATIVE_ACCURACY = 10
"""
Relative accuracy in number of significant digits used in dreye for any
float. For example, it is used
to compare if to arrays/signals are equal
"""
ABSOLUTE_ACCURACY = 1e-10
"""
Absolute accuracy used in dreye for any float. For example, it is used
to compare if to arrays/signals are equal
"""
