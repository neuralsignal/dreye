"""
Default global variables
"""

import numpy as np


DEFAULT_FLOAT_DTYPE = np.float64
DEFAULT_INT_DTYPE = np.int64
EPS_NP64 = np.finfo(np.float64).eps
EPS_NP32 = np.finfo(np.float32).eps

RELATIVE_ACCURACY = 5
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

DEFAULT_WL_RANGE = np.arange(300, 701, 1.0)
