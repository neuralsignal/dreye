"""Test environment
"""

import os
import sys
import numpy as np
np.random.seed(10)

test_datapath = os.path.join(os.path.dirname(__file__), 'test_data')
if not os.path.exists(test_datapath):
    os.makedirs(test_datapath)

# insert dreye path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

import dreye
from dreye import err
