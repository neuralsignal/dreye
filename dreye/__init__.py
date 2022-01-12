"""
*dreye* (drosophila eye) is a *Python* photoreceptor and (color) vision
package implementing various photoreceptor and (color) vision models and
algorithms, mainly for drosophila.
Part of the code base is inspired by the Colour GitHub project
(https://github.com/colour-science/colour; BSD-3-Clause license),
which is a package implementing color vision models and algorithms for human
observers.
"""

__author__ = """gucky92"""
__email__ = 'gucky@gucky.eu'
__version__ = '1.0.0dev1'

import os
DREYE_DIR = os.path.dirname(__file__)

from dreye.api import calculate_capture, ReceptorEstimator

__all__ = [
    'calculate_capture', 
    'ReceptorEstimator', 
]