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

from dreye.api.capture import calculate_capture
from dreye.api.estimator import ReceptorEstimator
from dreye.api.units.convert import irr2flux, flux2irr
from dreye.api.filter_templates import gaussian_template, govardovskii2000_template, stavenga1993_template


__all__ = [
    'calculate_capture',
    'ReceptorEstimator', 
    'irr2flux', 
    'flux2irr', 
    'gaussian_template',
    'govardovskii2000_template', 
    'stavenga1993_template'
]