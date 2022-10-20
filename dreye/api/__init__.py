"""
Main API functions and classes
"""

from dreye.api.capture import calculate_capture
from dreye.api.estimator import ReceptorEstimator
from dreye.api.units.convert import irr2flux, flux2irr
from dreye.api.filter_templates import gaussian_template, govardovskii2000_template, stavenga1993_template
from dreye.api.utils import l1norm, l2norm, integral


__all__ = [
    'calculate_capture',
    'ReceptorEstimator', 
    'irr2flux', 
    'flux2irr', 
    'gaussian_template',
    'govardovskii2000_template', 
    'stavenga1993_template', 
    'l1norm', 'l2norm', 
    'integral'
]