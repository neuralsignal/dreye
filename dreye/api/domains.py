"""
Handling domains
"""

from scipy.interpolate import interp1d

def equalize_domains(
    domains, arrs, axes=None
):
    """
    equalize domains between different arrays.
    """