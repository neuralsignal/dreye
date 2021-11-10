"""
Handling domains
"""

from numbers import Number
import numpy as np
from scipy.interpolate import interp1d

from dreye.api.utils import arange


def equalize_domains(
    domains, arrs, axes=None, fill_value=0, 
    bounds_error=False, stack_axis=None, concatenate=False
):
    """
    equalize domains between different arrays.
    """
    count = len(domains)

    if axes is None:
        axes = [-1] * count
    elif isinstance(axes, Number):
        axes = [axes] * count

    lemin = -np.inf
    lemax = np.inf
    lediff = 0
    for domain in domains:
        lemin = np.maximum(lemin, np.min(domain))
        lemax = np.minimum(lemax, np.max(domain))
        lediff = np.maximum(lediff, np.mean(np.diff(np.sort(domain))))
        
    if (lemin >= lemax) or ((lemax - lemin) < lediff):
        raise ValueError("Cannot equalize domains.")

    new_domain = arange(lemin, lemax, lediff)
    # new_domain = np.arange(lemin, lemax+lediff-lemax%lediff, lediff)
    new_arrs = []
    for domain, arr, axis in zip(domain, arrs, axes):
        arr = interp1d(domain, arr, axis=axis, fill_value=fill_value, bounds_error=bounds_error)(new_domain)
        new_arrs.append(arr)

    if stack_axis is None:
        pass
    elif concatenate:
        new_arrs = np.concatenate(new_arrs, axis=stack_axis)
    else:
        new_arrs = np.stack(new_arrs, axis=stack_axis)

    return new_domain, new_arrs


# TODO smooth
# TODO integral, normalize
