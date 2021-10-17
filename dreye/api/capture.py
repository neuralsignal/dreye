"""
Methods for capture calculation.
"""

import numpy as np
import jax.numpy as jnp
from numbers import Number

from dreye.utilities.common import is_numeric


def calculate_capture(
    filters, 
    signals,
    domain=1.0, 
    trapz=True, 
):
    """
    Calculate the light-induced capture.

    Parameters
    ----------
    filters : ndarray (..., filter_dim, domain_dim)
    signals : ndarray (..., signal_dim, domain_dim)
    domain : float or ndarray (domain_dim)
    trapz : bool

    Returns
    -------
    captures : ndarray (signal_dim x filter_dim)
    """
    filters = np.asarray(filters)
    signals = np.asarray(signals)

    # broadcasting rules when arrays were passed
    if (filters.ndim > 1) and (signals.ndim > 1):
        filters = filters[..., None, :, :]
        signals = signals[..., :, None, :]


    if isinstance(domain, Number):
        if trapz:
            return np.trapz(filters * signals, dx=domain, axis=-1)
        else:
            return np.sum(filters * signals, axis=-1) * domain
    return np.trapz(filters * signals, x=domain, axis=-1)