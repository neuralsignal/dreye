"""
Methods for capture calculation.
"""

import numpy as np
from numbers import Number


def calculate_capture(
    filters, 
    signals,
    domain=1.0, 
    trapz=True, 
):
    """Calculate the stimulus-induced capture.

    Parameters
    ----------
    filters : array-like (..., filter_dim, domain_dim)
        [description]
    signals : array-like (..., signal_dim, domain_dim)
        [description]
    domain : float or array-like (domain_dim), optional
        If float, it is assumed to be the step size in the domain (e.g. dx=1nm for wavelengths). 
        If array-like, it is assumed to be an ascending array where each element is the
        value in domain coordinates (e.g. [340, 350, ..., 670, 680]nm for wavelengths).
        By default 1.0.
    trapz : bool, optional
        Whether to use the trapezoidal method to calculate the integral or simply use the sum.
        If domain is array-like, this argument is ignored and the trapezoidal method is
        always used. By default True.

    Returns
    -------
    captures : array-like (..., signal_dim, filter_dim)
        [description]
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
            return np.sum(filters * signals * domain, axis=-1)
    return np.trapz(filters * signals, x=np.asarray(domain), axis=-1)