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
    filters : ndarray of shape (..., n_filters, n_domain)
        Filter functions for each receptor type.
    signals : ndarray of shape (..., n_signals, n_domain)
        The signals exciting all receptor types.
    domain : float or ndarray of shape (n_domain), optional
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
    captures : ndarray of shape (..., n_signals, n_filters)
        Calculated capture values.
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
