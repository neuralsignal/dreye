"""various utility functions for handinglin distributions
"""

import numpy as np


def sample_truncated_distribution(
    distribution, lower=None, upper=None, method='clip', **kwargs
):
    """sample from a truncated distribution

    Parameters
    ----------
    distribution : scipy.stats.rv_continuous
        continuous scipy distribution
    lower : float
        lower bound of truncated distribution
    upper : float
        upper bound of truncated distribution
    """

    samples = distribution.rvs(**kwargs)

    if lower is None and upper is None:
        return samples

    if method == 'clip':
        samples = np.clip(samples, lower, upper)
    else:
        raise NameError(f'method {method} does not exist.')

    return samples


def convert_truncnorm_clip(a, b, loc, scale):
    """convert truncnorm clip values to standard form.
    """

    return (a - loc) / scale, (b - loc) / scale
