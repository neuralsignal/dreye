"""
Various general metrics
"""

import numpy as np
from numpy.random import default_rng
from scipy import stats


def compute_jensen_shannon_divergence(P, Q, base=2):
    """
    Jensen-Shannon divergence of P and Q.
    """
    assert P.shape == Q.shape, "`P` and `Q` must be the same shape"
    P = P.ravel()
    Q = Q.ravel()
    _P = P / np.linalg.norm(P, ord=1)
    _Q = Q / np.linalg.norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (
        stats.entropy(_P, _M, base=base)
        + stats.entropy(_Q, _M, base=base)
    )


def compute_jensen_shannon_similarity(P, Q):
    """
    Compute Jensen-Shannon divergence with base 2 and subtract it from 1,
    so that 1 is equality of distribution and 0 is no similarity.
    """
    return 1 - compute_jensen_shannon_divergence(P, Q)


def compute_mean_width(X, n=1000, vectorized=False, centered=False, seed=None):
    """
    Compute mean width by projecting `X` onto random vectors

    Parameters
    ----------
    X : numpy.ndarray
        n x m matrix with n samples and m features.
    n : int
        Number of random projections to calculate width

    Returns
    -------
    mean_width : float
        Mean width of `X`.
    """
    if not centered:
        X = X - X.mean(0)  # centering data
    rprojs = default_rng(seed).standard_normal(size=(X.shape[-1], n))
    rprojs /= np.linalg.norm(rprojs, axis=0)  # normalize vectors by l2-norm
    if vectorized:
        proj = X @ rprojs  # project samples onto random vectors
        max1 = proj.max(0)  # max across samples
        max2 = (-proj).max(0)  # max across samples
    else:
        max1 = np.zeros(n)
        max2 = np.zeros(n)
        for idx, rproj in enumerate(rprojs.T):
            proj = X @ rproj
            max1[idx] = proj.max()
            max2[idx] = (-proj).max()
    return (max1 + max2).mean()
