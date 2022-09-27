"""
Various general metrics
"""

import numpy as np
from numpy.random import default_rng
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

from dreye.api.barycentric import barycentric_dim_reduction
from dreye.api.project import proj_P_for_hull, proj_P_to_simplex


def compute_jensen_shannon_divergence(P, Q, base=2):
    """Jensen-Shannon divergence of P and Q.

    Parameters
    ----------
    P : [type]
        [description]
    Q : [type]
        [description]
    base : int, optional
        [description], by default 2

    Returns
    -------
    [type]
        [description]
    """
    P = np.asarray(P)
    Q = np.asarray(Q)

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

    Parameters
    ----------
    P : [type]
        [description]
    Q : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return 1 - compute_jensen_shannon_divergence(P, Q)


def compute_mean_width(X, n=1000, vectorized=False, center=False, seed=None):
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
    X = np.asarray(X)
    
    if (X.ndim == 1) or (X.shape[-1] < 2):
        return np.max(X) - np.min(X)
    if not center:
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


def compute_mean_correlation(X):
    """[summary]

    Parameters
    ----------
    X : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    X = np.asarray(X)
    
    # compute correlation of each feature
    cc = np.abs(np.corrcoef(X, rowvar=False))
    return (cc - np.eye(cc.shape[0])).mean()


def compute_mean_mutual_info(X, **kwargs):
    """[summary]

    Parameters
    ----------
    X : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    X = np.asarray(X)
    
    mis = []
    for idx in range(X.shape[1] - 1):
        mi = mutual_info_regression(X[idx], X[idx + 1:], **kwargs)
        mis.append(mi)
    return np.concatenate(mis).mean()


def compute_volume(X):
    """[summary]

    Parameters
    ----------
    X : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    X = np.asarray(X)

    if (X.ndim == 1) or (X.shape[1] < 2):
        return np.max(X) - np.min(X)
    hull = proj_P_for_hull(X)
    # if one-dimensional a simple array is returned
    if isinstance(hull, np.ndarray):
        return np.max(hull) - np.min(hull)
    # if multi-dimensional then a convex hull object is returned
    return hull.volume


def compute_gamut(
    X,
    at_l1=None, 
    relative_to=None,
    center_to_neutral=False,
    metric='width', 
    center=True,
    seed=None
):
    """
    Compute absolute gamut
    """
    X = np.asarray(X)
    
    if metric == 'width':
        func = compute_mean_width
        kwargs = {
            'seed': seed, 
            'center': center
        }
    elif metric == 'volume':
        func = compute_volume
        kwargs = {}
    else:
        raise NameError(f"Metric must be `width` or `volume`, but is `{metric}`.")

    l1 = X.sum(axis=-1)
    
    if at_l1 is not None:
        if np.all(l1 <= at_l1):
            return 0
        elif np.all(l1 > at_l1):
            return 0
        # just keep current points if zero q the only one below at_l1
        elif np.all(l1[l1 < at_l1] == 0):
            pass
        else:
            X = proj_P_to_simplex(X, at_l1)
            l1 = X.sum(axis=-1)
    
    X = X[l1 != 0]
    if X.shape[0] == 0:
        return 0
    X = barycentric_dim_reduction(X, center=center_to_neutral)
    num = func(X, **kwargs)
    
    if relative_to is not None:
        denom = compute_gamut(
            relative_to, 
            metric=metric, 
            at_l1=None,
            relative_to=None,
            center=center,
            center_to_neutral=center_to_neutral, 
            seed=seed
        )
        return num / denom
    return num