"""
This module contains various general metrics computation functions.

These functions include:
    - Jensen-Shannon divergence and similarity computation.
    - Mean width computation.
    - Mean correlation computation.
    - Mean mutual information computation.
    - Volume computation.
    - Gamut computation.
"""

import numpy as np
from numpy.random import default_rng
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

from dreye.api.barycentric import barycentric_dim_reduction
from dreye.api.project import proj_P_for_hull, proj_P_to_simplex


def compute_jensen_shannon_divergence(P, Q, base=2):
    """
    Compute Jensen-Shannon divergence between two probability distributions P and Q.

    Parameters
    ----------
    P : numpy.ndarray
        The first probability distribution.
    Q : numpy.ndarray
        The second probability distribution.
    base : int, optional
        The logarithmic base to use, by default 2

    Returns
    -------
    float
        The Jensen-Shannon divergence between P and Q.
    """
    P, Q = np.asarray(P).ravel(), np.asarray(Q).ravel()

    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError("Probabilities must be non-negative.")

    P, Q = P / np.linalg.norm(P, ord=1), Q / np.linalg.norm(Q, ord=1)
    M = 0.5 * (P + Q)
    return 0.5 * (stats.entropy(P, M, base=base) + stats.entropy(Q, M, base=base))


def compute_jensen_shannon_similarity(P, Q):
    """
    Compute Jensen-Shannon similarity between two probability distributions P and Q.
    This is computed as 1 minus Jensen-Shannon divergence.

    Parameters
    ----------
    P : numpy.ndarray
        The first probability distribution.
    Q : numpy.ndarray
        The second probability distribution.

    Returns
    -------
    float
        The Jensen-Shannon similarity between P and Q.
    """
    return 1 - compute_jensen_shannon_divergence(P, Q)


def compute_mean_width(X, n=1000, vectorized=False, center=False, seed=None):
    """
    Compute mean width of a matrix X by projecting it onto random vectors.

    Parameters
    ----------
    X : numpy.ndarray
        An n x m matrix with n samples and m features.
    n : int, optional
        The number of random projections to calculate width, by default 1000.
    vectorized : bool, optional
        If True, use vectorized operations for speed up, by default False.
    center : bool, optional
        If True, center the data before computing, by default False.
    seed : int, optional
        The seed for the random number generator, by default None.

    Returns
    -------
    float
        The mean width of X.
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
        max1, max2 = proj.max(0), (-proj).max(0)  # max across samples
    else:
        max1 = np.zeros(n)
        max2 = np.zeros(n)
        for idx, rproj in enumerate(rprojs.T):
            proj = X @ rproj
            max1[idx] = proj.max()
            max2[idx] = (-proj).max()
    return (max1 + max2).mean()


def compute_mean_correlation(X):
    """
    Compute mean correlation of a matrix X.

    Parameters
    ----------
    X : numpy.ndarray
        An n x m matrix with n samples and m features.

    Returns
    -------
    float
        The mean correlation of X.
    """
    X = np.asarray(X)
    cc = np.abs(np.corrcoef(X, rowvar=False))
    return (cc - np.eye(cc.shape[0])).mean()


def compute_mean_mutual_info(X, **kwargs):
    """
    Compute mean mutual information of a matrix X.

    Parameters
    ----------
    X : numpy.ndarray
        An n x m matrix with n samples and m features.

    Returns
    -------
    float
        The mean mutual information of X.
    """
    X = np.asarray(X)
    mis = []
    for idx in range(X.shape[1]):
        X_ = np.delete(X, idx, axis=1)
        y_ = X[:, idx]
        mi = mutual_info_regression(X_, y_, **kwargs)
        mis.append(mi)

    return np.concatenate(mis).mean()


def compute_volume(X):
    """
    Compute volume of a matrix X.

    Parameters
    ----------
    X : numpy.ndarray
        An n x m matrix with n samples and m features.

    Returns
    -------
    float
        The volume of X.
    """
    X = np.asarray(X)
    if (X.ndim == 1) or (X.shape[1] < 2):
        return np.max(X) - np.min(X)
    if np.allclose(X, X[0]):
        return 0.0
    hull = proj_P_for_hull(X)
    return (
        hull.volume if not isinstance(hull, np.ndarray) else np.max(hull) - np.min(hull)
    )


def compute_gamut(
    X,
    at_l1=None,
    relative_to=None,
    center_to_neutral=False,
    metric="width",
    center=True,
    seed=None,
):
    """
    Compute absolute gamut of a matrix X.

    Parameters
    ----------
    X : np.ndarray
        An n x m matrix with n samples and m features.
    at_l1 : float, optional
        [description], by default None
    relative_to : np.ndarray, optional
        Calculate relative to another array with samples, by default None
    center_to_neutral : bool, optional
        Center for barycentric dimensionality reduction, by default False
    metric : str, optional
        The metric to use when computing the gamut, either 'width' or 'volume', by default 'width'
    center : bool, optional
        If True, center the data before computing, by default True
    seed : int, optional
        The seed for the random number generator, by default None

    Returns
    -------
    float
        The gamut of X.
    """
    X = np.asarray(X)
    func = compute_mean_width if metric == "width" else compute_volume
    kwargs = {"seed": seed, "center": center} if metric == "width" else {}
    l1 = X.sum(axis=-1)

    if at_l1 is not None:
        assert at_l1 > 0, "at_l1 must be greater than 0"
        if np.all(l1 <= at_l1) or np.all(l1 > at_l1):
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
            seed=seed,
        )
        return num / denom
    return num
