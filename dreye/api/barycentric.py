"""
Handling barycentric coordinates
"""

import numpy as np
from sklearn.preprocessing import normalize


def barycentric_dim_reduction(X, center=False):
    """
    Reduce dimensionality of `X` to N-1 by l1-normalizing each sample 
    and using a barycentric to cartesian coordinate transformation

    Parameters
    ----------
    X : [type]
        [description]
    center : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    X = normalize(X, norm='l1', axis=1)
    return barycentric_to_cartesian(X, center=center)


def barycentric_to_cartesian(X, center=False):
    """
    Convert from barycentric coordinates to cartesian coordinates

    Parameters
    ----------
    X : [type]
        [description]
    center : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    n = X.shape[1]
    A = barycentric_to_cartesian_transformer(n)
    if center:
        return X @ A - (np.ones(n)/n) @ A
    else:
        return X @ A


def cartesian_to_barycentric(X, I=None, centered=False):
    """Convert from cartesian to barycentric coordinates.

    Parameters
    ----------
    X : [type]
        [description]
    I : [type], optional
        [description], by default None
    centered : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    n = X.shape[1] + 1
    A = barycentric_to_cartesian_transformer(n)
    if centered:
        X = X + (np.ones(n)/n) @ A
    
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    A = np.hstack([A, np.ones((A.shape[0], 1))])
    X = X @ np.linalg.inv(A)
    
    if I is None:
        return X
    else:
        I = np.atleast_1d(I)
        return X * I[..., None]


def barycentric_to_cartesian_transformer(n):
    """Get linear transformation from barycentric to cartesian coordinates.

    Parameters
    ----------
    n : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    assert n > 1
    A = np.zeros((n, n-1))
    A[1, 0] = 1
    for i in range(2, n):
        A[i, :i-1] = np.mean(A[:i, :i-1], axis=0)
        dis = np.sum((A[:i, :i-1] - A[i, :i-1])**2, axis=1)
        # sanity check
        assert np.isclose(dis, dis[0]).all(), "non-unique rows"
        x = np.sqrt(1 - dis.mean())
        A[i, i-1] = x
    return A