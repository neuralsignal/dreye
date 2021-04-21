"""
Utility functions to convert from barycentric coordinates
"""

import numpy as np
from sklearn.preprocessing import normalize


def barycentric_dim_reduction(X):
    """
    Reduce dimensionality of `X` to N-1 using barycentric to cartesian
    cooredinate transformation
    """
    X = np.abs(X)
    X = normalize(X, norm='l1', axis=1)
    return barycentric_to_cartesian(X)


def barycentric_to_cartesian(X):
    n = X.shape[1]
    A = barycentric_to_cartesian_transformer(n)
    return X @ A


def barycentric_to_cartesian_transformer(n):
    assert n > 1
    A = np.zeros((n, n-1))
    A[1, 0] = 1
    for i in range(2, n):
        A[i, :i-1] = np.mean(A[:i, :i-1], axis=0)
        dis = np.sum((A[:i, :i-1] - A[i, :i-1])**2, axis=1)
        assert np.unique(dis).size == 1
        x = np.sqrt(1 - dis.mean())
        A[i, i-1] = x
    return A
