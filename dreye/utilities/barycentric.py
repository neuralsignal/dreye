"""
Utility functions to convert from barycentric coordinates
"""

from itertools import product
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


def line_intersection_simplex(x1, x2, c, axis=-1, checks=True):
    """
    Line intersection with simplex plane with sum of `c`

    See: https://math.stackexchange.com/questions/151064/calculating-line-intersection-with-hypersphere-surface-in-mathbbrn?rq=1
    """
    if checks:
        assert np.all(c > 0)
        assert np.all(x1 >= 0)
        assert np.all(x2 >= 0)

    t = (
        c - np.sum(x1, axis=axis, keepdims=True)
    ) / (
        np.sum(x2 - x1, axis=axis, keepdims=True)
    )
    return x1 + t * (x2 - x1)


def _find_points_to_intersect_for_simplex(points, c):
    """
    For `line_intersection_simplex`

    Parameters
    ----------
    points: numpy.ndarray (n_samples x n_dim)
    """
    psum = np.sum(points, axis=-1)
    threshbool = psum <= c

    assert np.any(threshbool), "target `c` too low."
    assert not np.all(threshbool), "target `c` too high."

    p1 = points[threshbool]
    p2 = points[~threshbool]
    return product(p1, p2)


def simplex_plane_points_in_hull(points, c):
    assert np.all(points >= 0), 'all `points` must be positive'
    assert np.all(c > 0), '`c` must be positive'
    # TODO efficiency
    return np.array([
        line_intersection_simplex(x1, x2, c, checks=False)
        for x1, x2 in _find_points_to_intersect_for_simplex(points, c)
    ])
