"""
Algorithms related to
convex combinations and convex hulls
"""

import numpy as np
from scipy.optimize import nnls
from scipy.spatial import Delaunay
from scipy.spatial.qhull import ConvexHull, QhullError
from quadprog import solve_qp


def in_hull(points, x, bounded=True):
    """
    Compute if `x` is in the convex hull of `points` - i.e. a convex combination.
    """
    if bounded:
        try:
            # hull algorithm is more efficient
            if not isinstance(points, Delaunay):
                points = Delaunay(points)
            inhull = points.find_simplex(x) >= 0
        except QhullError:
            _, _, inhull = convex_combination(points, x, bounded=bounded)
    else:
        _, _, inhull = convex_combination(points, x, bounded=bounded)
    return inhull


def convex_combination(points, x, bounded=True):
    """
    Return convex combination.

    Returns
    -------
    w : numpy.ndarray
    norm : float
    in_hull : bool
    """
    if bounded:
        # add one to ensure a convex combination
        A = np.r_[points.T, np.ones((1, points.shape[0]))]
        x = np.r_[x, np.ones(1)]
    else:
        A = points.T

    # perform nnls to find a solution
    w, norm = nnls(A, x)
    
    # check if norm is close to zero - that is a optimal solution was found
    return w, norm, np.isclose(norm, 0)


def proj2hull(z, equations):
    """
    Project `z` to the convex hull defined by the
    hyperplane equations of the facets

    Parameters
    ----------

    z : numpy.ndarray (..., ndim)
    equations : numpy.ndarray (nfacets, ndim + 1)

    Returns
    -------
    x: numpy.ndarray (..., ndim)
    """
    Q = np.eye(z.shape[-1])
    def helper(a):
        return solve_qp(
            Q, a, 
            -equations[:, :-1].T, equations[:, -1], 
            meq=0, factorized=True
        )[0]
    return np.apply_along_axis(helper, -1, z)


def hull_intersection_alpha(U, hull):
    """
    Get multiple of vector that intersects with hull

    Adapted from:  https://stackoverflow.com/questions/30486312/intersection-of-nd-line-with-convex-hull-in-python
    """
    if not isinstance(hull, ConvexHull):
        hull = ConvexHull(hull)
    eq = hull.equations.T
    V, b = eq[:-1], eq[-1]
    alpha = -b / (U @ V)
    # mask smaller equal to zero values
    alpha[alpha <= 0] = np.nan
    alpha = np.nanmin(alpha, axis=-1)
    return alpha


def find_hull_intersection(U, hull):
    """
    Find hull intersection of vector U.
    """
    return hull_intersection_alpha(U, hull)[..., None] * U

