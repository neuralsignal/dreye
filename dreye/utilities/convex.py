"""
Algorithms related to
convex combinations and convex hulls
"""

import numpy as np
from scipy.optimize import nnls
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError


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

