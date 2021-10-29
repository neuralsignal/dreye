"""
Algorithms related to
convex combinations and convex hulls.

P : numpy.ndarray (npoints x ndim)
    Set of vectors that entail a convex hull
    or whose projection into a lower dimension 
    describes a convex hull
X : numpy.ndarray (npoints x ndim)
    Arbitary set of vectors
"""

import numpy as np
from scipy.optimize import nnls
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError


def in_hull(P, x, bounded=True):
    """
    Compute if `x` is in the convex hull of `points`
    or a convex combination of `points` if not bounded.
    """
    if bounded:
        # TODO check math for projection -> probably doesn't work
        # hull, ndim, svd = projP4hull(
        #     P, hull_class=Delaunay, return_ndim=True, return_transformer=True
        # )
        # if svd is not None:
        #     x = svd.transform(x[None])[0]
        # if ndim == 1:
        #     inhull = (x <= hull.max()) & (x >= hull.min())
        # else:
        #     inhull = hull.find_simplex(x) >= 0
        try:
            # hull algorithm is more efficient
            hull = Delaunay(P)
            inhull = hull.find_simplex(x) >= 0
        except QhullError:
            _, _, inhull = convex_combination(P, x, bounded=bounded)
    else:
        _, _, inhull = convex_combination(P, x, bounded=bounded)
    return inhull


def convex_combination(P, x, bounded=True):
    """
    Return convex combination.

    Returns
    -------
    w : numpy.ndarray
    norm : float
    in_hull : bool
    """
    # TODO vectorize
    if bounded:
        # add one to ensure a convex combination
        A = np.r_[P.T, np.ones((1, P.shape[0]))]
        x = np.r_[x, np.ones(1)]
    else:
        A = P.T

    # perform nnls to find a solution
    # TODO efficiency - use cvxpy for vectorization?
    w, norm = nnls(A, x)
    
    # check if norm is close to zero - that is a optimal solution was found
    return w, norm, np.isclose(norm, 0)