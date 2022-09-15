"""
Projection methods.

P : numpy.ndarray (npoints x ndim)
    Set of vectors that entail a convex hull
    or whose projection into a lower dimension 
    describes a convex hull
B : numpy.ndarray (npoints x ndim)
    Arbitary set of vectors
"""

from scipy.spatial import ConvexHull, QhullError
from itertools import product
import numpy as np
from sklearn.decomposition import PCA

from quadprog import solve_qp


def proj_P_for_hull(P, hull_class=ConvexHull, return_ndim=False, return_hull=True, return_transformer=False):
    """
    Project `P` until it defines a convex hull.
    """
    try:
        if P.shape[-1] == 1:
            hull = P
            ndim = 1
        else:
            hull = hull_class(P)
            ndim = P.shape[-1]
        svd = None
    except QhullError:
        # reduce dimensionality
        svd = PCA(P.shape[1]-1)
        P = svd.fit_transform(P)
        evar = np.cumsum(svd.explained_variance_ratio_)
        ndim = np.min(np.flatnonzero(np.isclose(evar, 1))) + 1
        if ndim == 1:
            hull = P[:, :ndim]
        else:
            hull = hull_class(P[:, :ndim])

    if return_hull and not return_ndim and not return_transformer:
        return hull
    elif return_hull and return_ndim and return_transformer:
        return hull, ndim, svd
    elif return_hull and return_ndim:
        return hull, ndim
    elif return_hull and return_transformer:
        return hull, svd
    elif return_transformer:
        return svd
    else:
        return ndim


def line_to_simplex(x1, x2, c, axis=-1, checks=True):
    """
    Return the point where the line defined by vectors `x1`
    and `x2` intersects with the simplex with sum of `c`
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


def yieldPpairs4proj2simplex(P, c):
    """
    Yield the combination of points in `P` 
    that are between the simplex with sum of `c` 
    and whose intersection with the simplex 
    would result in the projection of the
    convex hull described by `P` onto the simplex.

    Parameters
    ----------
    P : numpy.ndarray (n_samples x n_dim)
    c : float
    """
    psum = np.sum(P, axis=-1)
    threshbool = psum <= c

    assert np.any(threshbool), "target `c` too low."
    assert not np.all(threshbool), "target `c` too high."

    if P.shape[0] > P.shape[1]:
        hull, ndim = proj_P_for_hull(P, return_ndim=True)

        # points lie in one dimension
        if ndim == 1:  # or if hull is None
            p1 = P[threshbool][:1]
            p2 = P[~threshbool][-1:]
            return product(p1, p2)

        idcs1 = np.flatnonzero(threshbool)  # smaller
        idcs2 = np.flatnonzero(~threshbool)  # bigger
        edges = []
        for e in hull.simplices:
            for idx, jdx in product(e, e):
                if idx == jdx:
                    continue
                if ((idx, jdx) in edges) or ((jdx, idx) in edges):
                    continue
                # can't be both below threshold
                if idx in idcs1 and jdx in idcs1:
                    continue
                # can't be both above threshold
                if idx in idcs2 and jdx in idcs2:
                    continue
                # idx must be the lower point
                if psum[idx] > psum[jdx]:
                    continue

                edges.append((idx, jdx))
                yield P[idx], P[jdx]
    else:
        p1 = P[threshbool]
        p2 = P[~threshbool]
        return product(p1, p2)


def proj_P_to_simplex(P, c):
    """
    Project the points that project the convex hull entailed
    by `P` onto the simplex with sum of `c`.
    Each returned point should sum up to `c`.
    """
    assert np.all(P >= 0), 'all `points` must be positive'
    assert np.all(c > 0), '`c` must be positive'
    # TODO efficiency
    return np.array([
        line_to_simplex(x1, x2, c, checks=False)
        for x1, x2 in yieldPpairs4proj2simplex(P, c)
    ])


def proj_B_to_hull(B, equations):
    """
    Project `B` to the convex hull defined by the
    hyperplane equations of the facets

    Parameters
    ----------
    B : numpy.ndarray (..., ndim)
    equations : numpy.ndarray (nfacets, ndim + 1)

    Returns
    -------
    X : numpy.ndarray (..., ndim)
    """
    Q = np.eye(B.shape[-1])
    # TODO replace with cvxpy or more general qpsolvers (for solutions)? - However, quadprog is probably faster?
    # TODO TEST
    # TODO efficiency - vectorization
    def helper(a):
        return solve_qp(
            Q, a, 
            -equations[:, :-1].T, equations[:, -1], 
            meq=0, factorized=True
        )[0]
    return np.apply_along_axis(helper, -1, B)


def alpha_for_B_with_P(B, equations):
    """
    Get multiple of vector that intersects with hull
    """
    # need to transpose
    equations = equations.T
    # transform and bias
    V, b = equations[:-1], equations[-1]
    alpha = -b / (B @ V)
    # mask smaller equal to zero values
    alpha[alpha <= 0] = np.nan
    alpha = np.nanmin(alpha, axis=-1)
    return alpha


def B_with_P(B, equations):
    """
    Find hull intersection of vectors `B`.
    """
    return alpha_for_B_with_P(B, equations)[..., None] * B

