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
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
import cvxpy as cp

from dreye.api.utils import l2norm


def getPfromA(A, lb, ub, K=None, baseline=None):
    pass


def in_hull(P, X, bounded=True):
    """
    Compute if `x` is in the convex hull of `points`
    or a convex combination of `points` if not bounded.
    """
    if bounded:
        try:
            # try hull algorithm before relying on convex combination algorithm
            hull = Delaunay(P)
            inhull = hull.find_simplex(X) >= 0
        except QhullError:
            _, _, inhull = convex_combination(P, X, bounded=bounded)
    else:
        _, _, inhull = convex_combination(P, X, bounded=bounded)
    return inhull


def convex_combination(P, X, bounded=True, **kwargs):
    """
    Return convex combination.

    Returns
    -------
    w : numpy.ndarray
    norm : float
    in_hull : bool
    """
    ndim = X.ndim
    X = np.atleast_2d(X)
    nsamples = X.shape[0]
    
    if bounded:
        # add one to ensure a convex combination
        A = np.vstack([P.T, np.ones((1, P.shape[0]))])
        X = np.hstack([X, np.ones((X.shape[0], 1))])
    else:
        A = P.T

    w = cp.Variable(A.shape[1])
    x = cp.Parameter(A.shape[0])

    constraints = [w >= 0]
    objective = cp.Minimize(cp.sum_squares(A @ w - x))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dcp(dpp=True)

    # perform nnls to find a solution
    W = np.zeros((nsamples, A.shape[1]))
    # TODO parallelize? or batching?
    for idx, x_ in enumerate(X):
        x.value = x_
        problem.solve(**kwargs)
        if not np.isfinite(problem.value):
            raise RuntimeError("Convex Combination did not converge.")
        W[idx] = w.value

    norms = l2norm(W @ A.T - X, axis=-1)
    # check if norm is close to zero - that is a optimal solution was found
    in_hulls = np.isclose(norms, 0)

    if ndim == 1:
        return W[0], norms[0], in_hulls[0]

    return W, norms, in_hulls