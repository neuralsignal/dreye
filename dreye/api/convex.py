"""
Algorithms related to
convex combinations and convex hulls.

P : numpy.ndarray (npoints x ndim)
    Set of vectors that entail a convex hull
    or whose projection into a lower dimension 
    describes a convex hull
B : numpy.ndarray (npoints x ndim)
    Arbitary set of vectors
"""

import warnings
from itertools import combinations, product
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
import cvxpy as cp

from dreye.api.optimize.lsq_linear import lsq_linear
from dreye.api.utils import check_and_return_transformed_values, l2norm, get_prediction


def all_combinations_of_bounds(
    lb, ub,
    ratios=np.linspace(0., 1., 11), 
    include_ratios=False
):
    """
    Get all possible combinations of the lower bound `lb` 
    and `ub` as a two-dimensional array.
    """    
    # all possible combinations of the lower and upper bound
    samples = np.array(list(product([0., 1.], repeat=len(lb))))
    # rescaling to lower and upper bounds
    samples = samples * (ub - lb) + lb
    
    if include_ratios:
        samples_ = []
        for (idx, isample), (jdx, jsample) in product(enumerate(samples), enumerate(samples)):
            if idx >= jdx:
                continue
            s_ = (ratios[:, None] * isample) + ((1-ratios[:, None]) * jsample)
            samples_.append(s_)
        samples_ = np.vstack(samples_)
        samples = np.vstack([samples, samples_])
        # remove non-unique samples
        samples = np.unique(samples, axis=0)
    return samples


def get_P_from_A(A, lb, ub, K=None, baseline=None, bounded=True):
    """[summary]

    Parameters
    ----------
    A : [type]
        [description]
    lb : [type]
        [description]
    ub : [type]
        [description]
    K : [type], optional
        [description], by default None
    baseline : [type], optional
        [description], by default None
    bounded : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    A, lb, ub, baseline = check_and_return_transformed_values(A, lb, ub, K, baseline)
    # lb has to be finite
    assert np.all(np.isfinite(lb)), "The lower bound has to be finite to obtain vectors for convex combination or hull."
    # upper bound can be infinite but bounded must be False
    if not np.all(np.isfinite(ub)):
        if bounded:
            raise ValueError("Upper bound has infinite values and `bounded` was set to True.")
        # reset upper bound to one plus lower bound (usually zero), if not bounded
        ub = np.ones(ub.size) + lb
    # get all possible combinations for to describe the convex hull (before transformation)
    X = all_combinations_of_bounds(lb, ub)
    return get_prediction(X, A, baseline)


def in_hull(P, B, bounded=True, qhull_options=None):
    """
    Compute if `B` is in the convex hull of `points`
    or a convex combination of `points` if not bounded.

    Parameters
    ----------
    P : [type]
        [description]
    B : [type]
        [description]
    bounded : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """
    if bounded:
        try:
            # try hull algorithm before relying on convex combination algorithm
            hull = Delaunay(P, qhull_options=qhull_options)
            inhull = hull.find_simplex(B) >= 0
        except QhullError:
            _, _, inhull = convex_combination(P, B, bounded=bounded)
    else:
        _, _, inhull = convex_combination(P, B, bounded=bounded)
    return inhull


def convex_combination(P, B, bounded=True, **kwargs):
    """Return if `B` is a convex combination of `P`.

    Parameters
    ----------
    P : [type]
        [description]
    B : [type]
        [description]
    bounded : bool, optional
        [description], by default True

    Returns
    -------
    w : numpy.ndarray
    norm : float
    in_hull : bool

    Raises
    ------
    RuntimeError
        [description]
    """
    ndim = B.ndim
    B = np.atleast_2d(B)
    nsamples = B.shape[0]
    
    if bounded:
        # add one to ensure a convex combination
        A = np.vstack([P.T, np.ones((1, P.shape[0]))])
        B = np.hstack([B, np.ones((B.shape[0], 1))])
    else:
        A = P.T

    x = cp.Variable(A.shape[1])
    b = cp.Parameter(A.shape[0])

    constraints = [x >= 0]
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dcp(dpp=True)

    # perform nnls to find a solution
    X = np.zeros((nsamples, A.shape[1]))
    # TODO parallelize? or batching?
    for idx, b_ in enumerate(B):
        b.value = b_
        problem.solve(**kwargs)
        if not np.isfinite(problem.value):
            raise RuntimeError("Convex Combination did not converge.")
        X[idx] = x.value

    norms = l2norm(X @ A.T - B, axis=-1)
    # check if norm is close to zero - that is a optimal solution was found
    in_hulls = np.isclose(norms, 0)

    if ndim == 1:
        return X[0], norms[0], in_hulls[0]

    return X, norms, in_hulls


def in_hull_from_A(B, A, lb, ub, K=None, baseline=None):
    """[summary]

    Parameters
    ----------
    B : [type]
        [description]
    A : [type]
        [description]
    lb : [type]
        [description]
    ub : [type]
        [description]
    K : [type], optional
        [description], by default None
    baseline : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    B = np.asarray(B)
    # handling K
    A, lb, ub, baseline = check_and_return_transformed_values(
        A, lb, ub, K, baseline
    )
    
    # check if bounded
    bounded = np.all(np.isfinite(ub))
    # get overcomplete convex combinations
    P = get_P_from_A(A, lb, ub, baseline=baseline, bounded=bounded)
    
    # offset-subtraced P and B (contains baseline and lb)
    # to test for inhull
    # -> zero-minimum
    offset = np.min(P, axis=0)
    P_ = P - offset
    B_ = B - offset
    
    return in_hull(P_, B_, bounded=bounded)


def range_of_solutions(
    B, A, lb, ub, K=None, baseline=None, error='raise', 
    n=None, eps=1e-7
):
    """Range of solutions for a bound-constrained system of linear equations.

    Parameters
    ----------
    B : [type]
        [description]
    A : [type]
        [description]
    lb : [type]
        [description]
    ub : [type]
        [description]
    K : [type], optional
        [description], by default None
    baseline : [type], optional
        [description], by default None
    error : str, optional
        [description], by default 'raise'
    n : [type], optional
        [description], by default None
    eps : [type], optional
        [description], by default 1e-7

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    ValueError
        [description]
    """
    # handling dimensionality of X
    B = np.asarray(B)
    ndim = B.ndim
    B = np.atleast_2d(B)
    
    # handling K
    A, lb, ub, baseline = check_and_return_transformed_values(
        A, lb, ub, K, baseline
    )
    
    if A.shape[1] <= A.shape[0]:
        raise ValueError("System of equation is not underdetermined.")
    
    # check if bounded
    bounded = np.all(np.isfinite(ub))
    # get overcomplete convex combinations
    P = get_P_from_A(A, lb, ub, baseline=baseline, bounded=bounded)
    
    # offset-subtraced P and B (contains baseline and lb)
    # to test for inhull
    # -> zero-minimum
    offset = np.min(P, axis=0)
    P_ = P - offset
    B_ = B - offset
    
    inhull = in_hull(P_, B_, bounded=bounded)
    
    # X must have baseline removed for helper function
    B = B - baseline
    
    if np.any(~inhull):
        if error not in {'ignore', 'warn'}:
            raise ValueError(
                f"Some targets are outside the convex "
                "combination or hull (i.e. gamut of the system)."
            )  
        elif error == 'warn':
            warnings.warn(
                "Some targets are outside the convex "
                "combination or hull (i.e. gamut of the system). "
                "Choosing best possible solution.", 
                RuntimeWarning
            )    
        X = lsq_linear(A, B, lb=lb, ub=ub)

    Xmins = np.zeros((B.shape[0], A.shape[1]))
    Xmaxs = np.zeros((B.shape[0], A.shape[1]))
    Xs = np.zeros(B.shape[0], dtype=object)
    # loop over all samples in B
    for idx, (b, inside) in enumerate(zip(B, inhull)):        
        
        if not inside:
            Xmins[idx], Xmaxs[idx] = X[idx], X[idx]
            
            if n is not None:
                Xs[idx] = X[idx][None]
        
        else:
            Xmins[idx], Xmaxs[idx] = _range_of_solutions(A, b, lb, ub)
            
            if n is not None:
                Xs[idx] = _spaced_solutions(Xmins[idx], Xmaxs[idx], A, b, n=n, eps=eps)
        
    if ndim == 1:
        if n is None:
            return Xmins[0], Xmaxs[0]
        else:
            return Xmins[0], Xmaxs[0], Xs[0]
    
    if n is None:
        return Xmins, Xmaxs
    else:
        return Xmins, Xmaxs, Xs


def _spaced_solutions(
    Xmins, Xmaxs,
    A, b, n=20, 
    eps=1e-7
):
    """
    Spaced intensity solutions from `mins` to `maxs`.
    If the difference between the number
    of light sources and the number of opsins exceeds 1 
    than the number of samples cannot be predetermined 
    but will be greate than `n` ** `n_diff`.
    """
    n_diff = A.shape[1] - A.shape[0]

    if n_diff > 1:
        # go through first led -> remove -> find new range of solution
        # restrict by wmin and wmax
        # repeat until only one extra LED
        # then get equally spaced solutions
        Xs = []

        # for numerical stability
        eps_ = (Xmaxs - Xmins) * eps
        idcs = np.arange(A.shape[1])

        for idx in idcs:
            # for indexing
            not_idx = ~(idcs == idx)
            argsort = np.concatenate([[idx], idcs[not_idx]])
            
            for iX in np.linspace(Xmins[idx]+eps_[idx], Xmaxs[idx]-eps_[idx], n):
                Astar = A[:, not_idx]
                offset = A[:, idx] * iX
                bstar = b - offset
                
                Xminstar, Xmaxstar = _range_of_solutions(
                    Astar, bstar,
                    lb=Xmins[not_idx], 
                    ub=Xmaxs[not_idx]
                )
                if (Xminstar > Xmaxstar).any():
                    continue

                Xsstar = _spaced_solutions(
                    Xminstar, Xmaxstar, Astar, bstar, n=n, 
                    eps=eps
                )
                
                X_ = np.hstack([
                    np.ones((Xsstar.shape[0], 1)) * iX, 
                    Xsstar
                ])[:, argsort]
                
                Xs.append(X_)

        return np.vstack(Xs)

    else:
        # create equally space solutions
        Xs = np.zeros((n, A.shape[1]))
        
        # for numerical stability
        eps_ = (Xmaxs[0] - Xmins[0]) * eps
        
        idx = 0
        
        for iX in np.linspace(Xmins[0]+eps_, Xmaxs[0]-eps_, n):
            Astar = A[:, 1:]
            offset = A[:, 0] * iX
            sols = np.linalg.solve(
                Astar, 
                (b - offset)
            )
            Xs[idx] = np.concatenate([[iX], sols])
            idx += 1
            
        return Xs

    
def _range_of_solutions(A, b, lb, ub):
    """Helper function for range_of_solutions

    Parameters
    ----------
    A : [type]
        [description]
    b : [type]
        [description]
    lb : [type]
        [description]
    ub : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # TODO work on efficiency
    # x must have baseline removed already
    maxs = lb.copy()
    mins = ub.copy()
    # difference between number of opsins and leds
    n_diff = A.shape[1] - A.shape[0]
    # combinations x n_diff
    omat = np.array(list(product([0, 1], repeat=n_diff)))
    # range of indices for inputs
    idcs = np.arange(A.shape[1])  
    
    # loop over all possible input combinations by removing excessive inputs
    for ridcs in combinations(idcs, n_diff):
        # A
        ridcs = list(ridcs)
        Arest = A[:, ridcs]
        
        # combinations x n_diff
        # all possible combinations of the removed inputs being on or off
        offsets = omat * (
            ub[ridcs] - lb[ridcs]
        ) + lb[ridcs]
        
        # filter non-real values
        offsets[np.isnan(offsets)] = 0.0
        offsets = offsets[np.isfinite(offsets).all(axis=1)]
        
        # combinations x channels
        offset = (offsets @ Arest.T)
        
        # remove idcs not considered during linear regression
        Astar = np.delete(A, ridcs, axis=1)
        
        sols = np.linalg.solve(
            Astar, 
            (b - offset).T
        ).T
        # combinations x used inputs

        # allowed bounds for included 
        b0 = np.delete(lb, ridcs)
        b1 = np.delete(ub, ridcs)
        # all within bounds
        psols = np.all((sols >= b0) & (sols <= b1), axis=1)
        
        # continue if no solutions exist
        if not np.any(psols):
            continue

        # add minimum and maximum to global minimum and maximum
        rbool = np.isin(idcs, ridcs)
        
        # create minimum and maximum input values for this loop
        _mins = np.zeros(A.shape[1])
        _maxs = np.zeros(A.shape[1])
        _mins[rbool] = offsets[psols].min(axis=0)
        _maxs[rbool] = offsets[psols].max(axis=0)
        _mins[~rbool] = sols[psols].min(axis=0)
        _maxs[~rbool] = sols[psols].max(axis=0)

        # reset to new minimum and maximum
        mins = np.minimum(mins, _mins)
        maxs = np.maximum(maxs, _maxs)
        
    return mins, maxs