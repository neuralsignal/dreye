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

from typing import Tuple, Optional
import warnings
from itertools import combinations, product
import numpy as np
from scipy.spatial import Delaunay

try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError
import cvxpy as cp

from dreye.api.optimize.lsq_linear import lsq_linear
from dreye.api.utils import transform_values, l2norm, predict_values


def all_combinations_of_bounds(
    lb: np.ndarray,
    ub: np.ndarray,
    ratios: np.ndarray = np.linspace(0.0, 1.0, 11),
    include_ratios: bool = False,
) -> np.ndarray:
    """
    Get all possible combinations of the lower bound `lb`
    and upper bound `ub` as a two-dimensional array.

    Parameters
    ----------
    lb : numpy.ndarray
        Lower bound array.
    ub : numpy.ndarray
        Upper bound array.
    ratios : numpy.ndarray, optional
        Array of ratios for combination, by default np.linspace(0., 1., 11).
    include_ratios : bool, optional
        If True, include ratios in the combinations, by default False.

    Returns
    -------
    numpy.ndarray
        Array of all possible combinations of the lower bound and upper bound.
    """
    # all possible combinations of the lower and upper bound
    samples = np.array(list(product([0.0, 1.0], repeat=len(lb))))
    # rescaling to lower and upper bounds
    samples = samples * (ub - lb) + lb

    if include_ratios:
        samples_ = []
        for (idx, isample), (jdx, jsample) in product(
            enumerate(samples), enumerate(samples)
        ):
            if idx >= jdx:
                continue
            s_ = (ratios[:, None] * isample) + ((1 - ratios[:, None]) * jsample)
            samples_.append(s_)
        samples_ = np.vstack(samples_)
        samples = np.vstack([samples, samples_])
        # remove non-unique samples
        samples = np.unique(samples, axis=0)
    return samples


def get_P_from_A(
    A: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    K=None,
    baseline=None,
    bounded: bool = True,
) -> np.ndarray:
    """
    Transforms the values and gets all possible combinations to describe the convex hull.

    Parameters
    ----------
    A : numpy.ndarray
        Array to be transformed.
    lb : numpy.ndarray
        Lower bound array.
    ub : numpy.ndarray
        Upper bound array.
    K : unknown type, optional
        Unknown variable, by default None.
    baseline : unknown type, optional
        Unknown variable, by default None.
    bounded : bool, optional
        If True, the function assumes that upper bound array is finite, by default True.

    Returns
    -------
    numpy.ndarray
        Array of all possible combinations for to describe the convex hull (before transformation).

    Raises
    ------
    ValueError
        If `bounded` is True and `ub` has infinite values.
    AssertionError
        If `lb` is not finite.
    """
    A, lb, ub, baseline = transform_values(A, lb, ub, K, baseline)
    # lb has to be finite
    assert np.all(
        np.isfinite(lb)
    ), "The lower bound has to be finite to obtain vectors for convex combination or hull."
    # upper bound can be infinite but bounded must be False
    if not np.all(np.isfinite(ub)):
        if bounded:
            raise ValueError(
                "Upper bound has infinite values and `bounded` was set to True."
            )
        # reset upper bound to one plus lower bound (usually zero), if not bounded
        ub = np.ones(ub.size) + lb
    # get all possible combinations for to describe the convex hull (before transformation)
    X = all_combinations_of_bounds(lb, ub)
    return predict_values(X, A, baseline)


def in_hull(
    P: np.ndarray, B: np.ndarray, bounded: bool = True, qhull_options: str = None
) -> bool:
    """
    Check if points `B` are in the convex hull of `P`.

    Parameters
    ----------
    P : np.ndarray
        Array of points defining the convex hull.
    B : np.ndarray
        Array of points to be checked if they are inside the convex hull.
    bounded : bool, optional
        If True (default), the solution is a convex combination. If False, the solution is an affine combination.
    qhull_options : str, optional
        Additional options to pass to Qhull via Delaunay.

    Returns
    -------
    bool
        True if `B` is inside the convex hull of `P`, False otherwise.
    """
    if bounded:
        try:
            # try hull algorithm before relying on convex combination algorithm
            if not isinstance(P, Delaunay):
                P = Delaunay(P, qhull_options=qhull_options)
            inhull = P.find_simplex(B) >= 0
        except QhullError:
            _, _, inhull = convex_combination(P, B, bounded=bounded)
    else:
        _, _, inhull = convex_combination(P, B, bounded=bounded)

    return inhull


def convex_combination(
    P: np.ndarray, B: np.ndarray, bounded: bool = True, **kwargs
) -> Tuple[np.ndarray, float, bool]:
    """
    Determine if `B` is a convex combination of `P`.

    Parameters
    ----------
    P : np.ndarray
        Array of points potentially forming the convex combination.
    B : np.ndarray
        Array of points to be expressed as a convex combination of `P`.
    bounded : bool, optional
        If True (default), the solution is a convex combination. If False, the solution is an affine combination.

    Returns
    -------
    w : np.ndarray
        Weight vector used to form the convex combination.
    norm : float
        The L2 norm of the residual of the convex combination.
    in_hull : bool
        True if `B` is a convex combination of `P`, False otherwise.

    Raises
    ------
    RuntimeError
        Raised if the optimization problem does not converge.
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


def in_hull_from_A(
    B: np.ndarray,
    A: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    K: Optional[np.ndarray] = None,
    baseline: Optional[np.ndarray] = None,
) -> bool:
    """
    Determine if a given point `B` lies within the convex hull formed by a set of points derived from `A`.

    Parameters
    ----------
    B : np.ndarray
        The point to check.
    A : np.ndarray
        Array used to generate the set of points forming the convex hull.
        Should be a 2-D array where each row is a point in the hull.
    lb : np.ndarray
        Lower bound for the convex combination coefficients. Should be a 1-D array.
    ub : np.ndarray
        Upper bound for the convex combination coefficients. Should be a 1-D array.
    K : np.ndarray, optional
        Transformation matrix, by default None.
    baseline : np.ndarray, optional
        The baseline point to be subtracted from all points in the convex hull, by default None.

    Returns
    -------
    bool
        True if `B` is in the convex hull, False otherwise.
    """
    B = np.asarray(B)
    # handling K
    A, lb, ub, baseline = transform_values(A, lb, ub, K, baseline)

    # check if bounded
    bounded = np.all(np.isfinite(ub))
    # get overcomplete convex combinations
    P = get_P_from_A(A, lb, ub, baseline=baseline, bounded=bounded)

    # offset-subtracted P and B (contains baseline and lb)
    # to test for in_hull
    # -> zero-minimum
    offset = np.min(P, axis=0)
    P_ = P - offset
    B_ = B - offset

    return in_hull(P_, B_, bounded=bounded)


def range_of_solutions(
    B: np.ndarray,
    A: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    K: Optional[np.ndarray] = None,
    baseline: Optional[np.ndarray] = None,
    error: str = "raise",
    n: Optional[int] = None,
    eps: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Calculate the range of solutions for a bound-constrained system of linear equations.

    Parameters
    ----------
    B : np.ndarray
        The target values to approximate. Can be a 1-D or 2-D array.
    A : np.ndarray
        The matrix of coefficients in the system of equations.
        Should be a 2-D array where each row is an equation and each column is a variable.
    lb : np.ndarray
        The lower bounds for the solutions. Should be a 1-D array of the same length as the number of columns in `A`.
    ub : np.ndarray
        The upper bounds for the solutions. Should be a 1-D array of the same length as the number of columns in `A`.
    K : np.ndarray, optional
        Transformation matrix for `A`, `lb`, `ub`, and `baseline`, by default None.
    baseline : np.ndarray, optional
        The baseline values for the solutions. Should be a 1-D array of the same length as the number of columns in `A`, by default None.
    error : str, optional
        The error handling mode. If 'raise', a ValueError is raised if a target value is outside the gamut.
        If 'warn', a warning is issued and the best possible solution is chosen. If 'ignore', the error is ignored, by default 'raise'.
    n : int, optional
        The number of equally spaced solutions to generate between the minimum and maximum solutions, by default None.
    eps : float, optional
        A small value added to the minimum and subtracted from the maximum to improve numerical stability, by default 1e-7.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
        The minimum and maximum solutions. If `n` is not None, also returns an array of `n` equally spaced solutions.

    Raises
    ------
    ValueError
        If the system of equations is not underdetermined, or if any target values are outside the gamut and `error` is 'raise'.
    """
    # handling dimensionality of X
    B = np.asarray(B)
    ndim = B.ndim
    B = np.atleast_2d(B)

    # handling K
    A, lb, ub, baseline = transform_values(A, lb, ub, K, baseline)

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
        if error not in {"ignore", "warn"}:
            raise ValueError(
                "Some targets are outside the convex "
                "combination or hull (i.e. gamut of the system)."
            )
        elif error == "warn":
            warnings.warn(
                "Some targets are outside the convex "
                "combination or hull (i.e. gamut of the system). "
                "Choosing best possible solution.",
                RuntimeWarning,
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
    Xmins: np.ndarray,
    Xmaxs: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    n: int = 20,
    eps: float = 1e-7,
) -> np.ndarray:
    """Generate equally spaced intensity solutions between minimum and maximum solutions.

    Parameters
    ----------
    Xmins : np.ndarray
        The minimum solutions.
    Xmaxs : np.ndarray
        The maximum solutions.
    A : np.ndarray
        The matrix of coefficients in the system of equations.
    b : np.ndarray
        The target value to approximate.
    n : int, optional
        The number of solutions to generate, by default 20.
    eps : float, optional
        A small value added to the minimum and subtracted from the maximum to improve numerical stability, by default 1e-7.

    Returns
    -------
    np.ndarray
        An array of equally spaced solutions between `Xmins` and `Xmaxs`.
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

            for iX in np.linspace(Xmins[idx] + eps_[idx], Xmaxs[idx] - eps_[idx], n):
                Astar = A[:, not_idx]
                offset = A[:, idx] * iX
                bstar = b - offset

                Xminstar, Xmaxstar = _range_of_solutions(
                    Astar, bstar, lb=Xmins[not_idx], ub=Xmaxs[not_idx]
                )
                if (Xminstar > Xmaxstar).any():
                    continue

                Xsstar = _spaced_solutions(
                    Xminstar, Xmaxstar, Astar, bstar, n=n, eps=eps
                )

                X_ = np.hstack([np.ones((Xsstar.shape[0], 1)) * iX, Xsstar])[:, argsort]

                Xs.append(X_)

        return np.vstack(Xs)

    else:
        # create equally space solutions
        Xs = np.zeros((n, A.shape[1]))

        # for numerical stability
        eps_ = (Xmaxs[0] - Xmins[0]) * eps

        idx = 0

        for iX in np.linspace(Xmins[0] + eps_, Xmaxs[0] - eps_, n):
            Astar = A[:, 1:]
            offset = A[:, 0] * iX
            sols = np.linalg.solve(Astar, (b - offset))
            Xs[idx] = np.concatenate([[iX], sols])
            idx += 1

        return Xs


def _range_of_solutions(
    A: np.ndarray, b: np.ndarray, lb: np.ndarray, ub: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the range of solutions for a bound-constrained system of linear equations.

    Parameters
    ----------
    A : np.ndarray
        The matrix of coefficients in the system of equations.
        Should be a 2-D array where each row is an equation and each column is a variable.
    b : np.ndarray
        The target value to approximate. Should be a 1-D array.
    lb : np.ndarray
        The lower bounds for the solutions. Should be a 1-D array of the same length as the number of columns in `A`.
    ub : np.ndarray
        The upper bounds for the solutions. Should be a 1-D array of the same length as the number of columns in `A`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The minimum and maximum solutions.

    Notes
    -----
    This is a helper function for `range_of_solutions`. It assumes that `b` is a single target value (1-D array),
    and that any necessary error checking and transformation of `A`, `lb`, `ub`, and `b` has already been done.
    """
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
        offsets = omat * (ub[ridcs] - lb[ridcs]) + lb[ridcs]

        # filter non-real values
        offsets[np.isnan(offsets)] = 0.0
        offsets = offsets[np.isfinite(offsets).all(axis=1)]

        # combinations x channels
        offset = offsets @ Arest.T

        # remove idcs not considered during linear regression
        Astar = np.delete(A, ridcs, axis=1)

        sols = np.linalg.solve(Astar, (b - offset).T).T
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
