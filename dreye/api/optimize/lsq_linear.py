"""
The scipy lsq_linear algorithm implemented using jax.numpy
"""

from numbers import Number
import warnings
from scipy import optimize
import numpy as np
from numpy.random import default_rng
import cvxpy as cp
from sklearn.decomposition import NMF

from dreye.api.optimize.parallel import batched_iteration, diagonal_stack, concat
from dreye.api.optimize.utils import FAILURE_MESSAGE, error_propagation, get_batch_size, prepare_parameters_for_linear
from dreye.constants.common import EPS_NP64
# TODO Huber loss instead of just sum squares? -> outliers less penalized


def lsq_linear(
    A, B, 
    lb=None, ub=None, W=None,
    K=None, baseline=None, 
    error='raise', 
    n_jobs=None, 
    batch_size=1,
    verbose=0, 
    return_pred=False, 
    **opt_kwargs
):
    """
    Solve a linear least-squares problem with bounds on the variables.

    Currently uses `scipy.optimize.lsq_linear` function.

    A (channels x inputs)
    B (samples x channels)
    K (channels) or (channels x channels)
    baseline (channels)
    ub (inputs)
    lb (inputs)
    w (channels)
    """
    A, B, lb, ub, W, baseline = prepare_parameters_for_linear(A, B, lb, ub, W, K, baseline)
    B = B - baseline
    batch_size = get_batch_size(batch_size, B.shape[0])

    if n_jobs is not None:
        raise NotImplementedError("parallel jobs")

    X = np.zeros((B.shape[0], A.shape[-1]))
    count_failure = 0
    for idx, (b, w), (A_, lb_, ub_) in batched_iteration(B.shape[0], (B, W), (A, lb, ub), batch_size=batch_size):
        # TODO parallelizing
        # TODO test using sparse matrices when batching
        # TODO substitute with faster algorithm
        result = optimize.lsq_linear(
            A_ * w[:, None], b * w, bounds=(lb_, ub_), 
            **opt_kwargs
        )
        X[idx * batch_size : (idx+1) * batch_size] = result.x.reshape(-1, A.shape[-1])
        count_failure += int(result.status <= 0)

    if count_failure:
        if error == "ignore":
            pass
        elif error == "warn":
            warnings.warn(FAILURE_MESSAGE.format(count=count_failure), RuntimeWarning)
        else:
            raise RuntimeError(FAILURE_MESSAGE.format(count=count_failure)) 

    if return_pred:
        return X, (X @ A.T + baseline)
    return X


def lsq_linear_cp(
    A, B, 
    lb=None, ub=None, W=None,
    K=None, baseline=None, 
    n_jobs=None, 
    batch_size=1,
    verbose=0, 
    return_pred=False,
    l2_eps=None,
    underdetermined_opt='l2',
    **opt_kwargs
):
    """
    Solve a linear least-squares problem with bounds on the variables.

    This method uses `cvxpy` to solve the problem

    A (channels x inputs)
    B (samples x channels)
    K (channels) or (channels x channels)
    baseline (channels)
    ub (inputs)
    lb (inputs)
    w (channels)
    """
    A, B, lb, ub, W, baseline = prepare_parameters_for_linear(A, B, lb, ub, W, K, baseline)
    B = B - baseline
    batch_size = get_batch_size(batch_size, B.shape[0])

    if n_jobs is not None:
        raise NotImplementedError("parallel jobs")
    
    # set up cvxpy problem
    # constants
    A_ = diagonal_stack(A, batch_size)
    lb_ = concat(lb, batch_size)
    ub_ = concat(ub, batch_size)

    # parameters
    w_ = cp.Parameter((batch_size * W.shape[1]), pos=True)
    b_ = cp.Parameter((batch_size * B.shape[1]), )
    
    # variable and constraints
    if np.all(lb >= 0):
        xkwargs = {'pos': True}
    else:
        xkwargs = {}
    x_ = cp.Variable(A_.shape[1], **xkwargs)
    constraints = []
    if np.all(np.isfinite(lb)):
        constraints.append(x_ >= lb_)
    if np.all(np.isfinite(ub)):
        constraints.append(x_ <= ub_)
    
    # objective function
    if l2_eps is None:
        objective = cp.Minimize(
            cp.sum_squares((cp.multiply(A_, w_[:, None]) @ x_ - b_))
        )
    else:
        assert batch_size == 1, "For underdetermined optimization batch_size has to be 1"
        constraint = cp.norm2(cp.multiply(A_, w_[:, None]) @ x_ - b_) <= l2_eps
        constraints.append(constraint)
        # TODO add indexing of intensities
        # TODO vectorization
        if isinstance(underdetermined_opt, Number):
            objective = cp.Minimize(cp.sum_squares(cp.sum(x_) - underdetermined_opt))
        elif isinstance(underdetermined_opt, np.ndarray):
            raise NotImplementedError("Target intensities for light sources.")
        elif not isinstance(underdetermined_opt, str):
            raise TypeError(f"`{underdetermined_opt}` is not a underdetermined_opt option.")
        elif underdetermined_opt == 'l2':
            objective = cp.Minimize(cp.norm2(x_))
        elif underdetermined_opt == 'min':
            objective = cp.Minimize(cp.sum(x_))
        elif underdetermined_opt == 'max':
            objective = cp.Maximize(cp.sum(x_))
        elif underdetermined_opt == 'var':
            objective = cp.Minimize(x_ - cp.sum(x_)/x_.size)
        else:
            raise NameError(f"`{underdetermined_opt}` is not a underdetermined_opt option.")

    problem = cp.Problem(objective, constraints)
    assert problem.is_dcp(dpp=True)

    # empty X
    X = np.zeros((B.shape[0], A.shape[-1]))
    last_batch_size = X.shape[0] % batch_size
    # iterate over batches - and pad last batch
    for idx, (b, w), _ in batched_iteration(B.shape[0], (B, W), (), batch_size=batch_size, pad=True):
        w_.value = w
        b_.value = b * w  # ensures that objective is dpp compliant
        
        problem.solve(**opt_kwargs)
        if not np.isfinite(problem.value):
            raise RuntimeError("Optimization did not converge.")

        x = x_.value.reshape(-1, A.shape[-1])

        if ((idx+1) * batch_size) > X.shape[0]:
            X[idx * batch_size:] = x[:last_batch_size]
        else:
            X[idx * batch_size:(idx+1) * batch_size] = x

    if return_pred:
        return X, (X @ A.T + baseline)
    return X


def lsq_linear_minimize(
    A, B, Epsilon, 
    lb=None, ub=None, W=None,
    K=None, baseline=None, 
    norm=None,
    delta=None,
    n_jobs=None, 
    batch_size=1,
    verbose=0, 
    return_pred=False,
    **opt_kwargs
):
    """
    Linear minimization approach
    """
    delta = (EPS_NP64 if delta is None else delta)
    
    Epsilon = error_propagation(Epsilon, K)
    A, B, lb, ub, W, baseline = prepare_parameters_for_linear(A, B, lb, ub, W, K, baseline)
    
    if norm is None:
        _, B0 = lsq_linear_cp(
            A, B, lb=lb, ub=ub, W=W, K=K, 
            baseline=baseline, n_jobs=n_jobs, 
            batch_size=batch_size, verbose=verbose, 
            return_pred=True,
            **opt_kwargs
        )
        norm = np.sum((W*B0 - W*B)**2, axis=-1)
    
    B = B - baseline
    total_delta = np.broadcast_to(np.atleast_1d(delta + norm), B.shape[0])
    batch_size = get_batch_size(batch_size, B.shape[0])

    if n_jobs is not None:
        raise NotImplementedError("parallel jobs")

    # set up cvxpy problem
    # constants
    A_ = diagonal_stack(A, batch_size)
    Epsilon_ = diagonal_stack(Epsilon, batch_size)
    lb_ = concat(lb, batch_size)
    ub_ = concat(ub, batch_size)

    # parameters
    w_ = cp.Parameter((batch_size * W.shape[1]), pos=True)
    # this b_ has w_ multiplied into it to aid compilation
    b_ = cp.Parameter((batch_size * B.shape[1]), )
    t_delta_ = cp.Parameter((batch_size), pos=True)
    
    # variable and constraints
    if np.all(lb >= 0):
        xkwargs = {'pos': True}
    else:
        xkwargs = {}
    x_ = cp.Variable(A_.shape[1], **xkwargs)
    constraints = [
        # proper reshaping
        cp.sum(
            cp.reshape(
                cp.multiply(A_, w_[:, None]) @ x_ - b_, 
                (batch_size, A.shape[0])
            ) ** 2, axis=1
        ) <= t_delta_
    ]
    if np.all(np.isfinite(lb)):
        constraints.append(x_ >= lb_)
    if np.all(np.isfinite(ub)):
        constraints.append(x_ <= ub_)
    
    # objective function
    objective = cp.Minimize(
        cp.sum(Epsilon_ @ x_**2)
    )
    problem = cp.Problem(objective, constraints)
    assert problem.is_dcp(dpp=True)

    # empty X
    X = np.zeros((B.shape[0], A.shape[-1]))
    last_batch_size = X.shape[0] % batch_size
    # iterate over batches - and pad last batch
    for idx, (b, w, t_delta), _ in batched_iteration(B.shape[0], (B, W, total_delta), (), batch_size=batch_size, pad=True):
        w_.value = w
        b_.value = b * w  # ensures that objective is dpp compliant
        if ((idx+1) * batch_size) > X.shape[0]:
            t_delta = np.atleast_1d(t_delta).copy
            t_delta[last_batch_size:] = 1e10  # some big number
            t_delta_.value = t_delta
        else:
            t_delta_.value = np.atleast_1d(t_delta)
        
        problem.solve(**opt_kwargs)
        if not np.isfinite(problem.value):
            raise RuntimeError("Optimization did not converge.")

        x = x_.value.reshape(-1, A.shape[-1])

        if ((idx+1) * batch_size) > X.shape[0]:
            X[idx * batch_size:] = x[:last_batch_size]
        else:
            X[idx * batch_size:(idx+1) * batch_size] = x

    if return_pred:
        return X, (X @ A.T + baseline), (X**2 @ Epsilon.T)
    return X


def lsq_linear_decomposition(
    A, B, 
    n_layers=None,
    mask=None,
    lb=None, ub=None, W=None,
    lbp=0, ubp=1, 
    K=None, baseline=None,
    max_iter=200,
    init_iter=1000,
    seed=None, 
    subsample=None,
    verbose=0,
    solver=cp.SCS, 
    return_pred=False,
    **opt_kwargs
):
    A, B, lb, ub, W, baseline = prepare_parameters_for_linear(A, B, lb, ub, W, K, baseline)
    B = B - baseline
    lbp = np.atleast_2d(lbp)
    ubp = np.atleast_2d(ubp)
    
    size = B.shape[0]
    inputs = A.shape[1]
    channels = B.shape[1]

    if subsample:
        if isinstance(subsample, str):
            if subsample == 'fast':
                subsample = min(1, 1028 / size)
            else:
                raise NameError("subsample string must be `fast`.")
        Btotal = B
        Wtotal = W
        total_size = size
        
        size = int(size * subsample)
        rng = default_rng(seed)
        idcs = rng.choice(total_size, size=size, replace=False)
        B = B[idcs]
        W = W[idcs]

    if mask is None and n_layers is None:
        n_layers = (A.shape[0]-1)
        mask = np.ones((n_layers, inputs))
    elif mask is None:
        mask = np.ones((n_layers, inputs))
    else:
        mask = np.asarray(mask)

    Pvar = cp.Variable((size, n_layers), pos=True)
    Xvar = cp.Variable((n_layers, inputs), pos=True)
    Ppar = cp.Parameter((size, n_layers), pos=True)
    Xpar = cp.Parameter((n_layers, inputs), pos=True)
    # P @ X @ A.T

    # initialize P matrix (usually pixel intensities)
    nmf = NMF(
        n_components=n_layers, 
        random_state=seed, 
        init=('random' if (n_layers > channels) else 'nndsvda'), 
        max_iter=init_iter
    )
    P0 = nmf.fit(B.T).components_.T
    P0 = np.abs(P0) / np.max(np.abs(P0))  # range is 0-1
    P0 = (P0 - lbp) / (ubp - lbp) 
    Ppar.value = P0

    # p constraints
    p_constraints = [
        # TODO broadcasting
        Pvar >= lbp, 
        Pvar <= ubp
    ]

    # x constraints
    x_constraints = []
    if np.any(mask == 0):
        x_constraints.append(
            Xvar[mask == 0] == 0
        )
    # all subframes have the same overall intensity
    if n_layers > 1:
        x_constraints.append(
            cp.diff(cp.sum(Xvar, axis=1)) == 0
        )
    if np.all(np.isfinite(lb)):
        x_constraints.append(Xvar >= np.atleast_2d(lb))
    if np.all(np.isfinite(ub)):
        x_constraints.append(Xvar <= np.atleast_2d(ub))

    # x objective function
    x_objective = cp.Minimize(
        cp.norm(cp.multiply(W, Ppar @ Xvar @ A.T - B), 'fro')
    )
    x_problem = cp.Problem(x_objective, x_constraints)
    assert x_problem.is_dcp(dpp=True)

    # x objective function
    p_objective = cp.Minimize(
        cp.norm(cp.multiply(W, Pvar @ Xpar @ A.T - B), 'fro')
    )
    p_problem = cp.Problem(p_objective, p_constraints)
    assert p_problem.is_dcp(dpp=True)

    # TODO convergence criteria and early stopping
    for _ in range(max_iter):
        x_problem.solve(solver=solver, **opt_kwargs)
        if not np.isfinite(x_problem.value):
            raise RuntimeError("Optimization did not converge.")

        X = Xpar.value = Xvar.value
        p_problem.solve(solver=solver, **opt_kwargs)
        if not np.isfinite(p_problem.value):
            raise RuntimeError("Optimization did not converge.")

        P = Ppar.value = Pvar.value
    
    x_problem.solve(solver=solver, **opt_kwargs)
    if not np.isfinite(x_problem.value):
        raise RuntimeError("Optimization did not converge.")

    X = Xvar.value

    if subsample:
        Pvar = cp.Variable((total_size, n_layers), pos=True)
        p_constraints = [
            # TODO broadcasting
            Pvar >= lbp, 
            Pvar <= ubp
        ]
        p_objective = cp.Minimize(
            cp.norm(cp.multiply(Wtotal, Pvar @ X @ A.T - Btotal), 'fro')
        )
        p_problem = cp.Problem(p_objective, p_constraints)
        p_problem.solve(solver=solver, **opt_kwargs)
        if not np.isfinite(x_problem.value):
            raise RuntimeError("Optimization did not converge.")
        P = Pvar.value

    if return_pred:
        return X, P, (P @ X @ A.T + baseline)
    return X, P


def lsq_linear_adaptive(
    A, B, 
    lb=None, ub=None, W=None,
    K=None, baseline=None, 
    neutral_point=None,
    verbose=0, 
    delta=0,
    scale_w=1,
    solver=cp.ECOS, 
    return_pred=False,
    **opt_kwargs
):
    """
    Adaptively solve a linear least-squares problem with bounds on the variables.

    This method uses `cvxpy` to solve the problem.

    A (channels x inputs)
    B (samples x channels)
    K (channels) or (channels x channels)
    baseline (channels)
    ub (inputs)
    lb (inputs)
    w (channels)
    """
    A, B, lb, ub, W, baseline = prepare_parameters_for_linear(A, B, lb, ub, W, K, baseline)
    size = B.shape[0]
    inputs = A.shape[1]

    scale_w = np.atleast_1d(scale_w)

    # "intensity" dimension
    Bsum = B.sum(axis=-1)
    # "radius" dimensions
    if neutral_point is None:
        neutral_point = np.ones((1, B.shape[-1]))
    else:
        neutral_point = np.atleast_2d(neutral_point)
    neutral_points = neutral_point / neutral_point.sum() * Bsum[:, None]
    Brad = B - neutral_points

    if np.all(lb >= 0):
        xkwargs = {'pos': True}
    else:
        xkwargs = {}
    X = cp.Variable((size, inputs), **xkwargs)
    scales = cp.Variable(2, pos=True)
    
    B_pred = X @ A.T + baseline[None]
    
    radii_vec_pred = B_pred - cp.multiply(neutral_points, scales[0])
    radii_vec_actual = cp.multiply(scales[1], Brad)
    int_pred =  cp.sum(B_pred, axis=1) 
    int_actual = cp.multiply(Bsum, scales[0])
    
    bound_constraints = []
    if np.all(np.isfinite(lb)):
        bound_constraints.append(X >= lb[None])
    if np.all(np.isfinite(ub)):
        bound_constraints.append(X <= ub[None])

    constraints = [
        # intensity constraint
        int_pred == int_actual, 
        # radii constraint
        (
            cp.sum_squares(radii_vec_actual - radii_vec_pred) <= delta
            if delta
            else radii_vec_pred == radii_vec_actual
        )
    ] + bound_constraints
    # scale things as little as possible
    objective = cp.Minimize(cp.sum_squares(cp.multiply(scale_w, (scales - 1))))
    # problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=bool(verbose), **opt_kwargs)
    if not np.isfinite(problem.value):
        raise RuntimeError("Optimization did not converge.")

    X, scales = X.value, scales.value
    if return_pred:
        return X, scales, (X @ A.T + baseline)

    return X, scales