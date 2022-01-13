"""
The scipy lsq_linear algorithm

A (channels x inputs)
B (samples x channels)
K (channels) or (channels x channels)
baseline (channels)
ub (inputs)
lb (inputs)
W (samples x channels)
"""

from numbers import Number
import warnings
from tqdm import tqdm

import numpy as np
from numpy.random import default_rng
import cvxpy as cp
from sklearn.decomposition import NMF

from dreye.api.optimize.parallel import batched_iteration, diagonal_stack, concat
from dreye.api.optimize.utils import get_batch_size, prepare_parameters_for_linear
from dreye.api.utils import l2norm, error_propagation, get_prediction
from dreye.api.defaults import EPS_NP64
# TODO-later Huber loss instead of just sum squares? -> outliers less penalized


def _prepare_parameters(A, B, lb, ub, W, K, baseline, batch_size, subtract=True):
    """Prepare least-squares problem.

    Parameters
    ----------
    A : [type]
        [description]
    B : [type]
        [description]
    lb : [type]
        [description]
    ub : [type]
        [description]
    W : [type]
        [description]
    K : [type]
        [description]
    baseline : [type]
        [description]
    batch_size : [type]
        [description]
    subtract : bool, optional
        Whether to subtract the baseline from `B` or not.

    Returns
    -------
    [type]
        [description]
    """
    A, B, lb, ub, W, baseline = prepare_parameters_for_linear(A, B, lb, ub, W, K, baseline)
    if subtract:
        B = B - baseline
    batch_size = get_batch_size(batch_size, B.shape[0])
    return A, B, lb, ub, W, baseline, batch_size


def _prepare_variables(A, B, lb, ub, W, batch_size):
    """[summary]

    Parameters
    ----------
    A : [type]
        [description]
    B : [type]
        [description]
    lb : [type]
        [description]
    ub : [type]
        [description]
    W : [type]
        [description]
    batch_size : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
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
        
    return A_, x_, w_, b_, constraints


def lsq_linear(
    A, B, 
    lb=None, ub=None, W=None,
    K=None, baseline=None, 
    n_jobs=None, 
    batch_size=1,
    model='gaussian',
    verbose=0, 
    return_pred=False,
    **opt_kwargs
):
    """
    Least-squares linear optimization problem.

    Parameters
    ----------
    A : [type]
        [description]
    B : [type]
        [description]
    lb : [type], optional
        [description], by default None
    ub : [type], optional
        [description], by default None
    W : [type], optional
        [description], by default None
    K : [type], optional
        [description], by default None
    baseline : [type], optional
        [description], by default None
    n_jobs : [type], optional
        [description], by default None
    batch_size : int, optional
        [description], by default 1
    model : str, optional
        [description]. By default 'gaussian'.
    verbose : int, optional
        [description], by default 0
    return_pred : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NotImplementedError
        [description]
    TypeError
        [description]
    NameError
        [description]
    RuntimeError
        [description]
    """
    A, B, lb, ub, W, baseline, batch_size = _prepare_parameters(
        A, B, lb, ub, W, K, baseline, batch_size
    )
    A_, x_, w_, b_, constraints = _prepare_variables(
        A, B, lb, ub, W, batch_size
    )

    if n_jobs is not None:
        raise NotImplementedError("parallel jobs")
     
    if model == 'gaussian':
        objective = cp.Minimize(
            cp.sum_squares((cp.multiply(A_, w_[:, None]) @ x_ - b_))
        )
    elif model == 'poisson':
        objective = cp.Minimize(
            -cp.sum(cp.multiply(b_, cp.log(A_ @ x_)) - (cp.multiply(A_, w_[:, None]) @ x_))
        )
    else:
        raise NameError(f"Model string name must be `gaussian` or `poisson`, but is {model}")
    
    problem = cp.Problem(objective, constraints)
    assert problem.is_dcp(dpp=True)

    # empty X
    X = np.zeros((B.shape[0], A.shape[-1]))
    last_batch_size = X.shape[0] % batch_size
    # iterate over batches - and pad last batch
    for idx, (b, w), _ in batched_iteration(B.shape[0], (B, W), (), batch_size=batch_size, pad=True, verbose=verbose):
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
        return X, get_prediction(X, A, baseline)
    return X


def lsq_linear_excitation(
    A, B, 
    lb=None, ub=None, W=None,
    K=None, baseline=None, 
    n_jobs=None, 
    batch_size=1,
    verbose=0, 
    return_pred=False,
    solver=cp.SCS,
    **opt_kwargs
):
   
    A, B, lb, ub, W, baseline, batch_size = _prepare_parameters(
        A, B, lb, ub, W, K, baseline, batch_size, subtract=False
    )
    A_, x_, w_, b_, constraints = _prepare_variables(
        A, B, lb, ub, W, batch_size
    )

    if n_jobs is not None:
        raise NotImplementedError("parallel jobs")

    btarget_ = cp.multiply(A_, w_[:, None]) @ x_
    denom = cp.multiply((1+b_), (1+btarget_))
    num = b_ - btarget_
    objective = cp.Minimize(cp.max(cp.abs(num)/denom))
    
    problem = cp.Problem(objective, constraints)
    assert problem.is_dqcp(), "Problem is not quasi-convex."

    # empty X
    X = np.zeros((B.shape[0], A.shape[-1]))
    last_batch_size = X.shape[0] % batch_size
    # iterate over batches - and pad last batch
    for idx, (b, w), _ in batched_iteration(B.shape[0], (B, W), (), batch_size=batch_size, pad=True, verbose=verbose):
        w_.value = w
        b_.value = b * w  # ensures that objective is dpp compliant
        
        problem.solve(solver=solver, qcp=True, **opt_kwargs)
        if not np.isfinite(problem.value):
            raise RuntimeError("Optimization did not converge.")

        x = x_.value.reshape(-1, A.shape[-1])

        if ((idx+1) * batch_size) > X.shape[0]:
            X[idx * batch_size:] = x[:last_batch_size]
        else:
            X[idx * batch_size:(idx+1) * batch_size] = x

    if return_pred:
        return X, get_prediction(X, A, baseline)
    return X


def lsq_linear_underdetermined(
    A, B, 
    lb=None, ub=None, W=None,
    K=None, baseline=None, 
    n_jobs=None, 
    batch_size=1,
    verbose=0, 
    return_pred=False,
    l2_eps=EPS_NP64,
    underdetermined_opt=None,
    **opt_kwargs
):
    A, B, lb, ub, W, baseline, batch_size = _prepare_parameters(
        A, B, lb, ub, W, K, baseline, batch_size
    )
    A_, x_, w_, b_, constraints = _prepare_variables(
        A, B, lb, ub, W, batch_size
    )

    if n_jobs is not None:
        raise NotImplementedError("parallel jobs")
    
    # objective function
    if underdetermined_opt is None: 
        underdetermined_opt = 'l2'
    
    # TODO-later vectorize l2_eps and underdetermined_opt, idcs
    # TODO-later batching
    assert A.shape[1] > A.shape[0], "System is not underdetermined."
    assert batch_size == 1, "For underdetermined optimization batch_size has to be 1."

    if isinstance(underdetermined_opt, tuple):
        underdetermined_opt, idcs = underdetermined_opt
        xselect_ = x_[idcs]
    else:
        xselect_ = x_
    
    # TODO-docstring-warning this constraint assumes that things are within the hull - TEST and FIT normal
    # TODO-later add poisson error term instead? and excitation error term instead
    constraint = cp.norm2(cp.multiply(A_, w_[:, None]) @ x_ - b_) <= l2_eps
    constraints.append(constraint)
    
    if isinstance(underdetermined_opt, Number):
        objective = cp.Minimize(cp.sum_squares(cp.sum(xselect_) - underdetermined_opt))
    elif isinstance(underdetermined_opt, np.ndarray):
        objective = cp.Minimize(cp.sum_squares(xselect_ - underdetermined_opt))
    elif not isinstance(underdetermined_opt, str):
        raise TypeError(f"Type `{type(underdetermined_opt)}` is not the correct underdetermined_opt type.")
    elif underdetermined_opt == 'l2':
        objective = cp.Minimize(cp.norm2(xselect_))
    elif underdetermined_opt == 'min':
        objective = cp.Minimize(cp.sum(xselect_))
    elif underdetermined_opt == 'max':
        objective = cp.Maximize(cp.sum(xselect_))
    elif underdetermined_opt == 'var':
        objective = cp.Minimize(cp.sum_squares(xselect_ - cp.sum(xselect_)/xselect_.size))
    else:
        raise NameError(f"`{underdetermined_opt}` is not a underdetermined_opt option.")
            
    problem = cp.Problem(objective, constraints)
    assert problem.is_dcp(dpp=True)

    # empty X
    X = np.zeros((B.shape[0], A.shape[-1]))
    last_batch_size = X.shape[0] % batch_size
    # iterate over batches - and pad last batch
    for idx, (b, w), _ in batched_iteration(B.shape[0], (B, W), (), batch_size=batch_size, pad=True, verbose=verbose):
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
        return X, get_prediction(X, A, baseline)
    return X


def lsq_linear_minimize(
    A, B, Epsilon, 
    lb=None, ub=None, W=None,
    K=None, baseline=None, 
    norm=None,
    l2_eps=EPS_NP64,
    L1=None,
    l1_eps=EPS_NP64,
    n_jobs=None, 
    batch_size=1,
    verbose=0, 
    return_pred=False,
    **opt_kwargs
):
    """Minimize Least-squares problem.

    Parameters
    ----------
    A : [type]
        [description]
    B : [type]
        [description]
    Epsilon : [type]
        [description]
    lb : [type], optional
        [description], by default None
    ub : [type], optional
        [description], by default None
    W : [type], optional
        [description], by default None
    K : [type], optional
        [description], by default None
    baseline : [type], optional
        [description], by default None
    norm : [type], optional
        [description], by default None
    l2_eps : [type], optional
        [description], by default EPS_NP64
    L1 : [type], optional
        [description], by default None
    l1_eps : [type], optional
        [description], by default EPS_NP64
    n_jobs : [type], optional
        [description], by default None
    batch_size : int, optional
        [description], by default 1
    verbose : int, optional
        [description], by default 0
    return_pred : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NameError
        [description]
    NotImplementedError
        [description]
    RuntimeError
        [description]
    """
    # prepare parameters
    # NB: baseline gets removed from B
    A, B, lb, ub, W, baseline, batch_size = _prepare_parameters(
        A, B, lb, ub, W, K, baseline, batch_size, 
    )
    
    # if norm is None perform normal fitting to obtain norm.
    if norm is None:
        # NB: baselines cancel
        _, B0 = lsq_linear(
            A, 
            B,
            lb=lb, ub=ub, W=W, 
            K=None,  # already prepared
            baseline=None,  # already prepared
            n_jobs=n_jobs, 
            batch_size=batch_size, verbose=verbose, 
            return_pred=True,
            **opt_kwargs
        )
        norm = l2norm((W*B0 - W*B), axis=-1)
    
    total_delta = np.broadcast_to(np.atleast_1d(l2_eps + norm), B.shape[0])
    # Format Epsilon
    if isinstance(Epsilon, str):
        if Epsilon == 'heteroscedastic':
            Epsilon = A.copy()
        else:
            raise NameError(f"Epsilon must be array or `heteroscedastic`, but is `{Epsilon}`")
    else:
        Epsilon = error_propagation(Epsilon, K)
    
    # L1-norm constraint implementation
    if L1 is None:
        l1_constraint = False
        L1 = np.zeros(B.shape[0])
    elif isinstance(L1, Number):
        l1_constraint = True
        L1 = np.array([L1]*B.shape[0])
    else:
        l1_constraint = True
        L1 = np.asarray(L1)
        assert L1.size == B.shape[0]

    if n_jobs is not None:
        raise NotImplementedError("parallel jobs")

    # set up cvxpy problem    
    A_, x_, w_, b_, constraints = _prepare_variables(
        A, B, lb, ub, W, batch_size
    )
    
    Epsilon_ = diagonal_stack(Epsilon, batch_size)
    t_delta_ = cp.Parameter((batch_size), pos=True)
    
    # TODO poisson and excitation constraint option?
    constraints.append(
        cp.norm2(
            cp.reshape(
                cp.multiply(A_, w_[:, None]) @ x_ - b_, 
                (batch_size, A.shape[0])
            ), axis=1
        ) <= t_delta_
    )
    
    # TODO test if necessary for parameter?
    # variable and constraints
    if np.all(lb >= 0):
        xkwargs = {'pos': True}
    else:
        xkwargs = {}
    l1_ = cp.Parameter(batch_size, **xkwargs)
    if l1_constraint:
        if np.all(lb >= 0):
            constraints.extend([
                cp.sum(cp.reshape(x_, (batch_size, A.shape[1])), axis=1) <= (l1_ + l1_eps), 
                cp.sum(cp.reshape(x_, (batch_size, A.shape[1])), axis=1) >= (l1_ - l1_eps), 
            ])
        else:
            constraints.extend([
                cp.sum(cp.abs(cp.reshape(x_, (batch_size, A.shape[1]))), axis=1) <= (l1_ + l1_eps), 
                cp.sum(cp.abs(cp.reshape(x_, (batch_size, A.shape[1]))), axis=1) >= (l1_ - l1_eps), 
            ])
    
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
    for idx, (b, w, t_delta, i), _ in batched_iteration(B.shape[0], (B, W, total_delta, L1), (), batch_size=batch_size, pad=True, verbose=verbose):
        w_.value = w
        b_.value = b * w  # ensures that objective is dpp compliant
        l1_.value = np.atleast_1d(i)
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
        return X, get_prediction(X, A, baseline), (X**2 @ Epsilon.T)
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
    equal_l1norm_constraint=True,
    xtol=1e-8, 
    ftol=1e-8,
    **opt_kwargs
):
    """[summary]

    Parameters
    ----------
    A : [type]
        [description]
    B : [type]
        [description]
    n_layers : [type], optional
        [description], by default None
    mask : [type], optional
        [description], by default None
    lb : [type], optional
        [description], by default None
    ub : [type], optional
        [description], by default None
    W : [type], optional
        [description], by default None
    lbp : int, optional
        [description], by default 0
    ubp : int, optional
        [description], by default 1
    K : [type], optional
        [description], by default None
    baseline : [type], optional
        [description], by default None
    max_iter : int, optional
        [description], by default 200
    init_iter : int, optional
        [description], by default 1000
    seed : [type], optional
        [description], by default None
    subsample : [type], optional
        [description], by default None
    verbose : int, optional
        [description], by default 0
    solver : [type], optional
        [description], by default cp.SCS
    return_pred : bool, optional
        [description], by default False
    equal_l1norm_constraint : bool, optional
        [description], by default True
    xtol : float, optional
    ftol : float, optional

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NameError
        [description]
    RuntimeError
        [description]
    RuntimeError
        [description]
    RuntimeError
        [description]
    RuntimeError
        [description]
    """
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
    # TODO faster version?
    if verbose:
        print("Initializing `P` with scikit-learn's NMF algorithm.")
    nmf = NMF(
        n_components=n_layers, 
        random_state=seed, 
        init=('random' if (n_layers > channels) else 'nndsvda'), 
        max_iter=init_iter, 
        verbose=verbose
    )
    P0 = nmf.fit(B.T).components_.T
    # rescaling initial `P`
    P0 = np.abs(P0) / np.max(np.abs(P0))  # range is 0-1
    P0 = P0 * (ubp - lbp) + lbp  # rescale to range
    # set inital P value for X problem
    Ppar.value = P0

    # p constraints
    p_constraints = [
        # TODO-later broadcasting
        Pvar >= lbp, 
        Pvar <= ubp
    ]
    
    # TODO poisson/excitation objective?

    # x constraints
    x_constraints = []
    if np.any(mask == 0):
        x_constraints.append(
            Xvar[mask == 0] == 0
        )
    # all subframes have the same overall intensity -> equal l1norm
    if (n_layers > 1) and equal_l1norm_constraint:
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
    
    if verbose:
        iters = tqdm(range(max_iter), desc="Decompose iteration", total=max_iter)
    else:
        iters = range(max_iter)

    # EM-type algorithm to optimize P and X
    for n in iters:
        
        if verbose > 1:
            print("Fitting `X` problem")
        
        x_problem.solve(solver=solver, verbose=bool(verbose > 1), **opt_kwargs)
        if not np.isfinite(x_problem.value):
            raise RuntimeError("Optimization did not converge.")

        X = Xpar.value = Xvar.value
        
        if verbose > 1:
            print("Fitting `P` problem.")
        
        p_problem.solve(solver=solver, verbose=bool(verbose > 1), **opt_kwargs)
        if not np.isfinite(p_problem.value):
            raise RuntimeError("Optimization did not converge.")

        P = Ppar.value = Pvar.value
        
        current_loss = np.linalg.norm(W * (P @ X @ A.T - B), ord='fro')
        # current variables
        current_vars = np.concatenate([np.ravel(X), np.ravel(P)])
        
        if n:
            # tolerance for terminatino by the change of the cost function
            if (prev_loss - current_loss) < (ftol * current_loss):
                if verbose:
                    print(f"Tolerance criteria met for cost function; stopping at {n} iterations")
                break
            
            # Tolerance for termination by the change of the independent variables. 
            if l2norm(prev_vars - current_vars) < (xtol * (xtol + l2norm(current_vars))):
                if verbose:
                    print(f"Tolerance criteria met for independen variables; stopping at {n} iterations.")
                break
        
        # reassign previous loss
        prev_loss = current_loss
        prev_vars = current_vars
        
    else:
        warnings.warn(
            f"Maximum number of iterations {max_iter} reached. Increase it to improve convergence.", 
            RuntimeWarning
        )
    
    if verbose:
        print("Refitting `X` values as final fitting step")
    x_problem.solve(solver=solver, verbose=bool(verbose > 1), **opt_kwargs)
    if not np.isfinite(x_problem.value):
        raise RuntimeError("Optimization did not converge.")

    X = Xvar.value

    if subsample:
        if verbose:
            print("Fitting complete `P` after fitting subsampled `P`")
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
        p_problem.solve(solver=solver, verbose=bool(verbose > 1), **opt_kwargs)
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
    delta_radius=EPS_NP64,
    delta_norm1=EPS_NP64,
    scale_w=1,
    adaptive_objective="unity",
    solver=cp.ECOS, 
    return_pred=False,
    **opt_kwargs
):
    """
    Adaptively solve a linear least-squares problem with bounds on the variables.

    Parameters
    ----------
    A : [type]
        [description]
    B : [type]
        [description]
    lb : [type], optional
        [description], by default None
    ub : [type], optional
        [description], by default None
    W : [type], optional
        [description], by default None
    K : [type], optional
        [description], by default None
    baseline : [type], optional
        [description], by default None
    neutral_point : [type], optional
        [description], by default None
    verbose : int, optional
        [description], by default 0
    delta_radius : [type], optional
        [description], by default EPS_NP64
    delta_norm1 : [type], optional
        [description], by default EPS_NP64
    scale_w : int, optional
        [description], by default 1
    adaptive_objective : str, optional
        [description], by default "unity"
    solver : [type], optional
        [description], by default cp.ECOS
    return_pred : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NameError
        [description]
    RuntimeError
        [description]
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
    
    # capture prediction minus the neutral point for that vector scaled by intensity
    radii_vec_pred = B_pred - cp.multiply(neutral_points, scales[0])
    # actual radius scalled by scalar radius
    radii_vec_actual = cp.multiply(scales[1], Brad)
    int_pred =  cp.sum(B_pred, axis=1) 
    int_actual = cp.multiply(Bsum, scales[0])
    
    bound_constraints = []
    if np.all(np.isfinite(lb)):
        bound_constraints.append(X >= lb[None])
    if np.all(np.isfinite(ub)):
        bound_constraints.append(X <= ub[None])

    constraints = [
        # intensity/l1norm constraint
        (
            cp.max(cp.abs(int_pred - int_actual)) <= delta_norm1
            if delta_norm1
            else int_pred == int_actual
        ),
        # radii constraint
        (
            cp.max(cp.abs(radii_vec_actual - radii_vec_pred)) <= delta_radius
            if delta_radius
            else radii_vec_pred == radii_vec_actual
        )
    ] + bound_constraints
    # scale things as little as possible
    if (adaptive_objective == 'unity') or (adaptive_objective is None):
        objective = cp.Minimize(cp.sum_squares(cp.multiply(scale_w, (scales - 1))))
    elif adaptive_objective == 'max':
        objective = cp.Maximize(cp.sum(cp.multiply(scale_w, scales)))
    else:
        raise NameError(f"adaptive_objective must be None, `unity`, or `max`, but is `{adaptive_objective}`.")
    # problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=bool(verbose), **opt_kwargs)
    if not np.isfinite(problem.value):
        raise RuntimeError("Optimization did not converge.")

    X, scales = X.value, scales.value
    if return_pred:
        return X, scales, get_prediction(X, A, baseline)

    return X, scales