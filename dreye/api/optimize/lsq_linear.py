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

from typing import Union, Tuple, Optional
from numbers import Number
import warnings
from tqdm import tqdm

import numpy as np
from numpy.random import default_rng
import cvxpy as cp
from sklearn.decomposition import NMF

from dreye.api.optimize.parallel import batched_iteration, diagonal_stack, concat
from dreye.api.optimize.utils import get_batch_size, prepare_parameters_for_linear
from dreye.api.utils import l2norm, propagate_error, predict_values
from dreye.api.defaults import EPS_NP64

# TODO-later Huber loss instead of just sum squares? -> outliers less penalized


def _prepare_parameters(
    A: np.ndarray,
    B: np.ndarray,
    lb: Union[float, np.ndarray],
    ub: Union[float, np.ndarray],
    W: np.ndarray,
    K: Optional[np.ndarray],
    baseline: Union[float, np.ndarray],
    batch_size: int,
    subtract: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Prepare parameters for batched linear least squares problem.

    This function prepares the parameters for the batched linear least squares problem
    by first checking and transforming the parameters using `prepare_parameters_for_linear`,
    and then adjusting the target matrix and batch size as necessary.

    Parameters
    ----------
    A : np.ndarray
        The input matrix.
    B : np.ndarray
        The target matrix.
    lb : float or np.ndarray
        The lower bounds for the parameters.
    ub : float or np.ndarray
        The upper bounds for the parameters.
    W : np.ndarray
        The weight matrix.
    K : np.ndarray, optional
        Transformation matrix for `A`, `lb`, `ub`, and `baseline`, by default None.
    baseline : float or np.ndarray
        The baseline to subtract from the target matrix.
    batch_size : int
        The size of the batch for the batched least squares problem.
    subtract : bool, optional
        Whether to subtract the baseline from `B` or not, by default True

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]
        The prepared parameters and the adjusted batch size.
    """

    A, B, lb, ub, W, baseline = prepare_parameters_for_linear(
        A, B, lb, ub, W, K, baseline
    )
    if subtract:
        B = B - baseline
    batch_size = get_batch_size(batch_size, B.shape[0])
    return A, B, lb, ub, W, baseline, batch_size


def _prepare_variables(
    A: np.ndarray,
    B: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    W: np.ndarray,
    batch_size: int,
):
    """
    Prepares variables and constraints for the cvxpy problem.

    Parameters
    ----------
    A : np.ndarray
        Coefficients of the linear equation.
    B : np.ndarray
        Right-hand side of the linear equation.
    lb : np.ndarray
        Lower bounds for the variables.
    ub : np.ndarray
        Upper bounds for the variables.
    W : np.ndarray
        Weights for the objective function.
    batch_size : int
        Size of batches for parallel processing.

    Returns
    -------
    Tuple[np.ndarray, cp.Variable, cp.Parameter, cp.Parameter, List[cp.constraints]]
        Prepared variables and constraints.
    """
    A_ = diagonal_stack(A, batch_size)
    lb_ = concat(lb, batch_size)
    ub_ = concat(ub, batch_size)

    kwargs = {"pos": True} if np.all(lb >= 0) else {}

    w_ = cp.Parameter((batch_size * W.shape[1]), pos=True)
    b_ = cp.Parameter((batch_size * B.shape[1]), **kwargs)

    x_ = cp.Variable(A_.shape[1], **kwargs)
    constraints = []
    if np.all(np.isfinite(lb)):
        constraints.append(x_ >= lb_)
    if np.all(np.isfinite(ub)):
        constraints.append(x_ <= ub_)

    return A_, x_, w_, b_, constraints


def _solve_problem(
    problem: cp.Problem,
    x_: cp.Variable,
    A: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    batch_size: int,
    w_: cp.Parameter,
    b_: cp.Parameter,
    verbose: int = 0,
    **opt_kwargs,
) -> np.ndarray:
    """
    Solves a cvxpy Problem in batches and returns the solution.

    Parameters
    ----------
    problem : cp.Problem
        The cvxpy Problem to solve.
    x_ : cp.Variable
        The cvxpy Variable to optimize.
    A : np.ndarray
        The input matrix.
    B : np.ndarray
        The target matrix.
    W : np.ndarray
        The weight matrix.
    batch_size : int
        The batch size for solving the problem.
    w_ : cp.Parameter
        The weight vector parameter.
    b_ : cp.Parameter
        The bias vector parameter.
    verbose : int, optional
        The verbosity level, by default 0.
    **opt_kwargs
        Additional keyword arguments for the solver.

    Returns
    -------
    np.ndarray
        The optimized x values.
    """
    # empty X
    X = np.zeros((B.shape[0], A.shape[-1]))
    last_batch_size = X.shape[0] % batch_size

    # iterate over batches - and pad last batch
    for idx, (b, w), _ in batched_iteration(
        B.shape[0], (B, W), (), batch_size=batch_size, pad=True, verbose=verbose
    ):
        w_.value = w
        b_.value = b * w  # ensures that objective is dpp compliant

        problem.solve(**opt_kwargs)
        if not np.isfinite(problem.value):
            raise RuntimeError("Optimization did not converge.")

        x = x_.value.reshape(-1, A.shape[-1])

        if ((idx + 1) * batch_size) > X.shape[0]:
            X[idx * batch_size :] = x[:last_batch_size]
        else:
            X[idx * batch_size : (idx + 1) * batch_size] = x

    return X


def lsq_linear(
    A: np.ndarray,
    B: np.ndarray,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    W: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    baseline: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    batch_size: int = 1,
    model: str = "gaussian",
    verbose: int = 0,
    return_pred: bool = False,
    **opt_kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Least-squares linear optimization problem.

    Parameters
    ----------
    A : np.ndarray
        Coefficients of the linear equation.
    B : np.ndarray
        Right-hand side of the linear equation.
    lb : np.ndarray, optional
        Lower bounds for the variables, by default None.
    ub : np.ndarray, optional
        Upper bounds for the variables, by default None.
    W : np.ndarray, optional
        Weights for the objective function, by default None.
    K : np.ndarray, optional
        Transformation matrix for `A`, `lb`, `ub`, and `baseline`, by default None.
    baseline : np.ndarray, optional
        Bias values for the linear equation, by default None.
    n_jobs : int, optional
        Not yet implemented, by default None.
    batch_size : int, optional
        Size of batches for parallel processing, by default 1.
    model : str, optional
        Model type ('gaussian' or 'poisson'), by default 'gaussian'.
    verbose : int, optional
        Level of verbosity, by default 0.
    return_pred : bool, optional
        Whether to return predicted values or not, by default False.

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        Coefficients of the solution, and possibly predicted values.

    Raises
    ------
    NotImplementedError
        If n_jobs is not None.
    ValueError
        If model type is neither 'gaussian' nor 'poisson'.
    RuntimeError
        If optimization did not converge.
    """
    if n_jobs is not None:
        raise NotImplementedError("Parallel jobs are not implemented.")

    if model not in ["gaussian", "poisson"]:
        raise ValueError(
            f"Model string name must be 'gaussian' or 'poisson', but is {model}."
        )

    A, B, lb, ub, W, baseline, batch_size = _prepare_parameters(
        A, B, lb, ub, W, K, baseline, batch_size, subtract=(model != "poisson")
    )
    A_, x_, w_, b_, constraints = _prepare_variables(A, B, lb, ub, W, batch_size)

    if n_jobs is not None:
        raise NotImplementedError("Parallel jobs are not implemented.")

    if model == "gaussian":
        objective = cp.Minimize(
            cp.sum_squares((cp.multiply(A_, w_[:, None]) @ x_ - b_))
        )
    elif model == "poisson":
        objective = cp.Minimize(
            -cp.sum(
                cp.multiply(b_, cp.log(A_ @ x_ + baseline))
                - (cp.multiply(A_, w_[:, None]) @ x_ + cp.multiply(w_, baseline))
            )
        )

    problem = cp.Problem(objective, constraints)
    if not problem.is_dcp(dpp=True):
        raise ValueError(
            "Problem is not convex. Positivity constraint probably not met."
        )

    X = _solve_problem(problem, x_, A, B, W, batch_size, w_, b_, verbose, **opt_kwargs)

    return (X, predict_values(X, A, baseline)) if return_pred else X


def lsq_linear_excitation(
    A: np.ndarray,
    B: np.ndarray,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    W: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    baseline: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    batch_size: int = 1,
    verbose: int = 0,
    return_pred: bool = False,
    solver=cp.SCS,
    **opt_kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Least-squares linear optimization problem for excitation model.

    Parameters
    ----------
    A : np.ndarray
        Coefficients of the linear equation.
    B : np.ndarray
        Right-hand side of the linear equation.
    lb : np.ndarray, optional
        Lower bounds for the variables, by default None.
    ub : np.ndarray, optional
        Upper bounds for the variables, by default None.
    W : np.ndarray, optional
        Weights for the objective function, by default None.
    K : np.ndarray, optional
        Transformation matrix for `A`, `lb`, `ub`, and `baseline`, by default None.
    baseline : np.ndarray, optional
        Bias values for the linear equation, by default None.
    n_jobs : int, optional
        Number of jobs to be used for parallel processing, by default None.
    batch_size : int, optional
        Size of batches for parallel processing, by default 1.
    verbose : int, optional
        Level of verbosity, by default 0.
    return_pred : bool, optional
        Whether to return predicted values or not, by default False.
    solver: cp.SCS, optional
        Solver to be used for the problem, by default cp.SCS

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        Coefficients of the solution, and possibly predicted values.

    Raises
    ------
    NotImplementedError
        If `n_jobs` is provided.
    RuntimeError
        If the optimization problem does not converge.
    """
    if n_jobs is not None:
        raise NotImplementedError("parallel jobs")

    A, B, lb, ub, W, baseline, batch_size = _prepare_parameters(
        A, B, lb, ub, W, K, baseline, batch_size, subtract=False
    )
    A_, x_, w_, b_, constraints = _prepare_variables(A, B, lb, ub, W, batch_size)

    btarget_ = cp.multiply(A_, w_[:, None]) @ x_
    denom = cp.multiply((1 + b_), (1 + btarget_))
    num = b_ - btarget_
    objective = cp.Minimize(cp.max(cp.abs(num) / denom))

    problem = cp.Problem(objective, constraints)
    assert problem.is_dqcp(), "Problem is not quasi-convex."

    X = _solve_problem(
        problem,
        x_,
        A,
        B,
        W,
        batch_size,
        w_,
        b_,
        verbose,
        **dict(solver=solver, qcp=True, **opt_kwargs),
    )

    return (X, predict_values(X, A, baseline)) if return_pred else X


def _get_underdetermined_objective(
    x_: cp.Variable, underdetermined_opt: Optional[Union[str, Number, Tuple]]
):
    """
    Helper function to get the objective for underdetermined system.
    """
    if isinstance(underdetermined_opt, str):
        if underdetermined_opt == "l2":
            return cp.Minimize(cp.norm2(x_))
        elif underdetermined_opt == "min":
            return cp.Minimize(cp.sum(x_))
        elif underdetermined_opt == "max":
            return cp.Maximize(cp.sum(x_))
        elif underdetermined_opt == "var":
            return cp.Minimize(cp.sum_squares(x_ - cp.sum(x_) / x_.size))
        else:
            raise ValueError(
                f"Invalid option '{underdetermined_opt}' for underdetermined_opt."
            )
    elif isinstance(underdetermined_opt, Number):
        return cp.Minimize(cp.sum_squares(cp.sum(x_) - underdetermined_opt))
    elif isinstance(underdetermined_opt, np.ndarray):
        return cp.Minimize(cp.sum_squares(x_ - underdetermined_opt))
    elif isinstance(underdetermined_opt, tuple):
        q, r = underdetermined_opt
        return cp.Minimize(q * cp.norm(x_, 1) + r * cp.norm(x_, 2))
    else:
        raise ValueError(
            f"Invalid type '{type(underdetermined_opt).__name__}' for underdetermined_opt."
        )


def lsq_linear_underdetermined(
    A: np.ndarray,
    B: np.ndarray,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    W: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    baseline: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    batch_size: int = 1,
    verbose: int = 0,
    return_pred: bool = False,
    l2_eps: float = EPS_NP64,
    underdetermined_opt: Optional[Union[str, Number, Tuple]] = None,
    **opt_kwargs,
):
    """
    Least-squares linear optimization problem for underdetermined system.

    Parameters
    ----------
    A : np.ndarray
        Coefficients of the linear equation.
    B : np.ndarray
        Right-hand side of the linear equation.
    lb : np.ndarray, optional
        Lower bounds for the variables, by default None.
    ub : np.ndarray, optional
        Upper bounds for the variables, by default None.
    W : np.ndarray, optional
        Weights for the objective function, by default None.
    K : np.ndarray, optional
        Transformation matrix for `A`, `lb`, `ub`, and `baseline`, by default None.
    baseline : np.ndarray, optional
        Bias values for the linear equation, by default None.
    n_jobs : int, optional
        Number of jobs to be used for parallel processing, by default None.
    batch_size : int, optional
        Size of batches for parallel processing, by default 1.
    verbose : int, optional
        Level of verbosity, by default 0.
    return_pred : bool, optional
        Whether to return predicted values or not, by default False.
    l2_eps: float, optional
        Tolerance for the least squares error, by default EPS_NP64.
    underdetermined_opt: str, Number, tuple, optional
        Specification for the type of underdetermined problem. Options include 'l2', 'min', 'max', 'var', or a specific value.

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        Coefficients of the solution, and possibly predicted values.

    Raises
    ------
    NotImplementedError
        If `n_jobs` is provided.
    RuntimeError
        If the optimization problem does not converge.
    """
    if n_jobs is not None:
        raise NotImplementedError("parallel jobs")

    A, B, lb, ub, W, baseline, batch_size = _prepare_parameters(
        A, B, lb, ub, W, K, baseline, batch_size
    )
    A_, x_, w_, b_, constraints = _prepare_variables(A, B, lb, ub, W, batch_size)

    # objective function
    if underdetermined_opt is None:
        underdetermined_opt = "l2"

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

    objective = _get_underdetermined_objective(xselect_, underdetermined_opt)

    problem = cp.Problem(objective, constraints)
    assert problem.is_dcp(dpp=True)

    X = _solve_problem(problem, x_, A, B, W, batch_size, w_, b_, verbose, **opt_kwargs)

    return (X, predict_values(X, A, baseline)) if return_pred else X


def lsq_linear_minimize(
    A,
    B,
    Epsilon=None,
    lb=None,
    ub=None,
    W=None,
    K=None,
    baseline=None,
    norm=None,
    l2_eps=EPS_NP64,
    L1=None,
    l1_eps=EPS_NP64,
    n_jobs=None,
    batch_size=1,
    verbose=0,
    return_pred=False,
    **opt_kwargs,
):
    """
    Minimize a least-squares linear problem.

    Parameters
    ----------
    A : np.ndarray
        Coefficients of the linear equation.
    B : np.ndarray
        Right-hand side of the linear equation.
    Epsilon : np.ndarray, optional
        Error term for the optimization problem. If not provided, defaults to None.
    lb : np.ndarray, optional
        Lower bounds for the variables. If not provided, defaults to None.
    ub : np.ndarray, optional
        Upper bounds for the variables. If not provided, defaults to None.
    W : np.ndarray, optional
        Weights for the objective function. If not provided, defaults to None.
    K : np.ndarray, optional
        Transformation matrix for `A`, `lb`, `ub`, and `baseline`, by default None.
    baseline : np.ndarray, optional
        Bias values for the linear equation. If not provided, defaults to None.
    norm : np.ndarray, optional
        The norm to use for error calculation. If not provided, defaults to None.
    l2_eps : float, optional
        Tolerance for the least squares error. If not provided, defaults to `EPS_NP64`.
    L1 : float, optional
        Value for L1-norm constraint. If not provided, defaults to None.
    l1_eps : float, optional
        Tolerance for the L1-norm constraint. If not provided, defaults to `EPS_NP64`.
    n_jobs : int, optional
        Number of jobs to be used for parallel processing. If not provided, defaults to None.
    batch_size : int, optional
        Size of batches for parallel processing. If not provided, defaults to 1.
    verbose : int, optional
        Level of verbosity. If not provided, defaults to 0.
    return_pred : bool, optional
        If True, return predicted values. If not provided, defaults to False.
    **opt_kwargs
        Additional keyword arguments for the problem solver.

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, np.ndarray, np.ndarray]
        Coefficients of the solution, possibly predicted values, and possibly the weighted sum of the squares of the solution coefficients.

    Raises
    ------
    NameError
        If `Epsilon` is not array or 'heteroscedastic'.
    NotImplementedError
        If `n_jobs` is provided.
    RuntimeError
        If the optimization problem does not converge.
    """
    if n_jobs is not None:
        raise NotImplementedError("parallel jobs")

    # prepare parameters
    # NB: baseline gets removed from B
    A, B, lb, ub, W, baseline, batch_size = _prepare_parameters(
        A,
        B,
        lb,
        ub,
        W,
        K,
        baseline,
        batch_size,
    )

    # if norm is None perform normal fitting to obtain norm.
    if norm is None:
        # NB: baselines cancel
        _, B0 = lsq_linear(
            A,
            B,
            lb=lb,
            ub=ub,
            W=W,
            K=None,  # already prepared
            baseline=None,  # already prepared
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
            return_pred=True,
            **opt_kwargs,
        )
        norm = l2norm((W * B0 - W * B), axis=-1)

    total_delta = np.broadcast_to(np.atleast_1d(l2_eps + norm), B.shape[0])
    # Format Epsilon
    if Epsilon is None:
        Epsilon = A.copy() ** 2  # Epsilon is the variance -> squaring necessary
    elif isinstance(Epsilon, str):
        if Epsilon == "heteroscedastic":
            Epsilon = A.copy() ** 2  # Epsilon is the variance -> squaring necessary
        else:
            raise NameError(
                f"Epsilon must be array or `heteroscedastic`, but is `{Epsilon}`"
            )
    else:
        Epsilon = propagate_error(Epsilon, K)

    # L1-norm constraint implementation
    if L1 is None:
        l1_constraint = False
        L1 = np.zeros(B.shape[0])
    elif isinstance(L1, Number):
        l1_constraint = True
        L1 = np.array([L1] * B.shape[0])
    else:
        l1_constraint = True
        L1 = np.asarray(L1)
        assert L1.size == B.shape[0]

    # set up cvxpy problem
    A_, x_, w_, b_, constraints = _prepare_variables(A, B, lb, ub, W, batch_size)

    Epsilon_ = diagonal_stack(Epsilon, batch_size)
    t_delta_ = cp.Parameter((batch_size), pos=True)

    constraints.append(
        cp.norm2(
            cp.reshape(
                cp.multiply(A_, w_[:, None]) @ x_ - b_, (batch_size, A.shape[0])
            ),
            axis=1,
        )
        <= t_delta_
    )

    # variable and constraints
    if np.all(lb >= 0):
        xkwargs = {"pos": True}
    else:
        xkwargs = {}
    l1_ = cp.Parameter(batch_size, **xkwargs)
    if l1_constraint:
        if np.all(lb >= 0):
            constraints.extend(
                [
                    cp.sum(cp.reshape(x_, (batch_size, A.shape[1])), axis=1)
                    <= (l1_ + l1_eps),
                    cp.sum(cp.reshape(x_, (batch_size, A.shape[1])), axis=1)
                    >= (l1_ - l1_eps),
                ]
            )
        else:
            constraints.extend(
                [
                    cp.sum(cp.abs(cp.reshape(x_, (batch_size, A.shape[1]))), axis=1)
                    <= (l1_ + l1_eps),
                    cp.sum(cp.abs(cp.reshape(x_, (batch_size, A.shape[1]))), axis=1)
                    >= (l1_ - l1_eps),
                ]
            )

    # objective function
    objective = cp.Minimize(cp.sum(Epsilon_ @ x_**2))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dcp(dpp=True)

    # empty X
    X = np.zeros((B.shape[0], A.shape[-1]))
    last_batch_size = X.shape[0] % batch_size
    # iterate over batches - and pad last batch
    for idx, (b, w, t_delta, i), _ in batched_iteration(
        B.shape[0],
        (B, W, total_delta, L1),
        (),
        batch_size=batch_size,
        pad=True,
        verbose=verbose,
    ):
        w_.value = w
        b_.value = b * w  # ensures that objective is dpp compliant
        l1_.value = np.atleast_1d(i)
        if ((idx + 1) * batch_size) > X.shape[0]:
            t_delta = np.atleast_1d(t_delta).copy
            t_delta[last_batch_size:] = 1e10  # some big number
            t_delta_.value = t_delta
        else:
            t_delta_.value = np.atleast_1d(t_delta)

        problem.solve(**opt_kwargs)
        if not np.isfinite(problem.value):
            raise RuntimeError("Optimization did not converge.")

        x = x_.value.reshape(-1, A.shape[-1])

        if ((idx + 1) * batch_size) > X.shape[0]:
            X[idx * batch_size :] = x[:last_batch_size]
        else:
            X[idx * batch_size : (idx + 1) * batch_size] = x

    if return_pred:
        return X, predict_values(X, A, baseline), (X**2 @ Epsilon.T)
    return X


def lsq_linear_decomposition(
    A,
    B,
    n_layers=None,
    mask=None,
    lb=None,
    ub=None,
    W=None,
    lbp=0,
    ubp=1,
    K=None,
    baseline=None,
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
    **opt_kwargs,
):
    """Solves a linear decomposition problem with constraints using a variant of the NMF algorithm.

    Parameters
    ----------
    A : np.ndarray
        Input data matrix.
    B : np.ndarray
        Target data matrix.
    n_layers : int, optional
        Number of layers in the model, by default None.
    mask : np.ndarray, optional
        Mask array, by default None.
    lb : Union[np.ndarray, float], optional
        Lower bounds for the variables, by default None.
    ub : Union[np.ndarray, float], optional
        Upper bounds for the variables, by default None.
    W : np.ndarray, optional
        Weight matrix, by default None.
    lbp : int, optional
        Lower bound for the 'P' matrix in the NMF algorithm, by default 0.
    ubp : int, optional
        Upper bound for the 'P' matrix in the NMF algorithm, by default 1.
    K : np.ndarray, optional
        Transformation matrix for `A`, `lb`, `ub`, and `baseline`, by default None.
    baseline : np.ndarray, optional
        Baseline values, by default None.
    max_iter : int, optional
        Maximum number of iterations, by default 200.
    init_iter : int, optional
        Initial iterations, by default 1000.
    seed : int, optional
        Seed for random number generator, by default None.
    subsample : Union[float, str], optional
        Fraction of data to use for subsampling, by default None.
    verbose : int, optional
        Control the verbosity of the function's output, by default 0.
    solver : str, optional
        Solver object from the CVXPY library, by default cp.SCS.
    return_pred : bool, optional
        Whether to return predictions or not, by default False.
    equal_l1norm_constraint : bool, optional
        Whether to enforce equal l1-norm constraint or not, by default True.
    xtol : float, optional
        Tolerance for solution's change, by default 1e-8.
    ftol : float, optional
        Tolerance for objective function's change, by default 1e-8.
    opt_kwargs : dict
        Additional keyword arguments for the optimization solver.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Returns a tuple containing the decomposition matrices and
        if return_pred is True the predicted/fitted values
    """
    A, B, lb, ub, W, baseline = prepare_parameters_for_linear(
        A, B, lb, ub, W, K, baseline
    )
    B = B - baseline
    lbp = np.atleast_2d(lbp)
    ubp = np.atleast_2d(ubp)

    size = B.shape[0]
    inputs = A.shape[1]
    channels = B.shape[1]

    if subsample:
        if isinstance(subsample, str):
            if subsample == "fast":
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
        n_layers = A.shape[0] - 1
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
    if verbose:
        print("Initializing `P` with scikit-learn's NMF algorithm.")
    nmf = NMF(
        n_components=n_layers,
        random_state=seed,
        init=("random" if (n_layers > channels) else "nndsvda"),
        max_iter=init_iter,
        verbose=verbose,
    )
    P0 = nmf.fit(B.T).components_.T
    # rescaling initial `P`
    P0 = np.abs(P0) / np.max(np.abs(P0))  # range is 0-1
    P0 = P0 * (ubp - lbp) + lbp  # rescale to range
    # set inital P value for X problem
    Ppar.value = P0

    # p constraints
    p_constraints = [
        Pvar >= lbp,
        Pvar <= ubp,
    ]

    # x constraints
    x_constraints = []
    if np.any(mask == 0):
        x_constraints.append(Xvar[mask == 0] == 0)
    # all subframes have the same overall intensity -> equal l1norm
    if (n_layers > 1) and equal_l1norm_constraint:
        x_constraints.append(cp.diff(cp.sum(Xvar, axis=1)) == 0)
    if np.all(np.isfinite(lb)):
        x_constraints.append(Xvar >= np.atleast_2d(lb))
    if np.all(np.isfinite(ub)):
        x_constraints.append(Xvar <= np.atleast_2d(ub))

    # x objective function
    x_objective = cp.Minimize(cp.norm(cp.multiply(W, Ppar @ Xvar @ A.T - B), "fro"))
    x_problem = cp.Problem(x_objective, x_constraints)
    assert x_problem.is_dcp(dpp=True)

    # x objective function
    p_objective = cp.Minimize(cp.norm(cp.multiply(W, Pvar @ Xpar @ A.T - B), "fro"))
    p_problem = cp.Problem(p_objective, p_constraints)
    assert p_problem.is_dcp(dpp=True)

    if verbose:
        iters = tqdm(range(max_iter), desc="Decompose iteration", total=max_iter)
    else:
        iters = range(max_iter)

    # for linting purposes
    prev_vars = None
    prev_loss = None

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

        current_loss = np.linalg.norm(W * (P @ X @ A.T - B), ord="fro")
        # current variables
        current_vars = np.concatenate([np.ravel(X), np.ravel(P)])

        if n:
            # tolerance for terminatino by the change of the cost function
            if (prev_loss - current_loss) < (ftol * current_loss):
                if verbose:
                    print(
                        f"Tolerance criteria met for cost function; stopping at {n} iterations"
                    )
                break

            # Tolerance for termination by the change of the independent variables.
            if l2norm(prev_vars - current_vars) < (
                xtol * (xtol + l2norm(current_vars))
            ):
                if verbose:
                    print(
                        f"Tolerance criteria met for independen variables; stopping at {n} iterations."
                    )
                break

        # reassign previous loss
        prev_loss = current_loss
        prev_vars = current_vars

    else:
        warnings.warn(
            f"Maximum number of iterations {max_iter} reached. Increase it to improve convergence.",
            RuntimeWarning,
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
            Pvar >= lbp,
            Pvar <= ubp,
        ]
        p_objective = cp.Minimize(
            cp.norm(cp.multiply(Wtotal, Pvar @ X @ A.T - Btotal), "fro")
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
    A,
    B,
    lb=None,
    ub=None,
    W=None,
    K=None,
    baseline=None,
    neutral_point=None,
    verbose=0,
    delta_radius=EPS_NP64,
    delta_norm1=EPS_NP64,
    scale_w=1,
    adaptive_objective="unity",
    solver=cp.ECOS,
    return_pred=False,
    **opt_kwargs,
):
    """
    Adaptively solve a linear least-squares problem with bounds on the variables.

    Parameters
    ----------
    A : np.ndarray
        Input data matrix.
    B : np.ndarray
        Target data matrix.
    lb : Union[np.ndarray, float], optional
        Lower bounds for the variables, by default None.
    ub : Union[np.ndarray, float], optional
        Upper bounds for the variables, by default None.
    W : np.ndarray, optional
        Weight matrix, by default None.
    K : np.ndarray, optional
        Transformation matrix for `A`, `lb`, `ub`, and `baseline`, by default None.
    baseline : np.ndarray, optional
        Baseline values, by default None.
    neutral_point : np.ndarray, optional
        The neutral point used for scaling radius dimensions, by default None.
    verbose : int, optional
        Control the verbosity of the function's output, by default 0.
    delta_radius : float, optional
        Tolerance for radii constraint, by default EPS_NP64.
    delta_norm1 : float, optional
        Tolerance for intensity/l1norm constraint, by default EPS_NP64.
    scale_w : int, optional
        Scaling factor for the weights, by default 1.
    adaptive_objective : str, optional
        Objective of the adaptive algorithm, could be 'unity' or 'max', by default "unity".
    solver : str, optional
        Solver object from the CVXPY library, by default cp.ECOS.
    return_pred : bool, optional
        Whether to return predictions or not, by default False.
    opt_kwargs : dict
        Additional keyword arguments for the optimization solver.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
        The optimized variables, scales and optionally the predictions.

    Raises
    ------
    NameError
        If the adaptive_objective is not None, 'unity', or 'max'.
    RuntimeError
        If the optimization did not converge.
    """

    A, B, lb, ub, W, baseline = prepare_parameters_for_linear(
        A, B, lb, ub, W, K, baseline
    )
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
        xkwargs = {"pos": True}
    else:
        xkwargs = {}
    X = cp.Variable((size, inputs), **xkwargs)
    scales = cp.Variable(2, pos=True)

    B_pred = X @ A.T + baseline[None]

    # capture prediction minus the neutral point for that vector scaled by intensity
    radii_vec_pred = B_pred - cp.multiply(neutral_points, scales[0])
    # actual radius scalled by scalar radius
    radii_vec_actual = cp.multiply(scales[1], Brad)
    int_pred = cp.sum(B_pred, axis=1)
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
        ),
    ] + bound_constraints
    # scale things as little as possible
    if (adaptive_objective == "unity") or (adaptive_objective is None):
        objective = cp.Minimize(cp.sum_squares(cp.multiply(scale_w, (scales - 1))))
    elif adaptive_objective == "max":
        objective = cp.Maximize(cp.sum(cp.multiply(scale_w, scales)))
    else:
        raise NameError(
            f"adaptive_objective must be None, `unity`, or `max`, but is `{adaptive_objective}`."
        )
    # problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=bool(verbose), **opt_kwargs)
    if not np.isfinite(problem.value):
        raise RuntimeError("Optimization did not converge.")

    X, scales = X.value, scales.value
    if return_pred:
        return X, scales, predict_values(X, A, baseline)

    return X, scales
