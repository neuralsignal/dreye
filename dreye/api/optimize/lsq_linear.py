"""
The scipy lsq_linear algorithm implemented using jax.numpy
"""

import warnings
from scipy import optimize
import numpy as np
import cvxpy as cp

from dreye.api.optimize.parallel import batched_iteration, diagonal_stack, concat
from dreye.api.optimize.utils import FAILURE_MESSAGE, prepare_parameters_for_linear
    


def lsq_linear(
    A, B, 
    lb=None, ub=None, W=None,
    K=None, baseline=None, 
    error='raise', 
    n_jobs=None, 
    batch_size=1,
    verbose=0, 
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

    return X


def lsq_linear_cp(
    A, B, 
    lb=None, ub=None, W=None,
    K=None, baseline=None, 
    n_jobs=None, 
    batch_size=1,
    verbose=0, 
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
    x_ = cp.Variable(A_.shape[1])
    constraints = [
        x_ >= lb_, 
        x_ <= ub_
    ]
    
    # objective function
    objective = cp.Minimize(
        cp.sum_squares((cp.multiply(A_, w_[:, None]) @ x_ - b_))
    )
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

        x = x_.value.reshape(-1, A.shape[-1])

        if ((idx+1) * batch_size) > X.shape[0]:
            X[idx * batch_size:] = x[:last_batch_size]
        else:
            X[idx * batch_size:(idx+1) * batch_size] = x

    return X


# import jax.numpy as jnp
# import jax.scipy as jsc
# from scipy.optimize._lsq.givens_elimination import givens_elimination
# from scipy.optimize._lsq.lsq_linear import TERMINATION_MESSAGES

# from dreye.constants.common import EPS_NP64
# from dreye.api.optimize.utils import CL_scaling_vector, compute_grad, in_bounds, make_strictly_feasible, reflective_transformation

# def _lsq_linear(
#     A, b, 
#     lb, ub, 
#     max_iter=100, 
#     tol=1e-10
# ):
#     "Solve a linear least-squares with bound on the variables."

#     x_lsq = jnp.linalg.lstsq(A, b, rcond=-1)[0]

#     if in_bounds(x_lsq, lb, ub):
#         return x_lsq

#     # trf linear algorithm
#     m, n = A.shape
#     x, _ = reflective_transformation(x_lsq, lb, ub)
#     x = make_strictly_feasible(x, lb, ub, rstep=0.1)

#     QT, R, perm = jsc.linalg.qr(A, mode='economic', pivoting=True)
#     QT = QT.T

#     if m < n:
#         R = jnp.vstack((R, jnp.zero((n - m, n))))

#     QTr = jnp.zeros(n)
#     k = min(m, n)

#     r = A.dot(x) - b
#     g = compute_grad(A, r)
#     cost = 0.5 * jnp.dot(r, r)
#     initial_cost = cost

#     termination_status = None
#     step_norm = None
#     cost_change = None

#     for iteration in range(max_iter):
#         v, dv = CL_scaling_vector(x, g, lb, ub)
#         g_scaled = g * v
#         g_norm = jnp.linalg.norm(g_scaled, ord=jnp.inf)
        
#         if g_norm < tol:
#             break

#         diag_h = g * dv
#         diag_root_h = diag_h ** 0.5
#         d = v ** 0.5
#         g_h = d * g

#         # A_h = right_multiplied_operator(A, d)
#         QTr = QTr.at[:k].set(QT.dot(r))
#         p_h = -regularized_lsq_with_qr(
#             m, n, R * d[perm], QTr, perm,
#             diag_root_h, copy_R=False)

#         p = d * p_h
#         p_dot_g = jnp.dot(p, g)

#         if p_dot_g > 0:
#             raise RuntimeError(TERMINATION_MESSAGES[-1])

#         theta = 1 - min(0.005, g_norm)
#         step = select_step(x, A_h, g_h, diag_h, p, p_h, d, lb, ub, theta)
#         cost_change = -evaluate_quadratic(A, g, step)


#     else:
#         raise RuntimeError(TERMINATION_MESSAGES[0])


# def regularized_lsq_with_qr(m, n, R, QTb, perm, diag, copy_R=True):
#     """Solve regularized least squares using information from QR-decomposition.
#     The initial problem is to solve the following system in a least-squares
#     sense:
#     ::
#         A x = b
#         D x = 0
#     where D is diagonal matrix. The method is based on QR decomposition
#     of the form A P = Q R, where P is a column permutation matrix, Q is an
#     orthogonal matrix and R is an upper triangular matrix.
#     Parameters
#     ----------
#     m, n : int
#         Initial shape of A.
#     R : ndarray, shape (n, n)
#         Upper triangular matrix from QR decomposition of A.
#     QTb : ndarray, shape (n,)
#         First n components of Q^T b.
#     perm : ndarray, shape (n,)
#         Array defining column permutation of A, such that ith column of
#         P is perm[i]-th column of identity matrix.
#     diag : ndarray, shape (n,)
#         Array containing diagonal elements of D.
#     Returns
#     -------
#     x : ndarray, shape (n,)
#         Found least-squares solution.
#     """
#     if copy_R:
#         R = R.copy()
#     v = QTb.copy()

#     givens_elimination(R, v, diag[perm])

#     abs_diag_R = jnp.abs(jnp.diag(R))
#     threshold = EPS_NP64 * float(max(m, n)) * jnp.max(abs_diag_R)
#     nns, = jnp.nonzero(abs_diag_R > threshold)

#     R = R[jnp.ix_(nns, nns)]
#     v = v[nns]

#     x = jnp.zeros(n)
#     x[perm[nns]] = jsc.linalg.solve_triangular(R, v)

#     return x



# def select_step(x, A_h, g_h, c_h, p, p_h, d, lb, ub, theta):
#     """Select the best step according to Trust Region Reflective algorithm."""
#     if in_bounds(x + p, lb, ub):
#         return p

#     p_stride, hits = step_size_to_bound(x, p, lb, ub)
#     r_h = jnp.copy(p_h)
#     r_h = r_h.at[hits.astype(bool)]
#     r_h[hits.astype(bool)] *= -1
#     r = d * r_h

#     # Restrict step, such that it hits the bound.
#     p *= p_stride
#     p_h *= p_stride
#     x_on_bound = x + p

#     # Find the step size along reflected direction.
#     r_stride_u, _ = step_size_to_bound(x_on_bound, r, lb, ub)

#     # Stay interior.
#     r_stride_l = (1 - theta) * r_stride_u
#     r_stride_u *= theta

#     if r_stride_u > 0:
#         a, b, c = build_quadratic_1d(A_h, g_h, r_h, s0=p_h, diag=c_h)
#         r_stride, r_value = minimize_quadratic_1d(
#             a, b, r_stride_l, r_stride_u, c=c)
#         r_h = p_h + r_h * r_stride
#         r = d * r_h
#     else:
#         r_value = np.inf

#     # Now correct p_h to make it strictly interior.
#     p_h *= theta
#     p *= theta
#     p_value = evaluate_quadratic(A_h, g_h, p_h, diag=c_h)

#     ag_h = -g_h
#     ag = d * ag_h
#     ag_stride_u, _ = step_size_to_bound(x, ag, lb, ub)
#     ag_stride_u *= theta
#     a, b = build_quadratic_1d(A_h, g_h, ag_h, diag=c_h)
#     ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride_u)
#     ag *= ag_stride

#     if p_value < r_value and p_value < ag_value:
#         return p
#     elif r_value < p_value and r_value < ag_value:
#         return r
#     else:
#         return ag
