"""
Non-linear least squares

Currently uses scipy.optimize, but in the future will use
jax optimization to increase speed.
"""

import warnings 
import numpy as np
from scipy import optimize
from scipy.linalg import block_diag

try:
    JAX = True
    import jax.numpy as jnp
    from jax import jit, jacfwd, jacrev, grad, vmap
except (ImportError, RuntimeError):
    JAX = False

from dreye.api.optimize.parallel import batch_arrays, batched_iteration
from dreye.api.optimize.utils import get_batch_size, prepare_parameters_for_linear, FAILURE_MESSAGE, replace_numpy
from dreye.api.optimize.lsq_linear import lsq_linear
from dreye.api.utils import get_prediction


# B is assumed to have the affine transform already applied
# jacfwd uses forward-mode automatic differentiation, 
# which is more efficient for “tall” Jacobian matrices 
# (many functions/residuals/channels), 
# while jacrev uses reverse-mode, which is more efficient 
# for “wide” (many parameters to fit) Jacobian matrices.


class LeastSquaresObjective:

    def __init__(self, nb, ne, nx, nonlin, nonlin_prime=None, jac_prime=False):
        self.nonlin = nonlin
        self.nonlin_prime = nonlin_prime
        self.jac_prime = jac_prime  # nonlin prime returns the jacobian
        self.nb = nb  # handle batching
        self.ne = ne  # handle batching
        self.nx = nx  # handle batching

    def objective(self, x, A, e, w, baseline):
        if self.jac_prime:
            return w * (
                self.nonlin((A @ x + baseline).reshape(-1, self.nb)) 
                - e.reshape(-1, self.ne)
            ).ravel()
        else:
            return w * (self.nonlin(A @ x + baseline) - e)

    def objective_jac(self, x, A, e, w, baseline):
        if self.jac_prime:
            # TODO check if this is correct -> broadcasting?
            # nonlin prime would return (batch, ne, nb)
            return (
                w[..., None] 
                * 
                block_diag(
                    *[
                        # TODO vectorize
                        self.nonlin_prime(b)
                        for b in (A @ x + baseline).reshape(-1, self.nb)
                    ]
                ) 
                @ A
            )
        else:
            return w[..., None] * self.nonlin_prime(A @ x + baseline)[..., None] * A


def lsq_nonlinear(
    A, B, X0=None,
    lb=None, ub=None, W=None,
    K=None, baseline=None, 
    nonlin=None, 
    nonlin_prime=None,
    jac_prime=False,
    error='raise', 
    n_jobs=None, 
    batch_size=1,
    autodiff=True,
    verbose=0, 
    return_pred=False,
    linopt_kwargs={},
    **opt_kwargs
):
    """
    Nonlinear least-squares. 

    A (channels x inputs)
    B (samples x channels)
    K (channels) or (channels x channels)
    baseline (channels)
    ub (inputs)
    lb (inputs)
    w (channels)
    """
    if not JAX:
        raise RuntimeError("JAX not properly installed in environment in order to perform nonlinear least squares optimization.")
    A, B, lb, ub, W, baseline = prepare_parameters_for_linear(A, B, lb, ub, W, K, baseline)
    # get linear X0
    if X0 is None:
        # TODO-later test optimality
        linopt_kwargs['batch_size'] = max(1, int(2**10 * 1/np.prod(A.shape)))
        linopt_kwargs['n_jobs'] = None
        X0 = lsq_linear(
            A=A, B=B, lb=lb, ub=ub, W=W, 
            # K=K, -> transformation already applied above 
            baseline=baseline, return_pred=False,
            **linopt_kwargs
        )
    else:
        X0 = np.asarray(X0)
        assert X0.shape[0] == B.shape[0]
        assert A.shape[1] == X0.shape[1]
        
    # TODO-later if all in gamut skip whole fitting procedure

    if nonlin is None:
        if return_pred:
            return X0, (X0 @ A.T + baseline)
        return X0

    X0 = np.clip(X0, lb, ub)
    # get E
    E = nonlin(B)

    batch_size = get_batch_size(batch_size, B.shape[0])

    # setup function and jacobian
    if autodiff:
        if nonlin_prime is None:
            jnp_nonlin = replace_numpy(jnp, nonlin)

            if not jac_prime:
                jnp_nonlin_prime = jit(vmap(grad(jnp_nonlin)))
            
            elif (A.shape[1] * 2) > B.shape[1]:
                jnp_nonlin_prime = jit(jacrev(jnp_nonlin))
            
            else:
                jnp_nonlin_prime = jit(jacfwd(jnp_nonlin))

            def nonlin_prime(x):
                return np.asarray(jnp_nonlin_prime(x))

        lsq = LeastSquaresObjective(B.shape[-1], E.shape[-1], A.shape[-1], nonlin, nonlin_prime, jac_prime)
        opt_kwargs['jac'] = lsq.objective_jac
    else:
        lsq = LeastSquaresObjective(B.shape[-1], E.shape[-1], A.shape[-1], nonlin)

    if n_jobs is not None:
        raise NotImplementedError("parallel jobs")

    X = np.zeros((E.shape[0], A.shape[-1]))
    count_failure = 0
    for idx, (e, w, x0), (A_, baseline_, lb_, ub_) in batched_iteration(E.shape[0], (E, W, X0), (A, baseline, lb, ub), batch_size=batch_size, verbose=verbose):
        # TODO-later parallelizing
        # TODO-later padding
        # TODO-later test using sparse matrices when batching
        # TODO-later substitute with faster algorithm
        # TODO-later efficiency skipping in gamut solutions
        idx_slice = slice(idx * batch_size, (idx+1) * batch_size)
        # reshape resulting x
        x0 = x0.reshape(-1, A.shape[-1])
        # skip zero residual solutions
        res = nonlin(x0 @ A.T + baseline) - e.reshape(-1, E.shape[-1])
        in_gamut = np.isclose(res, 0).all(axis=-1)

        if np.all(in_gamut):
            # if all within gamut just assign to x0
            X[idx_slice] = x0
        
        elif ~np.any(in_gamut):
            # fit in nonlinear case
            result = optimize.least_squares(
                lsq.objective, 
                x0.ravel(), 
                args=(A_, e, w, baseline_), 
                bounds=(lb_, ub_),
                **opt_kwargs
            )
            X[idx_slice] = result.x.reshape(-1, A.shape[-1])

            count_failure += int(result.status <= 0)
        
        else:
            # assign within gamut x and fit the rest
            X[idx_slice][in_gamut] = x0[in_gamut]
            n_out = np.sum(~in_gamut)

            # rebatch out of gamut samples
            if n_out == 1:
                A_, baseline_, lb_, ub_ = A, baseline, lb, ub
            else:
                A_, baseline_, lb_, ub_ = batch_arrays([A, baseline, lb, ub], n_out)
            e, w = E[idx_slice][~in_gamut].ravel(), W[idx_slice][~in_gamut].ravel()

            result = optimize.least_squares(
                lsq.objective, 
                x0[~in_gamut].ravel(), 
                args=(A_, e, w, baseline_), 
                bounds=(lb_, ub_),
                **opt_kwargs
            )

            X[idx_slice][~in_gamut] = result.x.reshape(-1, A.shape[-1])
                
            count_failure += int(result.status <= 0)

    if count_failure:
        if error == "ignore":
            pass
        elif error == "warn":
            warnings.warn(FAILURE_MESSAGE.format(count=count_failure), RuntimeWarning)
        else:
            raise RuntimeError(FAILURE_MESSAGE.format(count=count_failure)) 

    if return_pred:
        B = get_prediction(X, A, baseline)
        return X, B
    return X


# TODO-later transfer old code for nonlinear cases
# TODO-later nonlinear variance minimization
# TODO-later nonlinear gamut adaptive minimization
# TODO-later nonlinear image decomposition