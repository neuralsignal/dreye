"""
Future functions that replicates common in scipy
"""

# import jax.numpy as jnp

# def prepare_bounds(bounds, n):
#     lb, ub = [jnp.asarray(b, dtype=float) for b in bounds]

#     if lb.ndim == 0:
#         lb = jnp.resize(lb, n)

#     if ub.ndim == 0:
#         ub = jnp.resize(ub, n)

#     return lb, ub


# def in_bounds(x, lb, ub):
#     """Check if a point lies within bounds."""
#     return jnp.all((x >= lb) & (x <= ub))


# def compute_grad(J, f):
#     """Compute gradient of the least-squares cost function."""
#     return J.T.dot(f)


# def reflective_transformation(y, lb, ub):
#     """Compute reflective transformation and its gradient."""
#     if in_bounds(y, lb, ub):
#         return y, jnp.ones_like(y)

#     lb_finite = jnp.isfinite(lb)
#     ub_finite = jnp.isfinite(ub)

#     x = y.copy()
#     g_negative = jnp.zeros_like(y, dtype=bool)

#     mask = lb_finite & ~ub_finite
#     x = x.at[mask].set(jnp.maximum(y[mask], 2 * lb[mask] - y[mask]))
#     g_negative = g_negative.at[mask].set(y[mask] < lb[mask])

#     mask = ~lb_finite & ub_finite
#     x = x.at[mask].set(jnp.minimum(y[mask], 2 * ub[mask] - y[mask]))
#     g_negative = g_negative.at[mask].set(y[mask] > ub[mask])

#     mask = lb_finite & ub_finite
#     d = ub - lb
#     t = jnp.remainder(y[mask] - lb[mask], 2 * d[mask])
#     x = x.at[mask].set(lb[mask] + jnp.minimum(t, 2 * d[mask] - t))
#     g_negative = g_negative.at[mask].set(t > d[mask])

#     g = jnp.ones_like(y)
#     g = g.at[g_negative] = -1

#     return x, g


# def make_strictly_feasible(x: jnp.ndarray, lb: jnp.ndarray, ub: jnp.ndarray, rstep=1e-10):
#     """Shift a point to the interior of a feasible region.
#     Each element of the returned vector is at least at a relative distance
#     `rstep` from the closest bound. If ``rstep=0`` then `np.nextafter` is used.
#     """
#     x_new = x.copy()

#     active = find_active_constraints(x, lb, ub, rstep)
#     lower_mask = jnp.equal(active, -1)
#     upper_mask = jnp.equal(active, 1)

#     if rstep == 0:
#         x_new = x_new.at[lower_mask].set(jnp.nextafter(lb[lower_mask], ub[lower_mask]))
#         x_new = x_new.at[upper_mask].set(jnp.nextafter(ub[upper_mask], lb[upper_mask]))
#     else:
#         x_new = x_new.at[lower_mask].set(
#             lb[lower_mask] +
#             rstep * jnp.maximum(1.0, jnp.abs(lb[lower_mask]))
#         )
#         x_new = x_new.at[upper_mask].set(
#             ub[upper_mask] -
#             rstep * jnp.maximum(1.0, jnp.abs(ub[upper_mask]))
#         )

#     tight_bounds = (x_new < lb) | (x_new > ub)
#     x_new = x_new.at[tight_bounds].set(
#         0.5 * (lb[tight_bounds] + ub[tight_bounds])
#     )

#     return x_new


# def find_active_constraints(x, lb, ub, rtol=1e-10):
#     """Determine which constraints are active in a given point.
#     The threshold is computed using `rtol` and the absolute value of the
#     closest bound.
#     Returns
#     -------
#     active : ndarray of int with shape of x
#         Each component shows whether the corresponding constraint is active:
#              *  0 - a constraint is not active.
#              * -1 - a lower bound is active.
#              *  1 - a upper bound is active.
#     """
#     active = jnp.zeros_like(x, dtype=int)

#     if rtol == 0:
#         active = active.at[x <= lb].set(-1)
#         active = active.at[x >= ub].set(1)
#         return active

#     lower_dist = x - lb
#     upper_dist = ub - x

#     lower_threshold = rtol * jnp.maximum(1.0, jnp.abs(lb))
#     upper_threshold = rtol * jnp.maximum(1.0, jnp.abs(ub))

#     lower_active = (jnp.isfinite(lb) &
#                     (lower_dist <= jnp.minimum(upper_dist, lower_threshold)))
#     active = active.at[lower_active].set(-1)

#     upper_active = (jnp.isfinite(ub) &
#                     (upper_dist <= jnp.minimum(lower_dist, upper_threshold)))
#     active = active.at[upper_active].set(1)

#     return active


# def CL_scaling_vector(x, g, lb, ub):
#     """Compute Coleman-Li scaling vector and its derivatives.
#     Components of a vector v are defined as follows:
#     ::
#                | ub[i] - x[i], if g[i] < 0 and ub[i] < np.inf
#         v[i] = | x[i] - lb[i], if g[i] > 0 and lb[i] > -np.inf
#                | 1,           otherwise
#     According to this definition v[i] >= 0 for all i. It differs from the
#     definition in paper [1]_ (eq. (2.2)), where the absolute value of v is
#     used. Both definitions are equivalent down the line.
#     Derivatives of v with respect to x take value 1, -1 or 0 depending on a
#     case.
#     Returns
#     -------
#     v : ndarray with shape of x
#         Scaling vector.
#     dv : ndarray with shape of x
#         Derivatives of v[i] with respect to x[i], diagonal elements of v's
#         Jacobian.
#     References
#     ----------
#     .. [1] M.A. Branch, T.F. Coleman, and Y. Li, "A Subspace, Interior,
#            and Conjugate Gradient Method for Large-Scale Bound-Constrained
#            Minimization Problems," SIAM Journal on Scientific Computing,
#            Vol. 21, Number 1, pp 1-23, 1999.
#     """
#     v = jnp.ones_like(x)
#     dv = jnp.zeros_like(x)

#     mask = (g < 0) & jnp.isfinite(ub)
#     v = v.at[mask].set(ub[mask] - x[mask])
#     dv = dv.at[mask].set(-1.0)

#     mask = (g > 0) & jnp.isfinite(lb)
#     v = v.at[mask].set(x[mask] - lb[mask])
#     dv = dv.at[mask].set(1.0)

#     return v, dv