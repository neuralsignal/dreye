"""
Wrapper for scipy.optimize.minimize
for automatic differentiation
"""

from scipy.optimize import minimize

from dreye.constants.packages import TORCH, JAX


# def torch_minimize(
#     fun, x0, args=(), method=None, jac=None, hess=None,
#     hessp=None, bounds=None, constraints=(), tol=None,
#     callback=None, options=None
# ):
#     f"""
#     Wrapper for `scipy.optimize.minimize` function 
#     using pytorch autograd tools. `x0` and `fun`

#     Copied scipy docstring for `scipy.optimize.minimize`:
#     {minimize.__doc__}
#     """