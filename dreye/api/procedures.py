"""
Various fitting procedures
"""

# X: fitted weights - array (samples x inputs)
# x0: initial guess for weights - array (inputs)
# A: linear transformation - array (outputs x inputs)
# Y: linear targets - array (samples x outputs)
# S: normalized variances - array (outputs x inputs)
# l: lower bound - array (inputs)
# u: upper bound - array (inputs)
# w: weighting - array (outputs OR shape of returned value of `f`)
# f: nonlinear transformation - callable accepts an array and returns an array


def lsq_linear(
    A, Y, l=None, u=None, w=None
):
    pass


def nnls(
    A, Y, w=None
):
    pass


def lsq_nonlinear(
    A, Y, 
    x0, 
    f=None,
    l=None, 
    u=None, 
    w=None
):
    pass


def minimize_var1(
    A, Y, S,
    x0, delta, 
    f=None,
    l=None, 
    u=None, 
    w=None
):
    pass


def minimize_var2(
    A, Y, sampled_As,
    x0, delta,
    f=None,
    l=None, 
    u=None, 
    w=None
):
    pass


def minimize_constrained(
    A, Y, x0, 
    
):
    pass


def nonnegative_decomposition(
    Y, ndim
):
    pass
