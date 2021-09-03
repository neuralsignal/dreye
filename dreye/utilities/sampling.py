"""
Even-ish sampling
"""

import numpy as np
from scipy.special import factorial
from numpy.random import default_rng
from numpy.linalg import det
from scipy.spatial import Delaunay, ConvexHull
from scipy.stats import dirichlet

from dreye.utilities.common import is_integer
from dreye.utilities import barycentric_to_cartesian_transformer
from dreye.utilities.convex import in_hull


def dist_in_hull(points, n, seed=None):
    """
    Sampling uniformly from convex hull as given by points.

    From: https://stackoverflow.com/questions/59073952/how-to-get-uniformly-distributed-points-in-convex-hull
    """
    rng = default_rng(seed)

    dims = points.shape[-1]
    hull = points[ConvexHull(points).vertices]
    deln = hull[Delaunay(hull).simplices]

    # volume of each simplex
    vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)
    # indices of each simplex (weighted by the volume of the simplex)
    sample = rng.choice(len(vols), size=n, p=vols/vols.sum())

    return np.einsum(
        'ijk, ij -> ik', 
        deln[sample], 
        dirichlet.rvs([1]*(dims + 1), size=n, random_state=rng)
    )


def phi(d, x=2.0, n_iter=10):
    for _ in range(n_iter):
        x = (1+x) ** (1/(d+1))
    return x


def create_spaced_samples(n, d=3, simplex=True, append_pure=False, seed=0):
    """
    Create evenly spaces samples

    Parameters
    ----------
    n : int
        Number of samples.
    d : int
        Dimensionality of samples
    simplex : bool
        Create samples in d-simplex.
    append_pure : bool
        Append samples where each dimension/feature is 1 and all others are 0.
    seed : int or float between [0, 1]
        Float between zero and one. If seed is of type `int`, then a 
        pseudorandom value is chosen from a uniform distribution using this seed number.
    """
    if seed is None:
        seed = default_rng().random()
    elif is_integer(seed):
        seed = default_rng(seed).random()
    else:
        assert seed >= 0, "seed must be bigger or equal to zero"
        assert seed <= 1, "seed must be smaller or equal to one"
    
    if simplex:
        d = d - 1
        n = (n if d == 2 else n * factorial(d))

    g = phi(d)
    alpha = (
        (1/g) ** (np.arange(d)+1)
    ) % 1
    z = (
        seed
        + alpha * (np.arange(n)+1)[:, None]
    ) % 1

    if not simplex:
        return z

    A = barycentric_to_cartesian_transformer(d+1)
    Ahat = A[1:]
    # new points in reduced space
    zt = z @ Ahat

    if d == 2:
        ztrig = zt.copy()
        ztrig[z.sum(1) > 1] = (
            1 - z[z.sum(1) > 1]
        ) @ Ahat
    else:  # TODO analytic solution for mapping
        insimplex = in_hull(zt, A)
        ztrig = zt[insimplex]

    # invert points
    ztrig = ztrig @ np.linalg.inv(Ahat)
    ztrig = np.hstack([1 - ztrig.sum(1, keepdims=True), ztrig])

    if append_pure:
        pure = np.eye(d+1)
        ztrig = np.vstack([ztrig, pure])
    return ztrig