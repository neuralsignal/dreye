"""
Sampling functions
"""

import numpy as np
from scipy.stats import dirichlet
from numpy.linalg import det

from numpy.random import default_rng
from scipy.spatial import ConvexHull, Delaunay


# TODO sample in hull with quasi markov chains
def sample_in_hull(P, n, seed=None):
    """
    Sampling uniformly from convex hull as given by points.

    From: https://stackoverflow.com/questions/59073952/how-to-get-uniformly-distributed-points-in-convex-hull
    """
    rng = default_rng(seed)

    dims = P.shape[-1]
    hull = P[ConvexHull(P).vertices]
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