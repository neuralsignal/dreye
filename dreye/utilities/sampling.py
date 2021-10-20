"""
Even-ish sampling
"""

import numpy as np
from scipy.special import factorial
from numpy.random import default_rng
from numpy.linalg import det
from scipy.linalg import norm
from scipy.spatial import Delaunay, ConvexHull
from scipy.stats import dirichlet
from functools import reduce
from itertools import product

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


def project_angles_on_hypersphere(angles, r=1):
    """
    Project n-dimensional angles onto n+1 hypersphere with a specific radius
    """
    d = angles.shape[1] + 1
    cosx = np.cos(angles)
    sinx = np.sin(angles)
    X = np.zeros((angles.shape[0], d))
    X[:, 0] = cosx[:, 0]
    for i in range(1, d-1):
        X[:, i] = cosx[:, i] * reduce(lambda x, y: x * y, sinx[:, :i].T)
    X[:, d-1] = reduce(lambda x, y: x * y, sinx.T)
    r = np.atleast_1d(r)
    return (X * r[..., None, None]).reshape(-1, d)


def from_spherical_to_cartesian(Y):
    """
    Convert from spherical to cartesian coordinates
    """
    if Y.shape[-1] == 1:
        return Y
    
    # first dimension are the radii, the rest are angles
    angles = Y[..., 1:]
    r = Y[..., :1]
    
    # cosine and sine of all angle values
    cosx = np.cos(angles)
    sinx = np.sin(angles)
    
    X = np.zeros(Y.shape)
    # first dimension
    X[..., 0] = cosx[:, 0]
    # second to second to last
    for i in range(1, Y.shape[-1]-1):
        X[..., i] = cosx[:, i] * reduce(lambda x, y: x * y, sinx[:, :i].T)
    # last
    X[..., -1] = reduce(lambda x, y: x * y, sinx.T)
    
    return X * r


def from_cartesian_to_spherical(X):
    
    if X.shape[-1] == 1:
        return X
    
    r = norm(X, axis=-1, ord=2)
    Y = np.zeros(X.shape)
    Y[..., 0] = r
    d = X.shape[1] - 1
    zeros = (X == 0)
    
    for i in range(d-1):
        Y[..., i+1] = np.where(
            zeros[:, i:].all(-1),
            0,
            np.arccos(X[..., i]/norm(X[..., i:], axis=-1, ord=2))
        )
    
    lasty = np.arccos(X[..., -2]/norm(X[..., -2:]))
    Y[..., -1] = np.where(
        zeros[:, -2:].all(-1), 
        0, 
        np.where(
            X[..., -2] >= 0, 
            lasty, 
            2*np.pi - lasty
        )
    )
    return Y


def d_equally_spaced(n, d, one_inclusive=True):
    """
    Creates n^d equally samples in d dimension
    """
    if one_inclusive:
        x = np.linspace(0, 1, n)
    else:
        x = np.arange(0, 1, 1/n)
    return np.array(list(product(*[x]*d)))


def hypersphere_samples(n, d, r=1, seed=None):
    """
    Obtain well-spaced samples that lie on a hypersphere. 
    If $n^{1/(d-1)}$ is a whole number samples are created simply by using equally spaced
    angles in each dimension (from 0 to 2 * np.pi)
    """
    # samples x (d-1)
    if d == 2:
        angles = np.arange(0, 2*np.pi, 2*np.pi/n)[:, None]
    else:
        if np.isclose((n ** (1/(d-1))) % 1, 0):
            angles = d_equally_spaced(int(n**(1/(d-1))), d-1, one_inclusive=False)
        else:
            angles = create_spaced_samples(n, d-1, simplex=False, seed=seed) * 2 * np.pi
        
    return project_angles_on_hypersphere(angles, r=r)  