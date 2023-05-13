"""
Sampling functions

References
----------
sample_in_hull:
    adapted from https://stackoverflow.com/questions/59073952/how-to-get-uniformly-distributed-points-in-convex-hull
"""

from itertools import product
from typing import Optional, Union

import numpy as np
from numpy.random import default_rng, Generator
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import dirichlet, qmc

from dreye.api.utils import l1norm


def sample_in_hull(
    P: np.ndarray,
    n: int,
    seed: Optional[Union[int, Generator]] = None,
    engine: Optional[Union[str, qmc.QMCEngine]] = None,
    qhull_options: Optional[str] = None,
) -> np.ndarray:
    """
    Sampling uniformly from the convex hull defined by points P.

    Parameters
    ----------
    P : array-like (..., ndim)
        Array of points defining the convex hull.
    n : int
        Number of samples to generate.
    seed : int, Generator, or None, optional
        Random seed for generating samples, by default None.
    engine : str or QMCEngine, optional
        Quasi-Monte Carlo engine to use for sampling, by default None.
    qhull_options : str, optional
        Additional options to pass to the Qhull algorithm, by default None.

    Returns
    -------
    samples : array-like (n, ndim)
        Uniformly sampled points within the convex hull.

    Raises
    ------
    TypeError
        If the seed is not None, int, or Generator type.
    NameError
        If the engine string does not match any known QMC engines.
    TypeError
        If the engine is not a string or QMCEngine type.
    """
    if (seed is None) or isinstance(seed, int):
        rng = default_rng(seed)
    elif isinstance(seed, Generator):
        rng = seed
    else:
        raise TypeError("seed must be None, int, or `numpy.random.Generator` type.")

    dims = P.shape[-1]
    hull = P[ConvexHull(P, qhull_options=qhull_options).vertices]
    deln = hull[Delaunay(hull, qhull_options=qhull_options).simplices]

    # volume of each simplex
    vols = np.abs(
        np.linalg.det(deln[:, :dims, :] - deln[:, dims:, :])
    ) / np.math.factorial(dims)

    # Initialize probabilities
    probs = None

    if engine is None:
        # indices of each simplex (weighted by the volume of the simplex)
        sample_indices = rng.choice(len(vols), size=n, p=vols / vols.sum())
        probs = dirichlet.rvs([1] * (dims + 1), size=n, random_state=rng)
    else:
        # handle various engine types
        if isinstance(engine, str):
            if engine == "Sobol":
                engine = qmc.Sobol(dims + 1, seed=rng)
            elif engine == "Halton":
                engine = qmc.Halton(dims + 1, seed=rng)
            elif engine == "LHC":
                engine = qmc.LatinHypercube(dims + 1, seed=rng)
            else:
                raise NameError(
                    f"engine `{engine}` does not exist in `scipy.stats.qmc`."
                )
        elif isinstance(engine, qmc.QMCEngine):
            pass
        else:
            raise TypeError(f"engine wrong type `{type(engine)}`.")

        pvals = vols / vols.sum()
        mult_qmc = qmc.MultinomialQMC(pvals, 1, engine=type(engine)(1, seed=rng), seed=rng)
        counts = mult_qmc.random(n)
        # init probs array
        probs = np.zeros((n, dims + 1))
        total = 0
        # get probabilities for each simplex
        for count in counts:
            probs_ = engine.random(count)
            probs[total : count + total] = probs_ / l1norm(
                probs_, axis=-1, keepdims=True
            )
            total += count
        # create and shuffle sample
        sample_indices = np.repeat(np.arange(len(vols)), counts)

    # vertices weighted by probability simplex
    return np.einsum("ijk, ij -> ik", deln[sample_indices], probs)


def d_equally_spaced(n: int, d: int, one_inclusive: bool = True) -> np.ndarray:
    """
    Creates equally spaced samples in d dimensions.

    Parameters
    ----------
    n : int
        Number of samples per dimension.
    d : int
        Number of dimensions.
    one_inclusive : bool, optional
        If True, the samples will include the end point 1. If False, the samples will not
        include the end point 1. Default is True.

    Returns
    -------
    samples : array-like (n^d, d)
        The equally spaced samples in d dimensions.
    """
    if one_inclusive:
        x = np.linspace(0, 1, n)
    else:
        x = np.arange(0, 1, 1 / n)
    return np.array(list(product(*[x] * d)))
