"""
Sampling functions

References
----------
sample_in_hull:
    adapted from https://stackoverflow.com/questions/59073952/how-to-get-uniformly-distributed-points-in-convex-hull
"""

from itertools import product

import numpy as np
from scipy.stats import dirichlet, qmc
from numpy.linalg import det
from numpy.random import default_rng, Generator
from scipy.spatial import ConvexHull, Delaunay

from dreye.api.utils import l1norm


def sample_in_hull(P, n, seed=None, engine=None, qhull_options=None):
    """Sampling uniformly from convex hull as given by points.

    Parameters
    ----------
    P : [type]
        [description]
    n : [type]
        [description]
    seed : [type], optional
        [description], by default None
    engine : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    TypeError
        [description]
    NameError
        [description]
    TypeError
        [description]
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
    # deln = P[Delaunay(P, qhull_options=qhull_options).simplices]

    # volume of each simplex
    vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)
    
    if engine is None:
        # indices of each simplex (weighted by the volume of the simplex)
        sample = rng.choice(len(vols), size=n, p=vols/vols.sum())
        probs =  dirichlet.rvs([1]*(dims + 1), size=n, random_state=rng)
        
    else:
        if isinstance(engine, str):
            if engine == 'Sobol':
                engine = qmc.Sobol(dims+1, seed=rng)
            elif engine == 'Halton':
                engine = qmc.Halton(dims+1, seed=rng)
            elif engine == 'LHC':
                engine = qmc.LatinHypercube(dims+1, seed=rng)
            else:
                raise NameError(f"engine `{engine}` does not exist in `scipy.stats.qmc`.")
        elif isinstance(engine, qmc.QMCEngine):
            pass
        else:
            raise TypeError(f"engine wrong type `{type(engine)}`.")
        
        pvals = vols/vols.sum()
        mult_qmc = qmc.MultinomialQMC(
            pvals, engine=type(engine)(1, seed=rng), seed=rng
        )
        counts = mult_qmc.random(n)
        # init probs array
        probs = np.zeros((n, dims+1))
        total = 0
        # get probabilities for each simplex
        for count in counts:
            probs_ = engine.random(count)
            # engine.reset()
            probs[total:count+total] = probs_ / l1norm(probs_, axis=-1, keepdims=True)
            total += count
        # create and shuffle sample
        sample = np.repeat(np.arange(len(vols)), counts)
    
    # vertices weighted by probability simplex
    return np.einsum(
        'ijk, ij -> ik', 
        deln[sample], 
        probs
    )
    

def d_equally_spaced(n, d, one_inclusive=True):
    """Creates n^d equally samples in d dimension

    Parameters
    ----------
    n : [type]
        [description]
    d : [type]
        [description]
    one_inclusive : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """
    if one_inclusive:
        x = np.linspace(0, 1, n)
    else:
        x = np.arange(0, 1, 1/n)
    return np.array(list(product(*[x]*d)))