"""
Batch and parallelize objective functions
"""

from tqdm import tqdm
import numpy as np
from scipy.linalg import block_diag


def diagonal_stack(A, n, pad_size=0, pad=False):
    """[summary]

    Parameters
    ----------
    A : [type]
        [description]
    n : [type]
        [description]
    pad_size : int, optional
        [description], by default 0
    pad : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    if pad:
        return block_diag(*([A]*n + [np.zeros(A.shape)*pad_size]))
    else:
        return block_diag(*[A]*n)


def concat(x, n, pad_size=0, pad=False):
    """[summary]

    Parameters
    ----------
    x : [type]
        [description]
    n : [type]
        [description]
    pad_size : int, optional
        [description], by default 0
    pad : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    if pad:
        return np.concatenate([x]*n + [np.zeros(x.shape)]*pad_size)
    else:
        return np.concatenate([x]*n)


def batch_arrays(arrays, batch_size, pad_size=0, pad=False):
    """[summary]

    Parameters
    ----------
    arrays : [type]
        [description]
    batch_size : [type]
        [description]
    pad_size : int, optional
        [description], by default 0
    pad : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    return [
        concat(arr, batch_size, pad_size, pad)
        if arr.ndim == 1
        else
        diagonal_stack(arr, batch_size, pad_size, pad)
        for arr in arrays
    ]


def ravel_iarrays(iter_arrays, batch_size, idx):
    """[summary]

    Parameters
    ----------
    iter_arrays : [type]
        [description]
    batch_size : [type]
        [description]
    idx : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return [
        arr[idx*batch_size:(idx+1)*batch_size].ravel()
        for arr in iter_arrays
    ]


def ravel_last_iarrays(iter_arrays, last_batch_size, pad_size=0, pad=False):
    """[summary]

    Parameters
    ----------
    iter_arrays : [type]
        [description]
    last_batch_size : [type]
        [description]
    pad_size : int, optional
        [description], by default 0
    pad : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    return [
        np.concatenate([
            arr[-last_batch_size:].ravel(), 
            np.zeros((pad_size,) + arr.shape[1:]).ravel()
        ])
        if pad
        else arr[-last_batch_size:].ravel()
        for arr in iter_arrays
    ]


def batched_iteration(
    n_samples,  # number of samples
    iter_arrays,  # iterate (n_samples) and ravel these arrays
    arrays,  # concat these arrays so that it is properly sized
    batch_size=1, 
    pad=False, 
    verbose=False
):
    """[summary]

    Parameters
    ----------
    n_samples : [type]
        [description]
    pad : bool, optional
        [description], by default False
    verbose : bool, optional
        [description], by default False

    Yields
    -------
    [type]
        [description]
    """
    if batch_size == 1:
        if verbose:
            iters = tqdm(enumerate(zip(*iter_arrays)), desc="Iterations", total=n_samples)
        else:
            iters = enumerate(zip(*iter_arrays))
        for idx, iarrays in iters:
            yield (idx, iarrays, arrays)

    else:
        if verbose:
            iters = tqdm(range(n_samples // batch_size), desc='Iterations', total=n_samples//batch_size)
        else:
            iters = range(n_samples // batch_size)
        
        batched_arrays = batch_arrays(arrays, batch_size)
        for idx in iters:
            raveled_iarrays = ravel_iarrays(iter_arrays, batch_size, idx)
            yield (idx, raveled_iarrays, batched_arrays)
        
        last_batch_size = n_samples % batch_size
        if last_batch_size:
            batched_arrays = batch_arrays(
                arrays, last_batch_size, 
                batch_size-last_batch_size, pad=pad
            )
            raveled_iarrays = ravel_last_iarrays(
                iter_arrays, last_batch_size, 
                batch_size-last_batch_size, pad=pad
            )
            yield (idx+1, raveled_iarrays, batched_arrays)