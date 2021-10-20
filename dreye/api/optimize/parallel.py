"""
Batch and parallelize objective functions
"""

import numpy as np
from scipy.linalg import block_diag


def diagonal_stack(A, n, pad_size=0, pad=False):
    if pad:
        return block_diag(*([A]*n + [np.zeros(A.shape)*pad_size]))
    else:
        return block_diag(*[A]*n)


def concat(x, n, pad_size=0, pad=False):
    if pad:
        return np.concatenate([x]*n + [np.zeros(x.shape)]*pad_size)
    else:
        return np.concatenate([x]*n)


def batch_arrays(arrays, batch_size, pad_size=0, pad=False):
    return [
        concat(arr, batch_size, pad_size, pad)
        if arr.ndim == 1
        else
        diagonal_stack(arr, batch_size, pad_size, pad)
        for arr in arrays
    ]


def ravel_iarrays(iter_arrays, batch_size, idx):
    return [
        arr[idx*batch_size:(idx+1)*batch_size].ravel()
        for arr in iter_arrays
    ]


def ravel_last_iarrays(iter_arrays, last_batch_size, pad_size=0, pad=False):
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
    pad=False
):
    # TODO batch last batch with zeros for efficiency with jit
    # TODO 2^n batching is most efficient for gpus
    if batch_size == 1:
        for idx, iarrays in enumerate(zip(*iter_arrays)):
            yield (idx, iarrays, arrays)

    else:
        batched_arrays = batch_arrays(arrays, batch_size)
        for idx in range(n_samples // batch_size):
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