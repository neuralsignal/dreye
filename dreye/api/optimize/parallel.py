"""
Module for batching and parallelizing objective functions
"""


from typing import List, Tuple
from tqdm import tqdm
import numpy as np
from scipy.linalg import block_diag


def diagonal_stack(A: np.ndarray, n: int, pad_size: int = 0, pad: bool = False) -> np.ndarray:
    """
    Stack the array diagonally.

    Parameters
    ----------
    A : np.ndarray
        Input 2D array.
    n : int
        Number of times to repeat A.
    pad_size : int, optional
        Size of padding to add, by default 0.
    pad : bool, optional
        Whether to add padding or not, by default False.

    Returns
    -------
    np.ndarray
        Diagonally stacked array.
    """
    if pad:
        return block_diag(*([A]*n + [np.zeros(A.shape)*pad_size]))
    else:
        return block_diag(*[A]*n)


def concat(x: np.ndarray, n: int, pad_size: int = 0, pad: bool = False) -> np.ndarray:
    """
    Concatenate array n times.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    n : int
        Number of times to repeat x.
    pad_size : int, optional
        Size of padding to add, by default 0.
    pad : bool, optional
        Whether to add padding or not, by default False.

    Returns
    -------
    np.ndarray
        Concatenated array.
    """
    if pad:
        return np.concatenate([x]*n + [np.zeros(x.shape)]*pad_size)
    else:
        return np.concatenate([x]*n)


def batch_arrays(arrays: List[np.ndarray], batch_size: int, pad_size: int = 0, pad: bool = False) -> List[np.ndarray]:
    """
    Batch arrays.

    Parameters
    ----------
    arrays : list of np.ndarray
        List of arrays to batch.
    batch_size : int
        Size of each batch.
    pad_size : int, optional
        Size of padding to add, by default 0.
    pad : bool, optional
        Whether to add padding or not, by default False.

    Returns
    -------
    list of np.ndarray
        Batched arrays.
    """
    return [
        concat(arr, batch_size, pad_size, pad)
        if arr.ndim == 1
        else
        diagonal_stack(arr, batch_size, pad_size, pad)
        for arr in arrays
    ]


def ravel_iarrays(iter_arrays: List[np.ndarray], batch_size: int, idx: int) -> List[np.ndarray]:
    """
    Ravel and index into iter_arrays.

    Parameters
    ----------
    iter_arrays : list of np.ndarray
        List of arrays to ravel and index into.
    batch_size : int
        Size of each batch.
    idx : int
        Index to use when indexing into iter_arrays.

    Returns
    -------
    list of np.ndarray
        Raveled and indexed arrays.
    """
    return [
        arr[idx*batch_size:(idx+1)*batch_size].ravel()
        for arr in iter_arrays
    ]


def ravel_last_iarrays(
    iter_arrays: List[np.ndarray], last_batch_size: int, 
    pad_size: int = 0, pad: bool = False
) -> List[np.ndarray]:
    """
    Ravel the last batch of iter_arrays.

    Parameters
    ----------
    iter_arrays : list of np.ndarray
        List of arrays to ravel and index into.
    last_batch_size : int
        Size of the last batch.
    pad_size : int, optional
        Size of padding to add, by default 0.
    pad : bool, optional
        Whether to add padding or not, by default False.

    Returns
    -------
    list of np.ndarray
        Raveled arrays for the last batch.
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
    n_samples: int,  # number of samples
    iter_arrays: List[np.ndarray],  # iterate (n_samples) and ravel these arrays
    arrays: List[np.ndarray],  # concat these arrays so that it is properly sized
    batch_size: int = 1, 
    pad: bool = False, 
    verbose: bool = False
) -> Tuple[int, List[np.ndarray], List[np.ndarray]]:
    """
    Perform batched iteration over arrays.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    iter_arrays : list of np.ndarray
        List of arrays to iterate over and ravel.
    arrays : list of np.ndarray
        List of arrays to concatenate.
    batch_size : int, optional
        Size of each batch, by default 1.
    pad : bool, optional
        Whether to add padding or not, by default False.
    verbose : bool, optional
        Whether to print progress or not, by default False.

    Yields
    -------
    tuple
        Tuple containing the index, raveled iter_arrays, and batched arrays.
    """
    if batch_size == 1:
        iterator = tqdm(enumerate(zip(*iter_arrays)), desc="Iterations", total=n_samples) if verbose else enumerate(zip(*iter_arrays))
        for idx, iarrays in iterator:
            yield (idx, iarrays, arrays)

    else:
        iterator = tqdm(range(n_samples // batch_size), desc='Iterations', total=n_samples//batch_size) if verbose else range(n_samples // batch_size)
        
        batched_arrays = batch_arrays(arrays, batch_size)
        for idx in iterator:
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
