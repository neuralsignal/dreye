"""
Module for handling barycentric coordinates and transformations.

This module provides a set of functions for handling barycentric coordinates
and performing transformations between barycentric and cartesian coordinates.

It includes functions for:

- Dimensionality reduction of a given data set X using l1-normalizing and
  barycentric to cartesian coordinate transformation.
- Conversion from barycentric coordinates to cartesian coordinates.
- Conversion from cartesian coordinates to barycentric coordinates.
- Obtaining a linear transformation matrix from barycentric to cartesian coordinates.

It is designed to handle data in numpy arrays, and uses sklearn's normalize
function for l1-normalization, and numpy's linear algebra functions for transformations.
"""

import numpy as np
from sklearn.preprocessing import normalize
from typing import Optional


def barycentric_dim_reduction(X: np.ndarray, center: bool = False) -> np.ndarray:
    """
    Reduce dimensionality of `X` to N-1 by l1-normalizing each sample
    and using a barycentric to cartesian coordinate transformation

    Parameters
    ----------
    X : np.ndarray
        Input array with points in higher dimensions.
    center : bool, optional
        If True, center the coordinates, by default False

    Returns
    -------
    np.ndarray
        The input array transformed to lower dimensions.
    """
    X = normalize(X, norm="l1", axis=1)
    return barycentric_to_cartesian(X, center=center)


def barycentric_to_cartesian(X: np.ndarray, center: bool = False) -> np.ndarray:
    """
    Convert from barycentric coordinates to cartesian coordinates

    Parameters
    ----------
    X : np.ndarray
        Input array with points in barycentric coordinates.
    center : bool, optional
        If True, center the coordinates, by default False

    Returns
    -------
    np.ndarray
        The input array converted to cartesian coordinates.
    """
    n = X.shape[1]
    A = barycentric_to_cartesian_transformer(n)
    if center:
        return X @ A - (np.ones(n) / n) @ A
    else:
        return X @ A


def cartesian_to_barycentric(
    X: np.ndarray, L1: Optional[np.ndarray] = None, centered: bool = False
) -> np.ndarray:
    """
    Convert from cartesian to barycentric coordinates.

    Parameters
    ----------
    X : np.ndarray
        Input array with points in cartesian coordinates.
    L1 : np.ndarray, optional
        L1 normalization values, by default None.
    centered : bool, optional
        If True, center the coordinates, by default False.

    Returns
    -------
    np.ndarray
        The input array converted to barycentric coordinates.
    """
    n = X.shape[1] + 1
    A = barycentric_to_cartesian_transformer(n)
    if centered:
        X = X + (np.ones(n) / n) @ A

    X = np.hstack([X, np.ones((X.shape[0], 1))])
    A = np.hstack([A, np.ones((A.shape[0], 1))])
    X = X @ np.linalg.inv(A)

    if L1 is None:
        return X
    else:
        L1 = np.atleast_1d(L1)
        return X * L1[..., None]


def barycentric_to_cartesian_transformer(n: int) -> np.ndarray:
    """
    Get linear transformation from barycentric to cartesian coordinates.

    Parameters
    ----------
    n : int
        The number of dimensions in barycentric coordinates.

    Returns
    -------
    np.ndarray
        Transformation matrix for converting barycentric to cartesian coordinates.
    """
    assert n > 1, "The number of dimensions should be more than 1"
    A = np.zeros((n, n - 1))
    A[1, 0] = 1
    for i in range(2, n):
        A[i, : i - 1] = np.mean(A[:i, : i - 1], axis=0)
        dis = np.sum((A[:i, : i - 1] - A[i, : i - 1]) ** 2, axis=1)
        # sanity check
        assert np.isclose(dis, dis[0]).all(), "non-unique rows"
        x = np.sqrt(1 - dis.mean())
        A[i, i - 1] = x
    return A
