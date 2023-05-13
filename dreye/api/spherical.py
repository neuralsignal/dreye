"""
Handling n-dimensional spherical coordinates
"""

import numpy as np
from typing import Union
from dreye.api.utils import l2norm


def spherical_to_cartesian(Y: Union[np.ndarray, list]) -> np.ndarray:
    """
    Convert from spherical to cartesian coordinates.

    Parameters
    ----------
    Y : array-like (..., ndim)
        Array where the last axis corresponds to the dimensions of each spherical coordinate
        starting with the radius and ending with the angle that spans 2pi. The other angles
        only span pi. All angle dimensions must be in radians.

    Returns
    -------
    X : array-like (..., ndim)
        `Y` in cartesian corrdinates.

    Raises
    ------
    ValueError
        If the input array `Y` is not at least 1-dimensional.
    """

    if Y.shape[-1] == 1:
        return Y

    if len(Y.shape) < 1:
        raise ValueError("Input array must be at least 1-dimensional.")

    # first dimension are the radii, the rest are angles
    radii = Y[..., :1]
    angles = Y[..., 1:]

    # cosine and sine of all angle values
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    # Initializing the result array
    cartesian_coords = np.zeros(Y.shape)
    # first dimension
    cartesian_coords[..., 0] = cos_angles[..., 0]
    # second to second to last
    for i in range(1, Y.shape[-1] - 1):
        cartesian_coords[..., i] = cos_angles[..., i] * np.prod(
            sin_angles[..., :i], axis=-1
        )
    # last
    cartesian_coords[..., -1] = np.prod(sin_angles, axis=-1)

    return cartesian_coords * radii


def cartesian_to_spherical(X: Union[np.ndarray, list]) -> np.ndarray:
    """
    Convert from cartesian to spherical coordinates.

    Parameters
    ----------
    X : array-like (..., ndim)
        Array where the last axis corresponds to the dimensions in cartesian coordinates.

    Returns
    -------
    Y : array-like (..., ndim)
        Array where the last axis corresponds to the dimensions of each spherical coordinate
        starting with the radius and ending with the angle that spans 2pi. The other angles
        only span pi. All angle dimensions must be in radians.

    Raises
    ------
    ValueError
        If the input array `X` is not at least 1-dimensional.
    """

    if X.shape[-1] == 1:
        return X

    if len(X.shape) < 1:
        raise ValueError("Input array must be at least 1-dimensional.")

    # Calculate the radius
    radius = l2norm(X, axis=-1)

    # Initialize the result array
    spherical_coords = np.zeros(X.shape)
    spherical_coords[..., 0] = radius

    dimension = X.shape[1] - 1
    zeros = X == 0

    # Convert each dimension
    for i in range(dimension - 1):
        spherical_coords[..., i + 1] = np.where(
            zeros[:, i:].all(-1), 0, np.arccos(X[..., i] / l2norm(X[..., i:], axis=-1))
        )

    # Handle last dimension separately due to 2pi span
    last_angle = np.arccos(X[..., -2] / l2norm(X[..., -2:], axis=-1))
    spherical_coords[..., -1] = np.where(
        zeros[:, -2:].all(-1),
        0,
        np.where(X[..., -1] >= 0, last_angle, 2 * np.pi - last_angle),
    )
    return spherical_coords
