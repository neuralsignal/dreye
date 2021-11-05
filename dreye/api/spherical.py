"""
Handling n-dimensional spherical coordinates
"""

from functools import reduce
import numpy as np

from dreye.api.utils import l2norm


# TODO test


def spherical2cartesian(Y):
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


def cartesian2spherical(X):
    """
    Convert from cartesian to spherical coordinates
    """
    
    if X.shape[-1] == 1:
        return X
    
    r = l2norm(X, axis=-1)
    Y = np.zeros(X.shape)
    Y[..., 0] = r
    d = X.shape[1] - 1
    zeros = (X == 0)
    
    for i in range(d-1):
        Y[..., i+1] = np.where(
            zeros[:, i:].all(-1),
            0,
            np.arccos(X[..., i]/l2norm(X[..., i:], axis=-1))
        )
    
    lasty = np.arccos(X[..., -2]/l2norm(X[..., -2:], axis=-1))
    Y[..., -1] = np.where(
        zeros[:, -2:].all(-1), 
        0, 
        np.where(
            X[..., -1] >= 0, 
            lasty, 
            2*np.pi - lasty
        )
    )
    return Y