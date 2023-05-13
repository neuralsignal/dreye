import pytest
import numpy as np

from dreye.api.barycentric import (
    barycentric_dim_reduction,
    barycentric_to_cartesian,
    cartesian_to_barycentric,
    barycentric_to_cartesian_transformer,
)


def test_barycentric_dim_reduction():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    result = barycentric_dim_reduction(X)
    assert result.shape == (2, 2)


def test_barycentric_to_cartesian():
    X = np.array([[0.25, 0.25, 0.5], [0.5, 0.25, 0.25]])
    result = barycentric_to_cartesian(X)
    assert result.shape == (2, 2)


def test_cartesian_to_barycentric():
    X = np.array([[1, 2], [3, 4]])
    result = cartesian_to_barycentric(X)
    assert result.shape == (2, 3)


def test_barycentric_to_cartesian_transformer():
    n = 3
    result = barycentric_to_cartesian_transformer(n)
    assert result.shape == (n, n - 1)


def test_cartesian_barycentric_conversion():
    # Test if the conversion to barycentric and back to cartesian preserves the dimensionality,
    # barycentric coordinates sum to 1, and are all non-negative
    X = np.random.rand(5, 3)  # 3-dimensional barycentric coordinates, L1-normalized
    X = X / np.sum(X, axis=1, keepdims=True)  # L1 normalization
    X = barycentric_to_cartesian(X)  # expected to be 2-dimensional

    X_barycentric = cartesian_to_barycentric(
        X
    )  # expected to be 3-dimensional and L1-normalized
    X_cartesian = barycentric_to_cartesian(X_barycentric)  # convert back to cartesian
    assert X_barycentric.shape == (5, 3)
    assert np.allclose(np.sum(X_barycentric, axis=1), 1.0)
    assert np.all(X_barycentric >= 0)
    assert np.allclose(X_cartesian, X)


def test_barycentric_cartesian_conversion():
    # Test if the conversion to cartesian and back to barycentric preserves the dimensionality,
    # and the resulting cartesian coordinates are not necessarily all non-negative
    X = np.random.rand(5, 3)  # 3-dimensional barycentric coordinates, L1-normalized
    X = X / np.sum(X, axis=1, keepdims=True)  # L1 normalization
    X_cartesian = barycentric_to_cartesian(X)  # expected to be 2-dimensional
    X_barycentric = cartesian_to_barycentric(
        X_cartesian, L1=np.sum(X, axis=1)
    )  # convert back to barycentric
    assert X_cartesian.shape == (5, 2)
    assert X_barycentric.shape == X.shape
    assert np.allclose(X_barycentric, X)


def test_barycentric_cartesian_centering():
    # Test if centering is working correctly in barycentric_to_cartesian
    X = np.ones((5, 3))  # 3-dimensional barycentric coordinates, L1-normalized
    X = X / np.sum(X, axis=1, keepdims=True)  # L1 normalization
    X_cartesian_centered = barycentric_to_cartesian(X, center=True)
    assert np.allclose(np.mean(X_cartesian_centered, axis=0), 0.0)  # centered around 0


def test_cartesian_barycentric_centering():
    # Test if centering is working correctly in cartesian_to_barycentric
    X = np.zeros((5, 2))  # 2-dimensional cartesian coordinates
    X_barycentric_centered = cartesian_to_barycentric(X, centered=True)
    assert np.allclose(
        np.mean(X_barycentric_centered, axis=0), 1 / 3
    )  # centered around 1/3 for 3-dimensional barycentric coordinates
