import numpy as np
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import pytest

from dreye.api.project import (
    proj_P_for_hull,
    line_to_simplex,
    proj_P_to_simplex,
    yieldPpairs4proj2simplex,
    proj_B_to_hull, 
    alpha_for_B_with_P,
    B_with_P,
)


def test_proj_P_for_hull_return_hull():
    P = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    hull = proj_P_for_hull(
        P, return_hull=True, return_ndim=False, return_transformer=False
    )
    assert isinstance(hull, ConvexHull)


def test_proj_P_for_hull_return_ndim():
    P = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    ndim = proj_P_for_hull(
        P, return_hull=False, return_ndim=True, return_transformer=False
    )
    assert ndim == 2


def test_proj_P_for_hull_return_transformer():
    P = np.array([[0, 0, 0], [0, 10, 0], [1, 0, 0], [1, 10, 0]])
    transformer = proj_P_for_hull(
        P, return_hull=False, return_ndim=False, return_transformer=True
    )
    assert np.allclose(
        np.abs(transformer.components_), np.array([[0, 1, 0], [1, 0, 0]])
    )
    assert isinstance(transformer, PCA)


def test_proj_P_for_hull_return_hull_ndim():
    P = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    hull, ndim = proj_P_for_hull(
        P, return_hull=True, return_ndim=True, return_transformer=False
    )
    assert isinstance(hull, ConvexHull)
    assert ndim == 2


def test_proj_P_for_hull_one_dimension():
    P = np.array([[0], [1], [2], [3]])
    hull = proj_P_for_hull(
        P, return_hull=True, return_ndim=False, return_transformer=False
    )
    assert isinstance(hull, np.ndarray)


def test_proj_P_for_hull_qhull_error():
    P = np.array([[0, 0, 0], [0, 10, 0], [1, 0, 0], [1, 10, 0]])
    hull, ndim, transformer = proj_P_for_hull(
        P, return_hull=True, return_ndim=True, return_transformer=True
    )
    assert isinstance(hull, ConvexHull)
    assert ndim == 2
    assert isinstance(transformer, PCA)


def test_line_to_simplex():
    x1 = np.array([0.1, 0.2])
    x2 = np.array([0.8, 0.4])
    # x1 + t * (x2 - x1) = c
    c = 1.0
    expected = x1 + 0.7 / 0.9 * np.array([0.7, 0.2])
    result = line_to_simplex(x1, x2, c, checks=True)
    assert np.allclose(result, expected)
    assert np.allclose(np.sum(result), c)


def test_line_to_simplex_negative_input():
    x1 = np.array([-0.1, -0.2, -0.3])
    x2 = np.array([-0.4, -0.5, -0.6])
    c = 1.0
    with pytest.raises(AssertionError):
        line_to_simplex(x1, x2, c, checks=True)


def test_yieldPpairs4proj2simplex():
    # first one is below simplex the other two above
    P = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    c = 1.0
    result = list(yieldPpairs4proj2simplex(P, c))
    assert len(result) == 2
    assert np.all(result[0][0] == np.array([0.1, 0.2, 0.3]))
    assert np.all(result[0][1] == np.array([0.4, 0.5, 0.6]))
    assert np.all(result[1][0] == np.array([0.1, 0.2, 0.3]))
    assert np.all(result[1][1] == np.array([0.7, 0.8, 0.9]))


def test_proj_P_to_simplex():
    P = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    c = 1.0
    result = proj_P_to_simplex(P, c)
    assert np.all(result >= 0)
    assert np.allclose(np.sum(result, axis=1), c)


def test_proj_P_to_simplex_negative_input():
    P = np.array([[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6], [-0.7, -0.8, -0.9]])
    c = 1.0
    with pytest.raises(AssertionError):
        proj_P_to_simplex(P, c)
    
    
def test_proj_B_to_hull():
    B = np.array([[0.1, 0.2], [0.5, 0.5], [0.3, 0.6]])
    P = np.array([
        [0, 0], [0, 0.5], [0.5, 0], [0.5, 0.5],
    ])
    equations = ConvexHull(P).equations
    result = proj_B_to_hull(B, equations)
    assert np.allclose(
        result, [[0.1, 0.2], [0.5, 0.5], [0.3, 0.5]]
    )
    
    B = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5], [0.3, 0.5, 0.6]])
    P = np.array([
        [0, 0, 0], [0, 0.5, 0], [0.5, 0, 0], [0.5, 0.5, 0], 
        [0, 0, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0.5]
    ])
    equations = ConvexHull(P).equations
    result = proj_B_to_hull(B, equations)
    assert np.allclose(
        result, [[0.1, 0.2, 0.3], [0.5, 0.5, 0.5], [0.3, 0.5, 0.5]]
    )

def test_alpha_for_B_with_P():
    B = np.array([[0.1, 0.2], [0.5, 0.5], [0.3, 0.6]])
    P = np.array([
        [0, 0], [0, 0.5], [0.5, 0], [0.5, 0.5],
    ])
    equations = ConvexHull(P).equations
    result = alpha_for_B_with_P(B, equations)
    assert np.allclose(
        result, [5/2, 1, 5/6]
    )
    
    B = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5], [0.3, 0.5, 0.6]])
    P = np.array([
        [0, 0, 0], [0, 0.5, 0], [0.5, 0, 0], [0.5, 0.5, 0], 
        [0, 0, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0.5]
    ])
    equations = ConvexHull(P).equations
    result = alpha_for_B_with_P(B, equations)
    assert np.allclose(
        result, [5/3, 1, 5/6]
    )

def test_B_with_P():
    B = np.array([[0.1, 0.2], [0.5, 0.5], [0.3, 0.6]])
    P = np.array([
        [0, 0], [0, 0.5], [0.5, 0], [0.5, 0.5],
    ])
    equations = ConvexHull(P).equations
    result = B_with_P(B, equations)
    assert np.allclose(
        result, np.array([5/2, 1, 5/6])[:, None] * B
    )
    
    B = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5], [0.3, 0.5, 0.6]])
    P = np.array([
        [0, 0, 0], [0, 0.5, 0], [0.5, 0, 0], [0.5, 0.5, 0], 
        [0, 0, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0.5]
    ])
    equations = ConvexHull(P).equations
    result = B_with_P(B, equations)
    assert np.allclose(
        result, np.array([5/3, 1, 5/6])[:, None] * B
    )

