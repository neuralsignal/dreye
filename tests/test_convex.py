import numpy as np
import pytest
from dreye.api.convex import (
    all_combinations_of_bounds,
    get_P_from_A,
    in_hull,
    convex_combination,
    range_of_solutions
)


def test_all_combinations_of_bounds():
    lb = np.array([0, 0])
    ub = np.array([1, 1])

    expected = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    np.testing.assert_array_equal(all_combinations_of_bounds(lb, ub), expected)


def test_get_P_from_A():
    A = np.eye(3)
    # bounds
    lb = np.array([0, 0, 0])
    ub = np.array([2, 2, 4])
    expected_P = all_combinations_of_bounds(lb, ub)
    np.testing.assert_array_equal(get_P_from_A(A, lb, ub), expected_P)

    # Test with infinite ub
    ub = np.array([np.inf, np.inf, np.inf])
    with pytest.raises(ValueError):
        get_P_from_A(A, lb, ub, bounded=True)

    # Test with non-finite lb
    lb = np.array([0, np.nan, 0])
    with pytest.raises(AssertionError):
        get_P_from_A(A, lb, ub)


def test_in_hull():
    P = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Point is inside the convex hull
    B = np.array([0.5, 0.5])
    assert in_hull(P, B)

    # Point is outside the convex hull
    B = np.array([1.5, 1.5])
    assert not in_hull(P, B)

    # Point is on the border of the convex hull
    B = np.array([0, 0.5])
    assert in_hull(P, B)

    # Point is a vertex of the convex hull
    B = np.array([1, 1])
    assert in_hull(P, B)

    # Multiple points, some inside and some outside
    B = np.array(
        [
            [0.5, 0.5],  # inside
            [1.5, 1.5],  # outside
            [0, 0.5],  # on the border
            [1, 1],  # a vertex
        ]
    )
    expected = np.array([True, False, True, True])
    np.testing.assert_array_equal(in_hull(P, B), expected)

    # Point is in 3D convex hull
    P = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0.5, 0.5, 1]])
    B = np.array([0.5, 0.5, 0.5])
    assert in_hull(P, B)

    # Point is outside 3D convex hull
    B = np.array([1.5, 1.5, 1.5])
    assert not in_hull(P, B)


def test_convex_combination():
    P = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    B = np.array([0.5, 0.5])

    X, norms, in_hulls = convex_combination(P, B)

    assert np.isclose(norms, 0)
    assert in_hulls


def test_range_of_solutions():
    # Test with a simple 2D system
    B = np.array([2.5, 5])
    A = np.array([[1, 2], [3, 4], [5, 6]]).T
    lb = np.array([0, 0, 0])
    ub = np.array([10, 10, 10])
    Xmin, Xmax = range_of_solutions(B, A, lb, ub)
    assert Xmin.shape == (3,)
    assert Xmax.shape == (3,)
    assert np.all(Xmin >= lb)
    assert np.all(Xmax <= ub)

    # Test with 3D system, multiple B
    B = np.array([[1, 2], [3, 4]])
    A = np.array([[1, 2, 3], [4, 5, 6]])
    lb = np.array([0, 0, 0])
    ub = np.array([2, 2, 2])
    Xmin, Xmax = range_of_solutions(B, A, lb, ub, error='ignore')
    assert Xmin.shape == (2, 3)
    assert Xmax.shape == (2, 3)
    assert np.all(Xmin >= lb)
    assert np.all(Xmax <= ub)

    # Test with target outside the gamut (raise)
    B = np.array([10, 20])
    A = np.array([[1, 2, 3], [4, 5, 6]])
    lb = np.array([0, 0, 0])
    ub = np.array([2, 2, 2])
    with pytest.raises(ValueError):
        Xmin, Xmax = range_of_solutions(B, A, lb, ub, error='raise')

    # Test with underdetermined system
    B = np.array([1, 2])
    A = np.array([[1, 2], [3, 4]])
    lb = np.array([0, 0])
    ub = np.array([2, 2])
    with pytest.raises(ValueError):
        Xmin, Xmax = range_of_solutions(B, A, lb, ub)
