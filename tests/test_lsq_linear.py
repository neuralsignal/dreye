import numpy as np
import pytest

from dreye.api.optimize.lsq_linear import lsq_linear


def test_lsq_linear():
    # A = channels x sources
    # Test 1: simple problem with no bounds, weights, transformation matrix or baseline
    A = np.array([[1, 1], [1, -1]])
    B = np.array([2, 0])
    X = lsq_linear(A, B)
    np.testing.assert_almost_equal(X, np.array([[1, 1]]), decimal=5)

    # Test 2: problem with bounds
    A = np.array([[1, 0], [0, 1]])
    B = np.array([1, 1])
    lb = np.array([0, 0])
    ub = np.array([0.5, 0.5])
    X = lsq_linear(A, B, lb=lb, ub=ub)
    np.testing.assert_almost_equal(X, np.array([[0.5, 0.5]]), decimal=5)

    # Test 3: problem with weights
    A = np.array([[1, 0.5], [0.5, 1]])
    B = np.array([0, 1])
    W = np.array([10, 1])
    X = lsq_linear(A, B, W=W, lb=lb, ub=ub)
    np.testing.assert_almost_equal(X, np.array([[0, 0.0385]]), decimal=2)
    A = np.array([[1, 0.5], [0.5, 1]])
    B = np.array([0, 1])
    W = np.array([1, 1])
    X = lsq_linear(A, B, W=W, lb=lb, ub=ub)
    np.testing.assert_almost_equal(X, np.array([[0.0, 0.5]]), decimal=2)

    # Test 4: problem with transformation matrix
    A = np.array([[1, 0], [0, 1]])
    B = np.array([0.2, 0.2])
    K = np.array([2, 1])
    X = lsq_linear(A, B, K=K, lb=lb, ub=ub)
    np.testing.assert_almost_equal(X, np.array([[0.1, 0.2]]), decimal=2)

    # Test 5: problem with baseline
    A = np.array([[1, 0], [0, 1]])
    B = np.array([0.2, 0.2])
    K = np.array([2, 1])
    baseline = np.array([0.1, 0.1])
    X = lsq_linear(A, B, K=K, lb=lb, ub=ub, baseline=baseline)
    np.testing.assert_almost_equal(X, np.array([[0.0, 0.1]]), decimal=2)

    # Test 8: problem with batch_size
    B = np.ones((100, 2))
    X = lsq_linear(A, B, batch_size=2)
    np.testing.assert_almost_equal(X, np.ones((100, 2)), decimal=5)
