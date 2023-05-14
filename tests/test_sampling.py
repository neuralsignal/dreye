import numpy as np
import pytest

from dreye.api.sampling import sample_in_hull, d_equally_spaced


def test_sample_in_hull():
    P = np.array([[0, 0], [1, 0], [0, 1]])
    n = 10
    seed = 42
    result = sample_in_hull(P, n, seed)
    assert result.shape == (n, P.shape[-1])  # check shape
    assert np.all(result >= 0)  # check all values are non-negative
    assert np.all(result <= 1)  # check all values are less than or equal to 1


def test_sample_in_hull_with_engine():
    P = np.array([[0, 0], [1, 0], [0, 1]])
    n = 10
    seed = 42
    engine = "Sobol"
    result = sample_in_hull(P, n, seed, engine)
    assert result.shape == (n, P.shape[-1])  # check shape
    assert np.all(result >= 0)  # check all values are non-negative
    assert np.all(result <= 1)  # check all values are less than or equal to 1


def test_d_equally_spaced():
    n = 5
    d = 3
    result = d_equally_spaced(n, d)
    assert result.shape == (n**d, d)  # check shape
    assert np.all(result >= 0)  # check all values are non-negative
    assert np.all(result <= 1)  # check all values are less than or equal to 1


def test_d_equally_spaced_not_one_inclusive():
    n = 5
    d = 3
    result = d_equally_spaced(n, d, one_inclusive=False)
    assert result.shape == (n**d, d)  # check shape
    assert np.all(result >= 0)  # check all values are non-negative
    assert np.all(result < 1)  # check all values are less than 1
