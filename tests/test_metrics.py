import numpy as np
import pytest
from dreye.api.metrics import (
    compute_jensen_shannon_divergence,
    compute_jensen_shannon_similarity,
    compute_mean_width,
    compute_mean_correlation,
    compute_mean_mutual_info,
    compute_volume,
    compute_gamut,
)


def test_compare_jensen_shannon():
    P = np.array([0.1, 0.2, 0.7])
    Q = np.array([0.2, 0.2, 0.6])
    div = compute_jensen_shannon_divergence(P, Q)
    sim = compute_jensen_shannon_similarity(Q, P)
    assert np.isclose(div, 1 - sim)


def test_compute_mean_width():
    X = np.random.rand(10, 3)
    assert compute_mean_width(X) > 0


def test_compute_mean_correlation():
    X = np.random.rand(10, 3)
    assert 0 <= compute_mean_correlation(X) <= 1


def test_compute_mean_mutual_info():
    X = np.random.rand(10, 3)
    assert compute_mean_mutual_info(X) >= 0


def test_compute_volume():
    X = np.random.rand(10, 3)
    assert compute_volume(X) > 0


def test_compute_gamut():
    X = np.random.rand(10, 3)
    assert compute_gamut(X) >= 0


def test_compute_jensen_shannon_divergence_identical():
    P = np.array([0.1, 0.2, 0.7])
    assert np.isclose(compute_jensen_shannon_divergence(P, P), 0.0)


def test_compute_jensen_shannon_similarity_identical():
    P = np.array([0.1, 0.2, 0.7])
    assert np.isclose(compute_jensen_shannon_similarity(P, P), 1.0)


def test_compute_mean_width_single_value():
    X = np.ones((10, 3))
    assert np.isclose(compute_mean_width(X), 0.0)


def test_compute_mean_correlation_single_value():
    X = np.ones((10, 3))
    assert np.isnan(compute_mean_correlation(X))


def test_compute_volume_single_value():
    X = np.ones((10, 3))
    assert np.isclose(compute_volume(X), 0.0)


def test_compute_gamut_single_value():
    X = np.ones((10, 3))
    assert np.isclose(compute_gamut(X), 0.0)


def test_compute_jensen_shannon_divergence_identical():
    P = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    Q = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    assert np.isclose(compute_jensen_shannon_divergence(P, Q), 0)


def test_compute_jensen_shannon_divergence_different():
    P = np.array([1, 0, 0, 0, 0])
    Q = np.array([0, 0, 0, 0, 1])
    assert np.isclose(compute_jensen_shannon_divergence(P, Q), 1)


def test_compute_jensen_shannon_divergence_negative():
    P = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    Q = np.array([-0.2, -0.2, -0.2, -0.2, -0.2])
    with pytest.raises(ValueError):
        compute_jensen_shannon_divergence(P, Q)


def test_compute_jensen_shannon_divergence_large_values():
    P = np.array([1000000, 0, 0, 0, 0])
    Q = np.array([0, 0, 0, 0, 1000000])
    assert np.isclose(compute_jensen_shannon_divergence(P, Q), 1)


def test_compute_jensen_shannon_similarity_identical():
    P = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    Q = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    assert np.isclose(compute_jensen_shannon_similarity(P, Q), 1)


def test_compute_jensen_shannon_similarity_different():
    P = np.array([1, 0, 0, 0, 0])
    Q = np.array([0, 0, 0, 0, 1])
    assert np.isclose(compute_jensen_shannon_similarity(P, Q), 0)


def test_compute_jensen_shannon_similarity_negative():
    P = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    Q = np.array([-0.2, -0.2, -0.2, -0.2, -0.2])
    with pytest.raises(ValueError):
        compute_jensen_shannon_similarity(P, Q)


def test_compute_jensen_shannon_similarity_large_values():
    P = np.array([1000000, 0, 0, 0, 0])
    Q = np.array([0, 0, 0, 0, 1000000])
    assert np.isclose(compute_jensen_shannon_similarity(P, Q), 0)


def test_compute_gamut_volume():
    X = np.random.rand(10, 3)
    assert compute_gamut(X, metric="volume") >= 0


def test_compute_gamut_relative():
    X = np.random.rand(10, 3)
    Y = np.random.rand(10, 3)
    assert compute_gamut(X, relative_to=Y) >= 0


def test_compute_gamut_center_false():
    X = np.random.rand(10, 3)
    assert compute_gamut(X, center=False) >= 0


def test_compute_gamut_center_to_neutral_true():
    X = np.random.rand(10, 3)
    assert compute_gamut(X, center_to_neutral=True) >= 0


def test_compute_gamut_at_l1():
    X = np.random.rand(10, 3)
    assert compute_gamut(X, at_l1=1.5) >= 0


def test_compute_gamut_seed():
    X = np.random.rand(10, 3)
    result1 = compute_gamut(X, seed=0)
    result2 = compute_gamut(X, seed=0)
    assert np.isclose(result1, result2)


def test_compute_gamut_empty():
    X = np.empty((0, 3))
    assert compute_gamut(X) == 0


def test_compute_gamut_negative_at_l1():
    X = np.random.rand(10, 3)
    with pytest.raises(AssertionError):
        compute_gamut(X, at_l1=-1)
