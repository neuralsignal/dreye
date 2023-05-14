import pytest
import numpy as np

from dreye.api.capture import calculate_capture

    
def test_calculate_capture_1D():
    # 1D inputs
    filters = np.array([0.5, 0.5])
    signals = np.array([1, 1])
    # The trapezoidal rule approximates the integral as the sum of trapezoids.
    # Here it should be equivalent to the simple sum, as we are integrating over a uniform grid.
    assert np.isclose(calculate_capture(filters, signals), 0.5)


def test_calculate_capture_2D():
    # 2D inputs
    filters = np.array([[0.5, 0.5], [0.25, 0.75]])
    signals = np.array([[1, 1], [1, 1]])
    assert np.allclose(
        calculate_capture(filters, signals), np.array([[0.5, 0.5], [0.5, 0.5]])
    )


def test_calculate_capture_trapz_false():
    # Test with trapz set to False
    filters = np.array([0.5, 0.5])
    signals = np.array([1, 1])
    assert np.isclose(calculate_capture(filters, signals, trapz=False), 1.0)


def test_calculate_capture_domain_array():
    # Test with domain as an array
    filters = np.array([0.5, 1.0])
    signals = np.array([1, 1])
    domain = np.array([0.0, 2.0])
    assert np.isclose(calculate_capture(filters, signals, domain=domain), 1.5)


def test_calculate_capture_broadcasting():
    # Test broadcasting rules
    filters = np.array([[0.5, 0.5], [0.0, 1.0]])
    signals = np.array([1, 1])
    assert np.allclose(calculate_capture(filters, signals), np.array([0.5, 0.5]))


def test_calculate_capture_incompatible_shapes():
    # Test incompatible shapes
    filters = np.array([0.5, 0.5])
    signals = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        calculate_capture(filters, signals)
