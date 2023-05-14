import numpy as np
import pytest

from dreye.api.utils import (
    integral,
    round_to_significant_digits,
    round_to_precision,
    arange_with_interval,
)


def test_integral():
    arr = np.array([1, 2, 3, 4])
    domain = 1
    expected_result = 7.5
    assert integral(arr, domain) == expected_result

    domain = np.array([0, 1, 2, 3])
    expected_result = 7.5
    assert integral(arr, domain) == expected_result


def test_round_to_significant_digits():
    x = np.array([1.2345, 6.7890, 0.1234, 6.789])
    p = 2
    expected_result = np.array([1.2, 6.8, 0.12, 6.8])
    np.testing.assert_almost_equal(
        round_to_significant_digits(x, p), expected_result, decimal=1
    )


def test_round_to_precision():
    x = np.array([1.2, 2.3, 3.4, 4.5])
    precision = 0.5
    expected_result = np.array([1.0, 2.5, 3.5, 4.5])
    np.testing.assert_almost_equal(
        round_to_precision(x, precision), expected_result, decimal=1
    )


def test_integral_with_multi_dimensional_input():
    arr = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])
    domain = 1
    expected_result = np.array([7.5, 10.5])
    np.testing.assert_equal(integral(arr, domain, axis=-1), expected_result)

    domain = np.array([0, 1, 2, 3])
    expected_result = np.array([7.5, 10.5])
    np.testing.assert_equal(integral(arr, domain, axis=-1), expected_result)

    domain = np.array([0, 2, 4, 8])
    expected_result = np.array([12 + 2 + 4 + 2 + 1 + 1, 16 + 4 + 6 + 2 + 1 + 1])
    np.testing.assert_equal(integral(arr, domain, axis=-1), expected_result)


def test_integral_with_keepdims():
    arr = np.array([1, 2, 3, 4])
    domain = 1
    expected_result = np.array([7.5])
    np.testing.assert_almost_equal(integral(arr, domain, keepdims=True), expected_result, decimal=1)


def test_round_to_significant_digits_with_various_numbers():
    x = np.array([123.456, 0.0006789, 0.1234, 678.9])

    result = round_to_significant_digits(x, 2)
    np.testing.assert_almost_equal(result, np.array([120.0, 0.00068, 0.12, 680.0]), decimal=1)


def test_round_to_precision_with_various_precisions():
    x = np.array([1.234, 5.678, 9.012, 3.456])
    precisions = [0.1, 0.5, 1, 2, 10]
    for precision in precisions:
        result = round_to_precision(x, precision)
        assert result.shape == x.shape
        np.testing.assert_almost_equal(result, np.round(x / precision) * precision, decimal=2)


def test_arange_with_interval():
    # Test when stop is exactly on a step
    start = 0
    stop = 1
    step = 0.2
    expected_result = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
    np.testing.assert_almost_equal(arange_with_interval(start, stop, step), expected_result, decimal=2)
    
    # Test when stop is not exactly on a step
    start = 0
    stop = 1
    step = 0.3
    expected_result = np.array([0, 1/3, 2/3, 1])
    np.testing.assert_almost_equal(arange_with_interval(start, stop, step), expected_result, decimal=2)

    # Test when stop is less than start
    start = 1
    stop = 0
    step = -0.2
    expected_result = np.array([1, 0.8, 0.6, 0.4, 0.2, 0])
    np.testing.assert_almost_equal(arange_with_interval(start, stop, step), expected_result, decimal=2)

    # Test when stop is less than start and not exactly on a step
    start = 1
    stop = 0
    step = -0.3
    expected_result = np.array([1, 2/3, 1/3, 0])
    np.testing.assert_almost_equal(arange_with_interval(start, stop, step), expected_result, decimal=2)

    # Test return_interval
    arr, interval = arange_with_interval(start, stop, step, return_interval=True)
    np.testing.assert_almost_equal(arr, expected_result, decimal=2)
    assert interval == -1/3

    # Test raise_on_step_change
    with pytest.raises(ValueError):
        arange_with_interval(start, stop, step, raise_on_step_change=True)
