import numpy as np
from numpy.testing import assert_almost_equal

from dreye.api.spherical import cartesian_to_spherical, spherical_to_cartesian


def test_spherical_to_cartesian():
    # Test 2D
    spherical_coords_2d = np.array([[1, np.pi / 2]])
    cartesian_coords_2d = spherical_to_cartesian(spherical_coords_2d)
    assert_almost_equal(cartesian_coords_2d, np.array([[0, 1]]))

    # Test 3D
    spherical_coords_3d = np.array([[1, np.pi / 2, 0]])
    cartesian_coords_3d = spherical_to_cartesian(spherical_coords_3d)
    assert_almost_equal(cartesian_coords_3d, np.array([[0, 1, 0]]))
    
    # Test with 2D spherical coordinates (radius, angle)
    Y = np.array([[1, np.pi / 4], [1, np.pi / 2]])
    expected = np.array([[np.sqrt(2) / 2, np.sqrt(2) / 2], [0, 1]])
    assert_almost_equal(spherical_to_cartesian(Y), expected)

    # Test with 3D spherical coordinates (radius, theta, phi)
    Y = np.array([[1, np.pi / 2, np.pi / 2]])
    expected = np.array([[0, 0, 1]])
    assert_almost_equal(spherical_to_cartesian(Y), expected)

    # Test with 1D spherical coordinates (radius only)
    Y = np.array([[1], [2], [3]])
    expected = Y.copy()
    assert_almost_equal(spherical_to_cartesian(Y), expected)

    # Test with invalid input
    Y = np.array(1)
    try:
        spherical_to_cartesian(Y)
    except IndexError:
        pass


def test_cartesian_to_spherical():
    # Test with 2D cartesian coordinates (x, y)
    X = np.array([[np.sqrt(2) / 2, np.sqrt(2) / 2], [0, 1]])
    expected = np.array([[1, np.pi / 4], [1, np.pi / 2]])
    assert_almost_equal(cartesian_to_spherical(X), expected)

    # Test with 3D cartesian coordinates (x, y, z)
    X = np.array([[0, 0, 1]])
    expected = np.array([[1, np.pi / 2, np.pi / 2]])
    assert_almost_equal(cartesian_to_spherical(X), expected)

    # Test with 1D cartesian coordinates (x only)
    X = np.array([[1], [2], [3]])
    expected = X.copy()
    assert_almost_equal(cartesian_to_spherical(X), expected)

    # Test with invalid input
    X = np.array(1)
    try:
        cartesian_to_spherical(X)
    except IndexError:
        pass
