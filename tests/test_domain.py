import pytest
import numpy as np

from dreye.api.domain import equalize_domains


def test_equalize_domains_same_domain():
    domains = [np.arange(0, 10), np.arange(0, 10)]
    arrs = [np.random.rand(10), np.random.rand(10)]
    new_domain, new_arrs = equalize_domains(domains, arrs)
    assert np.array_equal(new_domain, domains[0])
    assert np.array_equal(new_arrs, arrs)


def test_equalize_domains_different_domains():
    domains = [np.arange(0, 10), np.arange(5, 15)]
    arrs = [np.random.rand(10), np.random.rand(10)]
    new_domain, new_arrs = equalize_domains(domains, arrs)
    assert np.array_equal(new_domain, np.arange(5, 10))
    assert new_arrs[0].shape == new_arrs[1].shape


def test_equalize_domains_different_domains_and_axes():
    domains = [np.arange(0, 10), np.arange(5, 15)]
    arrs = [np.random.rand(10, 5), np.random.rand(10, 5)]
    axes = [0, 0]
    new_domain, new_arrs = equalize_domains(domains, arrs, axes=axes)
    assert np.array_equal(new_domain, np.arange(5, 10))
    assert new_arrs[0].shape == new_arrs[1].shape


def test_equalize_domains_incompatible_domains():
    domains = [np.arange(0, 10), np.arange(20, 30)]
    arrs = [np.random.rand(10), np.random.rand(10)]
    with pytest.raises(ValueError):
        equalize_domains(domains, arrs)


def test_equalize_domains_stack_axis():
    domains = [np.arange(0, 10), np.arange(0, 10)]
    arrs = [np.random.rand(10), np.random.rand(10)]
    new_domain, new_arrs = equalize_domains(domains, arrs, stack_axis=0)
    assert np.array_equal(new_domain, domains[0])
    assert new_arrs.shape == (2, 10)
    
    
def test_equalize_domains():
    # domains with equal spacing but different ranges
    domain1 = np.array([0, 1, 2, 3, 4, 5])
    domain2 = np.array([3, 4, 5, 6, 7])
    arr1 = np.array([1, 2, 3, 4, 5, 6])
    arr2 = np.array([7, 8, 9, 10, 11])

    new_domain, new_arrs = equalize_domains([domain1, domain2], [arr1, arr2])

    assert np.allclose(new_domain, np.array([3, 4, 5]))
    assert np.allclose(new_arrs[0], np.array([4, 5, 6]))
    assert np.allclose(new_arrs[1], np.array([7, 8, 9]))

def test_equalize_domains_interpolation():
    # domains with different spacing
    domain1 = np.array([0, 2, 4, 6, 8, 10])
    domain2 = np.array([3, 6, 9])
    arr1 = np.array([1, 2, 3, 4, 5, 6])
    arr2 = np.array([7, 8, 9])

    new_domain, new_arrs = equalize_domains([domain1, domain2], [arr1, arr2])

    assert np.allclose(new_domain, np.array([3, 6, 9]))
    assert np.allclose(new_arrs[0], np.array([2.5, 4, 5.5]))
    assert np.allclose(new_arrs[1], np.array([7, 8, 9]))

def test_equalize_domains_stack():
    # test stacking
    domain1 = np.array([0, 1, 2, 3, 4, 5])
    domain2 = np.array([3, 4, 5, 6, 7])
    arr1 = np.array([1, 2, 3, 4, 5, 6])
    arr2 = np.array([7, 8, 9, 10, 11])

    new_domain, new_arrs = equalize_domains([domain1, domain2], [arr1, arr2], stack_axis=0)

    assert np.allclose(new_domain, np.array([3, 4, 5]))
    assert np.allclose(new_arrs, np.array([[4, 5, 6], [7, 8, 9]]))

def test_equalize_domains_concatenate():
    # test concatenating
    domain1 = np.array([0, 1, 2, 3, 4, 5])
    domain2 = np.array([3, 4, 5, 6, 7])
    arr1 = np.array([1, 2, 3, 4, 5, 6])
    arr2 = np.array([7, 8, 9, 10, 11])

    new_domain, new_arrs = equalize_domains([domain1, domain2], [arr1, arr2], stack_axis=0, concatenate=True)

    assert np.allclose(new_domain, np.array([3, 4, 5]))
    assert np.allclose(new_arrs, np.array([4, 5, 6, 7, 8, 9]))
