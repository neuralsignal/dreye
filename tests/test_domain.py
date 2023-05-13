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
