import numpy as np
import pytest
from dreye.api.filter_templates import (
    stavenga1993_template,
    govardovskii2000_template,
    gaussian_template,
)


@pytest.fixture
def wavelengths():
    return np.linspace(300, 700, 100)


def test_gaussian_template(wavelengths):
    # Test with normal values
    mean = 550
    std = 50
    templates = gaussian_template(wavelengths, mean, std)
    assert templates.shape == wavelengths.shape
    assert np.isclose(np.max(templates), 1.0)

    # Test with edge case, zero std
    templates = gaussian_template(wavelengths, mean, 0)
    assert np.isnan(templates).any()



def test_stavenga1993_template(wavelengths):
    # Test with normal values
    alpha_max = 550
    templates = stavenga1993_template(wavelengths, alpha_max)
    assert templates.shape == wavelengths.shape

    # Test with alpha_max being outside of wavelengths range
    alpha_max = 800
    templates = stavenga1993_template(wavelengths, alpha_max)
    assert templates.shape == wavelengths.shape


def test_govardovskii2000_template(wavelengths):
    # Test with normal values
    alpha_max = 550
    templates = govardovskii2000_template(wavelengths, alpha_max)
    assert templates.shape == wavelengths.shape

    # Test with alpha_max being outside of wavelengths range
    alpha_max = 800
    templates = govardovskii2000_template(wavelengths, alpha_max)
    assert templates.shape == wavelengths.shape
