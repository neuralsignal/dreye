"""
Various sensitivity template functions
"""

from typing import Union

import numpy as np
from scipy.stats import norm


def stavenga1993_band_calculation(
    x: np.ndarray, a: Union[float, np.ndarray], b: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Calculate band according to Stavenga et al (1993).

    Parameters
    ----------
    x : np.ndarray
        Input array.
    a : Union[float, np.ndarray]
        Parameter related to the width and shape of the band.
    b : Union[float, np.ndarray]
        Parameter related to the width and shape of the band.

    Returns
    -------
    np.ndarray
        Band calculated using the formula from Stavenga et al (1993).
    """
    return np.exp(-a * x**2 * (1 + b * x + 3 / 8 * (b * x) ** 2))


def gaussian_template(
    wavelengths: np.ndarray,
    mean: Union[float, np.ndarray],
    std: Union[float, np.ndarray] = 30.0,
) -> np.ndarray:
    """
    Create Gaussian filter template normalized to the max.

    Parameters
    ----------
    wavelengths : np.ndarray
        The wavelength array.
    mean : Union[float, np.ndarray]
        The mean of each filter.
    std : Union[float, np.ndarray], optional
        The standard deviation of each filter, default is 30.

    Returns
    -------
    np.ndarray
        The filter templates as a numpy ndarray.
    """
    y = norm.pdf(wavelengths, mean, std)
    return y / np.max(y, axis=-1, keepdims=True)


def stavenga1993_template(
    wavelengths: np.ndarray,
    alpha_max: Union[float, np.ndarray],
    a_alpha: Union[float, np.ndarray] = 380.0,
    b_alpha: Union[float, np.ndarray] = 6.09,
    beta_max: Union[float, np.ndarray] = 350.0,
    A_beta: Union[float, np.ndarray] = 0.29,
    a_beta: Union[float, np.ndarray] = 247.0,
    b_beta: Union[float, np.ndarray] = 3.59,
) -> np.ndarray:
    """
    Calculate opsin template according to Stavenga et al (1993).

    Parameters
    ----------
    wavelengths : np.ndarray
        The wavelength array.
    alpha_max : Union[float, np.ndarray]
        The wavelength peak of each filter.
    a_alpha : Union[float, np.ndarray], optional
        Parameter related to the width and shape of the alpha band, default is 380.
    b_alpha : Union[float, np.ndarray], optional
        Parameter related to the width and shape of the alpha band, default is 6.09.
    beta_max : Union[float, np.ndarray], optional
        The wavelength peak of the beta band, default is 350.
    A_beta : Union[float, np.ndarray], optional
        Parameter that determines the proportion of the beta band, default is 0.29.
    a_beta : Union[float, np.ndarray], optional
        Parameter related to the width and shape of the beta band, default is 247.
    b_beta : Union[float, np.ndarray], optional
        Parameter related to the width and shape of the beta band, default is 3.59.

    Returns
    -------
    templates : np.ndarray of shape (n_filters, n_wls)
        The filter templates as a numpy ndarray.

    References
    ----------
    .. [1] D.G. Stavenga, R.P. Smits, B.J. Hoenders,
        Simple exponential functions describing the absorbance bands of visual pigment spectra,
        Vision Research, Volume 33, Issue 8, 1993.
    """
    x_alpha = np.log10(wavelengths / alpha_max)
    alpha_band = stavenga1993_band_calculation(x_alpha, a_alpha, b_alpha)

    x_beta = np.log10(wavelengths / beta_max)
    beta_band = stavenga1993_band_calculation(x_beta, a_beta, b_beta)

    return alpha_band + A_beta * beta_band


def govardovskii2000_template(
    wavelengths: np.ndarray,
    alpha_max: Union[float, np.ndarray],
    A_alpha: Union[float, np.ndarray] = 69.7,
    a_alpha1: Union[float, np.ndarray] = 0.8795,
    a_alpha2: Union[float, np.ndarray] = 0.0459,
    a_alpha3: Union[float, np.ndarray] = 300.0,
    a_alpha4: Union[float, np.ndarray] = 11940.0,
    B_alpha: Union[float, np.ndarray] = 28.0,
    b_alpha: Union[float, np.ndarray] = 0.922,
    C_alpha: Union[float, np.ndarray] = -14.9,
    c_alpha: Union[float, np.ndarray] = 1.104,
    D_alpha: Union[float, np.ndarray] = 0.674,
    A_beta: Union[float, np.ndarray] = 0.26,
    beta_max1: Union[float, np.ndarray] = 189.0,
    beta_max2: Union[float, np.ndarray] = 0.315,
    d_beta1: Union[float, np.ndarray] = -40.5,
    d_beta2: Union[float, np.ndarray] = 0.195,
) -> np.ndarray:
    """
    Calculate Opsin template according to Govardovskii et al (2000).

    Parameters
    ----------
    wavelengths : np.ndarray
        The wavelength array.
    alpha_max : Union[float, np.ndarray]
        The wavelength peak of each filter.
    A_alpha : Union[float, np.ndarray], optional
        Relates to the width and shape of the alpha band, default is 69.7
    a_alpha1 : Union[float, np.ndarray], optional
        Relates to the width and shape of the alpha band, default is 0.8795
    a_alpha2 : Union[float, np.ndarray], optional
        Relates to the width and shape of the alpha band, default is 0.0459
    a_alpha3 : Union[float, np.ndarray], optional
        Relates to the width and shape of the alpha band, default is 300
    a_alpha4 : Union[float, np.ndarray], optional
        Relates to the width and shape of the alpha band, default is 11940
    B_alpha : Union[float, np.ndarray], optional
        Relates to the width and shape of the alpha band, default is 28
    b_alpha : Union[float, np.ndarray], optional
        Relates to the width and shape of the alpha band, default is 0.922
    C_alpha : Union[float, np.ndarray], optional
        Relates to the width and shape of the alpha band, default is -14.9
    c_alpha : Union[float, np.ndarray], optional
        Relates to the width and shape of the alpha band, default is 1.104
    D_alpha : Union[float, np.ndarray], optional
        Relates to the width and shape of the alpha band, default is 0.674
    A_beta : Union[float, np.ndarray], optional
         Determines the proportion of the beta band, default is 0.26
    beta_max1 : Union[float, np.ndarray], optional
        Relates to the peak and shape of the beta band, default is 189
    beta_max2 : Union[float, np.ndarray], optional
        Relates to the peak and shape of the beta band, default is 0.315
    d_beta1 : Union[float, np.ndarray], optional
        Relates to the peak and shape of the beta band, default is -40.5
    d_beta2 : Union[float, np.ndarray], optional
        Relates to the peak and shape of the beta band, default is 0.195

    Returns
    -------
    templates : np.ndarray of shape (n_filters, n_wls)
        The filter templates as a numpy ndarray.

    References
    ----------
    .. [1] Govardovskii, V. I., Fyhrquist, N., Reuter, T., Kuzmin, D. G., & Donner, K.
        In search of the visual pigment template.
        Visual neuroscience, 17(4), 509-528, 2000.
    """
    x_alpha = (wavelengths / alpha_max) ** -1
    a_alpha = a_alpha1 + a_alpha2 * np.exp(-((alpha_max - a_alpha3) ** 2) / a_alpha4)

    alpha_band = (
        np.exp(A_alpha * (a_alpha - x_alpha))
        + np.exp(B_alpha * (b_alpha - x_alpha))
        + np.exp(C_alpha * (c_alpha - x_alpha))
        + D_alpha
    ) ** -1

    beta_max = beta_max1 + beta_max2 * alpha_max
    d_beta = d_beta1 + d_beta2 * alpha_max
    beta_band = np.exp(-(((wavelengths - beta_max) / d_beta) ** 2))

    return alpha_band + A_beta * beta_band
