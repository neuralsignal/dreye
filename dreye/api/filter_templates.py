"""
Various sensitivity template functions
"""

import numpy as np
from scipy.stats import norm


def stavenga1993_band_calculation(x, a, b):
    """Band calculation according to Stavenga et al (1993).

    Parameters
    ----------
    x : [type]
        [description]
    a : [type]
        [description]
    b : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return np.exp(
        -a * x ** 2 * (
            1 + b * x + 3 / 8 * (b * x) ** 2
        )
    )


def gaussian_template(wavelengths, mean, std=30):
    """[summary]

    Parameters
    ----------
    wavelengths : [type]
        [description]
    mean : [type]
        [description]
    std : int, optional
        [description], by default 30

    Returns
    -------
    [type]
        [description]
    """
    y = norm.pdf(wavelengths, mean, std)
    return y / np.max(y, axis=0, keepdims=True)


def stavenga1993_template(
    wavelengths,
    alpha_max,
    a_alpha=380,
    b_alpha=6.09,
    beta_max=350,
    A_beta=0.29,
    a_beta=247,
    b_beta=3.59,
):
    """Calculate opsin template according to Stavenga et al (1993).

    Parameters
    ----------
    wavelengths : [type]
        [description]
    alpha_max : [type]
        [description]
    a_alpha : int, optional
        [description], by default 380
    b_alpha : float, optional
        [description], by default 6.09
    beta_max : int, optional
        [description], by default 350
    A_beta : float, optional
        [description], by default 0.29
    a_beta : int, optional
        [description], by default 247
    b_beta : float, optional
        [description], by default 3.59

    Returns
    -------
    [type]
        [description]
    """

    x_alpha = np.log10(wavelengths / alpha_max)
    alpha_band = stavenga1993_band_calculation(x_alpha, a_alpha, b_alpha)

    beta_band = stavenga1993_beta_band_template(
        wavelengths, beta_max=beta_max, a_beta=a_beta, b_beta=b_beta
    )
    return alpha_band + A_beta * beta_band


def stavenga1993_beta_band_template(
    wavelengths,
    beta_max=350,
    a_beta=247,
    b_beta=3.59,
):
    """Beta band calculation according to Stavenga et al (1993).

    Parameters
    ----------
    wavelengths : [type]
        [description]
    beta_max : int, optional
        [description], by default 350
    a_beta : int, optional
        [description], by default 247
    b_beta : float, optional
        [description], by default 3.59

    Returns
    -------
    [type]
        [description]
    """
    x_beta = np.log10(wavelengths / beta_max)
    beta_band = stavenga1993_band_calculation(x_beta, a_beta, b_beta)
    return beta_band


def govardovskii2000_template(
    wavelengths,
    alpha_max,
    A_alpha=69.7,
    a_alpha1=0.8795,
    a_alpha2=0.0459,
    a_alpha3=300,
    a_alpha4=11940,
    B_alpha=28,
    b_alpha=0.922,
    C_alpha=-14.9,
    c_alpha=1.104,
    D_alpha=0.674,
    A_beta=0.26,
    beta_max1=189,
    beta_max2=0.315,
    d_beta1=-40.5,
    d_beta2=0.195
):
    """Calculate Opsin template according to Govardovskii et al (2000).

    Parameters
    ----------
    wavelengths : [type]
        [description]
    alpha_max : [type]
        [description]
    A_alpha : float, optional
        [description], by default 69.7
    a_alpha1 : float, optional
        [description], by default 0.8795
    a_alpha2 : float, optional
        [description], by default 0.0459
    a_alpha3 : int, optional
        [description], by default 300
    a_alpha4 : int, optional
        [description], by default 11940
    B_alpha : int, optional
        [description], by default 28
    b_alpha : float, optional
        [description], by default 0.922
    C_alpha : float, optional
        [description], by default -14.9
    c_alpha : float, optional
        [description], by default 1.104
    D_alpha : float, optional
        [description], by default 0.674
    A_beta : float, optional
        [description], by default 0.26
    beta_max1 : int, optional
        [description], by default 189
    beta_max2 : float, optional
        [description], by default 0.315
    d_beta1 : float, optional
        [description], by default -40.5
    d_beta2 : float, optional
        [description], by default 0.195

    Returns
    -------
    [type]
        [description]
    """
    x_alpha = (wavelengths / alpha_max) ** -1
    a_alpha = a_alpha1 + a_alpha2 * np.exp(-(alpha_max - a_alpha3)**2 / a_alpha4)

    alpha_band = (
        np.exp(
            A_alpha * (a_alpha - x_alpha)
        )
        + np.exp(
            B_alpha * (b_alpha - x_alpha)
        )
        + np.exp(
            C_alpha * (c_alpha - x_alpha)
        )
        + D_alpha
    ) ** -1

    beta_band = govardovskii2000_beta_band_template(
        wavelengths,
        alpha_max,
        beta_max1=beta_max1,
        beta_max2=beta_max2,
        d_beta1=d_beta1,
        d_beta2=d_beta2
    )
    return alpha_band + A_beta * beta_band


def govardovskii2000_beta_band_template(
    wavelengths,
    alpha_max,
    beta_max1=189,
    beta_max2=0.315,
    d_beta1=-40.5,
    d_beta2=0.195
):
    """Calculate beta band according to Govardovskii et al (2000).

    Parameters
    ----------
    wavelengths : [type]
        [description]
    alpha_max : [type]
        [description]
    beta_max1 : int, optional
        [description], by default 189
    beta_max2 : float, optional
        [description], by default 0.315
    d_beta1 : float, optional
        [description], by default -40.5
    d_beta2 : float, optional
        [description], by default 0.195

    Returns
    -------
    [type]
        [description]
    """
    beta_max = beta_max1 + beta_max2 * alpha_max
    d_beta = d_beta1 + d_beta2 * alpha_max
    beta_band = np.exp(
        -((wavelengths - beta_max) / d_beta)**2
    )
    return beta_band
