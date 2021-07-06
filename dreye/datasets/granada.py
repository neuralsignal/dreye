"""
Reference:

Color and spectral analysis of daylight in southern Europe
Javier Hernandez-Andres, Javier Romero, and Juan L. Nieves
Raymond L. Lee, Jr.
J. Opt. Soc. Am. A, 2001
"""

import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
from dreye.utilities import irr2flux
from dreye import DREYE_DIR, Signals

GRANADA_DATAFILE = os.path.join(DREYE_DIR, 'datasets', 'granada_daylight.mat')


def load_dataset(as_spectra=False):
    """
    Load Granada two-year daylight spectra as a dataframe.
    Largely urban daylight spectra.

    Parameters
    ----------
    as_spectra : bool, optional
        Whether to return a `dreye.IntensitySpectra`. If False,
        returns a long-format `pandas.DataFrame`. Defaults to False.

    Returns
    -------
    df : `pandas.DataFrame` or `dreye.IntensitySpectra`
        A long-format `pandas.DataFrame` with the following columns:
            * `data_id`
            * `wavelengths`
            * `spectralirradiance`
            * `microspectralphotonflux`
        Or a `dreye.IntensitySpectra` instance in units of
        microspectralphotonflux and `data_id` labels along the column axis.

    References
    ----------
    .. [1] Hernández-Andrés, J., Romero, J., Nieves, J.L., Lee, R.L. (2001)
        Color and spectral analysis of daylight in southern Europe.
        J. Opt. Soc. Am. A 18, 1325-1335.
    """
    irr = loadmat(GRANADA_DATAFILE)['final'].T
    wls = np.linspace(300, 1100, 161)

    assert wls.shape[-1] == irr.shape[-1]

    series = pd.DataFrame(
        irr,
        columns=pd.Index(wls, name='wavelengths'),
        index=pd.Index(np.arange(irr.shape[0]), name='data_id')
    ).stack().fillna(0)
    series[series < 0] = 0
    series.name = 'spectralirradiance'
    df = series.reset_index()
    df['microspectralphotonflux'] = irr2flux(
        df['spectralirradiance'], df['wavelengths']
    ) * 10 ** 6

    if as_spectra:
        return Signals(
            df.pivot(
                'wavelengths',
                'data_id',
                'microspectralphotonflux'
            ).fillna(0),
            units='uE', 
            domain_units='nm'
        )
    return df
