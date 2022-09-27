"""
Load human spectral sensitivity
"""

import os
import pandas as pd
import numpy as np
from dreye import DREYE_DIR

D2_FILENAME = os.path.join(DREYE_DIR, 'datasets', 'linss2_10e_1.csv')
D10_FILENAME = os.path.join(DREYE_DIR, 'datasets', 'linss10e_1.csv')


def load_dataset(as_wide=False, d10=False, ascending=False):
    """
    Load human spectral sensitivity.

    Parameters
    ----------
    as_wide : bool, optional
        Whether to return a `dreye.Photoreceptor`. If False,
        returns a long-format `pandas.DataFrame`. Defaults to False.
    d10 : bool, optional
        Whether to return the 10 degree measurements. Otherwise, fetches the
        2 degree measurements.

    Returns
    -------
    df : `pandas.DataFrame`
        A long-format `pandas.DataFrame` with the following columns:
            * `wavelengths`
            * `opsin`
            * `value`
        Or a wide-format `pandas.DataFrame` with columns `S`, `M`, and `L` and 
        row indices `wavelengths`.

    References
    ----------
    .. [1] Stockman, A, MacLeod, D.I.A, Johnson N.E. (1993)
        Spectral sensitivities of the human cones.
        J. Opt. Soc. Am. A 10, 12, 2491-2521.
    """
    if d10:
        data = pd.read_csv(D10_FILENAME, header=None)
    else:
        data = pd.read_csv(D2_FILENAME, header=None)

    data.columns = ['wavelengths', 'L', 'M', 'S']
    if ascending:
        data = data[['wavelengths', 'S', 'M', 'L']]
    
    nanmin = np.nanmin(data[['S', 'M', 'L']].to_numpy())
    
    if as_wide:
        data = data.set_index('wavelengths')
        data = data.fillna(nanmin)
        return data

    return data.melt(['wavelengths'], var_name='opsin').fillna(nanmin)
