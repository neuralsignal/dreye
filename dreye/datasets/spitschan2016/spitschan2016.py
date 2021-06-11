"""
Functions to process Spitschan 2016 (Outdoor illumination ...)
spectrum data files.

Reference:
Spitschan, M., Aguirre, G., Brainard, D. et al.
Variation of outdoor illumination as a function of solar elevation and light pollution.
Sci Rep 6, 26756 (2016). https://doi.org/10.1038/srep26756
"""

import os
import pandas as pd
import numpy as np
from dreye.utilities import irr2flux
from dreye import DREYE_DIR, Signals


_WL_FILE = 'srep26756-s1.csv'
_RURAL_FILE = 'srep26756-s2.csv'
_CITY_FILE = 'srep26756-s3.csv'
SPITSCHAN2016_DATA = {
    'rural': {
        'filepath': os.path.join(DREYE_DIR, 'datasets', 'spitschan2016', _RURAL_FILE),
        'wl_filepath': os.path.join(DREYE_DIR, 'datasets', 'spitschan2016', _WL_FILE)
    },
    'city': {
        'filepath': os.path.join(DREYE_DIR, 'datasets', 'spitschan2016', _CITY_FILE),
        'wl_filepath': os.path.join(DREYE_DIR, 'datasets', 'spitschan2016', _WL_FILE)
    }
}
SPITSCHAN2016_FILE = os.path.join(
    DREYE_DIR, 'datasets', 'spitschan2016', 'spitschan2016.feather'
)


def load_dataset(as_spectra=False, label_cols='data_id'):
    """
    Load Spitschan dataset of various spectra for different times of day as a dataframe.

    Parameters
    ----------
    as_spectra : bool, optional
        Whether to return a `dreye.IntensitySpectra`. If False,
        returns a long-format `pandas.DataFrame`. Defaults to False.
    label_cols : str or list-like, optional
        The label columns for the `dreye.IntensitySpectra` instance.
        Defaults to `data_id`.

    Returns
    -------
    df : `pandas.DataFrame` or `dreye.IntensitySpectra`
        A long-format `pandas.DataFrame` with the following columns:
            * `date_number`
            * `solar_elevation`
            * `lunar_elevation`
            * `fraction_moon_illuminated`
            * `timestamp`
            * `hour`
            * `month`
            * `data_id`
            * `wavelengths`
            * `spectralirradiance`
            * `microspectralphotonflux`
            * `location`
        Or a `dreye.IntensitySpectra` instance in units of
        microspectralphotonflux and labels along the column axis.
        The column labels are a `pandas.MultiIndex`.

    References
    ----------
    .. [1] Spitschan, M., Aguirre, G.K., Brainard, D.H., Sweeney, A.M. (2016)
       Variation of outdoor illumination as a function of solar elevation and light pollution.
       Sci Rep 6, 26756.
    """
    if os.path.exists(SPITSCHAN2016_FILE):
        df = pd.read_feather(SPITSCHAN2016_FILE)
    else:
        df = pd.DataFrame()
        for location, kwargs in SPITSCHAN2016_DATA.items():
            df_ = process_spitschan(**kwargs)
            df_['location'] = location
            if len(df) > 0:
                df_['data_id'] += (df['data_id'].max() + 1)
            df = df.append(df_, ignore_index=True)

    if as_spectra:
        return Signals(
            pd.pivot_table(
                df,
                'microspectralphotonflux',
                'wavelengths',
                label_cols
            ),
            units='uE', 
            domain_units='nm'
        )

    return df


def _process_spitschan(filepath, wl_filepath):
    """
    Process Spitschan2016 file and return metadata and data
    """
    wls = pd.read_csv(wl_filepath, header=None, names=['wavelengths'])
    df = pd.read_csv(filepath).iloc[:, :-1]
    # format metadata
    metadata = df.iloc[:4]
    metadata.index = [
        'date_number', 'solar_elevation',
        'lunar_elevation', 'fraction_moon_illuminated'
    ]
    metadata = metadata.T
    metadata['timestamp'] = pd.to_datetime(metadata.index)
    metadata['hour'] = metadata['timestamp'].dt.hour
    metadata['month'] = metadata['timestamp'].dt.month
    metadata.index = np.arange(metadata.shape[0])
    metadata['data_id'] = metadata.index
    # reformat data
    data = df.iloc[4:]
    data.index = wls['wavelengths']
    data.columns = np.arange(data.shape[1])
    data = data.T
    return metadata, data


def process_spitschan(filepath, wl_filepath):
    """
    Process Spitschan2016 file and return a long dataframe
    """
    metadata, data = _process_spitschan(filepath, wl_filepath)
    # create long format pandas dataframe
    data.index = pd.MultiIndex.from_frame(metadata)
    data = data.stack()
    data.name = 'spectralirradiance'
    data = data.reset_index()
    data['microspectralphotonflux'] = irr2flux(
        data['spectralirradiance'], data['wavelengths']
    ) * 10 ** 6
    return data
