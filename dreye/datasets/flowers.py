"""
Reference:
http://www.reflectance.co.uk
"""

import os
import pandas as pd

from dreye import DREYE_DIR, Signals


FLOWER_PATH = os.path.join(
    DREYE_DIR, 'datasets', 'flowers.feather'
)


def load_dataset(as_spectra=False, label_cols='data_id'):
    """
    Load a set of flower reflectances as a dataframe.

    Parameters
    ----------
    as_spectra : bool, optional
        Whether to return a `dreye.Spectra`. If False,
        returns a long-format `pandas.DataFrame`. Defaults to False.
    label_cols : str or list-like, optional
        The label columns for the `dreye.Spectra` instance.
        Defaults to `data_id`.

    Returns
    -------
    df : `pandas.DataFrame` or `dreye.IntensitySpectra`
        A long-format `pandas.DataFrame` with the following columns:
            * `country`
            * `flower_genus`
            * `human_color`
            * `is_main_color`
            * `info_id`
            * `bee_color`
            * `part_of_flower`
            * `data_id`
            * `flower_species`
            * `wavelengths`
            * `reflectance`
            * `flower_family`
        Or a `dreye.Spectra` instance in dimensionless units and labels
        along the column axis.
        The column labels are a `pandas.MultiIndex`.

    References
    ----------
    .. [1] Chittka, L., Shmida, A., Troje, N., & Menzel, R. (1994)
        Ultraviolet as a component of flower reflections, and the colour perception of Hymenoptera.
        Vision Research, 34, 1489-1508.

    .. [2] Chittka, L. (1996)
        Optimal sets of colour receptors and opponent processes for coding of natural objects in insect vision.
        Journal of Theoretical Biology, 181, 179-196.

    .. [3] Chittka, L. (1997)
        Bee color vision is optimal for coding flower color, but flower colors are not optimal for being coded - why?
        Israel Journal of Plant Sciences, 45, 115-127.

    .. [4] Gumbert, A., Kunze, J., & Chittka, L. (1999)
        Floral colour diversity in plant communities, bee colour space and a null model.
        Proceedings of the Royal Society B: Biological Sciences, 266, 1711-1716.

    .. [5] Menzel, R. & Shmida, A. (1993)
        The ecology of flower colours and the natural colour vision of insect pollinators: the Israeli flora as a study case.
        Biological Reviews of the Cambridge Philosophical Society, 68, 81-120.
    """
    df = pd.read_feather(FLOWER_PATH)
    # BUG in dataset with duplicate values
    df = df.drop_duplicates()

    if as_spectra:
        return Signals(
            pd.pivot_table(df, 'reflectance', 'wavelengths', label_cols), 
            domain_units='nm'
        )

    return df
