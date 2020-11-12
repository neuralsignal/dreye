"""
Test stimuli
"""

from . import context
from dreye import read_json, write_json

import pandas as pd
import numpy as np


def test_df_serialize():
    filename = 'test_data/test_df.json'
    df = pd.DataFrame(
        np.random.random((50, 50)),
        index=pd.MultiIndex.from_arrays(
            [np.arange(50), np.arange(50, 100)], names=['a', 'b']
        ),
        columns=pd.MultiIndex.from_arrays(
            [np.arange(50), np.arange(50, 100)], names=['a', 'b']
        )
    )
    df[('extra')] = 'c'
    df[('extra1', 'extra2')] = 'd'
    write_json(filename, df)
    df_ = read_json(filename)

    assert np.allclose(df.to_numpy() == df_.to_numpy())
    assert np.all(df.index == df_.index)
    assert np.all(df.columns == df_.columns)
