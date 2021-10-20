"""
Test stimuli
"""

from . import context
from .context import test_datapath
from dreye.io import read_json, write_json

import pandas as pd
import numpy as np
import os


def test_df_serialize():
    filename = os.path.join(test_datapath, 'test_df.json')
    df = pd.DataFrame(
        np.random.random((50, 50)),
        index=pd.MultiIndex.from_arrays(
            [np.arange(50), np.arange(50, 100)], names=['a', 'b']
        ),
        columns=pd.MultiIndex.from_arrays(
            [np.arange(50), np.arange(50, 100)], names=['a', 'b']
        )
    )
    write_json(filename, df)
    df_ = read_json(filename)

    assert np.allclose(df.to_numpy(), df_.to_numpy())
    assert np.all(df.index == df_.index)
    assert np.all(df.columns == df_.columns)
