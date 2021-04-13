import dreye
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_theme(
    style='ticks', 
    context='talk',
)

fly_opsins = dreye.Sensitivity(
    pd.read_csv('data/fly_opsins.csv').set_index('wls'), 
    name='fruit fly', 
    labels=['ultrashort', 'short', 'medium', 'long']
)
fly_model = dreye.LinearPhotoreceptor(fly_opsins)
fly_log_model = dreye.LogPhotoreceptor(fly_opsins)
wls = fly_model.sensitivity.domain.magnitude
fish_opsins = dreye.Sensitivity(
    [365, 416, 483, 567],
    domain=wls, 
    from_template=True, 
    name='zebrafish', 
    labels=['ultrashort', 'short', 'medium', 'long']
)
fish_model = dreye.LinearPhotoreceptor(fish_opsins)
fish_log_model = dreye.LogPhotoreceptor(fish_opsins)

cmap = {'ultrashort': 'deeppink', 'short': 'purple', 'medium': 'blue', 'long': 'green'}
n = 4