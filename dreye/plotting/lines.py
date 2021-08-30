"""
Line utility functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

def clip_scale(data, vmin, vmax):
    scaled = (data - vmin) / (vmax - vmin)
    return np.clip(scaled, 0, 1)

def gradient_color_lineplot(
    *xargs,
    c, cmap='viridis', add_colorbar=True,
    vmin=None, vmax=None, ax=None, **kwargs
):
    if ax is None:
        ax = plt.gca()
        
    # remove color from kwargs
    kwargs = kwargs.copy()
    kwargs.pop('color', None)
    color = c
        
    vmin = (np.min(color) if vmin is None else vmin)
    vmax = (np.max(color) if vmax is None else vmax)
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
        
    if add_colorbar:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        mpl.colorbar.ColorbarBase(
            ax=cax, cmap=cmap,
            norm=norm,
            orientation='vertical'
        )
        
    xs = np.stack(xargs).T
    for idx, (ixs) in enumerate(zip(xs[:-1], xs[1:])):
        xy_ = np.stack(ixs).T
        cvalue = (color[idx] + color[idx+1]) / 2
        cvalue = clip_scale(cvalue, vmin, vmax)
        ax.plot(
            *xy_, 
            color=cmap(cvalue), 
            **kwargs
        )
        
    return ax