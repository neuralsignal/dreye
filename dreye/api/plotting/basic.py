"""basic functions
"""

from numbers import Number
import numpy as np
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


def clip_scale(data, vmin, vmax):
    """[summary]

    Parameters
    ----------
    data : [type]
        [description]
    vmin : [type]
        [description]
    vmax : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    scaled = (data - vmin) / (vmax - vmin)
    return np.clip(scaled, 0, 1)


def simple_plotting_function(x, ys, labels=None, colors=None, ax=None, **kwargs):
    """[summary]

    Parameters
    ----------
    x : [type]
        [description]
    ys : [type]
        [description]
    labels : [type], optional
        [description], by default None
    colors : [type], optional
        [description], by default None
    ax : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    if ax is None:
        ax = plt.gca()
        
    if isinstance(x, Number):
        x = np.arange(ys.shape[-1])
        
    if colors is None:
        colors = sns.color_palette('tab10', ys.shape[0])
        
    if labels is None:
        labels = np.arange(ys.shape[0])

    for label, y, color in zip(labels, ys, colors):
        kwargs['label'] = label
        kwargs['color'] = color
        ax.plot(x, y, **kwargs)
    
    return ax


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
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        mcolorbar.ColorbarBase(
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


def hull_outline(hull, ax=None, color='black', **kwargs):
    """[summary]

    Parameters
    ----------
    hull : [type]
        [description]
    ax : [type], optional
        [description], by default None
    color : str, optional
        [description], by default 'black'
    """
    if ax is None:
        ax = plt.gca()
    
    if not isinstance(hull, ConvexHull):
        hull = ConvexHull(hull)
    
    pts = hull.points
    
    for simplex in hull.simplices:
        ax.plot(
            pts[simplex, 0], pts[simplex, 1],
            color=color, **kwargs
        )
        
    return ax


def vectors_plot(X, offsets=0, ax=None, colors=None, labels=None, **kwargs):
    if ax is None:
        ax = plt.gca()
        
    X = np.asarray(X)
    offsets = np.broadcast_to(offsets, X.shape)
        
    if colors is None:
        colors = sns.color_palette('tab10', len(X))
    if labels is None:
        labels = np.arange(len(X))
        
    for x, offset, color, label in zip(X, offsets, colors, labels):
        ax.arrow(
            offset[0], offset[1], 
            dx=x[0]-offset[0], 
            dy=x[1]-offset[1], 
            label=label, 
            color=color, 
            **kwargs
        )
        
    return ax