"""Basic plotting functions."""

from numbers import Number
from typing import Optional, Union, List

import numpy as np
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


def clip_scale(
    data: np.array, vmin: Union[Number, np.array], vmax: Union[Number, np.array]
) -> np.array:
    """
    Scales and clips the data between a given range.

    Parameters
    ----------
    data : np.array
        Input data array.
    vmin : Number or np.array
        Minimum value for scaling.
    vmax : Number or np.array
        Maximum value for scaling.

    Returns
    -------
    np.array
        Scaled and clipped data.
    """
    if vmin >= vmax:
        raise ValueError(f"vmin {vmin} should be less than vmax {vmax}.")
    scaled = (data - vmin) / (vmax - vmin)
    return np.clip(scaled, 0, 1)


def simple_plotting_function(
    x: Union[Number, np.array],
    ys: np.array,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Plots multiple y series over the same x values.

    Parameters
    ----------
    x : Number or np.array
        x values. If a single number is provided, it generates an array of that length.
    ys : np.array
        y values, each row is a series to be plotted.
    labels : List[str], optional
        Labels for the series, by default None.
    colors : List[str], optional
        Colors for the series, by default None.
    ax : plt.Axes, optional
        Matplotlib axes object to plot on, by default None.

    Returns
    -------
    plt.Axes
        The axes object with the plot.
    """
    ax = plt.gca() if ax is None else ax
    x = np.arange(ys.shape[-1]) if isinstance(x, Number) else x
    colors = sns.color_palette("tab10", ys.shape[0]) if colors is None else colors
    labels = np.arange(ys.shape[0]) if labels is None else labels

    for label, y, color in zip(labels, ys, colors):
        kwargs["label"] = label
        kwargs["color"] = color
        ax.plot(x, y, **kwargs)

    return ax


def gradient_color_lineplot(
    *xargs: np.array,
    c: np.array,
    cmap: str = "viridis",
    add_colorbar: bool = True,
    vmin: Optional[Number] = None,
    vmax: Optional[Number] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Plots a line with gradient color.

    Parameters
    ----------
    *xargs : np.array
        x and y values.
    c : np.array
        Color values.
    cmap : str, optional
        Colormap name, by default 'viridis'.
    add_colorbar : bool, optional
        Whether to add a colorbar, by default True.
    vmin : Number, optional
        Minimum value for color scaling, by default None.
    vmax : Number, optional
        Maximum value for color scaling, by default None.
    ax : plt.Axes, optional
        Matplotlib axes object to plot on, by default None.

    Returns
    -------
    plt.Axes
        The axes object with the plot.
    """
    ax = plt.gca() if ax is None else ax

    # remove color from kwargs
    kwargs = kwargs.copy()
    kwargs.pop("color", None)
    color = c

    vmin = np.min(color) if vmin is None else vmin
    vmax = np.max(color) if vmax is None else vmax
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    if add_colorbar:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        mcolorbar.ColorbarBase(ax=cax, cmap=cmap, norm=norm, orientation="vertical")

    xs = np.stack(xargs).T
    for idx, (ixs) in enumerate(zip(xs[:-1], xs[1:])):
        xy_ = np.stack(ixs).T
        cvalue = (color[idx] + color[idx + 1]) / 2
        cvalue = clip_scale(cvalue, vmin, vmax)
        ax.plot(*xy_, color=cmap(cvalue), **kwargs)

    return ax


def hull_outline(
    hull: Union[ConvexHull, np.array],
    ax: Optional[plt.Axes] = None,
    color: str = "black",
    **kwargs
) -> plt.Axes:
    """
    Plots the outline of a ConvexHull object.

    Parameters
    ----------
    hull : ConvexHull or np.array
        ConvexHull object or points to generate one.
    ax : plt.Axes, optional
        Matplotlib axes object to plot on, by default None.
    color : str, optional
        Outline color, by default 'black'.

    Returns
    -------
    plt.Axes
        The axes object with the plot.
    """
    ax = plt.gca() if ax is None else ax

    if not isinstance(hull, ConvexHull):
        hull = ConvexHull(hull)

    pts = hull.points

    for simplex in hull.simplices:
        ax.plot(pts[simplex, 0], pts[simplex, 1], color=color, **kwargs)

    return ax


def vectors_plot(
    X: np.array,
    offsets: Union[Number, np.array] = 0,
    ax: Optional[plt.Axes] = None,
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    **kwargs
) -> plt.Axes:
    """
    Plots vectors from given offsets.

    Parameters
    ----------
    X : np.array
        Vectors to plot.
    offsets : Number or np.array, optional
        Offsets for the vectors, by default 0.
    ax : plt.Axes, optional
        Matplotlib axes object to plot on, by default None.
    colors : List[str], optional
        Colors for the vectors, by default None.
    labels : List[str], optional
        Labels for the vectors, by default None.

    Returns
    -------
    plt.Axes
        The axes object with the plot.
    """
    ax = plt.gca() if ax is None else ax

    X = np.asarray(X)
    offsets = np.broadcast_to(offsets, X.shape)

    if colors is None:
        colors = sns.color_palette("tab10", len(X))
    if labels is None:
        labels = np.arange(len(X))

    for x, offset, color, label in zip(X, offsets, colors, labels):
        ax.arrow(
            offset[0],
            offset[1],
            dx=x[0] - offset[0],
            dy=x[1] - offset[1],
            label=label,
            color=color,
            **kwargs
        )

    return ax
