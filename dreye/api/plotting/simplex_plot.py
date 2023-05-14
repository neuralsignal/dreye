"""
Plot barycentric
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D, art3d
from scipy.spatial import ConvexHull
from itertools import combinations
import matplotlib.pyplot as plt

from dreye.api.plotting.basic import gradient_color_lineplot
from dreye.api.barycentric import (
    barycentric_dim_reduction,
    barycentric_to_cartesian,
    barycentric_to_cartesian_transformer,
)


def plot_simplex(*args, **kwargs):
    """
    Plots a simplex with points and/or a convex hull.

    Parameters
    ----------
    n : int, optional
        The dimension of the simplex, must be 2, 3, or 4. Default is 4.
    points : np.ndarray, optional
        The points to be plotted in the simplex. They must have `n` or `n-1` dimensions.
    hull : ConvexHull or np.ndarray, optional
        Convex hull to be plotted. If an array, it is converted to a ConvexHull object.
    gradient_line : np.ndarray, optional
        Array of points to plot as a gradient line. Must have `n` or `n-1` dimensions.
    lines : bool, optional
        If True, draws lines between simplex vertices. Default is True.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes are created.
    line_color : str, optional
        Color for the lines between vertices. Default is 'black'.
    hull_color : str, optional
        Color for the convex hull. Default is 'gray'.
    labels : list of str, optional
        Labels for the vertices of the simplex. Length must be `n`.
    label_size : int, optional
        Font size for labels. Default is 16.
    point_colors : str or list, optional
        Colors for the points to be plotted. Default is 'blue'.
    hull_kws : dict, optional
        Additional keyword arguments for the ConvexHull plot.
    point_scatter_kws : dict, optional
        Additional keyword arguments for the scatter plot of points.
    gradient_line_kws : dict, optional
        Additional keyword arguments for the gradient line plot.
    label_kws : dict or list of dicts, optional
        Additional text keyword arguments for each label.
    gradient_color : np.ndarray, optional
        Colors for the gradient line. Default is a linear gradient.
    fig_kws : dict, optional
        Additional keyword arguments for the figure.
    remove_axes : bool, optional
        If True, removes the axes from the plot. Default is True.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plotted simplex.

    Raises
    ------
    ValueError
        If `points` or `gradient_line` have wrong dimensions.
    AssertionError
        If `n` is not 2, 3, or 4.
    """
    plotter = SimplexPlot(*args, **kwargs)
    return plotter.plot()


class SimplexPlot:
    """
    Plots a simplex with points and/or a convex hull.

    Parameters
    ----------
    n : int, optional
        The dimension of the simplex, must be 2, 3, or 4. Default is 4.
    points : np.ndarray, optional
        The points to be plotted in the simplex. They must have `n` or `n-1` dimensions.
    hull : ConvexHull or np.ndarray, optional
        Convex hull to be plotted. If an array, it is converted to a ConvexHull object.
    gradient_line : np.ndarray, optional
        Array of points to plot as a gradient line. Must have `n` or `n-1` dimensions.
    lines : bool, optional
        If True, draws lines between simplex vertices. Default is True.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes are created.
    line_color : str, optional
        Color for the lines between vertices. Default is 'black'.
    hull_color : str, optional
        Color for the convex hull. Default is 'gray'.
    labels : list of str, optional
        Labels for the vertices of the simplex. Length must be `n`.
    label_size : int, optional
        Font size for labels. Default is 16.
    point_colors : str or list, optional
        Colors for the points to be plotted. Default is 'blue'.
    hull_kws : dict, optional
        Additional keyword arguments for the ConvexHull plot.
    point_scatter_kws : dict, optional
        Additional keyword arguments for the scatter plot of points.
    gradient_line_kws : dict, optional
        Additional keyword arguments for the gradient line plot.
    label_kws : dict or list of dicts, optional
        Additional text keyword arguments for each label.
    gradient_color : np.ndarray, optional
        Colors for the gradient line. Default is a linear gradient.
    fig_kws : dict, optional
        Additional keyword arguments for the figure.
    remove_axes : bool, optional
        If True, removes the axes from the plot. Default is True.
    """

    def __init__(
        self,
        n,
        points=None,
        hull=None,
        gradient_line=None,
        lines=True,
        ax=None,
        line_color="black",
        hull_color="gray",
        labels=None,
        label_size=16,
        point_colors="blue",
        hull_kws=None,
        point_scatter_kws=None,
        gradient_line_kws=None,
        label_kws=None,
        gradient_color=None,
        fig_kws=None,
        remove_axes=True,
    ):
        """
        Initializes the SimplexPlot class with the given parameters.

        Parameters are the same as in the original `plot_simplex` function.
        """
        self.n = n
        self.points = points
        self.hull = hull
        self.gradient_line = gradient_line
        self.lines = lines
        self.ax = ax
        self.line_color = line_color
        self.labels = labels
        self.label_size = label_size

        self.hull_kws = hull_kws.copy() if hull_kws is not None else {}
        self.hull_color = self.hull_kws.pop("color", hull_color)

        self.point_scatter_kws = (
            point_scatter_kws.copy() if point_scatter_kws is not None else {}
        )
        point_colors = self.point_scatter_kws.pop("c", point_colors)
        point_colors = self.point_scatter_kws.pop("color", point_colors)
        self.point_colors = point_colors

        self.gradient_line_kws = (
            gradient_line_kws if gradient_line_kws is not None else {}
        )
        self.label_kws = label_kws if label_kws is not None else {}
        self.gradient_color = gradient_color
        self.fig_kws = fig_kws if fig_kws is not None else {}
        self.remove_axes = remove_axes

        self.initialize_plot()

    def initialize_plot(self):
        """
        Initializes the plot based on the input parameters.
        """
        assert self.n in {2, 3, 4}

        if self.ax is None:
            if self.n == 4:
                fig = plt.figure(**self.fig_kws)
                self.ax = Axes3D(fig, auto_add_to_figure=False)
                fig.add_axes(self.ax)
            else:
                fig = plt.figure(**self.fig_kws)
                self.ax = plt.subplot(111)

    def plot_points(self):
        """
        Plots the points on the simplex.
        """
        kws = self.point_scatter_kws.copy()
        kws["c"] = self.point_colors
        if self.points is not None:
            if self.points.shape[1] == self.n:
                X = barycentric_dim_reduction(self.points)
            elif self.points.shape[1] == (self.n - 1):
                X = self.points
            else:
                raise ValueError(
                    "`points` argument is the wrong dimension, "
                    f"must be `{self.n}` or `{self.n-1}`, but is `{self.points.shape[1]}`."
                )
            if self.n == 2:
                self.ax.scatter(X.squeeze(), np.ones(X.size) * 0.25, **kws)
            else:
                self.ax.scatter(*X.T, **kws)

    def plot_hull(self):
        """
        Plots the convex hull on the simplex.
        """
        hull_kws = self.hull_kws.copy()

        if self.hull is not None:
            if self.n == 2:
                if self.hull.shape[1] == self.n:
                    pts = barycentric_dim_reduction(self.hull)
                else:
                    pts = self.hull
            else:
                if not isinstance(self.hull, ConvexHull):
                    if self.hull.shape[1] == self.n:
                        self.hull = barycentric_dim_reduction(self.hull)
                    assert self.hull.shape[1] == (self.n - 1)
                    self.hull = ConvexHull(self.hull)
                pts = self.hull.points

            if self.n == 3:
                hull_kws["color"] = self.hull_color
                for simplex in self.hull.simplices:
                    self.ax.plot(
                        pts[simplex, 0],
                        pts[simplex, 1],
                        **hull_kws,
                    )
            elif self.n == 2:
                hull_kws["color"] = self.hull_color
                hull_kws["linewidth"] = 3
                self.ax.plot(pts, np.ones(pts.size) * 0.2, **hull_kws)
            else:
                org_triangles = [pts[s] for s in self.hull.simplices]
                f = Faces(org_triangles)
                g = f.simplify()

                hull_kws_default = {
                    "facecolors": self.hull_color,
                    "edgecolor": "lightgray",
                    "alpha": 0.8,
                }
                hull_kws = {**hull_kws_default, **hull_kws}
                pc = art3d.Poly3DCollection(g, **hull_kws)
                self.ax.add_collection3d(pc)

    def plot_lines(self):
        """
        Plots the lines on the simplex.
        """
        if self.lines:
            A = barycentric_to_cartesian_transformer(self.n)
            lines = combinations(A, 2)
            for line in lines:
                line = np.transpose(np.array(line))
                if self.n == 4:
                    self.ax.plot3D(*line, c=self.line_color)
                elif self.n == 2:
                    self.ax.plot(
                        line.squeeze(),
                        np.ones(line.size) * 0.35,
                        c=self.line_color,
                        linewidth=3,
                    )
                else:
                    self.ax.plot(*line, c=self.line_color)

    def plot_gradient_line(self):
        """
        Plots the gradient line on the simplex.
        """
        gradient_kws = self.gradient_line_kws.copy()
        if self.gradient_line is not None:
            if self.gradient_line.shape[1] == self.n:
                X = barycentric_dim_reduction(self.gradient_line)
            elif self.gradient_line.shape[1] == (self.n - 1):
                X = self.gradient_line
            else:
                raise ValueError(
                    "`points` argument is the wrong dimension, "
                    f"must be `{self.n}` or `{self.n-1}`, but is `{self.gradient_line.shape[1]}`."
                )

            if self.gradient_color is None:
                self.gradient_color = np.linspace(0, 1, X.shape[0])

            if self.n == 2:
                gradient_kws["c"] = self.gradient_color
                gradient_kws["linewidth"] = 3
                gradient_color_lineplot(
                    X.squeeze(),
                    np.ones(X.size) * 0.3,
                    **gradient_kws,
                    ax=self.ax,
                )
            else:
                if self.n == 4:
                    # TODO currently color bar cannot be shown on 3D
                    gradient_kws["add_colorbar"] = False
                gradient_kws["c"] = self.gradient_color
                gradient_color_lineplot(*X.T, **gradient_kws, ax=self.ax)

    def plot_labels(self):
        """
        Plots the labels on the simplex.
        """
        if self.labels is not None and self.n != 2:
            eye = np.eye(self.n)
            eye_cart = barycentric_to_cartesian(eye)
            if isinstance(self.label_kws, dict):
                self.label_kws = [self.label_kws] * len(self.labels)
            for idx, (point, label, label_kw) in enumerate(
                zip(eye_cart, self.labels, self.label_kws)
            ):
                text_kws_default = {
                    "fontsize": self.label_size,
                    "ha": "center",
                    "va": "center",
                }
                text_kws = {**text_kws_default, **label_kw}
                if self.n == 3:
                    self.ax.text(*point, label, **text_kws)
                elif self.n == 4:
                    self.ax.text(*point, label, zdir=(0, 0, 1), **text_kws)

    def finalize_plot(self):
        """
        Sets the final visual properties of the plot.
        """
        if self.n != 2:
            self.ax.axis("off")
        else:
            if self.remove_axes:
                self.ax.axis("off")
            else:
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                self.ax.spines["top"].set_color("none")
                self.ax.spines["right"].set_color("none")
                self.ax.spines["bottom"].set_color("none")
                self.ax.spines["left"].set_color("none")

    def plot(self):
        """
        Calls all the plotting methods and shows the plot.
        """
        self.plot_points()
        self.plot_hull()
        self.plot_lines()
        self.plot_gradient_line()
        self.plot_labels()
        self.finalize_plot()
        return self.ax


class Faces:
    """
    Class used to organize the faces of a 3D Convex Hull. It groups neighboring faces with the same normal together.

    Based on the method described in:
    https://stackoverflow.com/questions/49098466/plot-3d-convex-closed-regions-in-matplot-lib/49115448

    Attributes
    ----------
    method : str
        The method used for face simplification. Choices are "convexhull" or others.
    tri : np.ndarray
        The triangles that constitute the convex hull, rounded to a specified number of significant digits.
    grpinx : list of int
        List of group indices for the triangles. Initially, each triangle is assigned to its own group.
    inv : np.ndarray
        Array of indices that can revert the unique array of normals back to the original array of normals.

    Methods
    -------
    norm(self, sq)
        Calculates the norm of a triangle.
    isneighbor(self, tr1, tr2)
        Checks if two triangles are neighbors.
    order(self, v)
        Orders a set of points in a counter-clockwise manner based on the selected method.
    simplify(self)
        Simplifies the faces by grouping together neighboring faces that have the same normal.

    Parameters
    ----------
    tri : np.ndarray
        Array representing the triangles that constitute the convex hull.
    sig_dig : int, optional
        Number of significant digits to consider when comparing triangle normals. Default is 12.
    method : str, optional
        Method used for ordering points when simplifying faces. Default is 'convexhull'.
    """

    def __init__(self, tri, sig_dig=12, method="convexhull"):
        self.method = method
        self.tri = np.around(np.array(tri), sig_dig)
        self.grpinx = list(range(len(tri)))
        norms = np.around([self.norm(s) for s in self.tri], sig_dig)
        _, self.inv = np.unique(norms, return_inverse=True, axis=0)

    def norm(self, sq):
        cr = np.cross(sq[2] - sq[0], sq[1] - sq[0])
        return np.abs(cr / np.linalg.norm(cr))

    def isneighbor(self, tr1, tr2):
        a = np.concatenate((tr1, tr2), axis=0)
        return len(a) == len(np.unique(a, axis=0)) + 2

    def order(self, v):
        if len(v) <= 3:
            return v
        v = np.unique(v, axis=0)
        n = self.norm(v[:3])
        y = np.cross(n, v[1] - v[0])
        y = y / np.linalg.norm(y)
        c = np.dot(v, np.c_[v[1] - v[0], y])
        if self.method == "convexhull":
            h = ConvexHull(c)
            return v[h.vertices]
        else:
            mean = np.mean(c, axis=0)
            d = c - mean
            s = np.arctan2(d[:, 0], d[:, 1])
            return v[np.argsort(s)]

    def simplify(self):
        for i, tri1 in enumerate(self.tri):
            for j, tri2 in enumerate(self.tri):
                if j > i:
                    if self.isneighbor(tri1, tri2) and self.inv[i] == self.inv[j]:
                        self.grpinx[j] = self.grpinx[i]
        groups = []
        for i in np.unique(self.grpinx):
            u = self.tri[self.grpinx == i]
            u = np.concatenate([d for d in u])
            u = self.order(u)
            groups.append(u)
        return groups
