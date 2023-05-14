import pytest
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dreye.api.plotting.simplex_plot import SimplexPlot


def test_SimplexPlot_initialization():
    # Test initialization with correct parameters
    simplex_plot = SimplexPlot(n=3, points=np.random.rand(10, 3), labels=['A', 'B', 'C'])
    assert simplex_plot.n == 3
    assert simplex_plot.points.shape == (10, 3)
    assert simplex_plot.labels == ['A', 'B', 'C']

    # Test initialization with incorrect parameters
    with pytest.raises(AssertionError):
        simplex_plot = SimplexPlot(n=5, points=np.random.rand(10, 3), labels=['A', 'B', 'C'])

def test_SimplexPlot_plot_points():
    simplex_plot = SimplexPlot(n=4, points=np.random.rand(10, 4), labels=['A', 'B', 'C'])
    simplex_plot.plot_points()
    plt.close()
    assert isinstance(simplex_plot.ax, Axes3D)  # ensure 3D plot

def test_SimplexPlot_plot_hull():
    simplex_plot = SimplexPlot(n=3, points=np.random.rand(10, 3), labels=['A', 'B', 'C'])
    simplex_plot.plot_hull()  
    plt.close()
    # Add some kind of check here to ensure hull has been plotted

def test_SimplexPlot_plot_gradient_line():
    simplex_plot = SimplexPlot(n=3, points=np.random.rand(10, 3), labels=['A', 'B', 'C'], gradient_line=np.random.rand(10, 3))
    simplex_plot.plot_gradient_line()
    plt.close()
    # Add some kind of check here to ensure gradient line has been plotted

def test_SimplexPlot_plot_labels():
    simplex_plot = SimplexPlot(n=3, points=np.random.rand(10, 3), labels=['A', 'B', 'C'])
    simplex_plot.plot_labels()
    plt.close()
    # Add some kind of check here to ensure labels have been plotted

def test_SimplexPlot_finalize_plot():
    simplex_plot = SimplexPlot(n=3, points=np.random.rand(10, 3), labels=['A', 'B', 'C'])
    simplex_plot.finalize_plot()
    plt.close()
    # Add some kind of check here to ensure final plot properties have been set

def test_SimplexPlot_plot():
    simplex_plot = SimplexPlot(n=3, points=np.random.rand(10, 3), labels=['A', 'B', 'C'])
    simplex_plot.plot()
    plt.close()
    # Add some kind of check here to ensure plot has been shown
