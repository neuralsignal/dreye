"""
Plotting for Signal Classes
"""

from abc import abstractmethod
import numpy as np
import pandas as pd
import seaborn as sns

from dreye.err import DreyeError
from dreye.utilities import optional_to


def _get_label(units):

    if units.dimensionless:
        label = 'a.u.'
    else:
        label = units.format_babel()

    return label


class _PlottingMixin:
    """
    Plotting Mixin for signal class and signal container class
    """

    # variables preset for class
    _xlabel = None
    _ylabel = None
    _cmap = 'tab10'

    def _get_cls_vars(self, cmap, xlabel, ylabel):
        if cmap is None:
            cmap = self._cmap
        if xlabel is None:
            xlabel = self._xlabel
        if ylabel is None:
            ylabel = self._ylabel

        return cmap, xlabel, ylabel

    def _plot(
        self,
        data,
        xlabel=None, ylabel=None,
        palette=None,
        highlight_col=None,
        highlight=None,
        **kwargs
    ):
        value = {
            col: 'None'
            for col in data.columns
            if col not in {'values', 'domain'}
        }
        data = data.fillna(value)

        default_kws = dict(
            hue=('labels' if 'labels' in data.columns else None),
            kind='line'
        )
        default_kws.update(kwargs)
        kwargs = default_kws

        default_facet_kws = dict(
            margin_titles=True,
        )
        default_facet_kws.update(kwargs.get('facet_kws', {}))
        kwargs['facet_kws'] = default_facet_kws

        palette, xlabel, ylabel = self._get_cls_vars(
            palette, xlabel, ylabel
        )

        if highlight_col is not None:
            data['highlight'] = False
            data.loc[data[highlight_col].isin(highlight), 'hightlight'] = True

        g = sns.relplot(
            data=data,
            x='domain',
            y='values',
            palette=palette,
            **kwargs
        )

        if xlabel is None:
            xlabel = _get_label(self.domain_units)
        if ylabel is None:
            ylabel = _get_label(self.units)

        g.set_xlabels(xlabel)
        g.set_ylabels(ylabel)

        return g

    def plot(
        self,
        xlabel=None, ylabel=None,
        palette=None,
        highlight_col=None,
        highlight=None,
        **kwargs
    ):
        """
        Plot data.

        Parameters
        ----------
        xlabel : str, optional
            Label for x-axis.
        ylabel : str, optional
            Label for y-axis
        palette : str or dict, optional
            Color palette to use. See `seaborn.color_palette`.
        kwargs : dict, optional
            Keyword arguments passed to `seaborn.relplot`.

        Returns
        -------
        g : `seaborn.FacetGrid`
            FacetGrid object from seaborn.

        See Also
        --------
        seaborn.relplot
        self.plotsmooth
        """
        data = self.to_longframe()
        return self._plot(
            data,
            xlabel=xlabel,
            ylabel=ylabel,
            palette=palette,
            highlight_col=highlight_col,
            highlight=highlight,
            **kwargs
        )

    def plotsmooth(
        self,
        min_window=None,
        max_window=None,
        steps=4,
        offset=0,
        **kwargs
    ):
        """
        Plot data for different smoothing levels.

        Parameters
        ----------
        min_window : numeric, optional
            The minimum smoothing window.
        max_window : numeric, optional
            The maximum smoothing window.
        steps : int, optional
            The number of smoothing windows to test between
            `min_window` and `max_window`.
        offset : numeric, optional
            The offset along the y-axis for each smoothing window, in order
            to better see differences between smoothing windows.
        kwargs : dict, optional
            Keyword arguments passed to the `plot` method.

        Returns
        -------
        g : `seaborn.FacetGrid`
            FacetGrid object from seaborn.

        See Also
        --------
        seaborn.relplot
        plot
        """

        # update default handler
        default_kws = dict(
            hue='applied_window_',
            col=('labels' if hasattr(self, 'labels') else None),
            col_wrap=(3 if hasattr(self, 'labels') else None),
        )
        default_kws.update(kwargs)

        if min_window is None and max_window is None:
            raise DreyeError('provide min_window or max_window')
        elif min_window is None:
            min_window = self.smoothing_window
        elif max_window is None:
            max_window = self.smoothing_window

        offset = optional_to(offset, self.units)
        windows = np.linspace(min_window, max_window, steps)

        container = [
            self
        ] + [
            self.smooth(window) + offset * (idx + 1) * self.units
            for idx, window in enumerate(windows)
        ]

        data = pd.concat(
            [ele.to_longframe() for ele in container],
            ignore_index=True, sort=True
        )
        return self._plot(
            data,
            **default_kws
        )

    @abstractmethod
    def to_longframe(self):
        pass
