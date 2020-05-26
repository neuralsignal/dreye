"""
Plotting for Signal Classes
"""

from abc import abstractmethod
import numpy as np
import pandas as pd
import seaborn as sns

from dreye.err import DreyeError


def get_labels_formatted(label):

    label = str(
        label
    ).replace(
        '[', ''
    ).replace(
        ']', ''
    ).replace(
        ' ** ', '^'
    ).replace(
        '*', '\\cdot'
    ).replace('_', '')

    if '/' in label:
        label = "\\frac{" + label.replace('/', '}{', 1) + "}"
        label = label.replace('/', '\\cdot')

    return label


def get_label(units):

    if units.dimensionless:
        label = 'a.u.'
    else:
        label = get_labels_formatted(units.dimensionality)
        units = get_labels_formatted(units)
        label = f'${label}$ (${units}$)'

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
        **kwargs
    ):
        value = {
            col: 'None'
            for col in data.columns
            if col not in {'values', 'domain'}
        }
        data = data.fillna(value)

        default_kws = dict(
            hue='labels',
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

        g = sns.relplot(
            data=data,
            x='domain',
            y='values',
            palette=palette,
            **kwargs
        )

        if xlabel is None:
            xlabel = get_label(self.domain_units)
        if ylabel is None:
            ylabel = get_label(self.units)

        g.set_xlabels(xlabel)
        g.set_ylabels(ylabel)

        return g

    def plot(
        self,
        xlabel=None, ylabel=None,
        palette=None,
        **kwargs
    ):
        """aggregate plot using long dataframe and seaborn
        """
        data = self.to_longframe()
        return self._plot(
            data,
            xlabel=xlabel,
            ylabel=ylabel,
            palette=palette,
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
        """plot spectrum with different smoothing parameters
        """

        # update default handler
        default_kws = dict(
            hue='applied_window_',
            col='labels',
            col_wrap=3,
        )
        default_kws.update(kwargs)

        if min_window is None and max_window is None:
            raise DreyeError('provide min_window or max_window')
        elif min_window is None:
            min_window = self.smoothing_window
        elif max_window is None:
            max_window = self.smoothing_window

        windows = np.linspace(min_window, max_window, steps)

        container = [self] + [self.smooth(window)+offset*(idx+1)
                              for idx, window in enumerate(windows)]

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
