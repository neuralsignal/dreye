"""Plotting Mixin for Signal Class
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_label_formatted(label):

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
        label = get_label_formatted(units.dimensionality)
        units = get_label_formatted(units)
        label = f'${label}$ (${units}$)'

    return label


class SignalPlottingMixin:

    _xlabel = None
    _ylabel = None
    _cmap = 'tab10'
    _colors = None

    def _get_cls_vars(
        self,
        cmap, colors, xlabel, ylabel
    ):
        if cmap is None:
            cmap = self._cmap
        if colors is None:
            colors = self._colors
        if xlabel is None:
            xlabel = self._xlabel
        if ylabel is None:
            ylabel = self._ylabel

        return cmap, colors, xlabel, ylabel

    def plot(
        self, ax=None,
        labels=False,
        cmap=None, colors=None,
        despine_kwargs={},
        legend_kwargs={},
        xlabel=None,
        ylabel=None,
        **plot_kwargs
    ):

        cmap, colors, xlabel, ylabel = self._get_cls_vars(
            cmap, colors, xlabel, ylabel
        )

        if ax is None:
            ax = plt.gca()

        if self.ndim == 2:
            if colors is None:
                colors = sns.color_palette(cmap, self.other_len)
            else:
                assert len(colors) == self.other_len

            for idx, values in enumerate(
                np.moveaxis(self.magnitude, self.other_axis, 0)
            ):

                ax.plot(
                    self.domain.magnitude,
                    values,
                    label=(self.labels[idx] if labels else None),
                    color=colors[idx],
                    **plot_kwargs
                )

        else:
            ax.plot(
                self.domain.magnitude,
                self.magnitude,
                label=(self.labels if labels else None),
                color=('black' if colors is None else colors),
                **plot_kwargs
            )

        if xlabel is None:
            xlabel = get_label(self.domain.units)
        if ylabel is None:
            ylabel = get_label(self.units)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if labels:
            ax.legend(**legend_kwargs)

        sns.despine(ax=ax, **despine_kwargs)

        return ax
