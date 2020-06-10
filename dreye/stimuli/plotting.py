"""
Plotting Mixin for stimuli
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

from dreye.stimuli.variables import DELAY_KEY, DUR_KEY
from dreye.utilities import asarray


class StimPlottingMixin:

    # variables preset for class
    _cmap = 'tab10'
    _color = 'black'
    _plot_attrs = ['signal', 'fitted_signal', 'stimulus']
    _colormaps = {}

    def _get_colors(self, name, length, cmap, color):
        if cmap is None:
            if color is None:
                if name in self._colormaps:
                    colors = self._colormaps[name]
                    assert len(colors) == length
                elif name.replace('fitted_', '') in self._colormaps:
                    colors = self._colormaps[name.replace('fitted_', '')]
                    assert len(colors) == length
                else:
                    colors = sns.color_palette(self._cmap, length)
            elif isinstance(color, str):
                colors = [color] * length
            else:
                assert len(color) == length
                colors = color
        else:
            if isinstance(cmap, str):
                colors = sns.color_palette(cmap, length)
            else:
                assert len(cmap) == length
                colors = cmap

        if not isinstance(colors, dict):
            colors = {
                idx: color
                for idx, color in enumerate(colors)
            }

        return colors

    def plot(
        self,
        skip_attrs=(),
        fig=None,
        subplot_spec=None,
        fig_kws={},
        gridspec_kws=None,
        subplot_kws=None,
        **kwargs
    ):
        """
        plot different stimulus attributes.
        """

        plot_attrs = [
            attr
            for attr in self._plot_attrs
            if attr not in skip_attrs and not (
                attr.startswith('fitted_')
                and attr.replace('fitted_', '') in self._plot_attrs
            )
        ]
        fitted_attrs = {
            attr.replace('fitted_', ''): attr
            for attr in self._plot_attrs
            if attr.replace('fitted_', '') not in skip_attrs
            and attr.startswith('fitted_')
            and attr.replace('fitted_', '') in self._plot_attrs
        }

        if fig is None:
            assert subplot_spec is None
            fig_kws['constrained_layout'] = fig_kws.get(
                'constrained_layout', True
            )
            # create subplots
            fig, axes = plt.subplots(
                nrows=len(plot_attrs),
                sharex=True,
                gridspec_kw=gridspec_kws,
                subplot_kw=subplot_kws,
                **fig_kws
            )
        else:
            assert subplot_spec is not None
            nrows = len(plot_attrs)
            g = gridspec.GridSpecFromSubplotSpec(
                nrows=len(plot_attrs),
                ncols=1,
                subplot_spec=subplot_spec,
                **({} if gridspec_kws is None else gridspec_kws)
            )

            subplot_kws = ({} if subplot_kws is None else subplot_kws)

            axes = np.empty(nrows, object)

            for irow in range(nrows):

                if irow > 0:
                    subplot_kws['sharex'] = axes[0]

                axes[irow] = plt.Subplot(
                    fig, g[irow], **subplot_kws
                )

                fig.add_subplot(axes[irow])

        for attr, ax in zip(plot_attrs, axes):

            plot_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, dict):
                    if not set(value) & set(plot_attrs + ['_default']):
                        plot_kwargs[key] = value
                    elif attr in value:
                        plot_kwargs[key] = value[attr]
                    elif '_default' in value:
                        plot_kwargs[key] = value['_default']
                else:
                    plot_kwargs[key] = value

            plot_kwargs['add_title'] = plot_kwargs.get('add_title', attr)

            self.plot_data(
                ax=ax,
                data=attr,
                **plot_kwargs
            )
            fitted_attr = fitted_attrs.get(attr, None)
            if fitted_attr is not None:
                plot_kwargs['linestyle'] = '--'
                self.plot_data(
                    ax=ax,
                    data=fitted_attr,
                    _skip=True,
                    **plot_kwargs
                )

        return axes

    def plot_data(
        self, data=None, ax=None,
        color=None, cmap=None,
        add_legend=False,
        add_title=None,
        xlabel=None,
        ylabel=None,
        yscale=None,
        despine=True,
        add_events=None,
        events_palette='rainbow',
        events_text=True,
        events_plot_kwargs={},
        marker_events=False,
        _skip=False,
        **plot_kws
    ):
        """
        plot a stimuli/signal data for 2D arrays
        """

        if data is None:
            data = self.stimulus
            name = 'stimulus'
        elif isinstance(data, str):
            name = data
            try:
                data = getattr(self, data)
            except AttributeError:
                data = self[data]
        # asarray to avoid unit warning
        data = asarray(data)
        # get axis
        if ax is None:
            ax = plt.gca()

        assert data.ndim == 2, "data must be 2-dimensional"

        if self.time_axis != 0:
            data = np.swapaxes(0, 1)

        # always choosing first axis
        length = data.shape[1]
        colors = self._get_colors(name, length, cmap, color)

        if self.rate is None:
            x = np.arange(data.shape[0])
            # default for sampled stimuli
            plot_kws['linestyle'] = plot_kws.get('linestyle', '')
            plot_kws['marker'] = plot_kws.get('marker', 'o')
        else:
            x = self.timestamps

        for idx, (label, color) in enumerate(colors.items()):
            ax.plot(
                x, data[:, idx],
                label=(label if _skip else None),
                color=color,
                **plot_kws
            )

        if add_events is not None and self.rate is not None and not _skip:
            self._events_bars(
                ax, add_events, events_palette,
                events_text, events_plot_kwargs,
                marker_events
            )

        if yscale is not None:
            ax.set_yscale(yscale)

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if despine:
            sns.despine(ax=ax)

        if add_legend:
            ax.legend()

        if add_title is not None:
            ax.set_title(add_title)

        return ax

    def _events_bars(
        self, ax, add_events, events_palette,
        events_text, events_plot_kwargs, marker_events
    ):
        """

        Parameters
        ----------
        add_events : str or list
        events_palette : str
        events_text : bool
        events_plot_kwargs : dict
        marker_events : bool
        """
        events_plot_kwargs['alpha'] = events_plot_kwargs.get('alpha', 0.5)

        if marker_events:
            events_plot_kwargs['linestyle'] = events_plot_kwargs.get(
                'linestyle', '')
            events_plot_kwargs['marker'] = events_plot_kwargs.get(
                'marker', 'o'
            )
        else:
            events_plot_kwargs['linewidth'] = events_plot_kwargs.get(
                'linewidth', 0.5)

        # add events bars above ymax
        ylim = ax.get_ylim()
        ymax = ylim[1]
        # unique add_events group - this also sorts keys
        grouped = self.events.groupby(add_events)
        ecolors = sns.color_palette(events_palette, len(grouped))
        for index, (names, group) in enumerate(grouped):
            color = ecolors[index]

            for time_tuple in group[
                [DELAY_KEY, DUR_KEY]
            ].itertuples(False, None):
                if marker_events:
                    x = [time_tuple[0]]
                    y = [ymax]
                else:
                    x = [time_tuple[0], time_tuple[0] + time_tuple[1]]
                    y = [ymax, ymax]
                # plot even bar
                ax.plot(x, y, color=color, **events_plot_kwargs)

                if events_text:
                    ax.text(
                        np.mean(x),
                        np.mean(y),
                        names,
                        verticalalignment='bottom',
                        horizontalalignment='center'
                    )
