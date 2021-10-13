"""
Class to calculate various metrics given
a photoreceptor model and measured spectra
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import product
import copy

from dreye.utilities.common import is_string
from dreye.utilities.abstract import _InitDict, inherit_docstrings
from dreye.estimators.base import _PrModelMixin
from dreye.estimators.utils import get_optimal_capture_samples, get_source_idcs, get_source_idx, get_spanning_intensities, spaced_solutions
from dreye.estimators.metric_functions import compute_est_score, compute_gamut, compute_metric_for_samples, compute_peak_set, get_metrics
from dreye.estimators.silent_substitution import BestSubstitutionFit
from dreye.estimators.excitation_models import IndependentExcitationFit
from dreye.plotting import plot_simplex
from dreye.utilities import barycentric_dim_reduction, is_numeric
from dreye.estimators.capture_tests import CaptureTests


# TODO add old metrics back in new format


@inherit_docstrings
class Metrics(_InitDict, _PrModelMixin):
    """
    Metrics to compute the "goodness-of-fit" for various light source
    combinations.
    """
    # never ignore bounds
    ignore_bounds = False

    def __init__(
        self, 
        combos,
        photoreceptor_model,
        measured_spectra,
        *,
        intensity_bounds=None,
        background=None,
        wavelengths=None, 
        capture_noise_level=None
    ):
        # set init
        self.combos = combos
        self.photoreceptor_model = photoreceptor_model
        self.measured_spectra = measured_spectra
        self.intensity_bound = intensity_bounds
        self.wavelengths = wavelengths
        self.background = background
        self.capture_noise_level = capture_noise_level

        self._set_objs()

    def reset_background(self, background):
        self.background = background
        self._set_objs(do_combos=False, do_samples=False)
        return self

    def reset_combos(self, combos):
        self.combos = combos
        self._set_objs(do_pr_model_objs=False, do_optimal=False, do_samples=False)
        return self

    def reset_intensity_bounds(self, intensity_bounds):
        self.intensity_bounds = intensity_bounds
        self._set_objs(do_combos=False, do_optimal=False)

    def _set_objs(
        self, do_pr_model_objs=True, do_combos=True, 
        do_optimal=True, do_samples=True
    ):
        # compute pr model objects
        if do_pr_model_objs:
            self._set_pr_model_related_objects()
        # compute source idcs
        if do_combos:
            self.source_idcs_ = get_source_idcs(
                self.measured_spectra_.names, self.combos
            )
        # compute perfect system
        if do_optimal:
            self.optimal_q_samples_ = get_optimal_capture_samples(
                self.photoreceptor_model_, 
                self.background_
            )
        # compute sample excitations/captures
        if do_samples:
            self.intensity_samples_ = get_spanning_intensities(
                self.intensity_bounds_
            )

    def capture_for_selected_combo(self, source_idx):  # sample func
        """
        Get capture values given selected LED set.
        """
        source_idx = get_source_idx(
            self.measured_spectra_.names, source_idx, 
            asbool=True
        )
        w = self.intensity_samples_.copy()
        w[:, ~source_idx] = 0
        return self.get_capture(w.T)

    def excitation_for_selected_combo(self, source_idx):  # sample func
        """
        Get excitation values given selected LED set.
        """
        q = self.capture_for_selected_combo(source_idx)
        return self.photoreceptor_model_.excitefunc(q)

    def compute_gamuts(self, at_overall_q=None, relative=True, seed=None):
        """
        Compute Gamuts
        """
        if relative:
            relative_to = self.optimal_q_samples_
            denom = compute_gamut(
                relative_to, 
                gamut_metric='mean_width', 
                at_overall_q=None,
                relative_to=None,
                center=True,
                centered=True,
                seed=seed,
            )
        else:
            denom = 1
        
        df = get_metrics(
            self.source_idcs_, 
            self.measured_spectra_.names, 
            None, 
            compute_metric_for_samples, 
            'gamut', 
            metric='gamut',
            sample_func=self.capture_for_selected_combo,
            gamut_metric='mean_width', 
            at_overall_q=at_overall_q, 
            relative_to=None,
            center=True,
            centered=True,
            seed=seed,
        )
        df['metric'] /= denom
        return df

    def compute_best_substitutions(self, **kwargs):
        """
        Maximum contrast achievable for different combinations of measured spectra.
        """
        X = np.eye(self.photoreceptor_model_.n_opsins).astype(bool)
        return compute_est_score(
            self.source_idcs_, 
            X, 
            self.measured_spectra_, 
            self.photoreceptor_model_, 
            self.background_, 
            BestSubstitutionFit, 
            cols=self.photoreceptor_model_.labels, 
            score_method='feature_scores', 
            name='best_substitution', 
            normalize=False,
            score_kws={'method':'rel', 'aggfunc':'max'},
            intensity_bounds=self.intensity_bounds_, 
            **kwargs
        )

    def compute_excitation_scores(self, X, method='r2', **kwargs):
        """
        Independent excitation fits for different measured spectra combos. 

        See Also
        --------
        IndependentExcitationFit
        """
        return compute_est_score(
            self.source_idcs_, 
            X, 
            self.measured_spectra_, 
            self.photoreceptor_model_, 
            self.background_, 
            IndependentExcitationFit, 
            cols=self.photoreceptor_model_.labels, 
            score_method='feature_scores', 
            name='independent_excitation', 
            normalize=False,
            score_kws=dict(method=method),  # just r2 scores for each photoreceptor
            intensity_bounds=self.intensity_bounds_, 
            **kwargs
        )

    def compute_capture_in_range(self, X, ignore_bounds=None):
        """
        Compute percent of excitations in hull.

        See Also
        --------
        CaptureTests
        """
        return compute_est_score(
            self.source_idcs_, 
            X, 
            self.measured_spectra_, 
            self.photoreceptor_model_, 
            self.background_,
            CaptureTests, 
            cols=None, 
            score_method='capture_in_range', 
            name='capture_in_range', 
            normalize=False, 
            intensity_bounds=self.intensity_bounds_,
            ignore_bounds=ignore_bounds
        )

    def compute_excitation_scores_for_spectra(
        self, 
        spectra, 
        method='r2',
        **kwargs
    ):
        """
        Independent excitation fits for different measured spectra combos
        given a set of spectra.

        See Also
        --------
        IndependentExcitationFit
        """
        X = self.photoreceptor_model_.excitation(
            spectra, 
            background=self.background_, 
            return_units=False
        )
        return self.compute_excitation_scores(X, method=method, **kwargs)

    def compute_capture_in_range_for_spectra(self, spectra, ignore_bounds=None):
        """
        Compute percent of excitations in hull.

        See Also
        --------
        CaptureTests
        """
        X = self.photoreceptor_model_.capture(
            spectra, 
            background=self.background_, 
            return_units=False
        )
        return self.compute_capture_in_range(X, ignore_bounds=ignore_bounds)

    def compute_best_peak_difference(
        self, 
        as_string=False,
        as_idx=False,
    ):
        """
        Compute peaks of measured spectra that best match peaks of 
        opsin sensitivities.
        """
        return compute_peak_set(  # formerly compute_as_peaks
            self.photoreceptor_model_.sensitivity, 
            self.measured_spectra_.normalized_spectra,
            as_string=as_string, as_idx=as_idx
        )

    def simplex_plot(
        self, 
        combo=None, 
        wls=None, 
        ax=None, cmap='rainbow', nonspectral_lines=False, 
        add_center=True, 
        add_solos=False,
        gradient_line_kws={},
        point_scatter_kws={},
    ):
        """
        Plot a simplex plot for a specific light source combination.
        """
        n = self.photoreceptor_model_.n_opsins
        assert n in {2, 3, 4}, "Simplex plots only works for tri- or tetrachromatic animals."
        
        wls_ = self.photoreceptor_model_.wls
        if wls is None:
            wls = 10
        # if numeric it indicates spacing
        if is_numeric(wls):
            wmin = np.min(wls_)
            wmax = np.max(wls_)
            # earliest round number
            wmin = np.ceil(wmin / wls) * wls
            wls = np.arange(wmin, wmax, wls)
        
        qpoints = self.optimal_q_samples_[
            np.argmin(
                np.abs(
                    wls_[:, None]-wls
                ), axis=0
            )
        ]
        
        gradient_line_kws['cmap'] = cmap
        gradient_line_kws['add_colorbar'] = (
            False if (n == 4) else gradient_line_kws.get('add_colorbar', True)
        )

        ax = plot_simplex(
            n, 
            ax=ax,
            gradient_line=qpoints, 
            gradient_color=wls, 
            gradient_line_kws=gradient_line_kws, 
        )

        if add_center:
            plot_simplex(
                n, 
                ax=ax, 
                points=np.ones((1, qpoints.shape[1]))/qpoints.shape[1], 
                point_colors='gray', 
                point_scatter_kws=point_scatter_kws, 
                lines=False
            )

        if add_solos and n != 2:
            for i in range(n):
                x_ = np.zeros((2, n))
                x_[0, i] = 1
                x_[1] = 1
                x_[1:, i] = 0
                xs_ = barycentric_dim_reduction(x_)
                ax.plot(*xs_.T, color='gray', linestyle='--', alpha=0.5)
        
        if nonspectral_lines and n != 2:
        
            qmaxs = barycentric_dim_reduction(
                qpoints[np.argmax(qpoints, 0)]
            )
            for idx, jdx in product(range(n), range(n)):
                if idx >= jdx:
                    continue
                # sensitivities next to each other
                if abs(idx - jdx) == 1:
                    continue
                _qs = qmaxs[[idx, jdx]].T
                ax.plot(
                    *_qs,
                    color='black', 
                    linestyle='--', 
                    alpha=0.8
                )
                
        if combo is not None:
            hullp = self.capture_for_selected_combo(
                combo
            )
            hull_kws = {
                'color':'lightgray', 
                'edgecolor': 'gray', 
                'linestyle':'--',
            }
            if n <= 3:
                hull_kws.pop('edgecolor')
            ax = plot_simplex(
                n, 
                hull=hullp[
                    hullp.sum(axis=1) > 0
                ], 
                hull_kws=hull_kws, 
                ax=ax, 
                lines=False
            )
        
        return ax

    def plot_peak_differences(
        self, 
        highlight=True, 
        ax=None, 
        cmap='coolwarm', 
        over=None, under=None,
        **kwargs
    ):
        """
        Plot heatmap of peak differences between opsins and light sources.
        """
        se = self.photoreceptor_model_.sensitivity
        ns = self.measured_spectra_.normalized_spectra
        
        peakidcs = compute_peak_set(
            se, ns,
            as_idx=True
        )
        sdmax = se.dmax
        ndmax = ns.dmax
        
        diffs = pd.DataFrame(
            sdmax[:, None] - ndmax, 
            index=self.photoreceptor_model_.labels, 
            columns=self.measured_spectra_.names
        )
        maxdiff = np.max(np.abs(diffs.to_numpy()))
        
        if ax is None:
            ax = plt.gca()

        kws = kwargs.copy()
        vmin = kws.pop('vmin', -maxdiff)
        vmax = kws.pop('vmax', maxdiff)
        square = kws.pop('square', True)
        linewidths = kws.pop('linewidths', 0.5)
        cbar_kws = kws.pop('cbar_kws', {})

        if is_string(cmap):
            cmap = copy.copy(sns.color_palette(cmap, as_cmap=True))

        if over is not None and under is not None:
            cmap.set_over(over)
            cmap.set_under(under)
            cbar_kws['extend'] = 'both'
        elif over is not None:
            cmap.set_over(over)
            cbar_kws['extend'] = 'max'
        elif under is not None:
            cmap.set_under(under)
            cbar_kws['extend'] = 'min'
        
        sns.heatmap(
            diffs, 
            vmin=vmin, 
            vmax=vmax, 
            cmap=cmap, 
            square=square, 
            linewidths=linewidths, 
            cbar_kws=cbar_kws,
            **kws
        )
        
        # highlight best match
        if highlight:
            xs = np.array([0, 1, 1, 0, 0])
            ys = np.array([0, 0, 1, 1, 0])

            for xidx, yidx in zip(peakidcs, np.arange(peakidcs.size)):
                ax.plot(
                    xs+xidx, ys+yidx,
                    linewidth=linewidths*4, color='black'
                )
        return ax
        