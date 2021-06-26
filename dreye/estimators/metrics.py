"""
Class to calculate various metrics given
a photoreceptor model and measured spectra
"""

from dreye.estimators.utils import check_background, check_measured_spectra, check_photoreceptor_model
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from itertools import combinations, product
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import clone
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import normalize

from dreye.core.signal import Signals
from dreye.utilities import (
    is_numeric, asarray, is_listlike, is_dictlike, is_string
)
from dreye.core.photoreceptor import Photoreceptor
from dreye.core.spectral_measurement import MeasuredSpectraContainer
from dreye.utilities.abstract import _InitDict, inherit_docstrings
from dreye.utilities.barycentric import (
    barycentric_dim_reduction, simplex_plane_points_in_hull
)
from dreye.utilities.metrics import (
    compute_jensen_shannon_similarity,
    compute_mean_width
)
from dreye.estimators.base import _SpectraStaticMethodsMixin
# TODO metrics that depend on estimators
# from dreye.estimators.excitation_models import IndependentExcitationFit
# from dreye.estimators.silent_substitution import BestSubstitutionFit
# from dreye.estimators.led_substitution import LedSubstitutionFit
# TODO simplify

# @property
# def sensitivity_ratios(self):
#     """
#     Compute ratios of the sensitivities
#     """
#     s = self.data + self.capture_noise_level
#     if np.any(s < 0):
#         warnings.warn(
#             "Zeros or smaller in sensitivities array!", RuntimeWarning
#         )
#         s[s < 0] = 0
#     return normalize(s, norm='l1')

# def best_isolation(self, method='argmax', background=None):
#     """
#     Find wavelength that best isolates each opsin.
#     """
#     ratios = self.sensitivity_ratios

#     if method == 'argmax':
#         return self.wls[np.argmax(ratios, axis=0)]
#     elif method == 'substitution':
#         from dreye import create_measured_spectra_container, BestSubstitutionFit
        
#         perfect_system = create_measured_spectra_container(
#             np.eye(self.wls.size)[:, 1:-1], wavelengths=self.wls
#         )
#         model = BestSubstitutionFit(
#             photoreceptor_model=self, 
#             measured_spectra=perfect_system, 
#             ignore_bounds=True, 
#             substitution_type=1, 
#             background=background
#         )
#         model.fit(np.eye(self.n_opsins).astype(bool))
#         return np.sum(model.fitted_intensities_ * self.wls, axis=1) / np.sum(model.fitted_intensities_, axis=1)
#     else:
#         raise NameError(
#             "Only available methods are `argmax` "
#             f"and `substitution` and not {method}"
#         )

# def wavelength_range(self, rtol=None, peak2peak=False):
#     """
#     Range of wavelengths that the photoreceptors are sensitive to.
#     Returns a tuple of the min and max wavelength value.
#     """
#     if peak2peak:
#         dmax = self.sensitivity.dmax
#         return np.min(dmax), np.max(dmax)
#     rtol = (RELATIVE_SENSITIVITY_SIGNIFICANT if rtol is None else rtol)
#     tol = (
#         (self.sensitivity.max() - self.sensitivity.min())
#         * rtol
#     )
#     return self.sensitivity.nonzero_range(tol).boundaries



@inherit_docstrings
class Metrics(_InitDict, _SpectraStaticMethodsMixin):
    """
    Metrics to compute the "goodness-of-fit" for various light source
    combinations.
    """

    def __init__(
        self, 
        combos,
        photoreceptor_model,
        measured_spectra,
        *,
        intensity_bounds=None,
        wavelengths=None, 
        capture_noise_level=1e-4
    ):
        # set init
        self.combos = combos
        self.photoreceptor_model = photoreceptor_model
        self.measured_spectra = measured_spectra
        self.intensity_bound = intensity_bounds
        self.wavelengths = wavelengths

        # compute rest
        self.photoreceptor_model_ = check_photoreceptor_model(
            photoreceptor_model, wavelengths=wavelengths, 
            capture_noise_level=capture_noise_level
        )
        self.measured_spectra_ = check_measured_spectra(
            measured_spectra, photoreceptor_model=self.photoreceptor_model_, 
            wavelengths=wavelengths, intensity_bounds=intensity_bounds
        )
        self.source_idcs_ = self._get_source_idcs()

    def _get_source_idcs(self):
        n = len(self.measured_spectra_)
        if is_numeric(self.combos):
            combos = int(self.combos)
            source_idcs = self._get_source_idcs_from_k(
                n, combos
            )
        elif is_listlike(self.combos):
            combos = asarray(self.combos).astype(int)
            if combos.ndim == 1:
                source_idcs = []
                for k in combos:
                    source_idx = self._get_source_idcs_from_k(n, k)
                    source_idcs.append(source_idx)
                source_idcs = np.vstack(source_idcs)
            elif combos.ndim == 2:
                source_idcs = self.combos.astype(bool)
            else:
                raise ValueError(
                    f"`combos` dimensionality is `{self.combos.ndim}`, "
                    "but needs to be 1 or 2."
                )
        else:
            raise TypeError(
                f"`combos` is of type `{type(self.combos)}`, "
                "but must be numeric or array-like."
            )

        return source_idcs

    @staticmethod
    def _get_source_idcs_from_k(n, k):
        idcs = np.array(list(combinations(np.arange(n), k)))
        source_idcs = np.zeros((len(idcs), n)).astype(bool)
        source_idcs[
            np.repeat(np.arange(len(idcs)), k),
            idcs.ravel()
        ] = True
        return source_idcs

    def set_background(self, background, wavelengths=None):
        self.background_ = check_background(
            background, self.measured_spectra_, 
            wavelengths=(self.wavelengths if wavelengths is None else wavelengths)
        )
        return self

    def set_samples(self):
        pass

    def set_perfect_system(self):
        pass


@inherit_docstrings
class MeasuredSpectraMetrics(_InitDict):
    """
    Metrics to compute the "goodness-of-fit" for various light source
    combinations.

    Parameters
    ----------
    combos : int, array-like
    photoreceptor_model : :obj:`~dreye.Photoreceptor`
    measured_spectra : :obj:`~dreye.MeasuredSpectraContainer`
    n_samples : int, optional
    seed : int, optional
    background : :obj:`~dreye.Spectrum`
    """

    def __init__(
        self,
        combos,
        photoreceptor_model,
        measured_spectra,
        n_samples=10000,
        seed=None,
        background=None,
        rtol=None,
        peak2peak=False,
        random=False,
        eps=0
    ):
        assert isinstance(photoreceptor_model, Photoreceptor)
        assert isinstance(measured_spectra, MeasuredSpectraContainer)

        self.photoreceptor_model = photoreceptor_model
        self.measured_spectra = measured_spectra
        self.combos = combos
        self.n_samples = n_samples
        self.seed = seed
        self.background = background
        self.peak2peak = peak2peak
        self.rtol = rtol
        self.random = random
        self.eps = eps

        # set seed if necessary
        if seed is not None:
            np.random.seed(seed)

        # opsin x led
        self.A = self.photoreceptor_model.capture(
            self.measured_spectra.normalized_spectra,
            background=self.background,
            return_units=False,
        ).T

        if self.background is not None:
            self.q_bg = self.photoreceptor_model.capture(
                self.background, return_units=False
            )
        self.bounds = self.measured_spectra.intensity_bounds
        self.normalized_spectra = self.measured_spectra.normalized_spectra
        self.n_sources = len(self.measured_spectra)

        # perfect excitation of single wavelengths
        peaks = self.photoreceptor_model.sensitivity.dmax
        wl_range = self.photoreceptor_model.wavelength_range(
            rtol=rtol, peak2peak=peak2peak
        )
        domain = self.normalized_spectra.domain.magnitude  # TODO Buggy if sensitivity and spectra do not have overlapping domain
        spectra = []
        idx_peaks = []
        labels = []
        for val in domain:
            if (val < wl_range[0]) or (val > wl_range[1]):
                continue
            spectrum = np.zeros(domain.size)
            spectrum[val == domain] = 1
            if val in peaks:
                idx = len(spectra)
                idx_peaks.append(idx)
            spectra.append(spectrum)
            labels.append("{:.1f}".format(val))
        spectra = np.array(spectra).T  # wls x dirac pulses
        labels = np.array(labels)
        spectra[:, idx_peaks]
        spectra_combos = self._get_samples(len(idx_peaks), eps=0)
        spectra_combos = spectra_combos[spectra_combos.sum(axis=1) > 1]
        spectra_ = spectra[:, idx_peaks] @ spectra_combos.T
        labels_ = [
            '-'.join(labels[idx_peaks][combo]) for combo in spectra_combos.astype(bool)
        ]
        spectra = np.hstack([spectra, spectra_])
        labels = np.concatenate([labels, labels_])
        self.labels_perfect = pd.Series(labels)
        self.s_perfect = Signals(
            spectra,
            domain=domain,
            domain_units='nm',
            labels=labels, units='uE'
        )
        # will not result in perfect excitation?
        # if self.background is not None:
        #     self.s_perfect = self.s_perfect + self.background
        self.capture_perfect = self.photoreceptor_model.capture(
            self.s_perfect,
            background=self.background,
            return_units=False,
        )
        self.excite_perfect = self.photoreceptor_model.excitefunc(
            self.capture_perfect
        )

        if is_numeric(self.combos):
            self.combos = int(self.combos)
            self.source_idcs = self._get_source_idcs(
                self.n_sources, self.combos
            )
        elif is_listlike(self.combos):
            self.combos = asarray(self.combos).astype(int)
            if self.combos.ndim == 1:
                source_idcs = []
                for k in self.combos:
                    source_idx = self._get_source_idcs(self.n_sources, k)
                    source_idcs.append(source_idx)
                self.source_idcs = np.vstack(source_idcs)
            elif self.combos.ndim == 2:
                self.source_idcs = self.combos.astype(bool)
            else:
                raise ValueError(
                    f"`combos` dimensionality is `{self.combos.ndim}`, "
                    "but needs to be 1 or 2."
                )
        else:
            raise TypeError(
                f"`combos` is of type `{type(self.combos)}`, "
                "but must be numeric or array-like."
            )

        # random light source intensity levels
        self.random_samples = self.get_samples(random=random, eps=eps)

        # names of light sources
        names = np.array(self.measured_spectra.names)
        light_combos = []
        for idx, source_idx in enumerate(self.source_idcs):
            light_combos.append('+'.join(names[source_idx]))
        self.light_combos = light_combos

    def _get_metrics(
        self, metric_func, metric_name, B=None, as_frame=True,
        normalize=False, B_name=None, **kwargs
    ):

        name = (
            metric_name if is_string(metric_name) else
            getattr(metric_name, '__name__', repr(callable))
        )

        def helper(B):
            return self._metric_constructor_helper(
                name, metric_func,
                metric_name, B,
                as_frame=as_frame,
                normalize=normalize,
                **kwargs
            )

        if is_dictlike(B):
            if as_frame:
                metrics = pd.DataFrame()
            else:
                metrics = {}
            for transformation, B_ in B.items():
                metrics_ = helper(B_)
                if as_frame:
                    metrics_['transformation'] = transformation
                    metrics = metrics.append(metrics_, ignore_index=True)
                else:
                    metrics[transformation] = metrics_
            return metrics
        else:
            metrics = helper(B)
            metrics['transformation'] = B_name
            return metrics

    def get_capture_metrics(
        self, B=None, metric='volume', as_frame=True,
        normalize=False, **kwargs
    ):
        return self._get_metrics(
            self.get_capture_metric,
            metric,
            B,
            as_frame,
            normalize,
            **kwargs
        )

    def get_excitation_metrics(
        self, B=None, metric='volume', as_frame=True,
        normalize=False, **kwargs
    ):
        return self._get_metrics(
            self.get_excitation_metric,
            metric,
            B,
            as_frame,
            normalize,
            **kwargs
        )

    def _get_samples(self, n_features, n_samples=None, random=False, eps=1e-4):
        if n_samples is None:
            n_samples = self.n_samples
        if random:
            return np.random.random((n_samples, n_features))
        return np.array(list(product(*([[0, 1]] * n_features)))) + eps  # eps

    def get_samples(self, n_samples=None, random=False, eps=1e-4):
        """
        Get random intensity samples.
        """
        # samples = np.random.random((n_samples, self.n_sources))
        samples = self._get_samples(
            self.n_sources, n_samples=n_samples,
            random=random, eps=eps
        )
        # scale to intensity bounds
        samples = samples * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        return samples

    def get_captures(self, source_idx):
        """
        Get capture values given selected LED set.
        """
        if isinstance(source_idx, str):
            source_idx = [
                self.measured_spectra.names.index(name)
                for name in source_idx.split('+')
            ]
        q = self.random_samples[:, source_idx] @ self.A[:, source_idx].T
        if self.random:
            return q
        return np.unique(q, axis=0)

    def get_measured_spectra(self, source_idx):
        """
        Get measured spectra for given LED set.
        """
        if isinstance(source_idx, str):
            source_idx = [
                self.measured_spectra.names.index(name)
                for name in source_idx.split('+')
            ]
        source_idx = np.asarray(source_idx)
        if source_idx.dtype == np.bool:
            source_idx = np.flatnonzero(source_idx)
        return self.measured_spectra[source_idx]

    def get_excitations(self, source_idx):
        """
        Get excitations values given selected LED set.
        """
        return self.photoreceptor_model.excitefunc(
            self.get_captures(source_idx)
        )

    def _plot_points(self, points_func, source_idx, B=None, B_columns=None):
        """
        """
        points = points_func(source_idx)

        def helper(points, B, B_columns, title=None):
            points = self.transform_points(points, B)
            sns.pairplot(
                data=pd.DataFrame(
                    points,
                    columns=B_columns
                ),
                plot_kws=dict(
                    color='gray',
                    alpha=0.6
                ),
                diag_kws=dict(
                    color='gray',
                    alpha=0.6
                )
            )
            if title is not None:
                plt.suptitle(title, y=1)

            plt.show()

        if is_dictlike(B):
            for transformation, B_ in B.items():
                if B_columns is None:
                    B_columns_ = B_columns
                else:
                    B_columns_ = B_columns.get(transformation, None)
                helper(points, B_, B_columns_, transformation)
        else:
            return helper(points, B, B_columns)

    def plot_excitation_points(self, source_idx, B=None, B_columns=None):
        """
        Plot excitation points.
        """
        return self._plot_points(
            self.get_excitations, source_idx, B,
            B_columns
        )

    def plot_capture_points(self, source_idx, B=None):
        """
        Plot capture points.
        """
        return self._plot_points(self.get_captures, source_idx, B)

    @staticmethod
    def transform_points(points, B=None):
        """
        Transform `points` with B

        Parameters
        ----------
        points : numpy.ndarray
            2D matrix.
        B : callable or numpy.ndarray, optional
            If callable, `B` is a function: B(points). If B is a
            `numpy.ndarray`, then `B` is treated as a matrix: points @ B.
        """
        if B is None:
            return points
        elif callable(B):
            return B(points)
        else:
            return points @ B

    @staticmethod
    def compute_volume(points):
        """
        Compute the volume from a set of points.
        """
        if (points.ndim == 1) or (points.shape[1] < 2):
            return np.max(points) - np.min(points)
        if points.shape[0] <= points.shape[1]:
            return 0
        try:
            convex_hull = ConvexHull(
                points, qhull_options="QJ Pp"
            )
            return convex_hull.volume
        except Exception:
            points = points - np.mean(points, axis=0)
            convex_hull = ConvexHull(
                points,
                qhull_options="QJ Pp"
            )
            return convex_hull.volume

    @staticmethod
    def compute_continuity(points, bins=100, **kwargs):
        """
        Compute continuity of data by binning. Useful for circular datapoints.

        See Also
        --------
        compute_jss_uniformity
        """
        if (points.ndim == 1) or (points.shape[1] < 2):
            H = np.histogram(points, bins, **kwargs)[0].astype(bool)
        else:
            H = np.histogramdd(points, bins, **kwargs)[0].astype(bool)
        return H.sum() / H.size

    @staticmethod
    def compute_jss_uniformity(points, bins=100, **kwargs):
        """
        Compute how similar the dataset is to a uniform distribution.
        """
        if (points.ndim == 1) or (points.shape[1] < 2):
            H = np.histogram(points, bins, **kwargs)[0]
        else:
            H = np.histogramdd(points, bins, **kwargs)[0]
        H_uniform = np.ones(H.shape)
        return compute_jensen_shannon_similarity(H, H_uniform)

    @staticmethod
    def compute_mean_width(points, n=1000):
        """
        Compute mean width.
        """
        if (points.ndim == 1) or (points.shape[1] < 2):
            return np.max(points) - np.min(points)
        return compute_mean_width(points, n)

    @staticmethod
    def compute_mean_correlation(points):
        # compute correlation of each feature
        cc = np.abs(np.corrcoef(points, rowvar=False))
        return (cc - np.eye(cc.shape[0])).mean()

    @staticmethod
    def compute_mean_mutual_info(points, **kwargs):
        mis = []
        for idx in range(points.shape[1] - 1):
            mi = mutual_info_regression(points[idx], points[idx + 1:])
            mis.append(mi)
        return np.concatenate(mis).mean()

    def compute_gamut(
        self, points, nonlin=None, relative=True, zscore=False,
        gamut_metric='volume', compare_to='simplex', c=None, **kwargs
    ):
        """
        Compute absolute gamut
        """
        gamut_metric = self._get_metric_func(gamut_metric)
        if nonlin is not None:
            points = nonlin(points)
        if zscore or np.any(points < 0):
            points = (points - np.min(points)) / (np.max(points) - np.min(points))
        if c is not None:
            assert not zscore
            assert nonlin is None
            psum = points.sum(axis=-1)
            if np.all(psum <= c):
                return 0
            elif np.all(psum > c):
                return 0
            points = simplex_plane_points_in_hull(points, c)
        points = points[points.sum(axis=-1) != 0]
        nsize = points.shape[-1]
        volume = gamut_metric(barycentric_dim_reduction(points), **kwargs)
        if relative:
            if compare_to == 'simplex':
                points1 = np.eye(nsize)
                points2 = 1 - points1
                points2 = points2 / np.sum(points2, axis=-1, keepdims=True)
                points = np.vstack([points1, points2])
                denom_volume = gamut_metric(
                    barycentric_dim_reduction(points), **kwargs
                )
            elif compare_to == 'capture_perfect':
                denom_volume = self.compute_gamut(
                    self.capture_perfect,
                    nonlin=nonlin,
                    relative=False,
                    zscore=zscore, gamut_metric=gamut_metric,
                    **kwargs
                )
            elif compare_to == 'excite_perfect':
                denom_volume = self.compute_gamut(
                    self.excite_perfect,
                    nonlin=nonlin,
                    relative=False,
                    zscore=zscore, gamut_metric=gamut_metric,
                    **kwargs
                )
            else:
                raise NameError(f"compare_to argument `{compare_to}` not recognized.")
            return volume / denom_volume
        return volume

    def compute_peak_metric(self, source_idx, abs=True):
        if isinstance(source_idx, str):
            source_idx = [
                self.measured_spectra.names.index(name)
                for name in source_idx.split('+')
            ]
        pr_max = self.photoreceptor_model.sensitivity.dmax
        s_max = self.normalized_spectra.dmax[source_idx]
        diff = pr_max[:, None] - s_max[None, :]
        if abs:
            diff = np.abs(diff)
        amin = np.min(diff, axis=1)
        return amin

    def compute_peak_metrics(
        self,
        normalize=False,
        abs=True
    ):
        """
        Compute peak metrics
        """
        return self._metric_constructor_helper(
            'peak', self.compute_peak_metric,
            as_frame=True,
            normalize=normalize,
            cols=self.photoreceptor_model.names,
            abs=abs
        )

    def compute_as_peaks(self, as_string=False, as_idx=False):
        """
        Compute best set according to peaks of opsins and LEDs
        """
        pr_max = self.photoreceptor_model.sensitivity.dmax
        s_max = self.normalized_spectra.dmax
        argmin = np.argmin(np.abs(s_max[:, None] - pr_max[None]), axis=0)
        if np.unique(argmin).size != argmin.size:
            argmin_ = argmin
            argmin = []
            for odx, amin in enumerate(argmin_):
                if amin not in argmin:
                    argmin.append(amin)
                    continue

                argsort = np.argsort(np.abs(pr_max[odx] - s_max))
                for amin_ in argsort:
                    if amin_ not in argmin:
                        amin = amin_
                        break
                else:
                    warnings.warn("Couldn't find unique set of LEDs for opsin set.")
                argmin.append(amin)
        best = self.normalized_spectra.labels[argmin]
        if as_string:
            return '+'.join(best)
        if as_idx:
            return np.array(argmin)
        return best

    def _get_metric_func(self, metric):
        if callable(metric):
            return metric
        elif metric in {'volume', 'vol'}:
            return self.compute_volume
        elif metric in {'jss_uniformity', 'uniformity_similarity'}:
            return self.compute_jss_uniformity
        elif metric in {'mean_width', 'mw'}:
            return self.compute_mean_width
        elif metric in {'continuity', 'cont'}:
            return self.compute_continuity
        elif metric in {'corr', 'correlation'}:
            return self.compute_mean_correlation
        elif metric in {'mi', 'mutual_info'}:
            return self.compute_mean_mutual_info
        elif metric in {'gamut', }:
            return self.compute_gamut

        raise NameError(
            f"Did not recognize metric `{metric}`. "
            "`metric` must be a callable or an accepted str: "
            "{"
            "'volume', 'vol', 'jss_uniformity', 'uniformity_similarity', "
            "'mean_width', 'mw', 'continuity', 'cont', 'corr', 'correlation', "
            "'mi', 'mutual_info'"
            "}."
        )

    def get_excitation_metric(
        self, source_idx, metric='volume', B=None, **kwargs
    ):
        """
        Compute metric for particular combination of source lights.
        """
        metric_func = self._get_metric_func(metric)
        # get excitations and transform
        points = self.get_excitations(source_idx)
        points = self.transform_points(points, B)
        return metric_func(points, **kwargs)

    def get_capture_metric(
        self, source_idx, metric='volume', B=None, **kwargs
    ):
        """
        Compute metric for particular combination of source lights.
        """
        metric_func = self._get_metric_func(metric)
        # get excitations and transform
        points = self.get_captures(source_idx)
        points = self.transform_points(points, B)
        return metric_func(points, **kwargs)

    @staticmethod
    def _get_source_idcs(n, k):
        idcs = np.array(list(combinations(np.arange(n), k)))
        source_idcs = np.zeros((len(idcs), n)).astype(bool)
        source_idcs[
            np.repeat(np.arange(len(idcs)), k),
            idcs.ravel()
        ] = True
        return source_idcs

    def compute_est_score(
        self, est, X,
        score_method='score',
        name=None,
        col_names=None,
        normalize=False,
        **score_kws
    ):
        """
        Compute score of estimator
        """
        assert hasattr(est, 'measured_spectra')
        assert hasattr(est, score_method)

        if name is None:
            name = score_method

        # names of light sources
        names = np.array(self.measured_spectra.names)

        df = pd.DataFrame(
            self.source_idcs,
            columns=names
        )
        if col_names is None:
            if score_method == 'feature_scores':
                col_names = [f"metric_{i}" for i in range(X.shape[1])]
            elif score_method == 'sample_scores':
                col_names = [f"metric_{i}" for i in range(X.shape[0])]
        for idx, source_idx in enumerate(self.source_idcs):
            est_ = clone(est)
            est_.measured_spectra = self.get_measured_spectra(source_idx)
            est_.fit(X)
            score = getattr(est_, score_method)(**score_kws)

            if score_method in {'feature_scores', 'sample_scores'}:
                df.loc[idx, col_names] = score
                df.loc[idx, 'metric'] = score.mean()
            else:
                df.loc[idx, 'metric'] = score
            df.loc[idx, 'light_combos'] = self.light_combos[idx]
            df.loc[idx, 'k'] = np.sum(source_idx)

        df['k'] = df['k'].astype(int)
        df['metric_name'] = name

        if normalize:
            # TODO types of normalizations
            df['metric'] /= df['metric'].abs().max()
            if score_method in {'feature_scores', 'sample_scores'}:
                df[col_names] /= df[col_names].abs().max(axis=0)
        return df

    def _metric_constructor_helper(
        self, name, metric_func, *args, as_frame=True, normalize=False,
        cols=None,
        **kwargs
    ):
        # names of light sources
        names = np.array(self.measured_spectra.names)
        # pass
        if as_frame:
            metrics = pd.DataFrame(
                self.source_idcs,
                columns=names
            )
        else:
            metrics = np.zeros(len(self.source_idcs))

        for idx, source_idx in enumerate(self.source_idcs):
            metric = metric_func(source_idx, *args, **kwargs)
            if as_frame:
                if cols is None:
                    metrics.loc[idx, 'metric'] = metric
                else:
                    metrics.loc[idx, cols] = metric
                    metrics.loc[idx, 'metric'] = metric.mean()
                metrics.loc[idx, 'light_combos'] = self.light_combos[idx]
                metrics.loc[idx, 'k'] = np.sum(source_idx)
            else:
                if cols is None:
                    metrics[idx] = metric
                else:
                    raise TypeError(
                        'Must use dataframe format'
                        f'`as_frame` for `{name}`'
                    )

        if as_frame:
            metrics['k'] = metrics['k'].astype(int)
            metrics['metric_name'] = name
        if normalize:
            # TODO types of normalizations
            if as_frame:
                metrics['metric'] /= metrics['metric'].abs().max()
            else:
                metrics /= metrics.abs().max()
            if cols is not None:
                metrics[cols] /= metrics[cols].abs().max(axis=0)
        return metrics
