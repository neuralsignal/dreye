"""
Class to calculate various metrics given
a photoreceptor model and measured spectra
"""

import warnings
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from itertools import combinations
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import clone
from sklearn.feature_selection import mutual_info_regression

from dreye.utilities import (
    is_numeric, asarray, is_listlike, is_dictlike, is_string
)
from dreye.core.photoreceptor import Photoreceptor
from dreye.core.spectral_measurement import MeasuredSpectraContainer
from dreye.utilities.abstract import _InitDict, inherit_docstrings
# TODO metrics that depend on estimators
# from dreye.estimators.excitation_models import IndependentExcitationFit
# from dreye.estimators.silent_substitution import BestSubstitutionFit
# from dreye.estimators.led_substitution import LedSubstitutionFit


def compute_from_simplex(X):
    """
    Compute a simplex to a random variable.
    """
    X = np.abs(X)
    X = X / np.sum(X, axis=-1, keepdims=True)
    return probs_vec_to_cart(X)


def probs_vec_to_cart(X):
    n = X.shape[1]
    assert n > 1
    A = np.zeros((n, n-1))
    A[1, 0] = 1
    for i in range(2, n):
        A[i, :i-1] = np.mean(A[:i, :i-1], axis=0)
        dis = np.sum((A[:i, :i-1] - A[i, :i-1])**2, axis=1)
        assert np.unique(dis).size == 1
        x = np.sqrt(1 - dis.mean())
        A[i, i-1] = x

    return X @ A


def compute_jensen_shannon_divergence(P, Q, base=2):
    """
    Jensen-Shannon divergence of P and Q.
    """
    assert P.shape == Q.shape, "`P` and `Q` must be the same shape"
    P = P.ravel()
    Q = Q.ravel()
    _P = P / np.linalg.norm(P, ord=1)
    _Q = Q / np.linalg.norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (
        stats.entropy(_P, _M, base=base)
        + stats.entropy(_Q, _M, base=base)
    )


def compute_jensen_shannon_similarity(P, Q):
    """
    Compute Jensen-Shannon divergence with base 2 and subtract it from 1,
    so that 1 is equality of distribution and 0 is no similarity.
    """
    return 1 - compute_jensen_shannon_divergence(P, Q)


def compute_mean_width(X, n=1000, vectorized=False):
    """
    Compute mean width by projecting `X` onto random vectors

    Parameters
    ----------
    X : numpy.ndarray
        n x m matrix with n samples and m features.
    n : int
        Number of random projections to calculate width

    Returns
    -------
    mean_width : float
        Mean width of `X`.
    """
    X = X - X.mean(0)  # centering data
    rprojs = np.random.normal(size=(X.shape[-1], n))
    rprojs /= np.linalg.norm(rprojs, axis=0)  # normalize vectors by l2-norm
    if vectorized:
        proj = X @ rprojs  # project samples onto random vectors
        max1 = proj.max(0)  # max across samples
        max2 = (-proj).max(0)  # max across samples
    else:
        max1 = np.zeros(n)
        max2 = np.zeros(n)
        for idx, rproj in enumerate(rprojs.T):
            proj = X @ rproj
            max1[idx] = proj.max()
            max2[idx] = (-proj).max()
    return (max1 + max2).mean()


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
        background=None
    ):
        assert isinstance(photoreceptor_model, Photoreceptor)
        assert isinstance(measured_spectra, MeasuredSpectraContainer)

        self.photoreceptor_model = photoreceptor_model
        self.measured_spectra = measured_spectra
        self.combos = combos
        self.n_samples = n_samples
        self.seed = seed
        self.background = background

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
            )[0]
        self.bounds = self.measured_spectra.intensity_bounds
        self.normalized_spectra = self.measured_spectra.normalized_spectra
        self.n_sources = len(self.measured_spectra)

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
        self.random_samples = self.get_random_samples()

    def _get_metrics(
        self, metric_func, metric_name, B=None, as_frame=True,
        normalize=False, B_name=None, **kwargs
    ):

        name = (
            metric_name if is_string(metric_name) else
            getattr(metric_name, '__name__', repr(callable))
        )

        # names of light sources
        names = np.array(self.measured_spectra.names)

        def helper(B):
            if as_frame:
                metrics = pd.DataFrame(
                    self.source_idcs,
                    columns=names
                )
            else:
                metrics = np.zeros(len(self.source_idcs))

            for idx, source_idx in enumerate(self.source_idcs):
                metric = metric_func(source_idx, metric_name, B, **kwargs)
                if as_frame:
                    metrics.loc[idx, 'metric'] = metric
                    metrics.loc[idx, 'light_combos'] = '+'.join(
                        names[source_idx]
                    )
                    metrics.loc[idx, 'k'] = np.sum(source_idx)
                else:
                    metrics[idx] = metric

            if as_frame:
                metrics['k'] = metrics['k'].astype(int)
                metrics['metric_name'] = name
            if normalize:
                # TODO types of normalizations
                metrics['metric'] /= metrics['metric'].abs().max()
            return metrics

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

    def get_random_samples(self, n_samples=None):
        """
        Get random intensity samples.
        """
        if n_samples is None:
            n_samples = self.n_samples
        samples = np.random.random((n_samples, self.n_sources))
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
        return self.random_samples[:, source_idx] @ self.A[:, source_idx].T

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
        convex_hull = ConvexHull(points, qhull_options="QJ Pp")
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
        cc = np.corrcoef(points, rowvar=False)
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
        gamut_metric='mean_width'
    ):
        """
        Compute absolute gamut
        """
        gamut_metric = self._get_metric_func(gamut_metric)
        if nonlin is not None:
            points = nonlin(points)
        if zscore or np.any(points < 0):
            points = (points - np.min(points)) / (np.max(points) - np.min(points))
        nsize = points.shape[-1]
        volume = gamut_metric(compute_from_simplex(points))
        if relative:
            points1 = np.eye(nsize)
            points2 = 1 - points1
            points2 = points2 / np.sum(points2, axis=-1, keepdims=True)
            points = np.vstack([points1, points2])
            denom_volume = gamut_metric(compute_from_simplex(points))
            return volume / denom_volume
        return volume

    def compute_as_peaks(self):
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
        return self.normalized_spectra.labels[argmin]

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
            df.loc[idx, 'light_combos'] = '+'.join(
                names[source_idx]
            )
            df.loc[idx, 'k'] = np.sum(source_idx)

        df['k'] = df['k'].astype(int)
        df['metric_name'] = name

        if normalize:
            # TODO types of normalizations
            df['metric'] /= df['metric'].abs().max()
            if score_method in {'feature_scores', 'sample_scores'}:
                df[col_names] /= df[col_names].abs().max(axis=0)
        return df
