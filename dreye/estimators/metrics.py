"""
Class to calculate various metrics given
a photoreceptor model and measured spectra
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from itertools import combinations

from dreye.utilities import is_numeric, asarray, is_listlike, is_dictlike
from dreye.core.photoreceptor import Photoreceptor
from dreye.core.spectral_measurement import MeasuredSpectraContainer
from dreye.utilities.abstract import _InitDict, inherit_docstrings


def compute_mean_width(X, n=1000):
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
    proj = X @ rprojs  # project samples onto random vectors
    max1 = proj.max(0)  # max across samples
    max2 = (-proj).max(0)  # max across samples
    return (max1 + max2).mean()


@inherit_docstrings
class MeasuredSpectraMetrics(_InitDict):
    """
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
            apply_noise_threshold=False
        ).T
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
                self.source_idcs = self.combos
            else:
                raise ValueError(
                    "`combos` dimensionality is `{self.combos.ndim}`, "
                    "but needs to be 1 or 2."
                )
        else:
            raise TypeError(
                "`combos` is of type `{type(self.combos)}`, "
                "but must be numeric or array-like."
            )

        # random light source intensity levels
        self.random_samples = self.get_random_samples()

    def _get_metrics(
        self, metric_func, B=None, as_frame=True,
        normalize=True, name='volume', **kwargs
    ):

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
                metric = metric_func(source_idx, B, **kwargs)
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
                metrics['metric'] /= metrics['metric'].max()
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
            return helper(B)

    def get_excitation_volumes(
        self, B=None, as_frame=True, normalize=True, **kwargs
    ):
        """
        Get all volumes for excitations.
        """
        return self._get_metrics(
            self.get_excitation_volume,
            B, as_frame, normalize, 'volume', **kwargs)

    def get_capture_volumes(
        self, B=None, as_frame=True, normalize=True, **kwargs
    ):
        """
        Get all volumes for captures.
        """
        return self._get_metrics(
            self.get_capture_volume,
            B, as_frame, normalize, 'volume', **kwargs)

    def get_excitation_continuities(
        self, B=None, as_frame=True, normalize=True, **kwargs
    ):
        """
        Get all continuity likelihood for excitations.
        """
        return self._get_metrics(
            self.get_excitation_continuity,
            B, as_frame, normalize, 'continuity', **kwargs)

    def get_capture_continuities(
        self, B=None, as_frame=True, normalize=True, **kwargs
    ):
        """
        Get all volumes for captures.
        """
        return self._get_metrics(
            self.get_capture_continuity,
            B, as_frame, normalize, 'continuity', **kwargs)

    def get_excitation_mean_widths(
        self, B=None, as_frame=True, normalize=True, **kwargs
    ):
        """
        Get all mean widths for excitations.
        """
        return self._get_metrics(
            self.get_excitation_mean_width,
            B, as_frame, normalize, 'mean_width', **kwargs)

    def get_capture_mean_widths(
        self, B=None, as_frame=True, normalize=True, **kwargs
    ):
        """
        Get all mean widths for captures.
        """
        return self._get_metrics(
            self.get_capture_mean_width,
            B, as_frame, normalize, 'mean_width', **kwargs)

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

    def get_excitations(self, source_idx):
        """
        Get excitations values given selected LED set.
        """
        return self.photoreceptor_model.excitefunc(
            self.get_captures(source_idx)
        )

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
        convex_hull = ConvexHull(points)
        return convex_hull.volume

    def get_capture_volume(self, source_idx, B=None):
        """
        Get volume for capture values.
        """
        points = self.get_captures(source_idx)
        points = self.transform_points(points, B)
        return self.compute_volume(points)

    def get_excitation_volume(self, source_idx, B=None):
        """
        Get volume for capture values.
        """
        points = self.get_excitations(source_idx)
        points = self.transform_points(points, B)
        return self.compute_volume(points)

    @staticmethod
    def compute_continuity(points, bins=100, **kwargs):
        """
        Compute continuity likelihood. Useful for circular datapoints
        """
        if (points.ndim == 1) or (points.shape[1] < 2):
            H = np.histogram(points, bins, **kwargs)[0].astype(bool)
        else:
            H = np.histogramdd(points, bins, **kwargs)[0].astype(bool)
        return H.sum() / H.size

    def get_excitation_continuity(self, source_idx, B=None, bins=100, **kwargs):
        """
        Get the continuity likelihood.
        """
        points = self.get_excitations(source_idx)
        points = self.transform_points(points, B)
        return self.compute_continuity(points, bins, **kwargs)

    def get_capture_continuity(self, source_idx, B=None, bins=100, **kwargs):
        """
        Get the continuity likelihood.
        """
        points = self.get_captures(source_idx)
        points = self.transform_points(points, B)
        return self.compute_continuity(points, bins, **kwargs)

    @staticmethod
    def compute_mean_width(points, n=1000):
        """
        Compute mean width.
        """
        if (points.ndim == 1) or (points.shape[1] < 2):
            return np.max(points) - np.min(points)
        return compute_mean_width(points, n)

    def get_excitation_mean_width(self, source_idx, B=None, n=1000):
        """
        Get the mean width.
        """
        points = self.get_excitations(source_idx)
        points = self.transform_points(points, B)
        return self.compute_mean_width(points, n)

    def get_capture_mean_width(self, source_idx, B=None, n=1000):
        """
        Get the mean width.
        """
        points = self.get_captures(source_idx)
        points = self.transform_points(points, B)
        return self.compute_mean_width(points, n)

    @staticmethod
    def _get_source_idcs(n, k):
        idcs = np.array(list(combinations(np.arange(n), k)))
        source_idcs = np.zeros((len(idcs), n)).astype(bool)
        source_idcs[
            np.repeat(np.arange(len(idcs)), k),
            idcs.ravel()
        ] = True
        return source_idcs
