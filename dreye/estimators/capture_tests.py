"""
Capture tests
"""

import numpy as np
from scipy.spatial import ConvexHull

from dreye.utilities.array import asarray
from dreye.utilities.abstract import _InitDict, inherit_docstrings
from dreye.estimators.base import _PrModelMixin
from dreye.estimators.utils import get_spanning_intensities, spaced_solutions
from dreye.utilities.barycentric import barycentric_dim_reduction, cartesian_to_barycentric
from dreye.utilities.convex import hull_intersection_alpha, in_hull


@inherit_docstrings
class CaptureTests(_InitDict, _PrModelMixin):
    """
    Test if capture values are within system.
    """

    def __init__(
        self, 
        photoreceptor_model,
        measured_spectra,
        *,
        intensity_bounds=None,
        background=None,
        wavelengths=None, 
        capture_noise_level=None, 
        ignore_bounds=None, 
        bg_ints=None,
        background_external=None
    ):
        # set init
        self.photoreceptor_model = photoreceptor_model
        self.measured_spectra = measured_spectra
        self.intensity_bound = intensity_bounds
        self.wavelengths = wavelengths
        self.background = background
        self.capture_noise_level = capture_noise_level
        self.ignore_bounds = ignore_bounds
        self.bg_ints = bg_ints
        self.background_external = background_external

        self._set_pr_model_related_objects()

        # used by _PrModelMixin class methods
        samples = get_spanning_intensities(self.intensity_bounds_)
        self._sample_points = self.get_capture(samples.T)

    @property
    def is_underdetermined(self):
        return self._is_underdetermined_

    def _q_iterator(self, q, func, **kwargs):
        # TODO parallelize
        q = asarray(q)
        if q.ndim == 1:
            return func(q, **kwargs)
        else:
            qs = []
            for iq in q:
                qs.append(func(iq, **kwargs))
            qs = np.array(qs)
            return qs

    @property
    def sample_points(self):
        return self._sample_points

    def range_of_solutions(self, q):
        """
        Find the range of solutions that fit the capture `q`
        """
        return self._q_iterator(q, self._range_of_solutions_)

    def linspace_of_solutions(self, q, n=20):
        """
        A set of points that fit `q` in an underdetermined system.
        """
        assert self.is_underdetermined, "System must be underdetermined."

        def helper(q):
            wmin, wmax = self.range_of_solutions(q)

            return spaced_solutions(
                wmin, wmax, self.A_, q, n=n
            )

        return self._q_iterator(q, helper)

    def capture_in_range(self, q):
        """
        Check if capture is in measured spectra convex hull.

        Parameters
        ----------
        q : numpy.array (n_opsins)
            One set of captures.

        Returns
        -------
        inhull : bool
        """
        return self._q_iterator(q, self._capture_in_range_)

    def nonlinear_gamut_correction(self, X, neutral_point=None, n_steps_gamut_vertex_estimation=10000):
        if neutral_point is None:
            neutral_point = np.mean(X, axis=0)

        spanning_ints = get_spanning_intensities(
            self.intensity_bounds_, 
            compute_ratios=True,
            ratios=np.linspace(0, 1, n_steps_gamut_vertex_estimation)
        )
        points = self.get_excitation(spanning_ints.T) - neutral_point
        X = X - neutral_point

        if points.shape[1] == 1:
            assert np.any(points < 0) and np.any(points > 0)
            points_ = np.array([np.min(points), np.max(points)])
            alphas = points_[:, None, None] / X[None, ...]
            alphas[alphas <= 0] = np.nan
            alpha = np.nanmin(alphas)
        else:
            assert np.all(in_hull(points, np.zeros(points.shape[1]))), "neutral point is not in hull."
            hull = ConvexHull(points)
            alphas = hull_intersection_alpha(X, hull)
            alpha = np.nanmin(alphas)

        return X * alpha + neutral_point

    def gamut_scaling(self, Q):
        """
        Scale q to fit within gamut of system.
        """
        # correct for offset
        Q = Q - self._q_offset_
        # scale intensities to span gamut
        Amax = self.A_ * self.intensity_bounds_[1]
        amax = np.min(np.max(Amax, axis=-1))
        qmax = np.max(Q)
        return Q * amax / qmax + self._q_offset_

    def gamut_clipping(self, Q, neutral_point=None):
        """
        Clip q to fit within gamut of system.
        """
        if neutral_point is None:
            neutral_point = np.ones(self.photoreceptor_model_.n_opsins)
        neutral_point = np.atleast_2d(neutral_point)
        
        points = self.sample_points
        # remove zero point, if exists
        points = points[(points > 0).any(1)]
        # replace zero point with one
        Q = Q.copy()
        zero_rows = ~(Q > 0).any(1)
        Q[zero_rows] = 1
        I = Q.sum(axis=-1)

        center = barycentric_dim_reduction(neutral_point)
        
        bpoints = barycentric_dim_reduction(points) - center
        bQ = barycentric_dim_reduction(Q) - center
        
        # find alphas 
        if bpoints.shape[1] == 1:
            assert np.any(bpoints < 0) and np.any(bpoints > 0)
            bpoints_ = np.array([np.min(bpoints), np.max(bpoints)])
            alphas = bpoints_[:, None, None] / bQ[None, ...]
            alphas[alphas <= 0] = np.nan
            alpha = np.nanmin(alphas)
        else:
            assert np.all(in_hull(bpoints, np.zeros(bpoints.shape[1]))), "neutral point is not in hull."
            hull = ConvexHull(bpoints)
            alphas = hull_intersection_alpha(bQ, hull)
            alpha = np.nanmin(alphas)

        bQ_scaled = bQ * alpha + center

        Q = cartesian_to_barycentric(bQ_scaled, I)
        Q[zero_rows] = 0
        return Q

