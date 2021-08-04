"""
Capture tests
"""

import numpy as np

from dreye.utilities.array import asarray
from dreye.utilities.abstract import _InitDict, inherit_docstrings
from dreye.estimators.base import _PrModelMixin
from dreye.estimators.utils import get_spanning_intensities, spaced_solutions


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

