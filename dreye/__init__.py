"""
"""

__author__ = """gucky92"""
__email__ = "gucky@gucky.eu"
__version__ = "1.0.2dev1"

import os

from dreye.api.capture import calculate_capture
from dreye.api.estimator import ReceptorEstimator
from dreye.api.units.convert import irr2flux, flux2irr
from dreye.api.filter_templates import (
    gaussian_template,
    govardovskii2000_template,
    stavenga1993_template,
)
from dreye.api.utils import (
    l1norm,
    l2norm,
    integral,
    round_to_precision,
    round_to_significant_digits,
    arange_with_interval,
)
from dreye.api.barycentric import barycentric_to_cartesian, cartesian_to_barycentric
from dreye.api.convex import in_hull, range_of_solutions
from dreye.api.capture import calculate_capture
from dreye.api.domain import equalize_domains
from dreye.api.metrics import (
    compute_jensen_shannon_divergence,
    compute_jensen_shannon_similarity,
    compute_mean_width,
    compute_mean_correlation,
    compute_mean_mutual_info,
    compute_volume,
    compute_gamut,
)
from dreye.api.project import (
    proj_P_for_hull,
    proj_P_to_simplex,
    line_to_simplex,
    proj_B_to_hull,
    alpha_for_B_with_P,
    B_with_P,
)
from dreye.api.sampling import (
    sample_in_hull,
    d_equally_spaced,
)
from dreye.api.spherical import (
    spherical_to_cartesian,
    cartesian_to_spherical,
)

DREYE_DIR = os.path.dirname(__file__)


__all__ = [
    "calculate_capture",
    "ReceptorEstimator",
    "irr2flux",
    "flux2irr",
    "gaussian_template",
    "govardovskii2000_template",
    "stavenga1993_template",
    "l1norm",
    "l2norm",
    "integral",
    "round_to_precision",
    "round_to_significant_digits",
    "arange_with_interval",
    "barycentric_to_cartesian",
    "cartesian_to_barycentric",
    "in_hull",
    "range_of_solutions",
    "calculate_capture",
    "equalize_domains",
    "compute_jensen_shannon_divergence",
    "compute_jensen_shannon_similarity",
    "compute_mean_width",
    "compute_mean_correlation",
    "compute_mean_mutual_info",
    "compute_volume",
    "compute_gamut",
    "proj_P_for_hull",
    "proj_P_to_simplex",
    "line_to_simplex",
    "proj_B_to_hull",
    "alpha_for_B_with_P",
    "B_with_P",
    "sample_in_hull",
    "d_equally_spaced",
    "spherical_to_cartesian",
    "cartesian_to_spherical",
    "DREYE_DIR"
]
