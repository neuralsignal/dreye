"""
Handling domains
"""

from typing import Union, List, Tuple, Optional
from numbers import Number
import numpy as np
from scipy.interpolate import interp1d

from dreye.api.utils import arange_with_interval


def equalize_domains(
    domains: List[np.ndarray],
    arrs: List[np.ndarray],
    axes: Optional[Union[int, List[int]]] = None,
    fill_value: int = 0,
    bounds_error: bool = False,
    stack_axis: Optional[int] = None,
    concatenate: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Equalize domains between different arrays.

    Parameters
    ----------
    domains : list of ndarray
        List of 1-D arrays representing the domains to be equalized.
    arrs : list of ndarray
        List of n-D arrays whose domains are to be equalized.
    axes : int or list of int, optional
        The axis or axes along which the domains need to be equalized.
        If None, it assumes the last axis for all arrays, by default None.
    fill_value : int, optional
        The value to use for points outside of the interpolation range, by default 0.
    bounds_error : bool, optional
        If True, an error is thrown any time interpolation is attempted on a value outside of the range
        of x (where extrapolation is necessary). If False, out of bounds values are assigned fill_value.
        By default, it is set to False.
    stack_axis : int, optional
        Axis along which the arrays have to be stacked. If None, the arrays are not stacked, by default None.
    concatenate : bool, optional
        If True, the arrays are concatenated along the stack_axis, by default False.

    Returns
    -------
    tuple of ndarray
        A tuple containing the new domain and the new arrays.

    Raises
    ------
    ValueError
        If the domains cannot be equalized due to the range and difference of the domain values.
    """
    new_domain = domains[0]
    if not _is_equal_domains(domains):
        new_domain, arrs = _interpolate_domains(
            domains, arrs, axes, fill_value, bounds_error
        )

    if stack_axis is not None:
        arrs = _stack_or_concatenate(arrs, stack_axis, concatenate)

    return new_domain, arrs


def _is_equal_domains(domains: List[np.ndarray]) -> bool:
    """
    Check if all domains are equal.
    """
    first_domain = domains[0]
    return all(np.array_equal(first_domain, domain) for domain in domains)


def _interpolate_domains(
    domains: List[np.ndarray],
    arrs: List[np.ndarray],
    axes: Optional[Union[int, List[int]]],
    fill_value: int,
    bounds_error: bool,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Interpolate domains to equalize them.
    """
    count = len(domains)
    if axes is None:
        axes = [-1] * count
    elif isinstance(axes, Number):
        axes = [axes] * count

    lemin, lemax, lediff = _get_domain_bounds_and_diff(domains)
    if (lemin >= lemax) or ((lemax - lemin) < lediff):
        raise ValueError("Cannot equalize domains.")

    new_domain = arange_with_interval(lemin, lemax, lediff)
    new_arrs = []
    for domain, arr, axis in zip(domains, arrs, axes):
        interpolator = interp1d(
            domain, arr, axis=axis, fill_value=fill_value, bounds_error=bounds_error
        )
        new_arrs.append(interpolator(new_domain))

    return new_domain, new_arrs


def _get_domain_bounds_and_diff(
    domains: List[np.ndarray],
) -> Tuple[float, float, float]:
    """
    Get the bounds and mean difference of the domain values.
    """
    lemin = -np.inf
    lemax = np.inf
    lediff = 0
    for domain in domains:
        lemin = np.maximum(lemin, np.min(domain))
        lemax = np.minimum(lemax, np.max(domain))
        lediff = np.maximum(lediff, np.mean(np.diff(np.sort(domain))))

    return lemin, lemax, lediff


def _stack_or_concatenate(
    arrs: List[np.ndarray], stack_axis: int, concatenate: bool
) -> np.ndarray:
    """
    Stack or concatenate arrays along a specific axis.
    """
    if concatenate:
        return np.concatenate(arrs, axis=stack_axis)
    else:
        return np.stack(arrs, axis=stack_axis)


# TODO smooth

# def filter(
#     domain, arr,
#     domain_interval,
#     method='savgol', extrapolate=False,
#     **method_args
# ):
#     """
#     Filter signal using `scipy.signal.windows` function or
#     the `savgol` method.

#     Parameters
#     ----------
#     domain_interval : numeric, optional
#         The domain interval window to use for filtering. This should
#         be in units of `domain_units` or be convertible to these
#         units using `pint`'s `to` method.
#     method : str, optional
#         The method used for filtering the signal. Defaults to 'savgol'.
#         See `scipy.signal.windows` for more options.
#     extrapolate : bool, optional
#         Whether to extrapolate when applying a window filter from
#         `scipy.signal.windows`, in order to deal with edge cases.
#     method_args : dict, optional
#         Arguments passed to the filter method as is.

#     Returns
#     -------
#     object : signal-type
#         Filtered version of `self`.

#     See Also
#     --------
#     dreye.utilities.Filter1D
#     """

#     assert self.domain.is_uniform, (
#         "signal domain must be uniform for filtering"
#     )

#     domain_interval = optional_to(domain_interval, self.domain.units)

#     M = domain_interval / self.domain.interval
#     if M % 1 != 0:
#         warnings.warn(
#             "Chosen domain interval must be rounded down for filtering",
#             RuntimeWarning
#         )
#     M = int(M)

#     if method == 'savgol':

#         method_args['polyorder'] = method_args.get('polyorder', 2)
#         method_args['axis'] = self.domain_axis
#         M = M + ((M + 1) % 2)
#         values = savgol_filter(self.magnitude, M, **method_args)

#     elif extrapolate:
#         # create filter instance
#         filter1d = Filter1D(method, M, **method_args)
#         # handle borders by interpolating
#         start_idx, end_idx = \
#             int(np.floor((M - 1) / 2)), int(np.ceil((M - 1) / 2))
#         # create new domain
#         new_domain = self.domain.extend(
#             start_idx, left=True
#         ).extend(
#             end_idx, left=False
#         ).magnitude

#         values = filter1d(
#             self(new_domain).magnitude,
#             axis=self.domain_axis,
#             mode='valid'
#         )

#     else:
#         # create filter instance
#         filter1d = Filter1D(method, M, **method_args)
#         # from function
#         values = filter1d(
#             self.magnitude,
#             axis=self.domain_axis,
#             mode='same'
#         )

#     # filtering is shape-preserving
#     # sanity check
#     assert values.shape == self.shape, "Shape mismatch for filtering."
#     new = self.copy()
#     new._values = values
#     return new
