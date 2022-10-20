"""
Handling domains
"""

from numbers import Number
import numpy as np
from scipy.interpolate import interp1d

from dreye.api.utils import arange


def equalize_domains(
    domains, arrs, axes=None, fill_value=0, 
    bounds_error=False, stack_axis=None, concatenate=False
):
    """
    Equalize domains between different arrays.

    Parameters
    ----------
    domains : [type]
        [description]
    arrs : [type]
        [description]
    axes : [type], optional
        [description], by default None
    fill_value : int, optional
        [description], by default 0
    bounds_error : bool, optional
        [description], by default False
    stack_axis : [type], optional
        [description], by default None
    concatenate : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    new_domain = domains[0]
    for domain in domains:
        if new_domain.shape != domain.shape:
            break
        all_equal = np.equal(new_domain, domain).all()
        if not all_equal:
            break
    else:
        if stack_axis is None:
            new_arrs = arrs
        elif concatenate:
            new_arrs = np.concatenate(arrs, axis=stack_axis)
        else:
            new_arrs = np.stack(arrs, axis=stack_axis)
        return new_domain, new_arrs
    
    count = len(domains)

    if axes is None:
        axes = [-1] * count
    elif isinstance(axes, Number):
        axes = [axes] * count

    lemin = -np.inf
    lemax = np.inf
    lediff = 0
    for domain in domains:
        lemin = np.maximum(lemin, np.min(domain))
        lemax = np.minimum(lemax, np.max(domain))
        lediff = np.maximum(lediff, np.mean(np.diff(np.sort(domain))))
        
    if (lemin >= lemax) or ((lemax - lemin) < lediff):
        raise ValueError("Cannot equalize domains.")

    new_domain = arange(lemin, lemax, lediff)
    # new_domain = np.arange(lemin, lemax+lediff-lemax%lediff, lediff)
    new_arrs = []
    for domain, arr, axis in zip(domain, arrs, axes):
        arr = interp1d(domain, arr, axis=axis, fill_value=fill_value, bounds_error=bounds_error)(new_domain)
        new_arrs.append(arr)

    if stack_axis is None:
        pass
    elif concatenate:
        new_arrs = np.concatenate(new_arrs, axis=stack_axis)
    else:
        new_arrs = np.stack(new_arrs, axis=stack_axis)

    return new_domain, new_arrs


# TODO smooth
# TODO integral, normalize

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
