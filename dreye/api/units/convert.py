"""
Convert units
"""

from dreye.api.units.pint import CONTEXTS, ureg


def has_units(value):
    """
    Check if has units via duck-typing.
    """
    return (
        hasattr(value, 'units')
        and hasattr(value, 'to')
        and hasattr(value, 'magnitude')
    )


def optional_to(obj, units, *args, **kwargs):
    """
    Optionally convert to units and return magnitude (numeric or array).
    """
    if has_units(obj):
        if units is None:
            obj = obj.magnitude
        else:
            obj = obj.to(units, *CONTEXTS, *args, **kwargs).magnitude
    return obj


def irr2flux(irradiance, wavelengths, return_units=None, prefix=None, irr_units='I'):
    """
    Convert from irradiance to photonflux.

    Parameters
    ----------
    irradiance : float or array-like
        Array in spectral irradiance units (I=W/m^2/nm) or units that
        can be converted to spectra irradiance.
    wavelengths : float or array-like
        Array that can be broadcast to irradiance array in nanometer units
        or units that can be converted to nanometer.
    return_units : bool, optional
        Whether to return a `pint.Quantity` or `numpy.ndarray` object.
        If None, the function will return a `pint.Quantity` if `irradiance`
        have units.
    prefix : str, optional
        Unit prefix for photonflux (e.g. `micro`).

    Returns
    -------
    photonflux : numpy.ndarray or pint.Quantity
        Values converted to photonflux (mol/m^2/s/nm).
    """
    prefix = '' if prefix is None else prefix
    if return_units is None:
        if has_units(irradiance):
            return_units = True
        else:
            return_units = False
    # convert units
    irradiance = (
        optional_to(irradiance, irr_units)
        * ureg(irr_units)
    )
    wavelengths = optional_to(wavelengths, 'nm') * ureg('nm')
    photonflux = irradiance * wavelengths / (
        ureg.planck_constant
        * ureg.speed_of_light
        * ureg.N_A
    )
    if return_units:
        return photonflux.to(f'{prefix}E')
    else:
        return photonflux.to(f'{prefix}E').magnitude


def flux2irr(photonflux, wavelengths, return_units=None, prefix=None, flux_units='E'):
    """
    Convert from photonflux to irradiance.

    Parameters
    ----------
    photonflux : float or array-like
        Array in spectral photonflux (E=mol/m^2/s/nm) or units that
        can be converted to photonflux.
    wavelengths : float or array-like
        Array that can be broadcast to irradiance array in nanometer units
        or units that can be converted to nanometer.
    return_units : bool, optional
        Whether to return a `pint.Quantity` or `numpy.ndarray` object.
        If None, the function will return a `pint.Quantity` if `photonflux`
        have units.
    prefix : str, optional
        Unit prefix for irradiance (e.g. `micro`).

    Returns
    -------
    irradiance : numpy.ndarray or pint.Quantity
        Values converted to photonflux (W/m^2/nm).
    """
    prefix = '' if prefix is None else prefix
    if return_units is None:
        if has_units(photonflux):
            return_units = True
        else:
            return_units = False
    # convert units
    photonflux = (
        optional_to(photonflux, flux_units)
        * ureg(flux_units)
    )
    wavelengths = optional_to(wavelengths, 'nm') * ureg('nm')
    irradiance = (
        photonflux * (
            ureg.planck_constant
            * ureg.speed_of_light
            * ureg.N_A)
    ) / wavelengths
    if return_units:
        return irradiance.to(f'{prefix}spectralirradiance')
    else:
        return irradiance.to(f'{prefix}spectralirradiance').magnitude
