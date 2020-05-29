"""
Unit Definitions and Registry
=============================

Defines units and set unit registry.
"""

import pint
import numpy as np


ureg = pint.UnitRegistry()
"""
Standard unit registry as defined by pint package.
"""

ureg.define('radiant_energy = joule = radiantenergy')
ureg.define('radiant_energy_density = joule / '
            'meter ** 3 = radiantenergydensity')
ureg.define('radiant_flux = watt = radiantflux')
ureg.define('spectral_flux = radiant_flux / nanometer = spectralflux')
ureg.define('radiant_intensity = radiant_flux / steradian = radiantintensity')
ureg.define('spectral_intensity = radiant_intensity /'
            ' nanometer = spectralintensity')
ureg.define('radiance = radiant_intensity / meter ** 2')
ureg.define('spectral_radiance = radiance / nanometer = spectralradiance')
ureg.define('irradiance = radiant_flux / meter ** 2 = irrad = flux_density')
ureg.define('spectral_irradiance = irradiance / '
            'nanometer = spectral_flux_density = spectralirradiance = I')
ureg.define('E_Q = mole / meter^2 / second = photonflux = photon_flux')
ureg.define(
    'spectral_E_Q = mole / meter^2 / second / nanometer = spectral_photonflux'
    ' = spectral_photon_flux = spectralphotonflux = E')

c = pint.Context('flux')


def gradient(domain):
    """calculate gradient and preserve units
    """
    slices = tuple(
        None
        if i == 1
        else slice(None, None, None)
        for i in domain.shape
    )
    return np.gradient(np.squeeze(domain.magnitude))[slices] * domain.units


c.add_transformation(
    '[length] * [mass] / [time] ** 3',
    '[substance] / [length] ** 2 / [time]',
    lambda ureg, x: (
        x
        / (
            ureg.planck_constant
            * ureg.speed_of_light
            * ureg.N_A)
    )
)


c.add_transformation(
    '[mass] / [time] ** 3',
    '[substance] / [length] ** 2 / [time]',
    lambda ureg, x, domain: (
        x * domain / (
            ureg.planck_constant
            * ureg.speed_of_light
            * ureg.N_A)
    )
)


c.add_transformation(
    '[substance] / [length] ** 2 / [time]',
    '[length] * [mass] / [time] ** 3',
    lambda ureg, x: (
        x
        * (
            ureg.planck_constant
            * ureg.speed_of_light
            * ureg.N_A)
    )
)


c.add_transformation(
    '[substance] / [length] ** 2 / [time]',
    '[mass] / [time] ** 3',
    lambda ureg, x, domain: (
        x * (
            ureg.planck_constant
            * ureg.speed_of_light
            * ureg.N_A)
    ) / domain
)


c.add_transformation(
    '[substance] / [length] ** 2 / [time]',
    '[mass] / [length] / [time] ** 3',
    lambda ureg, x, domain: (
        x
        * (
            ureg.planck_constant
            * ureg.speed_of_light
            * ureg.N_A)
    ) / domain / gradient(domain)
)

c.add_transformation(
    '[mass] / [length] / [time] ** 3',
    '[substance] / [length] ** 2 / [time]',
    lambda ureg, x, domain: (
        x
        / (
            ureg.planck_constant
            * ureg.speed_of_light
            * ureg.N_A)
    ) * domain * gradient(domain)
)


c.add_transformation(
    '[substance] / [length] ** 3 / [time]',
    '[mass] / [time] ** 3 / [length]',
    lambda ureg, x, domain: (
        x
        * (
            ureg.planck_constant
            * ureg.speed_of_light
            * ureg.N_A
        )
    ) / domain
)


c.add_transformation(
    '[mass] / [time] ** 3 / [length]',
    '[substance] / [length] ** 3 / [time]',
    lambda ureg, x, domain: (
        x
        / (
            ureg.planck_constant
            * ureg.speed_of_light
            * ureg.N_A
        )
    ) * domain
)


c.add_transformation(
    '[mass] / [time] ** 3',
    '[mass] / [length] / [time] ** 3',
    lambda ureg, x, domain: x / gradient(domain)
)


c.add_transformation(
    '[mass] / [length] / [time] ** 3',
    '[mass] / [time] ** 3',
    lambda ureg, x, domain: x * gradient(domain)
)


c.add_transformation(
    '[substance] / [length] ** 3 / [time]',
    '[substance] / [length] ** 2 / [time]',
    lambda ureg, x, domain: x * gradient(domain)
)


c.add_transformation(
    '[substance] / [length] ** 2 / [time]',
    '[substance] / [length] ** 3 / [time]',
    lambda ureg, x, domain: x / gradient(domain)
)

c.add_transformation(
    '[substance]',
    '',
    lambda ureg, x: x * ureg.N_A
)

c.add_transformation(
    '',
    '[substance]',
    lambda ureg, x: x / ureg.N_A
)


ureg.add_context(c)
ureg.enable_contexts('flux')
ureg.setup_matplotlib()

pint.set_application_registry(ureg)
