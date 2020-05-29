"""
*dreye* (drosophila eye) is a *Python* photoreceptor and (color) vision
package implementing various photoreceptor and (color) vision models and
algorithms, mainly for drosophila.
Part of the code base is inspired by the Colour GitHub project
(https://github.com/colour-science/colour; BSD-3-Clause license),
which is a package implementing color vision models and algorithms for human
observers.
"""

# import all core elements and constants
from dreye.core.signal import Signals, DomainSignal, Signal
from dreye.core.signal_container import (
    SignalsContainer, DomainSignalContainer
)
from dreye.core.domain import Domain
from dreye.core.spectrum import (
    DomainSpectrum, Spectra,
    IntensitySpectra, IntensityDomainSpectrum,
    Spectrum, IntensitySpectrum
)
from dreye.core.spectral_measurement import (
    CalibrationSpectrum, MeasuredSpectrum, MeasuredSpectraContainer
)
from dreye.core.measurement_utils import (
    convert_measurement, create_measured_spectrum, create_measured_spectra,
    get_led_spectra_container
)
from dreye.core.spectrum_utils import create_gaussian_spectrum
from dreye.core.spectral_sensitivity import Sensitivity
from dreye.core.photoreceptor import LinearPhotoreceptor, LogPhotoreceptor


__all__ = [
    # domain
    'Domain',
    # signal
    'Signal',
    'Signals',
    'DomainSignal',
    'SignalsContainer',
    'DomainSignalContainer',
    # spectrum
    'DomainSpectrum',
    'Spectra',
    'Spectrum',
    'IntensitySpectrum',
    'IntensitySpectra',
    'IntensityDomainSpectrum',
    'CalibrationSpectrum',
    'MeasuredSpectrum',
    'MeasuredSpectraContainer',
    # measurement
    'convert_measurement',
    'create_measured_spectrum',
    'create_measured_spectra',
    # sensitivity
    'Sensitivity',
    # photoreceptor
    'LinearPhotoreceptor',
    'LogPhotoreceptor',
    'create_gaussian_spectrum',
    'get_led_spectra_container'
]
