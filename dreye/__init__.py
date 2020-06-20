"""
*dreye* (drosophila eye) is a *Python* photoreceptor and (color) vision
package implementing various photoreceptor and (color) vision models and
algorithms, mainly for drosophila.
Part of the code base is inspired by the Colour GitHub project
(https://github.com/colour-science/colour; BSD-3-Clause license),
which is a package implementing color vision models and algorithms for human
observers.
"""

__author__ = """gucky92"""
__email__ = 'gucky@gucky.eu'
__version__ = '0.0.0'

# import all core elements and constants
from dreye.constants.units import ureg
from dreye.algebra.filtering import Filter1D
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
from dreye.core.photoreceptor import (
    LinearPhotoreceptor, LogPhotoreceptor,
    get_photoreceptor_model
)
from dreye.estimators.excitation_models import (
    IndependentExcitationFit, TransformExcitationFit,
    ReflectanceExcitationFit
)

from dreye.estimators.intensity_models import (
    IntensityFit, RelativeIntensityFit
)

# import modules
from dreye import stimuli
from dreye import utilities
from dreye import io
from dreye import hardware
from dreye.utilities import abstract


__all__ = [
    'ureg',
    'Filter1D',
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
    'get_led_spectra_container',
    'get_photoreceptor_model',
    # estimators
    'IndependentExcitationFit',
    'TransformExcitationFit',
    'ReflectanceExcitationFit',
    'IntensityFit',
    'RelativeIntensityFit',
    # modules
    'stimuli', 'utilities',
    'io', 'hardware', 'algebra',
    'constants', 'abstract'
]
