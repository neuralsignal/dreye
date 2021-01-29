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
__version__ = '0.0.11'

# import all core elements and constants
from dreye.constants.units import ureg
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
    convert_measurement, create_measured_spectrum,
    get_led_spectra_container
)
from dreye.core.spectrum_utils import create_gaussian_spectrum
from dreye.core.spectral_sensitivity import Sensitivity
from dreye.core.photoreceptor import (
    LinearPhotoreceptor, LogPhotoreceptor,
    get_photoreceptor_model, HyperbolicPhotoreceptor,
    Photoreceptor, LinearContrastPhotoreceptor
)
from dreye.estimators.excitation_models import (
    IndependentExcitationFit, TransformExcitationFit,
    ReflectanceExcitationFit
)

from dreye.estimators.intensity_models import (
    IntensityFit, RelativeIntensityFit
)
from dreye.io.serialization import (
    read_json, write_json, read_pickle, write_pickle
)

# modules
from dreye import hardware
from dreye import utilities
from dreye import stimuli


__all__ = [
    'hardware', 'utilities', 'stimuli',
    # io
    'read_json',
    'write_json',
    'read_pickle',
    'write_pickle',
    # misc
    'ureg',
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
    # sensitivity
    'Sensitivity',
    # photoreceptor
    'Photoreceptor',
    'LinearPhotoreceptor',
    'LinearContrastPhotoreceptor',
    'LogPhotoreceptor',
    'HyperbolicPhotoreceptor',
    'create_gaussian_spectrum',
    'get_led_spectra_container',
    'get_photoreceptor_model',
    # estimators
    'IndependentExcitationFit',
    'TransformExcitationFit',
    'ReflectanceExcitationFit',
    'IntensityFit',
    'RelativeIntensityFit',
]
