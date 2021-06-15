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
__version__ = '0.2.0dev2'

# import all core elements and constants
from dreye.constants.units import ureg
from dreye.core.signal import (
    Signals, DomainSignal, Signal, domain_concat, labels_concat
)
from dreye.core.opsin_template import (
    stavenga1993_template, govardovskii2000_template
)
from dreye.core.signal_container import (
    SignalsContainer, DomainSignalContainer
)
from dreye.core.domain import Domain
from dreye.core.spectral_measurement import (
    CalibrationSpectrum, MeasuredSpectrum, MeasuredSpectraContainer
)
from dreye.core.measurement_utils import (
    convert_measurement, create_measured_spectrum,
    create_measured_spectra_container, create_measured_spectra_container
)
from dreye.core.spectrum_utils import create_gaussian_spectrum
from dreye.core.photoreceptor import (
    LinearPhotoreceptor, LogPhotoreceptor,
    create_photoreceptor_model, HyperbolicPhotoreceptor,
    Photoreceptor, LinearContrastPhotoreceptor,
    create_photoreceptor_model
)
from dreye.estimators.excitation_models import (
    IndependentExcitationFit, TransformExcitationFit,
    NonlinearTransformExcitationFit
)
from dreye.estimators.led_substitution import LedSubstitutionFit
from dreye.estimators.silent_substitution import BestSubstitutionFit

from dreye.estimators.intensity_models import (
    RelativeIntensityFit
)
from dreye.io.serialization import (
    read_json, write_json, read_pickle, write_pickle
)
from dreye.estimators.metrics import (
    MeasuredSpectraMetrics
)
from dreye.utilities import (
    irr2flux, flux2irr
)

# modules
from dreye import hardware
from dreye import utilities
from dreye import stimuli

# This directory
import os
DREYE_DIR = os.path.dirname(__file__)


__all__ = [
    'hardware', 'utilities', 'stimuli',
    # io
    'read_json',
    'write_json',
    'read_pickle',
    'write_pickle',
    # misc
    'ureg',
    'stavenga1993_template',
    'govardovskii2000_template',
    'irr2flux', 'flux2irr',
    # domain
    'Domain',
    # signal
    'Signal',
    'Signals',
    'DomainSignal',
    'SignalsContainer',
    'DomainSignalContainer',
    'labels_concat',
    'domain_concat',
    'LedSubstitutionFit',
    'BestSubstitutionFit',
    'MeasuredSpectraMetrics',
    # spectrum
    'CalibrationSpectrum',
    'MeasuredSpectrum',
    'MeasuredSpectraContainer',
    # measurement
    'convert_measurement',
    'create_measured_spectrum',
    # photoreceptor
    'Photoreceptor',
    'LinearPhotoreceptor',
    'LinearContrastPhotoreceptor',
    'LogPhotoreceptor',
    'HyperbolicPhotoreceptor',
    'create_gaussian_spectrum',
    'create_measured_spectra_container',
    'create_measured_spectra_container',
    'create_photoreceptor_model',
    'create_photoreceptor_model',
    # estimators
    'IndependentExcitationFit',
    'TransformExcitationFit',
    'RelativeIntensityFit',
    'NonlinearTransformExcitationFit'
]
