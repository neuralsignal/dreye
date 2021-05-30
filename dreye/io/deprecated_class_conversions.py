"""
Converted deprecated classes during serialization
"""

from dreye.core.signal import Signal, Signals, DomainSignal
from dreye.estimators.intensity_models import RelativeIntensityFit
from dreye.estimators.excitation_models import IndependentExcitationFit


class ReflectanceExcitationFit:

    _rm_kwargs = [
        'reflectances', 
        'add_background', 
        'filter_background',
    ]

    @classmethod
    def from_dict(cls, data):
        for k in cls._rm_kwargs:
            data.pop(k)
        return IndependentExcitationFit.from_dict(data)


class IntensityFit:

    @classmethod
    def from_dict(cls, data):
        data['rtype'] = 'absolute'
        return RelativeIntensityFit.from_dict(data)


_deprecated_classes = {
    'ReflectanceExcitationFit': ReflectanceExcitationFit, 
    'IntensityFit': IntensityFit, 
    'Spectrum': Signal, 
    'Spectra': Signals, 
    'IntensitySpectrum': Signal, 
    'Sensitivity': Signals, 
    'IntensitySpectra': Signals, 
    'DomainSpectrum': DomainSignal, 
    'IntensityDomainSpectrum': DomainSignal, 
}