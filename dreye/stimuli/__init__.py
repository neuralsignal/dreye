"""
Stimuli package
"""

from dreye.stimuli.base import BaseStimulus, ChainedStimuli
from dreye.stimuli.temporal.step import StepStimulus, RandomSwitchStimulus
from dreye.stimuli.temporal.noise import WhiteNoiseStimulus, BrownNoiseStimulus
from dreye.stimuli.chromatic.stimset import StimSet


__all__ = [
    'BaseStimulus', 'ChainedStimuli',
    'StepStimulus', 'RandomSwitchStimulus',
    'WhiteNoiseStimulus', 'BrownNoiseStimulus',
    'StimSet'
]
