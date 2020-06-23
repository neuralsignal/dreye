"""
=====================
Stimuli API Reference
=====================

.. currentmodule:: dreye.stimuli


Abstract stimuli classes
========================

.. autosummary::
   :toctree: api/

   BaseStimulus
   DynamicStimulus


Stimuli Containers
==================

.. autosummary::
   :toctree: api/

   ChainedStimuli
   RandomizeChainedStimuli


Temporal Stimuli
================

.. autosummary::
   :toctree: api/

   StepStimulus
   RandomSwitchStimulus
   NoiseStepStimulus
   WhiteNoiseStimulus
   BrownNoiseStimulus


Stimuli Sets
============

.. autosummary::
   :toctree: api/

   StimSet

"""

from dreye.stimuli.base import (
    BaseStimulus, ChainedStimuli, DynamicStimulus,
    RandomizeChainedStimuli
)
from dreye.stimuli.temporal.step import (
    StepStimulus, RandomSwitchStimulus, NoiseStepStimulus
)
from dreye.stimuli.temporal.noise import WhiteNoiseStimulus, BrownNoiseStimulus
from dreye.stimuli.chromatic.stimset import StimSet


__all__ = [
    'BaseStimulus', 'ChainedStimuli',
    'StepStimulus', 'RandomSwitchStimulus',
    'WhiteNoiseStimulus', 'BrownNoiseStimulus',
    'StimSet', 'DynamicStimulus', 'NoiseStepStimulus',
    'RandomizeChainedStimuli'
]
