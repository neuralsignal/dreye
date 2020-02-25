"""Chromatic Step Stimuli
"""

from dreye.stimuli.temporal.step import StepStimulus, RandomSwitchStimulus
from dreye.stimuli.chromatic.transformers import (
    CaptureTransformerMixin, LinearTransformCaptureTransformerMixin,
    IlluminantCaptureTransformerMixin, IlluminantBgCaptureTransformerMixin
)


class PRStepStimulus(CaptureTransformerMixin, StepStimulus):
    time_axis = 0
    channel_axis = 1
    alter_events = True


class TransformStepStimulus(
    LinearTransformCaptureTransformerMixin, StepStimulus
):
    time_axis = 0
    channel_axis = 1
    alter_events = True


class IlluminantStepStimulus(
    IlluminantCaptureTransformerMixin, StepStimulus
):
    time_axis = 0
    channel_axis = 1
    alter_events = True


class IlluminantBgStepStimulus(
    IlluminantBgCaptureTransformerMixin, StepStimulus
):
    time_axis = 0
    channel_axis = 1
    alter_events = True


class PRRandomSwitchStimulus(CaptureTransformerMixin, RandomSwitchStimulus):
    time_axis = 0
    channel_axis = 1
    alter_events = True


class TransformRandomSwitchStimulus(
    LinearTransformCaptureTransformerMixin, RandomSwitchStimulus
):
    time_axis = 0
    channel_axis = 1
    alter_events = True


class IlluminantRandomSwitchStimulus(
    IlluminantCaptureTransformerMixin, RandomSwitchStimulus
):
    time_axis = 0
    channel_axis = 1
    alter_events = True


class IlluminantBgRandomSwitchStimulus(
    IlluminantBgCaptureTransformerMixin, RandomSwitchStimulus
):
    time_axis = 0
    channel_axis = 1
    alter_events = True
