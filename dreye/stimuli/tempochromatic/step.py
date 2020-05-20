"""Chromatic Step Stimuli
"""

from dreye.stimuli.temporal.step import StepStimulus, RandomSwitchStimulus
from dreye.stimuli.chromatic.transformers import (
    CaptureTransformerMixin, LinearTransformCaptureTransformerMixin,
    IlluminantCaptureTransformerMixin, IlluminantBgCaptureTransformerMixin,
    SignalTransformerMixin
)


class LEDStepStimulus(SignalTransformerMixin, StepStimulus):
    fit_only_uniques = True
    alter_events = True


class PRStepStimulus(CaptureTransformerMixin, StepStimulus):
    fit_only_uniques = True
    alter_events = True


class TransformStepStimulus(
    LinearTransformCaptureTransformerMixin, StepStimulus
):
    fit_only_uniques = True
    alter_events = True


class IlluminantStepStimulus(
    IlluminantCaptureTransformerMixin, StepStimulus
):
    fit_only_uniques = True
    alter_events = True


class IlluminantBgStepStimulus(
    IlluminantBgCaptureTransformerMixin, StepStimulus
):
    fit_only_uniques = True
    alter_events = True


class LEDRandomSwitchStimulus(SignalTransformerMixin, RandomSwitchStimulus):
    fit_only_uniques = True
    alter_events = True


class PRRandomSwitchStimulus(CaptureTransformerMixin, RandomSwitchStimulus):
    fit_only_uniques = True
    alter_events = True


class TransformRandomSwitchStimulus(
    LinearTransformCaptureTransformerMixin, RandomSwitchStimulus
):
    fit_only_uniques = True
    alter_events = True


class IlluminantRandomSwitchStimulus(
    IlluminantCaptureTransformerMixin, RandomSwitchStimulus
):
    fit_only_uniques = True
    alter_events = True


class IlluminantBgRandomSwitchStimulus(
    IlluminantBgCaptureTransformerMixin, RandomSwitchStimulus
):
    fit_only_uniques = True
    alter_events = True
