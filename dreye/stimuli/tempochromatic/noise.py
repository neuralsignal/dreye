"""
"""

from dreye.stimuli.base import DUR_KEY, DELAY_KEY
from dreye.stimuli.temporal.noise import WhiteNoiseStimulus, BrownNoiseStimulus
from dreye.stimuli.chromatic.transformers import (
    CaptureTransformerMixin, LinearTransformCaptureTransformerMixin,
    IlluminantCaptureTransformerMixin, IlluminantBgCaptureTransformerMixin
)


def _add_fitted_random_signal(self):
    # add fitted random signal array in metadata
    self.metadata['target_random_signal'] = self.metadata['random_signal']
    dur_length = int(self.events[DUR_KEY].iloc[0] * self.rate)
    delay_idx = int(self.events[DELAY_KEY].iloc[0] * self.rate)
    random_signal = self.signal[delay_idx:dur_length+delay_idx]
    self.metadata['random_signal'] = random_signal


class PRWhiteNoiseStimulus(CaptureTransformerMixin, WhiteNoiseStimulus):
    time_axis = 0
    channel_axis = 1
    fit_only_uniques = False
    add_to_events_mean = False
    alter_events = True

    def create(self):
        super().create()
        _add_fitted_random_signal(self)


class PRBrownNoiseStimulus(CaptureTransformerMixin, BrownNoiseStimulus):
    time_axis = 0
    channel_axis = 1
    fit_only_uniques = False
    add_to_events_mean = False
    alter_events = True

    def create(self):
        super().create()
        _add_fitted_random_signal(self)


class TransformWhiteNoiseStimulus(
    LinearTransformCaptureTransformerMixin, WhiteNoiseStimulus
):
    time_axis = 0
    channel_axis = 1
    fit_only_uniques = False
    add_to_events_mean = False
    alter_events = True

    def create(self):
        super().create()
        _add_fitted_random_signal(self)


class TransformBrownNoiseStimulus(
    LinearTransformCaptureTransformerMixin, BrownNoiseStimulus
):
    time_axis = 0
    channel_axis = 1
    fit_only_uniques = False
    add_to_events_mean = False
    alter_events = True

    def create(self):
        super().create()
        _add_fitted_random_signal(self)


class IlluminantWhiteNoiseStimulus(
    IlluminantCaptureTransformerMixin, WhiteNoiseStimulus
):
    time_axis = 0
    channel_axis = 1
    fit_only_uniques = False
    add_to_events_mean = False
    alter_events = True

    def create(self):
        super().create()
        _add_fitted_random_signal(self)


class IlluminantBrownNoiseStimulus(
    IlluminantCaptureTransformerMixin, BrownNoiseStimulus
):
    time_axis = 0
    channel_axis = 1
    fit_only_uniques = False
    add_to_events_mean = False
    alter_events = True

    def create(self):
        super().create()
        _add_fitted_random_signal(self)


class IlluminantBgWhiteNoiseStimulus(
    IlluminantBgCaptureTransformerMixin, WhiteNoiseStimulus
):
    time_axis = 0
    channel_axis = 1
    fit_only_uniques = False
    add_to_events_mean = False
    alter_events = True

    def create(self):
        super().create()
        _add_fitted_random_signal(self)


class IlluminantBgBrownNoiseStimulus(
    IlluminantBgCaptureTransformerMixin, BrownNoiseStimulus
):
    time_axis = 0
    channel_axis = 1
    fit_only_uniques = False
    add_to_events_mean = False
    alter_events = True

    def create(self):
        super().create()
        _add_fitted_random_signal(self)
