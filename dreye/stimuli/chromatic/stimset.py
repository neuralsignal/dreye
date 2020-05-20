"""
"""

import numpy as np
import pandas as pd


from dreye.stimuli.base import BaseStimulus, DUR_KEY, DELAY_KEY
from dreye.stimuli.mixin import SetStepMixin, SetBaselineMixin
from dreye.utilities import asarray
from dreye.stimuli.chromatic.transformers import (
    CaptureTransformerMixin, LinearTransformCaptureTransformerMixin,
    IlluminantCaptureTransformerMixin, IlluminantBgCaptureTransformerMixin,
    SignalTransformerMixin
)


class StimSet(BaseStimulus, SetBaselineMixin, SetStepMixin):

    def __init__(
        self,
        values=1,
        separate_channels=False,
        baseline_values=0,
    ):

        # call init method of BaseStimulus class
        super().__init__(
            values=values,
            separate_channels=separate_channels,
            baseline_values=baseline_values,
        )

        # sets values and baseline values attribute correctly
        self.values, self.baseline_values = self._set_values(
            values=values, baseline_values=baseline_values,
            separate_channels=separate_channels
        )

    # --- standard create and transform methods --- #

    def create(self):
        """create events, metadata, and signal
        """
        self._events, self._metadata = self._create_events()
        self._signal = self._create_signal(self._events)

    def transform(self):
        """transform signal to stimulus to be sent
        """
        self._stimulus = asarray(self.signal)

    # --- methods for create method --- #

    def _create_events(self):
        """create event dataframe
        """
        events = self.values.copy()
        # add necessary DUR and DELAY key
        events[DUR_KEY] = 0
        events[DELAY_KEY] = np.arange(len(events))
        events['name'] = self.name
        return events, {}

    def _create_signal(self, events):
        """create signal attribute
        """

        return asarray(self.values)


class LEDStimSet(SignalTransformerMixin, StimSet):
    alter_events = True


class PRStimSet(CaptureTransformerMixin, StimSet):
    alter_events = True


class TransformStimSet(
    LinearTransformCaptureTransformerMixin, StimSet
):
    alter_events = True


class IlluminantStimSet(
    IlluminantCaptureTransformerMixin, StimSet
):
    alter_events = True


class IlluminantBgStimSet(
    IlluminantBgCaptureTransformerMixin, StimSet
):
    alter_events = True
