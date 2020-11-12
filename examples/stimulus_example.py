"""
Example Stimulus
"""

import numpy as np
import pandas as pd

from dreye.stimuli import BaseStimulus
# static column names for events dataframe
from dreye.stimuli.variables import DUR_KEY, PAUSE_KEY, DELAY_KEY
# DUR_KEY: dur of an event
# DELAY_KEY: start time of the event
# PAUSE_KEY: pause dur after the event ends


class RandomStimulus(BaseStimulus):

    # init not strictly necessary, but makes explicit which parameters
    # should be passed
    def __init__(
        self,
        rate=60,  # in Hz
        width=10, height=10,
        dur=1,  # in seconds here
        delay=1,  # in seconds here
        pause=1,  # in seconds here
        n_channels=1,  # number of channels (usually different LEDs, but can be number of opsins if estimator is passed)
        seed=None,
        estimator=None
    ):
        super().__init__(
            rate=rate, width=width,
            height=height, dur=dur,
            seed=seed,
            estimator=estimator,
            delay=delay, pause=pause,
            n_channels=n_channels
        )

    def create(self):
        # create stimulus parts
        np.random.seed(self.seed)

        n_frames = int(self.rate * self.dur)
        dur = n_frames / self.rate  # actual accuracy
        random_signal = np.random.random((
            n_frames, self.width, self.height, self.n_channels
        )) * 2 - 1  # values will be between -1 and 1

        n_frames = int(self.rate * self.delay)
        delay = n_frames / self.rate  # actual accuracy
        delay_signal = np.zeros((
            n_frames, self.width, self.height, self.n_channels
        ))

        n_frames = int(self.rate * self.pause)
        pause = n_frames / self.rate  # actual accuracy
        pause_signal = np.zeros((
            n_frames, self.width, self.height, self.n_channels
        ))

        # concatenate parts to create signal attribute
        self.signal = np.concatenate(
            [delay_signal, random_signal, pause_signal],
            axis=0
        )
        # dataframe that describe all events (only one here)
        # this can contain anything necessary to uniquely identify an event
        # and sometimes I put whole numpy arrays into single cells
        self.events = pd.DataFrame([[
            dur, delay, pause, 'random_signal'
        ]], columns=[
            DUR_KEY, DELAY_KEY, PAUSE_KEY, 'signal_type'
        ])
        # arbitrary metadata dictionary that can contain any stimulus
        # specific keys that are not event specific
        self.metadata = {}

        return self

# self.stimulus
