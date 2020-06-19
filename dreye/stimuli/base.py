"""Base class for stimuli
"""

from abc import ABC, abstractmethod
import random

import numpy as np
import pandas as pd

from dreye.io import read_json, write_json
from dreye.utilities import is_numeric, is_listlike, asarray, is_callable
from dreye.err import DreyeError
from dreye.stimuli.variables import DUR_KEY, DELAY_KEY, PAUSE_KEY
from dreye.stimuli.plotting import StimPlottingMixin


def _check_events(df):
    """check that event dataframe contains columns DELAY_KEY and DUR_KEY
    """

    if not isinstance(df, pd.DataFrame):
        raise DreyeError('Events frame must be dataframe.')

    elif len(df) == 0:

        cols = list(df.columns)
        if (DELAY_KEY in cols and DUR_KEY in cols):
            pass
        else:
            if DELAY_KEY not in cols:
                cols.append(DELAY_KEY)
            if DUR_KEY not in cols:
                cols.append(DUR_KEY)
            if PAUSE_KEY not in cols:
                cols.append(PAUSE_KEY)
            df = pd.DataFrame(columns=cols)

    else:
        assert not ({DELAY_KEY, DUR_KEY} - set(df.columns)), \
            'event frame must contain column start and dur.'
        if PAUSE_KEY not in df.columns:
            df[PAUSE_KEY] = 0.0

        # stringify columns - and copy dataframe WHY?
        cols = {
            col: str(col) for col in df.columns
            if not isinstance(col, str)
        }
        if cols:
            df = df.rename(columns=cols)

    return df


class BaseStimulus(ABC, StimPlottingMixin):

    time_axis = 0
    channel_axis = 1
    _add_mean_to_events = True

    def __init__(self, *, estimator=None, rate=None, seed=None, **kwargs):
        """base initialization
        """

        self.estimator = estimator
        self.seed = seed
        self.rate = rate
        if rate is not None:
            assert is_numeric(rate), 'rate must be numeric'
        if seed is not None:
            assert is_numeric(seed), 'seed must be numeric'
        if estimator is not None:
            assert hasattr(estimator, 'fit_transform'), (
                'Estimator requires methods `fit` and `transform`.'
            )

        # directly set elements
        for key, ele in kwargs.items():
            setattr(self, key, ele)

        self._stimulus = None
        self._signal = None
        self._fitted_signal = None
        self._metadata = {}
        self._events = pd.DataFrame()

        # if settings already exists simply update dictionary
        settings = {
            'rate': self.rate,
            'seed': self.seed,
            'estimator': self.estimator,
            **kwargs
        }
        if hasattr(self, '_settings'):
            self._settings.update(
                settings
            )
        else:
            self._settings = settings

    @property
    def _plot_attrs(self):
        plot_attrs = super()._plot_attrs
        if hasattr(self.estimator, '_X_length'):
            plot_attrs = (
                plot_attrs[:-1] + self.estimator._X_length + plot_attrs[-1:]
            )
        return plot_attrs

    @abstractmethod
    def create(self):
        """create signal and metadata
        """

    def transform(self):
        """transform signal to stimulus
        """

        if self.estimator is None:
            self._stimulus = self.signal
        else:
            self._stimulus = self.estimator.fit_transform(self.signal)

        if (
            hasattr(self.estimator, 'fitted_X')
            and hasattr(self.estimator, 'current_X_')
        ):
            if self.estimator.current_X_.shape == self.signal.shape:
                self._fitted_signal = self.estimator.fitted_X
            else:
                self._fitted_signal = self.signal
        else:
            self._fitted_signal = self.signal

        events = _check_events(self._events)
        # if channel names not in events add signal
        # HERE no overlap at all in names
        if not set(events.columns) & set(self.ch_names):
            events = self._add_to_events(
                events, self.ch_names, self.signal, '',
                'signal'
            )
        # MUST include all channel names
        elif (
            not set(self.ch_names) - set(events.columns)
            and hasattr(self, '_attrs_to_event_labels_mapping')
        ):
            # add signal to event labels mapping
            self._attrs_to_event_labels_mapping['signal'] = self.ch_names
        # add stimulus and fitted_signal to events
        events = self._add_to_events(
            events, self.ch_names, self.fitted_signal, 'fitted_',
            'fitted_signal'
        )
        if hasattr(self.estimator, '_X_length'):
            for attr in self.estimator._X_length:
                # special cases (handles photoreceptor model and spms)
                if (
                    'intensities_' in attr
                    and hasattr(self.estimator, 'measured_spectra_')
                ):
                    labels = self.estimator.measured_spectra_.names
                    prefix = attr.replace('intensities_', '')
                elif (
                    'excite_X_' in attr
                    and hasattr(self.estimator, 'photoreceptor_model_')
                ):
                    labels = self.estimator.photoreceptor_model_.names
                    prefix = attr.replace('excite_X_', '')
                elif (
                    'capture_X_' in attr
                    and hasattr(self.estimator, 'photoreceptor_model_')
                ):
                    labels = self.estimator.photoreceptor_model_.names
                    prefix = attr.replace('capture_X_', 'q_')
                else:
                    labels = None
                    prefix = attr
                events = self._add_to_events(
                    events, labels, getattr(self.estimator, attr), prefix,
                    attr
                )

        # ensure check again (stringify etc.)
        events = _check_events(events)
        self._events = events

    # --- short hand for metadata dictionary --- #

    def __getitem__(self, index):
        """
        """

        return self.metadata[index]

    def __setitem__(self, index, value):
        """
        """

        self.metadata[index] = value

    def __getattr__(self, name):
        if 'estimator' in vars(self) and hasattr(self.estimator, name):
            return getattr(self.estimator, name)
        else:
            raise AttributeError(
                f"`{type(self).__name__}` instance has not attribute `{name}`."
            )

    # --- properties of stimulus --- #

    @property
    def name(self):
        """short-hand for stimulus class name
        """

        return type(self).__name__

    @property
    def metadata(self):
        """a dictionary of the metadata
        """

        if self._signal is None:
            self.create()

        return self._metadata

    @property
    def ch_names(self):
        """
        Name for each channel
        """
        return list(range(self.signal.shape[self.channel_axis]))

    @property
    def events(self):
        """a pandas dataframe of events in the stimulus (custom-formatted).
        Each row is a single event.
        """

        if self._signal is None:
            self.transform()

        return self._events

    def time2frame(self, key=DELAY_KEY):
        """get idcs for delay period
        """

        if self.rate is None:
            return self.events[key].round(0).astype(int)
        else:
            return (self.events[key] * self.rate).round(0).astype(int)

    @property
    def signal(self):
        """unprocessed array of stimulus (not for sending to hardware)
        """

        if self._signal is None:
            self.create()

        return self._signal

    @property
    def fitted_signal(self):
        """
        fitted array using the estimator (if hasattr fitted_X)
        """

        if self._fitted_signal is None:
            self.transform()

        return self._fitted_signal

    @property
    def stimulus(self):
        """processed numpy array of stimulus (for sending to hardware)
        """

        if self._stimulus is None:
            self.transform()

        return self._stimulus

    @property
    def other_shape(self):
        """shape of stimulus (excluding time axis)
        """

        shape = list(self.stimulus.shape)
        shape.pop(self.time_axis)
        return tuple(shape)

    @property
    def time_len(self):
        """length of time axis
        """

        return self.stimulus.shape[self.time_axis]

    @property
    def channel_len(self):
        """length of channel axis
        """

        return self.stimulus.shape[self.channel_axis]

    @property
    def timestamps(self):
        """timestamps of stimulus same size of first axis
        """

        length = self.signal.shape[self.time_axis]
        arr = np.arange(0, length)

        if self.rate is None:
            return arr
        else:
            return arr / self.rate

    @property
    def duration(self):
        """duration of stimulus
        """

        length = self.signal.shape[self.time_axis]

        if self.rate is None:
            return length
        else:
            return length / self.rate

    @property
    def settings(self):
        """a dictionary of the settings
        """

        if not hasattr(self, '_settings'):
            return {}

        return self._settings

    # --- serialization methods --- #

    def to_dict(self):
        """dictionary format of stimulus class
        """

        return {
            'stimulus': self.stimulus,
            'signal': self.signal,
            'fitted_signal': self.fitted_signal,
            'metadata': self.metadata,
            'events': self.events.to_dict('list'),
            'settings': self.settings,
        }

    @classmethod
    def from_dict(cls, data):
        """
        """

        self = cls(**data['settings'])
        self._stimulus = data['stimulus']
        self._signal = data['signal']
        self._fitted_signal = data['fitted_signal']
        self._metadata = data['metadata']
        self._events = pd.DataFrame(data['events'])

        return self

    def save_settings(self, filename):
        """convenience function to save settings.
        """

        write_json(filename, self.settings)

    @classmethod
    def load_settings(cls, filename):
        """convenience function to load settings (just call read_json)
        """

        return read_json(filename)

    def save(self, filename):
        """function to save stimulus
        """

        write_json(filename, self)

    @classmethod
    def load(cls, filename):
        """function to load stimulus
        """

        return read_json(filename)

    def _add_to_events(self, events, labels, signal, prefix, attr_name):
        """
        method used to add transformations of signals
        """
        # has to be an array
        signal = asarray(signal)

        if labels is None:
            labels = [
                f'{prefix}{idx}'
                for idx in range(signal.shape[self.channel_axis])
            ]
        else:
            labels = [
                f"{prefix}{label}" for label in labels
            ]
        # add opsin channels to events
        # warn if channel names already exist
        truth = (
            set(labels)
            & set(events.columns)
        )
        if truth:
            raise DreyeError('Events Dataframe already contains columns'
                             f' for labels {labels}.')
        # assert sizes match
        assert len(labels) == signal.shape[self.channel_axis]
        # if signal high-dimensional then events columns need to be object type
        events_ = {}
        for channel in labels:
            events_[channel] = []

        for idx, row in events.iterrows():
            # get indices for event
            if self.rate is None:
                idcs = asarray([row[DELAY_KEY]]).astype(int)
            else:
                dur_length = row[DUR_KEY] * self.rate
                delay_idx = row[DELAY_KEY] * self.rate
                idcs = (np.arange(dur_length) + delay_idx).astype(int)
            # extract signal from event
            _signal = np.take(signal, idcs, axis=self.time_axis)
            # for each channel in labels take channel in signal
            # and average across time (and channel)
            for jdx, channel in enumerate(labels):
                # taking a list ensures that the axis are still aligned!
                value = np.take(_signal, [jdx], axis=self.channel_axis)
                if self._add_mean_to_events:
                    value = value.mean(
                        axis=(self.time_axis, self.channel_axis)
                    )
                else:
                    value = value.mean(axis=self.channel_axis)
                # assign entry
                events_[channel].append(value)

        events_ = pd.DataFrame(events_, index=events.index)
        # reassign events
        events = pd.concat([events, events_], axis=1)
        # assign attr_name to labels
        if hasattr(self, '_attrs_to_event_labels_mapping'):
            self._attrs_to_event_labels_mapping[attr_name] = labels

        return events


class DynamicStimulus(BaseStimulus):
    """
    Create a stimulus from a function dynamically.
    """

    def __init__(
        self, create_func,
        *,
        estimator=None,
        rate=None,
        seed=None,
        **kwargs
    ):
        assert is_callable(create_func), (
            "Create function must be callable."
        )
        super().__init__(
            create_func=create_func,
            estimator=estimator,
            rate=rate,
            seed=seed,
            **kwargs
        )

    def create(self):
        """create signal and metadata
        """
        output = self.create_func(self)
        if isinstance(output, tuple):
            if len(output) == 1:
                self._signal = output[0]
                self._events = pd.DataFrame()
                self._metadata = {}
            elif len(output) == 2:
                self._signal = output[0]
                self._events = output[1]
                self._metadata = {}
            elif len(output) == 3:
                self._signal = output[0]
                self._events = output[1]
                self._metadata = output[2]
            else:
                raise DreyeError("Output from `create_func` must contain "
                                 "only three tuples maximum, but contains "
                                 f"{len(output)}.")
        else:
            self._signal = output
            self._events = pd.DataFrame()
            self.metadata = {}


class ChainedStimuli:
    """
    Chain multiple stimuli together.
    """

    def __init__(self, stimuli, shuffle=False, seed=None):

        if isinstance(stimuli, type(self)):
            stimuli = list(stimuli.stimuli)
        elif is_listlike(stimuli):
            assert all(
                isinstance(stim, BaseStimulus)
                for stim in stimuli
            ), "All stimuli must be subclass of BaseStimulus."
            stimuli = list(stimuli)
        else:
            raise DreyeError('Stimuli must be ChainedStimuli or listlike.')

        if seed is not None:
            random.seed(seed)
        if shuffle:
            random.shuffle(stimuli)

        assert all(
            stimuli[0].time_axis == stim.time_axis
            for stim in stimuli
        ), "time axis must be aligned"
        assert all(
            stimuli[0].other_shape == stim.other_shape
            for stim in stimuli
        ), "shapes of stimuli must be the same"
        # assert rates are the same
        assert all(
            stimuli[0].rate == stim.rate
            for stim in stimuli
        ), "stimuli rates must be the same"

        self._stimuli = stimuli

    def __len__(self):
        return len(self.stimuli)

    def __iter__(self):
        return iter(self.stimuli)

    def __getitem__(self, key):
        return self.stimuli[key]

    @property
    def rate(self):
        return self.stimuli[0].rate

    @property
    def time_axis(self):
        return self.stimuli[0].time_axis

    @property
    def stimuli(self):
        return self._stimuli

    @property
    def events(self):
        """combine dataframe
        """
        events = pd.DataFrame()
        dur = 0.0
        for idx, stim in enumerate(self.stimuli):
            assert 'stim_index' not in stim.events
            event = stim.events.copy()
            # add duration
            event[DELAY_KEY] += dur
            # add index
            event['stim_index'] = idx
            events = events.append(event, ignore_index=True, sort=True)
            dur += stim.duration
        return events

    @property
    def metadata(self):
        return [
            stim.metadata
            for stim in self.stimuli
        ]

    @property
    def stimulus(self):
        """concatenate along time axis
        """
        return np.concatenate([
            stim.stimulus for stim in self.stimuli
        ], axis=self.time_axis)

    @property
    def timestamps(self):
        return np.concatenate([
            stim.timestamps
            + (0 if idx == 0 else np.cumsum(self.durations)[idx-1])
            for idx, stim in enumerate(self.stimuli)
        ])

    @property
    def duration(self):
        return np.sum(self.durations)

    @property
    def durations(self):
        """array of duration for each stimulus
        """
        return asarray([
            stim.duration
            for stim in self.stimuli
        ])

    @property
    def settings(self):
        return [
            stim.settings
            for stim in self.stimuli
        ]

    def to_dict(self):
        return self.stimuli

    @classmethod
    def from_dict(cls, data):
        return cls(data)

    def save(self, filename):
        write_json(filename, self)

    @classmethod
    def load(cls, filename):
        return read_json(filename)

# have alist of all attributes


class RandomizeChainedStimuli(ChainedStimuli):
    """
    Randomize chained stimuli by reordering events dataframe.
    """

    def __init__(self, stimuli, shuffle=True, seed=None):
        # initialize chained stimuli
        super().__init__(stimuli, shuffle=False, seed=None)

        # dataframe to shuffle
        if shuffle:
            events = super().events
            stimulus = super().stimulus
            # shuffle events
            events = events.sample(
                frac=1, replace=False, random_state=seed
            ).reset_index()
            # get index of first event (offset)
            offset_idx = int(events[DELAY_KEY].min() * self.rate)
            # copy stimulus
            new_stimulus = stimulus.copy()
            # loop over events
            for idx, event in events.iterrows():
                frames = int(
                    (event[DUR_KEY] + event[PAUSE_KEY]) * self.rate
                )
                # new start and end
                new_start_idx = offset_idx
                new_end_idx = offset_idx + frames
                # old start and end
                old_start_idx = int(event[DELAY_KEY] * self.rate)
                old_end_idx = old_start_idx + frames
                # change stimulus location
                new_stimulus[new_start_idx:new_end_idx] = stimulus[
                    old_start_idx:old_end_idx
                ]

                events.loc[idx, DELAY_KEY] = new_start_idx / self.rate

                offset_idx = new_end_idx

            self._events = events
            self._stimulus = new_stimulus
        else:
            self._events = super().events
            self._stimulus = super().stimulus

    @property
    def events(self):
        """events dataframe
        """
        return self._events

    @property
    def stimulus(self):
        """concatenate along time axis
        """
        return self._stimulus

    def to_dict(self):
        return {
            'stimuli': self.stimuli,
            'events': self.events,
            'stimulus': self.stimulus
        }

    @classmethod
    def from_dict(cls, data):
        self = cls(data['stimuli'], shuffle=False)
        self._events = data['events']
        self._stimulus = data['stimulus']
        return self
