"""Base class for stimuli
"""

from abc import ABC, abstractmethod
import inspect
import importlib
import warnings

import numpy as np
import pandas as pd

from dreye.io import read_json, write_json
from dreye.utilities import is_numeric


DUR_KEY = 'dur'
SYNCED_DUR_KEY = 'synced_dur'
DELAY_KEY = 'delay'
SYNCED_DELAY_KEY = 'synced_delay'


def check_events(df):
    """check that event dataframe contains columns DELAY_KEY and DUR_KEY
    """

    if len(df) == 0:

        cols = list(df.columns)
        if DELAY_KEY in cols and DUR_KEY in cols:
            pass
        else:
            if DELAY_KEY not in cols:
                cols.append(DELAY_KEY)
            if DUR_KEY not in cols:
                cols.append(DUR_KEY)
            df = pd.DataFrame(columns=cols)

    else:
        assert not ({DELAY_KEY, DUR_KEY} - set(df.columns)), \
            'event frame must contain column start and dur.'

        # stringify columns - and copy dataframe
        cols = {col: str(col) for col in df.columns}
        df = df.rename(columns=cols)

    return df


class BaseStimulus(ABC):

    time_axis = None

    def __init__(self, rate=None, seed=None, **kwargs):
        """base initialization
        """

        self.seed = seed
        self.rate = rate
        if rate is not None:
            assert is_numeric(rate), 'rate must be numeric'
        if seed is not None:
            assert is_numeric(seed), 'seed must be numeric'

        for key, ele in kwargs.items():
            setattr(self, key, ele)

        self._stimulus = None
        self._signal = None
        self._metadata = {}
        self._events = pd.DataFrame()

        # if settings already exists simply update dictionary
        settings = {
            'rate': self.rate,
            'seed': self.seed,
            **kwargs
        }
        if hasattr(self, '_settings'):
            self._settings.update(
                settings
            )
        else:
            self._settings = settings

    @abstractmethod
    def create(self):
        """create signal and metadata
        """

    @abstractmethod
    def transform(self):
        """transform signal to stimulus
        """

    # --- short hand for metadata dictionary --- #

    def __getitem__(self, index):
        """
        """

        return self.metadata[index]

    def __setitem__(self, index, value):
        """
        """

        self.metadata[index] = value

    # --- properties of stimulus --- #

    @property
    def name(self):
        """short-hand for stimulus class name
        """

        return self.__class__.__name__

    @property
    def module(self):
        """
        """

        # TODO module checking
        name = inspect.getmodule(self).__name__
        return name

    @property
    def metadata(self):
        """a dictionary of the metadata
        """

        if self._signal is None:
            self.create()

        return self._metadata

    @property
    def events(self):
        """a pandas dataframe of events in the stimulus (custom-formatted).
        Each row is a single event.
        """

        if self._signal is None:
            self.create()

        return check_events(self._events)

    def time2frame(self, key=DELAY_KEY):
        """get idcs for delay period
        """

        return (self.events[key] * self.rate).round(0).astype(int)

    @property
    def signal(self):
        """unprocessed array of stimulus (not for sending to hardware)
        """

        if self._signal is None:
            self.create()

        return self._signal

    @property
    def stimulus(self):
        """processed numpy array of stimulus (for sending to hardware)
        """

        if self._stimulus is None:
            if self._signal is None:
                self.create()
            self.transform()

        return self._stimulus

    @property
    def timestamps(self):
        """timestamps of stimulus same size of first axis
        """

        if self.rate is None:
            return None
        else:
            length = self.signal.shape[self.time_axis]
            return np.arange(0, length) / self.rate

    @property
    def duration(self):
        """duration of stimulus
        """

        if self.rate is None:
            return None
        else:
            length = self.signal.shape[self.time_axis]
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
            'metadata': self.metadata,
            'events': self.events.to_dict(),
            'settings': self.settings,
            'name': self.name,
            'module': self.module
        }

    @classmethod
    def from_dict(cls, data):
        """
        """

        # check if cls is the same as saved class otherwise
        # use correct class.
        if not cls.__name__ == data['name']:
            try:
                module = importlib.import_module(data['module'])
                try:
                    cls = getattr(module, data['name'])
                except AttributeError:
                    warnings.warn(f"Could not load class {data['name']}; "
                                  f"using '{cls.__name__}' as class instead.")
            except ModuleNotFoundError:
                warnings.warn(f"Could not import module '{data['module']}' "
                              f"for '{data['name']}'.")

        self = cls(**data['settings'])
        self._stimulus = data['stimulus']
        self._signal = data['signal']
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

        write_json(filename, self.to_dict())

    @classmethod
    def load(cls, filename):
        """function to load stimulus
        """

        data = read_json(filename)

        return cls.from_dict(data)


# TODO chained stimuli
# init check if correct instance
# shuffling of order option
# have alist of all attributes
# but also create complete events frame
# add queue parameter to events frame!!! and to each metadata dict
# and create complete stimulus (concatenate along time_axis)
# signal array can vary!!!
# no need to create complete
# Chain chained stimuli
