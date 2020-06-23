"""
"""

import numpy as np

from dreye.stimuli.base import BaseStimulus, DUR_KEY, DELAY_KEY
from dreye.stimuli.mixin import SetStepMixin, SetBaselineMixin
from dreye.utilities import asarray
from dreye.utilities.abstract import inherit_docstrings


@inherit_docstrings
class StimSet(BaseStimulus, SetBaselineMixin, SetStepMixin):
    """
    Parameters
    ----------
    estimator : scikit-learn type estimator
        Estimator that implements the `fit_transform` method.
    values : float or array-like or dict of arrays or dataframe
        Step values used.
    separate_channels : bool
        Whether to separate each channel for single steps.
        Works only if values are given as a dict. Each key represents a channel
    baseline_values : float or array-like or dict
        Baseline values when no stimulus is being presented. Defaults to 0.
    """

    def __init__(
        self,
        estimator=None,
        values=1,
        separate_channels=False,
        baseline_values=0,
    ):

        # call init method of BaseStimulus class
        super().__init__(
            estimator=estimator,
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
        self._events, self._metadata = self._create_events()
        self._signal = self._create_signal(self._events)

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
