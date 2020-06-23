"""Step stimulus
"""

from itertools import product
import pandas as pd
import numpy as np
from scipy import stats

from dreye.stimuli.base import BaseStimulus, DUR_KEY, DELAY_KEY, PAUSE_KEY
from dreye.stimuli.mixin import (
    SetBaselineMixin, SetStepMixin, SetRandomStepMixin
)
from dreye.utilities import is_numeric, convert_truncnorm_clip, asarray
from dreye.utilities.abstract import inherit_docstrings


@inherit_docstrings
class AbstractStepStimulus(BaseStimulus, SetBaselineMixin):
    """Abstract base class that has various helper functions
    """

    # --- standard create and transform methods --- #

    def create(self):

        self._events, self._metadata = self._create_events()
        self._signal = self._create_signal(self._events)


@inherit_docstrings
class StepStimulus(AbstractStepStimulus, SetStepMixin):
    """
    Step stimulus.

    Parameters
    ----------
    estimator : scikit-learn type estimator
        Estimator that implements the `fit_transform` method.
    rate : float
        The desired refresh rate.
    values : float or array-like or dict of arrays or dataframe
        Step values used.
    durations : float or array-like
        Different durations for the stimulus
    pause_durations : float or array-like
        Different durations after the stimulus; before the next stimulus
    repetitions : int
        How often to repeat each value. When randomize True, the value
        will not be repeated in a row.
    iterations : int
        How often to iterate over the whole stimulus. Does not rerandomize
        order for each iteration.
    randomize : bool
        randomize order of steps
    start_delay : float
        Duration before step stimulus starts (inverse units of rate)
    end_dur : float
        Duration after step stimulus ended (inverse units of rate)
    seed : int
        seed for randomization.
    aligned_durations : bool
        If True will check if durations and pause_durations are the same
        length and will iterate by zipping the lists for durations.
    separate_channels : bool
        Whether to separate each channel for single steps.
        Works only if values are given as a dict. Each key represents a channel
    baseline_values : float or array-like or dict
        Baseline values when no stimulus is being presented. Defaults to 0.
    func : callable
        Funtion applied to each step: f(t, values) = output <- (t x values).
        Defaults to None.
    """

    def __init__(
        self,
        *,
        estimator=None,
        values=1,
        durations=1,
        pause_durations=0,
        repetitions=1,
        iterations=1,
        randomize=False,
        rate=1,
        start_delay=0,
        end_dur=0,
        seed=None,
        aligned_durations=False,
        separate_channels=False,
        baseline_values=0,
        func=None
    ):

        # call init method of BaseStimulus class
        super().__init__(
            estimator=estimator,
            rate=rate,
            values=values,
            durations=durations,
            pause_durations=pause_durations,
            repetitions=repetitions,
            iterations=iterations,
            randomize=randomize,
            start_delay=start_delay,
            end_dur=end_dur,
            seed=seed,
            aligned_durations=aligned_durations,
            separate_channels=separate_channels,
            baseline_values=baseline_values,
            func=func
        )

        # sets values and baseline values attribute correctly
        self.values, self.baseline_values = self._set_values(
            values=values, baseline_values=baseline_values,
            separate_channels=separate_channels
        )

        # reset duration attributes correctly
        self.dur_iterable = self._set_durs(
            durations=durations, pause_durations=pause_durations,
            aligned_durations=aligned_durations
        )

    @property
    def ch_names(self):
        return list(self.values.columns)

    # --- methods for create method --- #

    def _create_events(self):
        """create event dataframe
        """

        # intialize event dataframe
        events = pd.DataFrame()

        for index, row in self.values.iterrows():
            for dur, pause in self.dur_iterable:
                row[DUR_KEY] = dur
                row[PAUSE_KEY] = pause
                df = pd.DataFrame([row] * self.repetitions)
                df['repeat'] = asarray(df.index)
                events = events.append(df, ignore_index=True, sort=False)

        if self.randomize:
            events = events.sample(
                frac=1, random_state=self.seed
            ).reset_index(drop=True)

        events['iter'] = 0
        # add copies of events dataframe to events dataframe (iterations)
        for n in range(self.iterations-1):
            _events = events.copy()
            # keep track of iteration number in copied dataframe
            _events['iter'] = n + 1
            events = events.append(_events, ignore_index=True, sort=False)

        delays = np.concatenate([
            [0],
            np.cumsum(asarray(events[[DUR_KEY, PAUSE_KEY]]).sum(1))[:-1]
        ])
        delays += self.start_delay

        events[DELAY_KEY] = delays
        events['name'] = self.name

        return events, {}

    def _create_signal(self, events):
        """create signal attribute
        """

        last_event = events.iloc[-1]
        total_dur = (
            last_event[DELAY_KEY]
            + last_event[DUR_KEY]
            + last_event[PAUSE_KEY]
            + self.end_dur
        )
        total_frames = int(np.ceil(total_dur * self.rate))
        times = np.arange(total_frames) / self.rate

        # intitialize signal array
        signal = np.ones((total_frames, len(self.values.columns)))
        signal *= self.baseline_values[None, :]

        # insert steps
        for index, row in events.iterrows():
            tbool = (
                (times >= row[DELAY_KEY])
                & (times < (row[DELAY_KEY] + row[DUR_KEY]))
            )
            # update events with true delay of signal due to frame rate
            true_delay = times[tbool][0]
            events.loc[index, DELAY_KEY] = true_delay
            # change signal to step change
            if self.func is None:
                signal[tbool] = asarray(row[self.values.columns])[None, :]
            else:
                signal[tbool] = self.func(
                    np.arange(tbool.size) / self.rate,
                    asarray(row[self.values.columns])
                )

        return signal

    # --- methods for setting attributes correctly (used in init) --- #

    def _set_durs(self, durations, pause_durations, aligned_durations):

        # convert durations into list object for iterations
        if is_numeric(durations):
            durations = [durations]
        if is_numeric(pause_durations):
            pause_durations = [pause_durations]

        if aligned_durations:
            assert len(durations) == len(pause_durations)
            dur_iterable = list(
                zip(durations, pause_durations)
            )
        else:
            dur_iterable = list(
                product(durations, pause_durations)
            )

        return dur_iterable


@inherit_docstrings
class NoiseStepStimulus(StepStimulus):
    """
    Step stimulus by choosing values from a truncated Gaussian.

    Parameters
    ----------
    estimator : scikit-learn type estimator
        Estimator that implements the `fit_transform` method.
    rate : float
        The desired refresh rate.
    n_samples : int
        Number of samples.
    n_channels : int
        Number of channels.
    channel_names : list-like
        Name of each channel.
    mean : float or array-like
        mean values for each channel.
    var : float or array-like
        variance for each channel.
    minimum : float
        minimum value
    maximum : float
        maximum value
    durations : float or array-like
        Different durations for the stimulus
    pause_durations : float or array-like
        Different durations after the stimulus; before the next stimulus
    repetitions : int
        How often to repeat each value. When randomize True, the value
        will not be repeated in a row.
    iterations : int
        How often to iterate over the whole stimulus. Does not rerandomize
        order for each iteration.
    randomize : bool
        randomize order of steps
    start_delay : float
        Duration before step stimulus starts (inverse units of rate)
    end_dur : float
        Duration after step stimulus ended (inverse units of rate)
    seed : int
        seed for randomization.
    aligned_durations : bool
        If True will check if durations and pause_durations are the same
        length and will iterate by zipping the lists for durations.
    separate_channels : bool
        Whether to separate each channel for single steps.
        Works only if values are given as a dict. Each key represents a channel
    baseline_values : float or array-like or dict
        Baseline values when no stimulus is being presented. Defaults to 0.
    func : callable
        Funtion applied to each step: f(t, values) = output <- (t x values).
        Defaults to None.
    """

    def __init__(
        self,
        *,
        estimator=None,
        n_samples=1,
        n_channels=None,
        mean=0,
        var=1,
        minimum=None,
        maximum=None,
        channel_names=None,
        durations=1,
        pause_durations=0,
        repetitions=1,
        iterations=1,
        randomize=False,
        rate=1,
        start_delay=0,
        end_dur=0,
        seed=None,
        values_seed=None,
        aligned_durations=False,
        func=None
    ):

        super().__init__(
            estimator=estimator,
            n_channels=n_channels,
            n_samples=n_samples,
            mean=mean,
            var=var,
            minimum=minimum,
            maximum=maximum,
            channel_names=channel_names,
            durations=durations,
            pause_durations=pause_durations,
            repetitions=repetitions,
            iterations=iterations,
            randomize=randomize,
            rate=rate,
            start_delay=start_delay,
            end_dur=end_dur,
            seed=seed,
            values_seed=values_seed,
            aligned_durations=aligned_durations,
            separate_channels=False,
            func=func
        )

        # TODO copy from whitenoise (make convenience function?)

        self.mean = asarray(self.mean)
        self.var = asarray(self.var)

        if n_channels is None and channel_names is None:
            self.n_channels = max([self.mean.size, self.var.size])
        elif n_channels is None:
            self.n_channels = len(self.channel_names)

        if channel_names is None:
            self.channel_names = list(range(self.n_channels))
        else:
            self.channel_names = list(channel_names)
            assert len(self.channel_names) == self.n_channels

        if minimum is None:
            self.minimum = self.mean - 5 * self.var
        else:
            self.minimum = asarray(self.minimum)
        if maximum is None:
            self.maximum = self.mean + 5 * self.var
        else:
            self.maximum = asarray(self.maximum)

        if np.all(self.maximum == self.minimum):
            self.maximum += 10**-5
            self.minimum -= 10**-5

        a, b = convert_truncnorm_clip(
            self.minimum, self.maximum, self.mean, self.var)

        distribution = stats.truncnorm(
            a=a,
            b=b,
            loc=self.mean,
            scale=self.var
        )

        values = pd.DataFrame(
            distribution.rvs(
                size=(n_samples, self.n_channels),
                random_state=self.values_seed
            ),
            columns=self.channel_names
        )

        # sets values and baseline values attribute correctly
        self.values, self.baseline_values = self._set_values(
            values=values, baseline_values=self.mean,
            separate_channels=False
        )

        # reset duration attributes correctly
        self.dur_iterable = self._set_durs(
            durations=durations, pause_durations=pause_durations,
            aligned_durations=aligned_durations
        )


@inherit_docstrings
class RandomSwitchStimulus(AbstractStepStimulus, SetRandomStepMixin):
    """
    Random switch stimulus using truncated Gaussian.

    Parameters
    ----------
    estimator : scikit-learn type estimator
        Estimator that implements the `fit_transform` method.
    rate : float
        The desired refresh rate.
    values : float or array-like or dict of arrays or dataframe
        Step values used.
    values_probs : array-like or dict of arrays
        Probability of each value. Default is None.
    loc : float
        Mean duration.
    scale : float
        variation in duration.
    clip_dur : float
        Max duration before switching.
    total_dur : float
        total duration of stimulus. Actual duration will be around the total
        duration specified, as the algorithm stops when the total duration
        is exceeded.
    iterations : int
        How often to iterate over the whole stimulus. Does not rerandomize
        order for each iteration.
    start_delay : float
        Duration before step stimulus starts (inverse units of rate)
    end_dur : float
        Duration after step stimulus ended (inverse units of rate)
    seed : int
        seed for randomization.
    baseline_values : float or array-like or dict
        Baseline values when no stimulus is being presented. Defaults to 0.
    """

    def __init__(
        self,
        *,
        estimator=None,
        values=[0, 1],
        values_probs=None,
        loc=0,
        scale=1,
        clip_dur=None,
        total_dur=10,
        rate=1,
        iterations=1,
        start_delay=0,
        end_dur=0,
        seed=None,
        baseline_values=0,
        func=None
    ):

        # call init of BaseStimulus class
        super().__init__(
            estimator=estimator,
            values=values,
            values_probs=values_probs,
            rate=rate,
            iterations=iterations,
            start_delay=start_delay,
            end_dur=end_dur,
            seed=seed,
            baseline_values=baseline_values,
            clip_dur=clip_dur,
            total_dur=total_dur,
            loc=loc,
            scale=scale,
            func=func
        )

        if self.clip_dur is None:
            self.clip_dur = self.total_dur

        # sets values and baseline values attribute correctly
        self.values, self.baseline_values, self.values_probs = \
            self._set_values(
                values=values, baseline_values=baseline_values,
                values_probs=values_probs
            )

    # --- methods for create method --- #

    @property
    def ch_names(self):
        return list(self.values.keys())

    def _create_events(self):
        """create events dataframe
        """

        a, b = convert_truncnorm_clip(0, self.clip_dur, self.loc, self.scale)

        # get distribution class from scipy.stats and initialize
        distribution = stats.truncnorm(
            loc=self.loc, scale=self.scale,
            a=a, b=b
        )

        events = pd.DataFrame()
        metadata = {}

        # set seed
        np.random.seed(self.seed)

        # iterate over each channel
        # since py_version > 3.6, dictionary is ordered
        for key, ele in self.values.items():
            # init cumulated duration
            metadata[key] = []
            cum_dur = distribution.rvs()
            while cum_dur < self.total_dur:

                # draw random duration (must be positive)
                dur = distribution.rvs()

                # draw random sample from values (uniform distribution)
                value = np.random.choice(ele, p=self.values_probs[key])

                row = pd.Series(dict(
                    delay=cum_dur,
                    dur=dur,
                    channel=key,
                    value=value,
                ))
                # to metadata dict only add delay and value
                metadata[key].append((cum_dur, value))
                # append events
                events = events.append(
                    row, ignore_index=True, sort=False
                )

                # add to cumulated duration
                cum_dur += dur

        events[DELAY_KEY] += self.start_delay
        events[PAUSE_KEY] = 0
        events['end'] = events[DELAY_KEY] + events[DUR_KEY]
        # sort events by delay and remove end column
        events.sort_values('end', inplace=True)
        events.drop('end', axis=1, inplace=True)

        events['iter'] = 0
        # add copies of events dataframe to events dataframe (iterations)
        for n in range(self.iterations-1):
            # copy events dataframe
            _events = events.copy()

            # keep track of iteration number in copied dataframe
            _events['iter'] = n + 1

            # get last previous event and change delay and dur
            previous_last_event = events.iloc[-1]
            previous_total_dur = (
                self.end_dur
                + previous_last_event[DELAY_KEY]
                + previous_last_event[DUR_KEY]
            )
            _events[DELAY_KEY] += previous_total_dur

            # append to event dataframe
            events = events.append(_events, ignore_index=True, sort=False)

        # add stimulus name to each event
        events['name'] = self.name

        return events, metadata

    def _create_signal(self, events):
        """create signal attribute
        """

        last_event = events.iloc[-1]
        total_dur = (
            last_event[DELAY_KEY]
            + last_event[DUR_KEY]
            + self.end_dur
        )
        total_frames = int(np.ceil(total_dur * self.rate))
        times = np.arange(total_frames) / self.rate

        # intitialize signal array
        signal = np.ones((total_frames, len(self.values)))
        signal *= self.baseline_values[None, :]

        # channel name to index mapping
        # Dictionary should be ordered since py_version > 3.6
        channel_mapping = {key: idx for idx, key in enumerate(self.values)}

        # insert steps
        for index, row in events.iterrows():
            # get interval boolean of event occurence
            tbool = (
                (times >= row[DELAY_KEY])
                & (times < (row[DELAY_KEY] + row[DUR_KEY]))
            )
            # update events with true delay of signal due to frame rate
            true_delay = times[tbool][0]
            events.loc[index, DELAY_KEY] = true_delay
            # change particular change to given value
            idx = channel_mapping[row['channel']]
            if self.func is None:
                signal[tbool, idx] = row['value']
            else:
                signal[tbool, idx] = self.func(
                    np.arange(tbool.size) / self.rate,
                    row['value']
                )

        return signal
