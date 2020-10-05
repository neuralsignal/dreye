"""
Sine waves
Chirps
Sawtooth
Square
Sweep poly
Gauss pulse
"""

from itertools import product
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import chirp, gausspulse, sawtooth, square, sweep_poly

from dreye.stimuli.base import BaseStimulus, DUR_KEY, DELAY_KEY


def sine(t, freq, phi=0):
    """normal sine-wave generator

    Parameters
    ----------
    t : ndarray
        Times at which to evaluate the waveform
    freq : float
        Frequency in Hz (full revolution)
    phi : float
        Phase offset, in degrees
    """


# iterations
class AbstractWaveformStimulus(BaseStimulus):
    """
    """

    def __init__(
        self,
        rate,
        func_name,
        func_kwargs,
        n_channels,
        amps,
        offsets,
        seed=None,
    ):
        pass
