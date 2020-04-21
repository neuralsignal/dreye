"""
"""

import sys
import numpy as np

from dreye.core.measurement_utils import create_measured_spectrum
from dreye.hardware.base_spectrometer import AbstractSpectrometer
from dreye.hardware.base_system import AbstractSystem


class MeasurementRunner:
    """
    MeasurementRunner is a class to which you must pass the spectrophotometer
    and system instance to before running measurements.

    Can be used with default measurement values if parameters are not
    specified.

    Parameters
    ----------
    system : dreye.hardware.AbstractSystem
        Defines the visual stimulation system.
    spec : dreye.hardware.AbstractSpectrometer
        Defines the spectrophotometer.
    wls : array-like, optional
        A numpy array with specific wavelength values. Alternatively, set
        wls=None, and wavelengths measured will be set to whatever is default
        for the spectrophotometer used.
    smoothing_window : float, optional
        Savgol filter window to smooth the spectrum of each averaged
        intensity value of each LED. polyorder argument for the
        scipy.signal.savgol_filter is set to 2.
    n_steps: int, optional
        Number of steps from 0 boundary to max boundary per LED, inclusive.
    n_avg : int, optional
        Number of times each step is averaged over.
    remove_zero: bool, optional
        substracts the zero boundary from all the other measurements. If set to
        False, zero boundary will not be subtracted.
     step_kwargs :
     sleep :

    """

    def __init__(
        self, system, spectrometer,
        n_steps=10, step_kwargs={},
        n_avg=10, sleep=None,
        wls=None, remove_zero=True,
        smoothing_window=None
    ):
        assert isinstance(system, AbstractSystem)
        assert isinstance(spectrometer, AbstractSpectrometer)
        system.check_channels()
        # TODO instance checking
        self.system = system
        self.spectrometer = spectrometer
        self.n_steps = n_steps
        self.step_kwargs = step_kwargs
        self.n_avg = n_avg
        self.sleep = sleep
        self.wls = wls
        self.remove_zero = remove_zero
        self.smoothing_window = smoothing_window

    def run(self, verbose=0):
        if verbose:
            sys.stdout.write(
                '\n---------------------STARTING MEASUREMENTS--------'
                '---------------\n'
            )
        # reset spms
        self.system._spms = None
        # iterate over system devices
        for n, output in enumerate(self.system):
            if verbose:
                sys.stdout.write(
                    f'\n---------------------------------------'
                    '--------------------------'
                    f'\nStarting measurement for {output.name}.\n'
                )
            output.open()
            # setup values to send to output device
            values = output.steps(self.n_steps, **self.step_kwargs)
            spectrum_array = np.zeros((self.spectrometer.size, len(values)))
            its = np.zeros(len(values))
            # zero output device
            output._zero()
            # verbosity
            if verbose:
                sys.stdout.write(
                    f'sending {len(values)} values\n'
                    '========================================='
                    '========================\n'
                )
            # substitute with tqdm?
            for idx, value in enumerate(values):
                output.send_value(value)
                spectrum_array[:, idx], its[idx] = (
                    self.spectrometer.perform_measurement(
                        self.n_avg, self.sleep, return_spectrum=False,
                        return_it=True, verbose=verbose
                    )
                )
                if verbose > 1:
                    photons_per_sec = np.sum(spectrum_array[:, idx])/its[idx]
                    sys.stdout.write(
                        f'step {idx}: {value*output.units} '
                        f'== {photons_per_sec} photons/second\n')
                elif verbose:
                    sys.stdout.write('.')
            # remove background zero from the rest
            if self.remove_zero:
                spectrum_array -= spectrum_array[:, :1]
            # flip if reversed
            if output.zero_boundary > output.max_boundary:
                spectrum_array = spectrum_array[:, ::-1]
                values = values[::-1]
                its = its[::-1]
            if verbose == 1:
                sys.stdout.write('\n')
            if verbose:
                sys.stdout.write(
                    f'==================================='
                    '=============================='
                    f'\nFinished measurement for "{output.name}".'
                    '\n------------------------------------'
                    '-----------------------------\n'
                )
            # zero output device
            output._zero()
            # convert everything correctly
            mspectrum = create_measured_spectrum(
                spectrum_array=spectrum_array,
                inputs=values,
                wavelengths=self.spectrometer.wavelengths,
                calibration=self.spectrometer.cal,
                integration_time=its,
                axis=0,
                smoothing_window=self.smoothing_window,
                input_units=output.units,
                zero_boundary=output.zero_boundary,
                max_boundary=output.max_boundary,
                zero_is_lower=output.zero_boundary < output.max_boundary,
                name=output.name
            )
            if self.wls is not None:
                mspectrum = mspectrum(self.wls)
            if self.smoothing_window is not None:
                mspectrum = mspectrum.smooth

            output.mspectrum = mspectrum
            output.close()

            if verbose:
                sys.stdout.write(
                    f'\nFinished conversion of '
                    f'measurement for "{output.name}".\n'
                )

        if verbose:
            sys.stdout.write(
                '\n---------------------MEASUREMENTS FINISHED--------'
                '---------------\n'
            )

        return self

    def save(self, filename):
        self.system.save(filename)
