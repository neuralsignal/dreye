"""
"""

import sys
import numpy as np

from dreye.core.spectral_measurement import MeasuredSpectrum
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
    spectrometer : dreye.hardware.AbstractSpectrometer
        Defines the spectrophotometer.
    wls : array-like, optional
        A numpy array with specific wavelength values. Alternatively, set
        wls=None, and wavelengths measured will be set to whatever is default
        for the spectrophotometer used.
    smoothing_window : float, optional
        Savgol filter window to smooth the spectrum of each averaged
        intensity value of each LED. `polyorder` argument for the
        `scipy.signal.savgol_filter` is set to 2.
    n_steps: int, optional
        Number of steps from 0 boundary to max intensity boundary per LED,
        inclusive.
    n_avg : int, optional
        Number of times each step is averaged over.
    remove_zero: bool, optional
        Substracts the background spectrum using the same optimal integration
        time from each spectrum.
    sleep : numeric, optional
        Seconds sleep between measurements.
    """

    def __init__(
        self, system, spectrometer,
        n_steps=10, step_kwargs={},
        n_avg=10, sleep=None,
        wls=None, remove_zero=False,
        smoothing_window=None
    ):
        assert isinstance(system, AbstractSystem)
        assert isinstance(spectrometer, AbstractSpectrometer)
        system.check_channels()
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
        """
        Run a measurement.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level.
        """
        if verbose:
            sys.stdout.write(
                '\n---------------------STARTING MEASUREMENTS--------'
                '---------------\n'
            )
        # reset spms
        self.system._measured_spectra = None
        # set outputs to zero
        for output in self.system:
            output.open()
            output._zero()
            output.close()
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
                # zero output device
                output._zero()
                # remove zero background after finding integration time
                if self.remove_zero:
                    bg_array_, bg_it_ = self.spectrometer.perform_measurement(
                        self.n_avg, self.sleep,
                        return_spectrum=False,
                        return_it=True,
                        optimize_it=False,
                        verbose=verbose
                    )
                    spectrum_array[:, idx] -= bg_array_
                    # assert integration time for safety
                    assert bg_it_ == its[idx], (
                        "Integration times don't match - this is a bug!"
                    )
                if verbose > 1:
                    photons_per_sec = np.sum(spectrum_array[:, idx])/its[idx]
                    sys.stdout.write(
                        f'step {idx}: {value*output.units} '
                        f'== {photons_per_sec} photons/second\n')
                elif verbose:
                    sys.stdout.write('.')
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
                output=values,
                wavelengths=self.spectrometer.wavelengths,
                calibration=self.spectrometer.cal,
                integration_time=its,
                smoothing_window=self.smoothing_window,
                output_units=output.units,
                zero_intensity_bound=output.zero_intensity_bound,
                max_intensity_bound=output.max_intensity_bound,
                name=output.name,
            )
            if self.wls is not None:
                mspectrum = mspectrum(self.wls)
            if self.smoothing_window is not None:
                mspectrum = mspectrum.smooth()

            output.measured_spectrum = mspectrum
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
        """
        Save `dreye.hardware.AbstractSystem` object.
        """
        self.system.save(filename)

    def __str__(self):
        return f"[{type(self).__name__}]\n\n{str(self.system)}"
