"""
"""

import sys
import numpy as np
import time
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from dreye.core.measurement_utils import create_measured_spectrum
from dreye.hardware.base_spectrometer import AbstractSpectrometer
from dreye.hardware.base_system import AbstractSystem


# TODO find the actual zero intensity bound


def _remove_spectrum_noise(
    wls, mean_counts, std_counts, n_avg,
    wls1=None, sigma=10, std_steps=4, axis=0
):
    """
    Remove noise from mean count array.
    """
    mean_counts = mean_counts.copy()
    # lower bound estimate
    lower = mean_counts - std_counts / np.sqrt(n_avg) * std_steps
    if wls1 is None:
        wl_diff = np.mean(np.diff(wls))
    else:
        lower = interp1d(wls, lower, axis=axis)(wls1)
        wl_diff = np.mean(np.diff(wls1))

    lower = gaussian_filter1d(lower, sigma/wl_diff, axis=axis)
    boolean = lower <= 0
    if wls1 is not None:
        boolean = interp1d(
            wls1, boolean.astype(float),
            axis=axis, bounds_error=False,
            fill_value=1
        )(wls).astype(int).astype(bool)

    mean_counts[boolean] = 0.0
    return mean_counts


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
    save_raw : bool, optional
        Whether to save raw photon count values in the output instance.
    sleep : numeric, optional
        Seconds sleep between measurements.
    zero_sleep : numeric, optional
        Seconds sleep between measurement and background measurement
    """

    def __init__(
        self, system, spectrometer,
        n_steps=10, step_kwargs={},
        n_avg=10, sleep=None,
        wls=None,
        remove_zero=False,
        smart_zero=None,
        smoothing_window=None,
        save_raw=True,
        zero_sleep=3
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
        self.smart_zero = smart_zero
        self.smoothing_window = smoothing_window
        self.save_raw = save_raw
        self.zero_sleep = zero_sleep

    def run(self, verbose=0, return_raw=False):
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
        # raw data
        raw_data = {}
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
            spectrum_sd = np.zeros(spectrum_array.shape)
            bg_array = np.zeros(spectrum_array.shape)
            bg_sd = np.zeros(spectrum_array.shape)
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
                if self.zero_sleep is not None:
                    time.sleep(self.zero_sleep)
                spectrum_array[:, idx], spectrum_sd[:, idx], its[idx] = (
                    self.spectrometer.perform_measurement(
                        self.n_avg, self.sleep, return_spectrum=False,
                        return_sd=True,
                        return_it=True, verbose=verbose
                    )
                )
                # zero output device
                output._zero()
                if self.zero_sleep is not None:
                    time.sleep(self.zero_sleep)
                # remove zero background after finding integration time
                bg_array[:, idx], bg_sd[:, idx], bg_it_ = \
                    self.spectrometer.perform_measurement(
                        self.n_avg, self.sleep,
                        return_spectrum=False,
                        return_it=True,
                        return_sd=True,
                        optimize_it=False,
                        verbose=verbose
                    )
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

            # define raw dictionary
            raw_data_ = {
                'volts': values,
                'wls': self.spectrometer.wavelengths,
                'spectra': spectrum_array,
                'spectra_sd': spectrum_sd,
                'bg_spectra': bg_array,
                'bg_spectra_sd': bg_sd,
                'integration_times': its
            }
            raw_data[output.name] = raw_data_

            if self.remove_zero:
                spectrum_array = spectrum_array - bg_array
                raw_data_['modified_spectra'] = spectrum_array
            if self.smart_zero is not None:
                spectrum_array = _remove_spectrum_noise(
                    self.spectrometer.wavelengths,
                    spectrum_array,
                    spectrum_sd,
                    self.n_avg,
                    wls1=self.wls,
                    **self.smart_zero
                )
                raw_data_['modified_spectra'] = spectrum_array
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
                mspectrum = mspectrum.smooth(
                    smoothing_window=self.smoothing_window
                )

            # raw_data
            if self.save_raw:
                output._raw_data = raw_data_

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

        if return_raw:
            return raw_data

        return self

    def save(self, filename):
        """
        Save `dreye.hardware.AbstractSystem` object.
        """
        self.system.save(filename)

    def __str__(self):
        return f"[{type(self).__name__}]\n\n{str(self.system)}"
