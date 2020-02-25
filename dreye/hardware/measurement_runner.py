"""
"""

import sys
import numpy as np

from dreye.core.measurement_utils import create_measured_spectrum
from dreye.hardware.base_spectrometer import AbstractSpectrometer
from dreye.hardware.base_system import AbstractSystem


class MeasurementRunner:

    def __init__(
        self, system, spectrometer,
        n_steps=10, step_kwargs={},
        n_avg=10, sleep=None,
        units='uE',
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
        self.units = units

    def run(self, verbose=0):
        if verbose:
            sys.stdout.wrtie(
                '\n---------STARTING MEASUREMENTS----------\n'
            )
        # iterate over system devices
        for n, output in enumerate(self.system):
            if verbose:
                sys.stdout.write(
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
                    '==============================================\n'
                )
            # substitute with tqdm?
            for idx, value in enumerate(values):
                if verbose > 1:
                    sys.stdout.write(f'number {idx}: value {value}\n')
                elif verbose:
                    sys.stdout.write('.')
                output.send_value(value)
                spectrum_array[:, idx], its[idx] = (
                    self.spectrometer.perform_measurement(
                        self.n_avg, self.sleep, return_spectrum=False,
                        return_it=True, verbose=verbose
                    )
                )
            if verbose == 1:
                sys.stdout.write('\n')
            if verbose:
                sys.stdout.write(
                    f'=============================================='
                    f'\nFinished measurement for "{output.name}".'
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
                input_units=output.units,
            )

            output.mspectrum = mspectrum
            output.close()

            if verbose:
                sys.stdout.write(
                    f'\nFinished conversion of '
                    f'measurement for "{output.name}".\n'
                )

        if verbose:
            sys.stdout.write(
                '\n---------MEASUREMENTS FINISHED----------\n'
            )

        return self

    def save(self, filename):
        self.system.save(filename)
