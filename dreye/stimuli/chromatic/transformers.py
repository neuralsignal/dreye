"""Basic Stimulus class with Transformer added
"""

import numpy as np
import pandas as pd

from dreye.err import DreyeError
from dreye.stimuli.base import BaseStimulus, DUR_KEY, DELAY_KEY
from dreye.core.photoreceptor import AbstractPhotoreceptor
from dreye.core.spectral_measurement import MeasuredSpectraContainer
from dreye.core.spectrum import AbstractSpectrum
from dreye.constants import ureg
from dreye.utilities import has_units, asarray, dissect_units

# TODO plotting for goodness-of-fits
# TODO nonlinear/functional transformation (requires providing inverse)


class SignalTransformerMixin:

    alter_events = False
    alter_events_raise_error = True
    fit_only_uniques = False
    add_to_events_mean = True  # when using add_to_events calculate the mean
    _plot_attrs = ['stimulus', 'signal']

    def __init__(
        self,
        *,
        measured_spectra,
        **kwargs
    ):
        """A Mixin class for transforming LED intensity values
        """

        assert isinstance(measured_spectra, MeasuredSpectraContainer), (
            "Argument 'measured_spectra' must be instance of "
            "MeasuredSpectraContainer, but instance of"
            f" '{type(measured_spectra)}'."
        )

        assert isinstance(self.time_axis, int)
        assert isinstance(self.channel_axis, int)

        # initialize mixin properties for settings file
        BaseStimulus.__init__(
            self,
            measured_spectra=measured_spectra,
        )

        super().__init__(**kwargs)

    @classmethod
    def forward_shape(cls, data):
        """reshapes data for time and channel axis to be in 0th and 1st axis.
        """

        if cls.channel_axis == 0:
            data = np.swapaxes(
                data, cls.time_axis, cls.channel_axis
            )
            data = np.moveaxis(
                data, cls.channel_axis, 1
            )
        else:
            if cls.time_axis != 0:
                data = np.moveaxis(
                    data, cls.time_axis, 0
                )
            if cls.channel_axis != 1:
                data = np.moveaxis(
                    data, cls.channel_axis, 1
                )

        return data

    @classmethod
    def backward_shape(cls, data):
        """reshape data for time and channel axis to be in original position.
        """

        if cls.channel_axis == 0:
            data = np.swapaxes(
                data, 0, 1
            )
            data = np.moveaxis(
                data, 1, cls.time_axis
            )
        else:
            if cls.channel_axis != 1:
                data = np.moveaxis(
                    data, 1, cls.channel_axis
                )
            if cls.time_axis != 0:
                data = np.moveaxis(
                    data, 0, cls.time_axis
                )

        return data

    @staticmethod
    def iterator_shape(data):
        # this will treat time_axis and channel_axis together when fitting
        # but assume the other dimensions are independent
        data_shape = data.shape
        data = data.reshape(
            data_shape[0], data_shape[1], int(np.prod(data_shape[2:]))
        )
        iterator = range(data.shape[-1])

        return iterator, data, data_shape

    def _add_to_events(self, labels, signal, prefix):
        """method used by _add_pr_to_events and _add_transform_to_events
        """
        if not self.alter_events:
            return

        # has to be an array
        signal = asarray(signal)

        if labels is None:
            labels = [
                f'{prefix}{idx}'
                for idx in range(signal.shape[self.channel_axis])
            ]
        else:
            labels = [
                str(label) for label in labels
            ]
        # add opsin channels to events
        # warn if channel names already exist
        truth = (
            set(labels)
            & set(self.events.columns)
        )
        if truth:
            labels = [
                f'{prefix}{channel}'
                for channel in labels
            ]
            truth = (
                set(labels)
                & set(self.events.columns)
            )
        if truth and self.alter_events_raise_error:
            raise DreyeError('Events Dataframe already contains columns'
                             f' for labels {labels}.')
        # assert sizes match
        assert len(labels) == signal.shape[self.channel_axis]
        # if signal high-dimensional then events columns need to be object type
        events = {}
        for channel in labels:
            events[channel] = []

        for idx, row in self.events.iterrows():
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
                if self.add_to_events_mean:
                    value = value.mean(
                        axis=(self.time_axis, self.channel_axis)
                    )
                else:
                    value = value.mean(axis=self.channel_axis)
                # assign entry
                events[channel].append(value)
                # self.events.loc[idx, channel] = value

        events = pd.DataFrame(events, index=self.events.index)
        # reassign events
        self._events = pd.concat([self.events, events], axis=1)

    @property
    def stimulus_signal(self):
        return self.signal

    def transform(self):

        # reshape appropriately and iterate to map values
        # keep track of units, if exist
        signal, units = dissect_units(self.stimulus_signal)
        signal = self.forward_shape(signal)
        iterator, signal, signal_shape = self.iterator_shape(signal)
        stimulus = np.zeros(signal.shape)

        for idx in iterator:
            if units is None:
                isignal = signal[..., idx]
            else:
                isignal = signal[..., idx] * units
            stimulus[..., idx] = asarray(self.measured_spectra.map(isignal))

        # reshape back into correct form and assign to stimulus
        # stimulus does not keep units
        # - should have label_units from measured_spectra
        self._stimulus = self.backward_shape(stimulus.reshape(signal_shape))
        self._add_to_events(
            self.measured_spectra.names, self.stimulus,
            prefix='stimout_'
        )


class CaptureTransformerMixin(SignalTransformerMixin):

    _plot_attrs = ['stimulus', 'stimulus_signal', 'signal', 'res_signal']

    def __init__(
        self,
        *,
        photoreceptor,
        measured_spectra,
        background=None,
        fit_kwargs=None,
        **kwargs
    ):
        """A Mixin class for transforming capture data using
        an AbstractPhotoreceptor instance (use case: limited number of LEDs)
        """

        assert isinstance(photoreceptor, AbstractPhotoreceptor), (
            "Argument 'photoreceptor' must be (subclass) instance of an "
            f"AbstractPhotoreceptor, but instance of '{type(photoreceptor)}'."
        )
        assert (
            isinstance(background, AbstractSpectrum)
            or (background is None)
        ), (
            "Argument 'background' must be instance of "
            "Spectrum, but instance of "
            f"'{type(background)}'"
        )

        assert isinstance(self.time_axis, int)
        assert isinstance(self.channel_axis, int)

        # initialize mixin properties for settings file
        BaseStimulus.__init__(
            self, photoreceptor=photoreceptor,
            background=background,
            fit_kwargs=({} if fit_kwargs is None else fit_kwargs)
        )

        if isinstance(background, AbstractSpectrum) and background.ndim == 2:
            self.background = background.sum(axis=self.other_axis)

        super().__init__(measured_spectra=measured_spectra, **kwargs)

    def signal_preprocess(self, signal):
        """preprocess signal before fitting
        """

        return signal

    def _add_pr_to_events(self):
        """add opsin channels to events dataframe.
        This is run after fitting (do not run externally)
        """
        self._add_to_events(
            self.photoreceptor.sensitivity.labels, self.signal,
            prefix='pr_'
        )

    def create_postprocess(self, skip_preprocess=False):
        """post processing after creating signal, metadata
        """

        fit_kwargs = self.fit_kwargs.copy()
        units = self.fit_kwargs.pop('units', True)

        # get normalized channel x LED matrix
        A = self.photoreceptor.get_A(
            self.measured_spectra,
            self.background,
            units=units
        )

        # preprocess signal
        if skip_preprocess:
            processed_signal = self.signal
        else:
            self.metadata['original_signal'] = self.signal
            processed_signal = self.signal_preprocess(self.signal)
        # reshape signal and create iterator
        signal, signal_units = dissect_units(processed_signal)
        # reshape
        signal = self.forward_shape(signal)
        iterator, signal, signal_shape = self.iterator_shape(signal)
        # create empty array for weights
        weights = np.zeros(
            (signal.shape[0], A.shape[1],) + signal.shape[2:]
        )
        fitted_signal = np.zeros(signal.shape)

        for idx in iterator:
            if signal_units is None:
                isignal = signal[..., idx]
            else:
                isignal = signal[..., idx] * signal_units

            # TODO change to ChromaticStimEstimator
            x = self.photoreceptor.fit_qs(
                isignal, A,
                bounds=self.measured_spectra.bounds,
                units=units,
                return_res=False,
                inverse=True,
                only_uniques=self.fit_only_uniques,
                **fit_kwargs
            )
            # just need to do for the first loop
            if idx == 0:
                if has_units(x):
                    weights_units = x.units
                else:
                    weights_units = 1
            weights[..., idx] = asarray(x)
            # TODO better way than just transforming
            fitted_signal[..., idx] = asarray(self.photoreceptor.fitted_qs(
                asarray(x.T),
                asarray(A)
            ).T)

        # reshape correctly
        weights = weights.reshape(
            (signal.shape[0], A.shape[1], ) + signal_shape[2:]
        )
        # reshape correctly
        fitted_signal = fitted_signal.reshape(signal_shape)

        # dealing with axes
        fitted_signal = self.backward_shape(fitted_signal)
        weights = self.backward_shape(weights)

        self.metadata['normalized_weight_matrix'] = A
        self.metadata['channel_weights'] = weights * weights_units

        # deal with units
        if signal_units is not None:
            fitted_signal = fitted_signal * signal_units

        target_signal = processed_signal
        res = (target_signal - fitted_signal)
        # r2s (across time and channel or just channel?)

        self.metadata['target_signal'] = target_signal
        self.metadata['res_signal'] = res

        # reassign signal
        self._signal = fitted_signal
        self._add_pr_to_events()

    def create(self):
        """
        """

        super().create()
        self.create_postprocess()

    @property
    def stimulus_signal(self):
        return asarray(self.metadata['channel_weights'])

    def transform(self):
        self._add_to_events(
            self.measured_spectra.names,
            self.stimulus_signal,
            prefix='stimin_'
        )
        super().transform()


class LinearTransformCaptureTransformerMixin(CaptureTransformerMixin):

    _plot_attrs = [
        'stimulus', 'stimulus_signal',
        'lineartransform_signal', 'lineartransform_res_signal',
        'signal', 'res_signal',
    ]

    def __init__(
        self,
        *,
        linear_transform,
        inverse_transform=None,
        linear_transform_labels=None,
        **kwargs
    ):
        BaseStimulus.__init__(
            self, linear_transform=linear_transform,
            inverse_transform=inverse_transform,
            linear_transform_labels=linear_transform_labels
        )

        linear_transform, transform_units = dissect_units(linear_transform)

        assert isinstance(linear_transform, np.ndarray)
        # check dimensionality
        assert linear_transform.ndim == 2
        if inverse_transform is not None:
            if transform_units is not None and has_units(inverse_transform):
                inverse_transform = inverse_transform.to(1/transform_units)
            elif transform_units is None and has_units(inverse_transform):
                transform_units = 1 / inverse_transform.units

            inverse_transform = asarray(inverse_transform)
            assert isinstance(inverse_transform, np.ndarray)
            assert inverse_transform.T.shape == linear_transform.shape

        # ignore dimensionless transform_units
        # if transform_units == ureg(None).units:
        #     transform_units = None
        #
        # if transform_units is not None:
        #     raise NotImplementedError(
        #         'Linear transform with units that have dimensionality.'
        #     )

        self.linear_transform = linear_transform
        self.inverse_transform = inverse_transform
        self.transform_units = transform_units

        super().__init__(**kwargs)

    def _add_transform_to_events(self):
        """This is run after transforming (do not run externally)
        """
        self._add_to_events(
            self.linear_transform_labels, self.signal, prefix='transform_'
        )

    def create_postprocess(self):
        """process capture data with linear transform
        """

        self.metadata['original_signal'] = self.signal
        original_signal = self.signal_preprocess(self.signal)
        # apply linear transform to signal
        signal, signal_units = dissect_units(original_signal)
        signal = np.moveaxis(signal, self.channel_axis, -1)
        # may take a long time if a large array!!!
        transformed_signal = signal @ self.linear_transform.T
        self._signal = np.moveaxis(
            transformed_signal, -1, self.channel_axis
        )
        if self.transform_units is None and signal_units is not None:
            self._signal = self._signal * signal_units
        elif self.transform_units is not None and signal_units is None:
            self._signal = self._signal * self.transform_units
        elif self.transform_units is not None and signal_units is not None:
            self._signal = self._signal * signal_units * self.transform_units

        # perform normal postprocess
        super().create_postprocess(skip_preprocess=True)

        # fitted signal transform
        self.metadata['lineartransform_target_signal'] = \
            self.metadata['target_signal']
        self.metadata['lineartransform_signal'] = self.signal
        self.metadata['lineartransform_res_signal'] = \
            self.metadata['res_signal']
        self.metadata['target_signal'] = original_signal

        # transform linearly transformed signal
        # reshape signal and create iterator
        # remove units
        signal, _ = dissect_units(self.signal)
        signal = self.forward_shape(signal)
        iterator, signal, signal_shape = self.iterator_shape(signal)
        new_signal = np.zeros((
            signal.shape[0], self.linear_transform.shape[1], signal.shape[-1]
        ))
        res = np.zeros((signal.shape[0], signal.shape[2]))
        rank = np.zeros(signal.shape[2])

        for idx in iterator:
            # need to fit in the case of non-invertable matrices
            if self.inverse_transform is None:
                _x, _res, _rank, _ = np.linalg.lstsq(
                    self.linear_transform, signal[..., idx].T,
                    rcond=-1
                )
                new_signal[..., idx] = _x.T
                # low-rank - perfect fit
                if _res.shape == (0,) or _res.shape == ():
                    pass
                else:
                    res[..., idx] = _res
                rank[idx] = _rank
            else:
                new_signal[..., idx] = (
                    self.inverse_transform @ signal[..., idx].T
                ).T
                # TODO rank?

        # reshape
        self._signal = self.backward_shape(
            new_signal.reshape(
                (signal_shape[0], self.linear_transform.shape[1])
                + signal_shape[2:]
            )
        )
        # reapply units from original_signal
        if signal_units is not None:
            self._signal = self._signal * signal_units
        #
        self._add_transform_to_events()
        self.metadata['res_signal'] = (original_signal - self.signal)
        self.metadata['linear_backtransform_res'] = res
        self.metadata['linear_backtransform_rank'] = rank


class IlluminantCaptureTransformerMixin(
    LinearTransformCaptureTransformerMixin
):

    def __init__(
        self,
        *,
        illuminant,
        photoreceptor,
        measured_spectra,
        background=None,
        **kwargs
    ):

        assert isinstance(photoreceptor, AbstractPhotoreceptor), (
            "Argument 'photoreceptor' must be (subclass) instance of an "
            f"AbstractPhotoreceptor, but instance of '{type(photoreceptor)}'."
        )
        assert isinstance(measured_spectra, MeasuredSpectraContainer), (
            "Argument 'measured_spectra' must be instance of "
            "MeasuredSpectraContainer, but instance of"
            f" '{type(measured_spectra)}'."
        )
        assert (
            isinstance(background, AbstractSpectrum)
            # TODO implement or (background is None)
        ), (
            "Argument 'background' must be instance of "
            "Spectrum, but instance of "
            f"'{type(background)}'"
        )

        assert isinstance(illuminant, AbstractSpectrum), (
            "Argument 'illuminant' must be instance of "
            "Spectrum, but instance of "
            f"'{type(illuminant)}'"
        )

        # equalize domains
        if background is not None:
            illuminant, background = illuminant.equalize_domains(
                background, equalize_dimensions=False, copy=True
            )
        # this would result in multiplication after capture calculation
        # i.e. after excitefunc
        # Other Option is to implement signal_preprocess instead of
        # using linear transform
        linear_transform = photoreceptor.excitation(
            illuminant, background=background
        )

        BaseStimulus.__init__(
            self, illuminant=illuminant
        )

        super().__init__(
            linear_transform=linear_transform,
            linear_transform_labels=illuminant.labels,
            photoreceptor=photoreceptor,
            measured_spectra=measured_spectra,
            background=background,
            **kwargs
        )


class IlluminantBgCaptureTransformerMixin(CaptureTransformerMixin):

    _plot_attrs = [
        'stimulus', 'stimulus_signal',
        'original_signal',
        'signal', 'res_signal'
    ]

    def __init__(
        self,
        *,
        illuminant,
        photoreceptor,
        measured_spectra,
        background=None,
        bg_func=None,
        **kwargs
    ):

        assert (
            isinstance(background, AbstractSpectrum)
        ), (
            "Argument 'background' must be instance of "
            "Spectrum, but instance of "
            f"'{type(background)}'"
        )

        assert isinstance(illuminant, AbstractSpectrum), (
            "Argument 'illuminant' must be instance of "
            "Spectrum, but instance of "
            f"'{type(illuminant)}'"
        )

        # convert to same units - depends if normalized spectrum or not
        # equalize domains
        illuminant, background = illuminant.equalize_domains(
            background, equalize_dimensions=False, copy=True
        )
        # equalize units (this is inplace!)
        if illuminant.units != ureg(None).units:
            background.units = illuminant.units

        BaseStimulus.__init__(
            self, illuminant=illuminant,
            bg_func=bg_func
        )

        if bg_func is None:
            def bg_func(signal, background):
                return signal
            self.bg_func = bg_func

        super().__init__(
            photoreceptor=photoreceptor,
            measured_spectra=measured_spectra,
            background=background,
            **kwargs
        )

    def signal_preprocess(self, signal):
        """preprocess signal before fitting. Signal here is dimensionless.
        It simply specifies the multiplier of the illuminant.
        """
        # Init already equalized units and domain correctly!
        # reshape signal to multiply with illuminant
        # TODO dissect_units(signal)
        signal, units = dissect_units(signal)
        if not (units is None or units == ureg(None).units):
            raise DreyeError('signal must be dimensionless.')

        signal = np.moveaxis(signal, self.channel_axis, -1)
        # shape of everything else x channels
        signal_shape = signal.shape
        # everything else x channels x None
        signal = signal.reshape(
            (int(np.prod(signal.shape[:-1])), signal.shape[-1], 1)
        )
        # reshape illuminant to multiply with signal
        # None x channels x wavelength
        illuminant = self.illuminant.moveaxis(
            self.illuminant.domain_axis, -1
        ).magnitude[None, ...]
        # multiply signal with illuminant and sum over channels
        signal = (signal * illuminant).sum(axis=-2)
        # apply function (everything else x wavelength)
        # None x wavelength
        background = self.background.magnitude[None, :]
        spectra = self.bg_func(signal, background)
        spectra_units = self.bg_func(
            1 * self.illuminant.units, 1 * self.background.units
        ).units
        # clip spectrum at zero
        spectra = np.clip(spectra, 0, None)
        # reshape spectra and create AbstractSpectrum
        # wavelength x everything else
        spectra = np.moveaxis(spectra, -1, 0)
        spectra = AbstractSpectrum(
            spectra,
            domain_axis=0,
            domain=self.illuminant.domain,
            units=spectra_units
        )

        # new_signal (opsin x everything else)
        new_signal = self.photoreceptor.excitation(
            spectra, background=self.background
        )
        # reshape new signal
        # opsin x shape of everything else
        new_signal = new_signal.reshape(
            (new_signal.shape[0], ) + signal_shape[:-1]
        )
        # move opsin channels to channel_axis
        new_signal = np.moveaxis(new_signal, 0, self.channel_axis)

        return new_signal
