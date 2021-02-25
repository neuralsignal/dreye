"""
LED substitution experiments
"""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, least_squares

from dreye.estimators.base import _RelativeMixin
from dreye.estimators.excitation_models import IndependentExcitationFit
from dreye.utilities.abstract import inherit_docstrings
from dreye.utilities import asarray

EPS = 1e-5
EPS1 = 1e-3
EPS2 = 1e-2


@inherit_docstrings
class LedSubstitution(IndependentExcitationFit, _RelativeMixin):
    """
    Led Substitution estimator.
    """

    _skip_bg_ints = False

    def __init__(
        self,
        *,
        photoreceptor_model=None,  # dict or Photoreceptor class
        fit_weights=None,
        background=None,  # dict or Spectrum instance or array-like
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        smoothing_window=None,  # float
        max_iter=None,
        hard_separation=False,  # bool or list-like (same length as number of LEDs)
        hard_sep_value=1.0,  # float in capture units (1 relative capture)
        bg_ints=None,
        fit_only_uniques=False,
        ignore_bounds=False,
        lsq_kwargs=None,
        ignore_capture_units=False,
        rtype='weber',  # {'fechner/log', 'weber', None}
        unidirectional=False,  # allow only increase or decreases of LEDs in simulation
        keep_proportions=False,
        keep_intensity=True
    ):
        super().__init__(
            photoreceptor_model=photoreceptor_model,
            measured_spectra=measured_spectra,
            smoothing_window=smoothing_window,
            background=background,
            max_iter=max_iter,
            hard_separation=hard_separation,
            hard_sep_value=hard_sep_value,
            fit_weights=fit_weights,
            fit_only_uniques=fit_only_uniques,
            lsq_kwargs=lsq_kwargs,
            ignore_bounds=ignore_bounds,
            bg_ints=bg_ints,
            ignore_capture_units=ignore_capture_units
        )
        self.rtype = rtype
        self.unidirectional = unidirectional
        self.keep_proportions = keep_proportions
        self.keep_intensity = keep_intensity

    def fit(self, X, y=None):
        """
        LED substitution experiment.

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional array with two columns. The first column contains
            the indices of the LED to simulate and the second column contains
            the intensity for the LED to reach and simulate.
            TODO: if NaN, skip those entries and substitute with the background
            intensity.
        """

        # set required objects
        self._set_required_objects(None)
        self._set_A()

        X = asarray(X)
        assert (X.shape[1] % 2) == 0

        if X.shape[1] == 2:
            led_idcs = X[:, 0].astype(int)  # every second one
            led_bounds = X[:, 1]  # every second one starting at 1
        else:
            # TODO simulate combinations of LEDs
            raise NotImplementedError("Combinations of LED steps")
            led_idcs = X[:, 0::2].astype(int)  # every second one
            led_bounds = X[:, 1::2]  # every second one starting at 1

        led_bgs = self.bg_ints_[led_idcs]
        if self.rtype in {'fechner', 'weber', 'log', 'total_weber', 'diff'}:
            led_simulate_maxs = led_bounds > 0
        elif self.rtype in {'ratio', 'linear'}:
            led_simulate_maxs = led_bounds > 1
        elif self.rtype == 'absolute':
            led_simulate_maxs = led_bounds > led_bgs
        else:
            raise ValueError(f"rtype `{self.rtype}` not recognized.")

        led_abs_bounds = self._to_absolute_intensity(
            led_bounds, led_bgs
        )

        # weighting for each photoreceptor
        if self.fit_weights is None:
            fit_weights = np.ones(self.photoreceptor_model_.pr_number)
        else:
            fit_weights = asarray(self.fit_weights)

        weighted_init_fit = fit_weights.size > 1

        fitted_intensities = []
        fitted_info = []

        already_fitted = {}

        for led_idx, led_bound, led_simulate_max in zip(
            led_idcs, led_abs_bounds, led_simulate_maxs
        ):
            w_solo = self._get_w_solo(led_idx, led_bound)
            if np.any(w_solo > self.bounds_[1]):
                warnings.warn(
                    "Absolute intensity goes beyond measurment bounds! - "
                    "Change target intensity values."
                )
            # keep proportions or not
            if (
                self.keep_proportions
                and (led_idx in already_fitted)
                # TODO test with other rtypes
                and (self.rtype in {'diff', 'weber', 'total_weber'})
            ):
                old_w, old_w_solo = already_fitted[led_idx]

                old_rel_w = self._to_relative_intensity(
                    old_w,
                    self.bg_ints_,
                )
                old_rel_solo = self._to_relative_intensity(
                    old_w_solo[led_idx],
                    self.bg_ints_[led_idx],
                )
                new_rel_solo = self._to_relative_intensity(
                    w_solo[led_idx],
                    self.bg_ints_[led_idx],
                )

                factor = np.sum(new_rel_solo / old_rel_solo)
                new_rel_w = factor * old_rel_w

                w = self._to_absolute_intensity(
                    new_rel_w,
                    self.bg_ints_,
                )

                if np.any(w > self.bounds_[1]):
                    warnings.warn(
                        "Proportions go beyond measurement bounds! - "
                        "Turn off `keep_proportions` to avoid this error or "
                        "switch the order of the first sample fitted to be "
                        "the maximum intensity."
                    )
            else:
                w, w_solo = self._fit_sample(
                    w_solo,
                    led_idx,
                    led_bound,
                    led_simulate_max,
                    fit_weights,
                    weighted_init_fit,
                )

                already_fitted[led_idx] = (w, w_solo)

            new_relative_int = self._to_relative_intensity(
                w_solo[led_idx],
                self.bg_ints_[led_idx],
            )

            fitted_intensities.append(w)
            fitted_intensities.append(w_solo)

            fitted_info.append({
                'led': self.measured_spectra_.names[led_idx],
                'simulate': True,
                'rel_led_int': new_relative_int,
                'rtype': self.rtype
            })
            fitted_info.append({
                'led': self.measured_spectra_.names[led_idx],
                'simulate': False,
                'rel_led_int': new_relative_int,
                'rtype': self.rtype
            })

        fitted_intensities = np.array(fitted_intensities)
        fitted_excitations = self.fitted_excite_X_ = np.array([
            self._get_x_pred_noise(w)
            for w in fitted_intensities
        ])

        # pass
        self.fitted_intensities_ = fitted_intensities[::2]
        self.fitted_solo_intensities_ = fitted_intensities[1::2]
        self.fitted_relative_intensities_ = self._to_relative_intensity(
            self.fitted_intensities_
        )
        self.fitted_solo_relative_intensities_ = self._to_relative_intensity(
            self.fitted_solo_intensities_
        )
        self.fitted_excite_X_ = fitted_excitations[::2]
        self.fitted_solo_excite_X_ = fitted_excitations[1::2]
        self.current_X_ = self.fitted_solo_excite_X_

        self.fitted_capture_X_ = self.photoreceptor_model_.inv_excitefunc(
            self.fitted_excite_X_
        )
        self.fitted_solo_capture_X_ = self.photoreceptor_model_.inv_excitefunc(
            self.fitted_solo_excite_X_
        )

        index = pd.MultiIndex.from_frame(pd.DataFrame(fitted_info))
        fitted_excitations = pd.DataFrame(
            fitted_excitations,
            index=index,
            columns=[f"fitted_{rh}" for rh in self.channel_names_]
        ).reset_index()
        self.fitted_excitations_df_ = fitted_excitations

        fitted_intensities = pd.DataFrame(
            fitted_intensities,
            index=pd.MultiIndex.from_frame(fitted_excitations),
            columns=self.measured_spectra_.names
        )
        self.fitted_intensities_df_ = fitted_intensities
        return self

    def _fit_sample(
        self,
        w_solo,
        led_idx,
        led_bound,
        led_simulate_max,
        fit_weights,
        weighted_init_fit,
    ):
        # adjust A matrix
        target_capture_x = self._get_x_capture(w_solo)

        # check if simulating max and adjust bounds
        if led_simulate_max:
            # TODO add noise from capture noise level
            assert np.all(target_capture_x >= 1 - EPS), (
                str(target_capture_x)
            )
            # adjust lower bound to be the led intensity of interest
            if self.unidirectional:
                min_bound = self.bg_ints_.copy()
            else:
                min_bound = self.bounds_[0].copy()
            min_bound[led_idx] = led_bound
            max_bound = self.bounds_[1].copy()
            if self.keep_intensity:
                max_bound[led_idx] = led_bound + EPS2
            bounds_ = (
                min_bound, max_bound
            )
        else:
            assert np.all(target_capture_x <= 1 + EPS), (
                str(target_capture_x)
            )
            # adjust upper bound to be the led intensity of interest
            if self.unidirectional:
                max_bound = self.bg_ints_.copy()
            else:
                max_bound = self.bounds_[1].copy()
            max_bound[led_idx] = led_bound
            min_bound = self.bounds_[0].copy()
            if self.keep_intensity:
                min_bound[led_idx] = led_bound - EPS2
            bounds_ = (
                min_bound, max_bound
            )

        # bounds for the LED not simulated and the other LEDs during simulation
        if self.unidirectional and led_simulate_max:
            min_bound = self.bg_ints_.copy()
            max_bound = self.bounds_[1].copy()
        elif self.unidirectional:
            min_bound = self.bounds_[0].copy()
            max_bound = self.bg_ints_.copy()
        else:
            min_bound = self.bounds_[0].copy()
            max_bound = self.bounds_[1].copy()
        min_bound[led_idx] = self.bg_ints_[led_idx]
        max_bound[led_idx] = self.bg_ints_[led_idx] + EPS
        bounds_without = (
            min_bound, max_bound
        )

        # try to simulate the lower bound of the LED simulation
        w0 = self._init_sample(
            target_capture_x, bounds_without,
            fit_weights, weighted_init_fit
        ).copy()
        if led_simulate_max:
            w0[led_idx] = led_bound + EPS1
        else:
            w0[led_idx] = led_bound - EPS1

        # simulatenously fit the upper and lower bound
        result = least_squares(
            self._objective,
            x0=w0,
            args=(
                led_idx, fit_weights,
            ),
            bounds=bounds_,
            max_nfev=self.max_iter,
            **({} if self.lsq_kwargs is None else self.lsq_kwargs)
        )
        w = result.x.copy()
        w[led_idx] = self.bg_ints_[led_idx]
        w_solo = self.bg_ints_.copy()
        w_solo[led_idx] = result.x[led_idx]

        return w, w_solo

    def _get_w_solo(self, led_idx, led_bound):
        w_solo = self.bg_ints_.copy()
        w_solo[led_idx] = led_bound
        return w_solo

    def _init_sample(
        self,
        capture_x,
        bounds_without,
        fit_weights,
        weighted_init_fit
    ):
        # weighted fit is better for substitution types of fits
        if weighted_init_fit:
            result = lsq_linear(
                fit_weights[:, None] * self.A_,
                fit_weights * capture_x,
                bounds=tuple(bounds_without),
                max_iter=self.max_iter
            )
        else:
            result = lsq_linear(
                self.A_, capture_x,
                bounds=tuple(bounds_without),
                max_iter=self.max_iter
            )
        # return fitted intensity (w)
        return result.x

    def _objective(
        self,
        w, led_idx, fit_weights
    ):
        w_solo = self.bg_ints_.copy()
        w_solo[led_idx] = w[led_idx]
        w = w.copy()
        w[led_idx] = self.bg_ints_[led_idx]
        x_pred = self._get_x_pred(w)
        excite_x = self._get_x_pred(w_solo)
        return fit_weights * (excite_x - x_pred)


# deprecated version
def fit_led_substitution(
    X,
    photoreceptor_model,
    background,
    measured_spectra,
    bg_ints,
    fit_weights,
    rtype='weber',
    max_iter=None,
    lsq_kwargs=None,
    unidirectional=False,  # allow only increase or decreases of LEDs in simulation
    keep_proportions=False,
    keep_intensity=True
):
    """
    LED substitution experiment

    Parameters
    ----------
    X : numpy.ndarray
        Two-dimensional array with two columns. The first column contains
        the indices of the LED to simulate and the second column contains
        the intensity for the LED to reach and simulate.
        TODO: if NaN, skip those entries and substitute with the background
        intensity.
    photoreceptor_model : dreye.Photoreceptor
    background : dreye.Signal or array-like
    measured_spectra : dreye.MeasuredSpectraContainer
    bg_ints : array-like
    fit_weights : array-like
    rtype : str, optional
    max_iter : int, optional
    lsq_kwargs : dict, optional
    unidirectional : bool, optional
    keep_proportions : bool, optional
    """

    model = LedSubstitution(
        photoreceptor_model=photoreceptor_model,
        background=background, measured_spectra=measured_spectra,
        bg_ints=bg_ints, fit_weights=fit_weights,
        rtype=rtype, max_iter=max_iter,
        lsq_kwargs=lsq_kwargs, unidirectional=unidirectional,
        keep_proportions=keep_proportions, keep_intensity=keep_intensity
    )
    model.fit(X)
    return model.fitted_intensities_df_
