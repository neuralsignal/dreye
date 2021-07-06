"""
Model for silent substitution
"""

import numpy as np
import cvxpy as cp
import pandas as pd

from dreye.estimators.base import _RelativeMixin
from dreye.estimators.excitation_models import IndependentExcitationFit
from dreye.utilities.abstract import inherit_docstrings
from dreye.utilities import asarray
from dreye.utilities import is_numeric, is_string, is_callable

EPS = 1e-4
N = 1000
# TODO docstrings


@inherit_docstrings
class BestSubstitutionFit(IndependentExcitationFit, _RelativeMixin):
    """
    Substitution estimator.
    """

    def __init__(
        self,
        *,
        photoreceptor_model=None,  # dict or Photoreceptor class
        fit_weights=None,
        background=None,  # dict or Spectrum instance or array-like
        measured_spectra=None,  # dict, or MeasuredSpectraContainer
        bg_ints=None,
        ignore_bounds=None,
        background_external=None,
        substitution_type='diff',
        eps=EPS,
        q_aggregator='min',
        cp_kwargs=None,
        linear_transform=None,
        nonlin=None, 
        intensity_bounds=None, 
        wavelengths=None, 
        capture_noise_level=None
    ):
        super().__init__(
            photoreceptor_model=photoreceptor_model,
            measured_spectra=measured_spectra,
            background=background,
            fit_weights=fit_weights,
            ignore_bounds=ignore_bounds,
            bg_ints=bg_ints,
            background_external=background_external, 
            intensity_bounds=intensity_bounds, 
            wavelengths=wavelengths, 
            capture_noise_level=capture_noise_level
        )
        self.substitution_type = substitution_type
        self.eps = eps
        self.q_aggregator = q_aggregator
        self.cp_kwargs = cp_kwargs
        self.linear_transform = linear_transform
        self.nonlin = nonlin

    def fit(self, X):
        # format X
        X = asarray(X)
        if X.ndim == 1:
            self._set_required_objects()
            X_ = np.zeros((len(X), self.photoreceptor_model_.n_opsins)).astype(bool)
            for idx, ix in enumerate(X):
                X_[idx, ix] = True
            X = X_
        else:
            # set all required objects
            self._set_required_objects(X.shape[1])

        self.boolean_X_ = X

        if self.photoreceptor_model_.filterfunc is not None:
            raise NotImplementedError(
                "Substitution with `filterfunc` in photoreceptor model."
            )

        if is_numeric(self.substitution_type):
            assert self.substitution_type > 0, "`substitution_type` must be a positive float."
        else:
            if self.substitution_type not in {'diff', 'max', 'min'}:
                raise NameError(
                    f"substitution_type `{self.substitution_type}`"
                    " not `diff`, `max`, or `min`."
                )

            if self.substitution_type != 'diff' and self.background_ is None:
                raise ValueError(
                    f"For substitution_type `{self.substitution_type}`, "
                    "a background needs to be supplied via `bg_ints` "
                    "or `background`."
                )

        if self.linear_transform is None:
            self.channels_ = self.photoreceptor_model_.labels
        else:
            self.channels_ = [f"comp{i}" for i in range(self.photoreceptor_model_.n_opsins)]

        intensities = np.zeros((len(X), len(self.measured_spectra_)))
        intensities2 = np.zeros(intensities.shape)
        captures = np.zeros((len(X), self.photoreceptor_model_.n_opsins))
        captures2 = np.zeros(captures.shape)
        r = np.zeros(captures.shape)
        r2 = np.zeros(captures2.shape)
        active = []

        for idx, ix in enumerate(X):
            q, i, q2, i2 = self._fit_sample(ix)
            captures[idx] = self.get_capture(i)
            captures2[idx] = self.get_capture(i2)
            r[idx] = q
            r2[idx] = q2
            intensities[idx] = i
            intensities2[idx] = i2
            active.append(np.array(self.channels_)[ix])

        excitations = self.photoreceptor_model_.excitefunc(captures)
        excitations2 = self.photoreceptor_model_.excitefunc(captures2)

        self.fitted_intensities_ = intensities
        self.fitted_other_intensities_ = intensities2
        self.fitted_excite_X_ = excitations
        self.fitted_other_excite_X_ = excitations2
        self.fitted_other_capture_X_ = captures2
        self.fitted_capture_X_ = captures
        self.fitted_r_ = r
        self.fitted_other_r_ = r2
        self.active_ = pd.Series(active).apply(tuple)

        if self.substitution_type == 'diff':
            df1 = self._create_dataframe(
                self.fitted_intensities_,
                self.fitted_excite_X_,
                self.active_,
                'max'
            )
            df2 = self._create_dataframe(
                self.fitted_other_intensities_,
                self.fitted_other_excite_X_,
                self.active_,
                'min'
            )
            self.fitted_intensities_df_ = pd.concat([df1, df2], axis=0)

        else:
            self.fitted_intensities_df_ = self._create_dataframe(
                self.fitted_intensities_,
                self.fitted_excite_X_,
                self.active_,
                self.substitution_type
            )

        return self

    @property
    def X_(self):
        return self.fitted_other_excite_X_

    def _create_dataframe(self, i, e, active, direction):
        """
        """
        edf = pd.DataFrame(
            e,
            columns=[f"fitted_{rh}" for rh in self.channels_]
        )
        edf['active'] = active
        edf['direction'] = direction
        edf['substitution_type'] = self.substitution_type
        return pd.DataFrame(
            i, columns=self.measured_spectra_.names,
            index=pd.MultiIndex.from_frame(edf)
        )

    def _get_i_constraints(self, i):
        constraints = [
            i >= self.intensity_bounds_[0],
        ]
        if np.isfinite(self.intensity_bounds_[1]).all():
            constraints.append(
                i <= self.intensity_bounds_[1]
            )
        return constraints

    def _get_q(self, i):
        q = self.get_capture(i)
        if self.nonlin is not None:
            q = self.nonlin(q)
        if self.linear_transform is not None:
            q = q @ self.linear_transform
        return q

    def _fit_sample(self, x):
        # q is usually the capture, but could also be excitations (not tested)
        i = cp.Variable(len(self.measured_spectra_), nonneg=True, name='i')
        q = self._get_q(i)

        constraints = []
        constraints.extend(self._get_i_constraints(i))

        if is_numeric(self.substitution_type):  # TODO must be a positive numeric type?
            constraints.extend([
                cp.sum(q - self.capture_border_) == self.substitution_type
            ])
        elif self.substitution_type == 'diff':
            i2 = cp.Variable(len(self.measured_spectra_), nonneg=True, name='i2')
            q2 = self._get_q(i2)

            constraints.extend(self._get_i_constraints(i2))
            constraints.extend([
                q[~x] <= (q2[~x] + self.eps),
                q[~x] >= (q2[~x] - self.eps)
            ])

        else:
            constraints.extend([
                q[~x] >= (self.capture_border_ - self.eps),
                q[~x] <= (self.capture_border_ + self.eps),
            ])

        if is_string(self.q_aggregator):
            q_aggregator = getattr(cp, self.q_aggregator)
        elif is_callable(self.q_aggregator):
            q_aggregator = self.q_aggregator

        if is_numeric(self.substitution_type):
            obj = (
                cp.Maximize(q_aggregator(q[x]))
                + cp.Maximize(q_aggregator(-q[~x]))
            )
        elif self.substitution_type == 'max':
            obj = cp.Maximize(q_aggregator(q[x]))
        elif self.substitution_type == 'min':
            obj = cp.Minimize(q_aggregator(q[x]))
        else:
            obj = (
                cp.Maximize(q_aggregator(q[x]))
                + cp.Maximize(q_aggregator(-q2[x]))
            )

        prob = cp.Problem(obj, constraints)

        _ = prob.solve(
            **({} if self.cp_kwargs is None else self.cp_kwargs)
        )

        q_value = q.value
        i_value = i.value
        if self.substitution_type == 'diff':
            q2_value = q2.value
            i2_value = i2.value
        else:
            q2_value = np.ones(self.photoreceptor_model_.n_opsins) * self.capture_border_
            i2_value = (
                np.zeros(len(self.measured_spectra_)) if self.bg_ints_ is None
                else self.bg_ints_
            )

        return q_value, i_value, q2_value, i2_value
