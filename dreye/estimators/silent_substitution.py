"""
Model for silent substitution
"""

import warnings

import numpy as np
import cvxpy as cp
import pandas as pd

from dreye.estimators.base import _RelativeMixin
from dreye.estimators.excitation_models import IndependentExcitationFit
from dreye.utilities.abstract import inherit_docstrings
from dreye.utilities import asarray

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
        ignore_bounds=False,
        ignore_capture_units=True,
        background_only_external=False,
        substitution_type='diff',
        eps=EPS,
        q_aggregator=cp.min,
        cp_kwargs=None,
        linear_transform=None,
    ):
        super().__init__(
            photoreceptor_model=photoreceptor_model,
            measured_spectra=measured_spectra,
            background=background,
            fit_weights=fit_weights,
            ignore_bounds=ignore_bounds,
            bg_ints=bg_ints,
            ignore_capture_units=ignore_capture_units,
            background_only_external=background_only_external
        )
        self.substitution_type = substitution_type
        self.eps = eps
        self.q_aggregator = q_aggregator
        self.cp_kwargs = cp_kwargs
        self.linear_transform = linear_transform

    def fit(self, X):
        # format X
        X = asarray(X)
        if X.ndim == 1:
            self._set_required_objects()
            X_ = np.zeros((len(X), self.n_features_)).astype(bool)
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
            self.channels_ = self.channel_names_
        else:
            self.channels_ = [f"comp{i}" for i in range(self.n_features_)]

        intensities = np.zeros((len(X), self.n_leds_))
        intensities2 = np.zeros(intensities.shape)
        captures = np.zeros((len(X), self.n_features_))
        captures2 = np.zeros(captures.shape)
        active = []

        for idx, ix in enumerate(X):
            q, i, q2, i2 = self._fit_sample(ix)
            captures[idx] = q
            captures2[idx] = q2
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
            i >= self.bounds_[0],
        ]
        if np.isfinite(self.bounds_[1]).all():
            constraints.append(
                i <= self.bounds_[1]
            )
        return constraints

    def _get_q_constraints(self, i, q):
        pred_q = self._get_x_capture(i)
        if self.linear_transform is not None:
            pred_q = pred_q @ self.linear_transform
        return [pred_q == q]

    def _fit_sample(self, x):
        # q is usually the capture, but could also be excitations
        q = cp.Variable(self.n_features_, name='q')
        i = cp.Variable(self.n_leds_, nonneg=True, name='i')

        constraints = []

        constraints.extend(self._get_i_constraints(i))
        constraints.extend(self._get_q_constraints(i, q))

        if self.substitution_type == 'diff':
            q2 = cp.Variable(self.n_features_, name='q2')
            i2 = cp.Variable(self.n_leds_, nonneg=True, name='i2')

            constraints.extend(self._get_i_constraints(i2))
            constraints.extend(self._get_q_constraints(i2, q2))

            constraints.extend([
                q[~x] <= (q2[~x] + self.eps),
                q[~x] >= (q2[~x] - self.eps)
            ])

        else:
            constraints.extend([
                q[~x] >= (self.capture_border_ - self.eps),
                q[~x] <= (self.capture_border_ + self.eps),
            ])

        if self.substitution_type == 'max':
            obj = cp.Maximize(self.q_aggregator(q[x]))
        elif self.substitution_type == 'min':
            obj = cp.Minimize(self.q_aggregator(q[x]))
        else:
            obj = (
                cp.Maximize(self.q_aggregator(q[x]))
                + cp.Maximize(self.q_aggregator(-q2[x]))
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
            q2_value = np.ones(self.n_features_) * self.capture_border_
            i2_value = (
                np.zeros(self.n_leds_) if self.bg_ints_ is None
                else self.bg_ints_
            )

        return q_value, i_value, q2_value, i2_value
