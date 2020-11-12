"""
Reshaper for sklearn-type estimators
"""

from sklearn.base import BaseEstimator, TransformerMixin


class Reshaper(BaseEstimator, TransformerMixin):

    def __init__(self, estimator):
        self.estimator = estimator

    def __getattr__(self, name):
        if hasattr(self.estimator, name):
            pass
        else:
            raise AttributeError

# TODO class reshaper (base class?)
# anything in _X_length with be reshaped
# transform, and inv_transform need to return reshaped X or output
# (except for last axis)
# any callable with X will be reshaped before passing
# if exists, then fitted_X, current_X_
# and anything that corresponds to fitted_X and current_X_ (all_equal)
