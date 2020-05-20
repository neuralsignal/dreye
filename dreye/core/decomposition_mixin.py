"""
"""

# TODO only focus on decomposition
# TODO rename decomposition mixin
# TODO add preprocessing steps?
# TODO run for multiple components at once
# TODO for multiple n_components put into long dataframe (the different attributes)

import numpy as np
import pandas as pd
from sklearn import decomposition as decomp
from joblib import Parallel, delayed

from dreye.utilities.abstract import AbstractContainer


class _BootstrappedContainer(AbstractContainer):
    _allowed_instances = (
        decomp.PCA, decomp.FastICA, decomp.NMF, decomp.SparsePCA,
        decomp.KernelPCA, decomp.FactorAnalysis, decomp.TruncatedSVD
    )


class DecompositionPlotter:
    """mixin class to plot `sklearn.decomposition` estimators
    """

    def __init__(self, obj, X, estimator, bs_estimators=None):
        self._obj = obj
        self._estimator = estimator
        self._bs_estimators = (bs_estimators
                               if bs_estimators is None
                               else _BootstrappedContainer(bs_estimators))

        # initialize properties
        self._bs_frame = None

    def __getattr__(self, name):
        if {'_obj', '_bs_estimators', '_estimator'} - set(vars(self)):
            raise AttributeError
        elif name.startswith('bs_'):
            if self._bs_estimators is None:
                raise AttributeError(f'{name} is not an attribute, '
                                     'as not boostrapped estimators exist.')
            name = name.replace('bs_', '')
            if name in self._estimator.__dict__:
                return np.stack(getattr(self._bs_estimators, name))
            else:
                return getattr(self._bs_estimators, name)
        else:
            return getattr(self._estimator, name)

    def plot(self, **kwargs):
        """figure-level plot (provide axes)?
        """
        raise NotImplementedError('')

    @property
    def bs_frame(self):
        if self._bs_estimators is None:
            return
        elif self._bs_frame is None:
            df = pd.DataFrame([est.__dict__ for est in self._bs_estimators])
            df.index.name = 'bs_iter'
            df = df.reset_index()
            self._bs_frame = df

        return self._bs_frame


class MultiDecompositionPlotter:
    """plot decomposition of multiple estimators with different n_components
    """

    # TODO plot pages
    # TODO plot frobenius norm and plot components
    # TODO plot transformation?


class PCAPlotter(DecompositionPlotter):
    pass


class FastICAPlotter(DecompositionPlotter):
    pass


class NMFPlotter(DecompositionPlotter):
    pass


class FactorAnalysisPlotter(DecompositionPlotter):
    pass


class KernelPCAPlotter(DecompositionPlotter):
    pass


class SparsePCAPlotter(DecompositionPlotter):
    pass


class TruncatedSVDPlotter(DecompositionPlotter):
    pass


class DecompositionMixin:
    """Mixin for signal class to apply `sklearn.decomposition` estimators
    """

    def _prepare_self_to_decompose(self, sample_axis):
        """
        Move sample_axis to zeroth axis if necessary.

        Returns
        -------
        self_copy : object
            Returns a copy of the self instance.
        """

        assert self.ndim == 2

        if sample_axis is None:
            sample_axis = self.domain_axis

        # move axis if necessary
        if int(sample_axis % 2) != 0:
            return self.moveaxis(sample_axis, 0)
        else:
            return self.copy()

    def _decompose(
        self,
        estimator_class,
        plotter_class,
        n_components=2,
        n_boots=None,
        n_samples=None,
        n_jobs=None,
        seed=None,
        **estimator_kwargs
    ):
        """apply sklearn estimator

        Returns
        -------
        plotter : list of `dreye.DecompositionPlotter`
        """

        X = self.magnitude

        estimator = _fit_decomp(
            estimator_class,
            X,
            n_components, **estimator_kwargs)

        if n_boots is None:
            bs_estimators = []
        else:  # get boostrapped estimators
            length = X.shape[0]
            n_samples = length if n_samples is None else n_samples
            rng = np.random.default_rng(seed)
            if n_jobs is None:
                bs_estimators = [
                    _fit_decomp(
                        estimator_class,
                        rng.choice(X, size=n_samples),
                        n_components, **estimator_kwargs
                    )
                    for i in range(n_boots)
                ]
            else:  # parallelization
                bs_estimators = Parallel(n_jobs=n_jobs)(
                    delayed(_fit_decomp)(
                        estimator_class,
                        rng.choice(X, size=n_samples),
                        n_components, **estimator_kwargs
                    )
                    for i in range(n_boots)
                )

        return plotter_class(self, X, estimator, bs_estimators)

    # TODO individual decomposition methods
    # TODO compare decomposition methods

    def pca(self, sample_axis=None, **estimator_kwargs):
        """Perform PCA on signal.
        """
        self = self._prepare_self_to_decompose(sample_axis)
        return self._decompose(
            decomp.PCA, PCAPlotter,
            **estimator_kwargs
        )

    def pca_plus(
        self, list_n_components, sample_axis=None, **estimator_kwargs
    ):
        """Perform PCA multiple times with varying components.
        """
        # pop n_components
        estimator_kwargs.pop('n_components', None)
        # TODO frobenius norm
        self = self._prepare_self_to_decompose(sample_axis)
        # TODO MultiPCAPlotter etc.
        return MultiDecompositionPlotter([
            self._decompose(
                decomp.PCA, PCAPlotter,
                n_components=n_components,
                **estimator_kwargs
            )
            for n_components in list_n_components
        ])

    def nmf(self, sample_axis=None, **estimator_kwargs):
        """Perform NMF on signal.
        """
        self = self._prepare_self_to_decompose(sample_axis)
        return self._decompose(
            decomp.NMF, NMFPlotter,
            **estimator_kwargs
        )


# --- helper functions --- #
def _fit_decomp(estimator_class, X, n_components, **estimator_kwargs):
    estimator = estimator_class(
        n_components=n_components,
        **estimator_kwargs)
    return estimator.fit(X)
