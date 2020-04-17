"""
"""

from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA, NMF

from dreye.utilities.abstract import AbstractContainer


class EstimatorHolder:
    """estimator holder that also implements plotting functionality
    """

    def __init__(self, estimator, X, y):
        self._estimator = estimator
        self._X = X
        self._y = y

    def __getattr__(self, name):
        return getattr(self._estimator, name)

    def _pca_plot(self):
        """plot for PCA
        """

        assert isinstance(self._estimator, PCA)
        raise NotImplementedError('pca_plot')

    def _nmf_plot(self):
        """plot for NMF
        """

        assert isinstance(self._estimator, NMF)
        raise NotImplementedError('nmf_plot')

    def _decomp_subplot(self):
        """
        """


class EstimatorContainer(AbstractContainer):
    """Container to hold multiple estimators from a cross validation run.
    """
    _allowed_instances = EstimatorHolder

    def __init__(self, container, **kwargs):
        self._kwargs = kwargs
        super().__init__(container)

    def __getitem__(self, key):
        if isinstance(key, str) and key in self._kwargs:
            return self._kwargs[key]
        return super().__getitem__(key)

    @property
    def scores(self):
        return self._kwargs


class EstimatorMixin:
    """Mixin for signal class to apply sklearn estimators
    """

    def sklearn_estimator(
        self,
        estimator_class,
        sample_axis=None,
        pass_domain=None,
        cv=None,
        fit_params=None,
        **estimator_kwargs
    ):
        """apply sklearn estimator

        Returns
        -------
        obj : estimator or EstimatorContainer
            If cross_validate is not None this function returns an
            EstimatorContainer that can be used like the sklearn.BaseEstimator
            and contains the train, test scores as returned by
            sklearn.model_selection.cross_validate.
            If cross_validate is None, the function simply returns the
            EstimatorHolder that works like an instance of the estimator class
            passed and also implements plotting functionality.
        """

        assert self.ndim == 2

        enforced_cv_kws = {
            'fit_params': fit_params,
            'return_train_score': True,
            'return_estimator': True,
        }

        if sample_axis is None:
            sample_axis = self.domain_axis

        if int(sample_axis % 2) != 0:
            self = self.moveaxis(sample_axis, 0)

        if pass_domain is None:
            X = self.magnitude
            y = None
        elif pass_domain == 'y':
            X = self.magnitude
            y = self.domain_magnitude[:, None]
        elif pass_domain == 'X':
            X = self.domain_magnitude[:, None]
            y = self.magnitude

        if isinstance(cv, dict):
            cv.update(enforced_cv_kws)
        elif cv is not None:
            enforced_cv_kws['cv'] = cv
            cv = enforced_cv_kws

        estimator = estimator_class(**estimator_kwargs)

        if cv is None:
            estimator.fit(X, y)
            return EstimatorHolder(estimator, X, y)
        else:
            scores = cross_validate(estimator, X, y, **cv)
            estimators = [
                EstimatorHolder(est, X, y)
                for est in scores['estimator']
            ]
            return EstimatorContainer(estimators, **scores)

    def pca(self, sample_axis=None, cv=None, **estimator_kwargs):
        """Perform PCA on signal.
        """

        return self.sklearn_estimator(
            PCA, sample_axis=sample_axis, cv=cv, **estimator_kwargs
        )

    def nmf(self, sample_axis=None, cv=None, **estimator_kwargs):
        """Perform NMF on signal.
        """

        return self.sklearn_estimator(
            NMF, sample_axis=sample_axis, cv=cv, **estimator_kwargs
        )
