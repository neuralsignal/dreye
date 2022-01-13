"""Object-oriented Estimator class for all fitting procedures
"""

from numbers import Number
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dreye.api.capture import calculate_capture
from dreye.api.convex import get_P_from_A, in_hull_from_A, range_of_solutions
from dreye.api.domain import equalize_domains
from dreye.api.optimize.lsq_linear import lsq_linear, lsq_linear_adaptive, lsq_linear_decomposition, lsq_linear_excitation, lsq_linear_minimize, lsq_linear_underdetermined
from dreye.api.optimize.lsq_nonlinear import lsq_nonlinear
from dreye.api.sampling import sample_in_hull
from dreye.api.utils import check_bounds
from dreye.api.metrics import compute_gamut


class ReceptorEstimator:
    """[summary]
    
    Parameters
    ----------
    filters : [type]
        [description]
    domain : float, optional
        [description], by default 1.0
    filters_uncertainty : [type], optional
        [description], by default None
    w : [type], optional
        [description], by default None
    labels : [type], optional
        [description], by default None
    """
    
    def __init__(
        self, 
        filters,
        domain=1.0,
        filters_uncertainty=None,
        w=1.0,
        labels=None, 
        K=1.0, 
        baseline=0.0
    ):
        self.filters = np.asarray(filters)
        self.domain = (domain if isinstance(domain, Number) else np.asarray(domain))
        self.filters_uncertainty = (
            filters_uncertainty if filters_uncertainty is None 
            else np.asarray(filters_uncertainty)
        )
        self.w = np.broadcast_to(w, (self.filters.shape[0],))
        self.labels = (np.arange(self.filters.shape[0]) if labels is None else np.asarray(labels))
        
        self.register_adaptation(K)
        self.register_baseline(baseline)
        
    def register_uncertainty(self, filters_uncertainty):
        """[summary]

        Parameters
        ----------
        filters_uncertainty : [type]
            [description]
        """
        self.filters_uncertainty = (
            filters_uncertainty 
            if filters_uncertainty is None 
            else np.asarray(filters_uncertainty)
        )
    
    ### register state of adaptation and baseline
    
    def register_adaptation(self, K):
        """[summary]

        Parameters
        ----------
        K : [type]
            [description]
        """
        # K being one d or two d
        self.K = np.atleast_1d(K)
    
    def register_background_adaptation(self, background, domain=None, add_baseline=True, add=False):
        """[summary]

        Parameters
        ----------
        background : [type]
            [description]
        domain : [type], optional
            [description], by default None
        add_baseline : bool, optional
            [description], by default True
        """
        qb = self.capture(background, domain=domain)
        if add_baseline:
            qb = qb + self.baseline
        if add:
            self.K = self.K + 1/qb
        else:
            self.K = 1/qb
    
    def register_baseline(self, baseline):
        """[summary]

        Parameters
        ----------
        baseline : [type]
            [description]
        """
        self.baseline = np.atleast_1d(baseline)
    
    def _check_domain(self, domain, signals, uncertainty=False):
        if uncertainty:
            filters = self.filters_uncertainty
            assert filters is not None, "uncertainty cannot be set to None to calculate uncertainty capture."
        else:
            filters = self.filters
        
        if domain is None:
            assert signals.shape[-1] == filters.shape[-1], "array lengths must match."
            return self.domain, filters, signals
        if isinstance(self.domain, Number):
            assert isinstance(domain, Number), "domain must be float, if filter domain was float."
            assert self.domain == domain, "domain length must match"
            assert signals.shape[-1] == filters.shape[-1], "array lengths must match."
            return self.domain, filters, signals
        assert not isinstance(domain, Number), "If filter domain is array-like, domain must be array-like."
        domain = np.asarray(domain)
        domain, [filters, signals] = equalize_domains([self.domain, domain], [filters, signals]) 
        return domain, filters, signals
    
    ### methods that only require filters
    
    def capture(self, signals, domain=None):
        """[summary]

        Parameters
        ----------
        signals : [type]
            [description]
        domain : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        signals = np.asarray(signals)
        domain, filters, signals = self._check_domain(domain, signals)
        # filterdim x signaldim
        return calculate_capture(filters, signals, domain=domain)
    
    def uncertainty_capture(self, signals, domain=None):
        """[summary]

        Parameters
        ----------
        signals : [type]
            [description]
        domain : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        ValueError
            [description]
        """
        signals = np.asarray(signals)
        domain, filters_uncertainty, signals = self._check_domain(domain, signals, uncertainty=True)
        if filters_uncertainty.ndim == 3:
            # assumes uncertainty are samples
            Q = calculate_capture(filters_uncertainty, signals, domain=domain)
            return np.var(Q, axis=0) 
        elif filters_uncertainty.ndim == 2:
            # assume uncertainty is standard deviation
            filters_var = filters_uncertainty ** 2
            signals_squared = signals ** 2
            return calculate_capture(filters_var, signals_squared, domain=domain)
        else:
            raise ValueError(f"filters uncertainty is wrong dimensionality: {filters_uncertainty.ndim}.")
    
    def _relative_capture(self, B):
        B = B + self.baseline
        if self.K.ndim <= 1:
            B = B * self.K
        else:
            B = (self.K @ B.T).T
        return B
        
    def relative_capture(self, signals, domain=None):
        """[summary]

        Parameters
        ----------
        signals : [type]
            [description]
        domain : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        B = self.capture(signals, domain)
        return self._relative_capture(B)
    
    ### register a system
    
    def register_system(self, sources, domain=None, lb=None, ub=None, labels=None, Epsilon=None):
        """[summary]

        Parameters
        ----------
        sources : [type]
            [description]
        domain : [type], optional
            [description], by default None
        lb : [type], optional
            [description], by default None
        ub : [type], optional
            [description], by default None
        labels : [type], optional
            [description], by default None
        """
        sources = np.asarray(sources)
        # assign sources and domain
        domain, _, sources = self._check_domain(domain, sources)
        self.sources = sources
        self.sources_domain = domain
        # lb and ub
        self.lb, self.ub = check_bounds(lb, ub, self.sources.shape[0])
        
        # labels
        self.sources_labels = (np.arange(self.sources.shape[0]) if labels is None else np.asarray(labels))
        
        # create A  (channels x sources)
        self.A = self.capture(sources, domain=domain).T
        
        if Epsilon is None:
            if self.filters_uncertainty is None:
                self.Epsilon = 'heteroscedastic'
            else:
                self.Epsilon = self.uncertainty_capture(sources, domain=domain).T
        else:
            self.Epsilon = np.asarray(Epsilon)
            assert self.Epsilon.shape == self.A.shape, "Shape of Epsilon must match (n_channels x n_sources)."
    
    def compute_gamut(
        self, 
        fraction=True,
        at_l1=None, metric='width', seed=None, 
        relative=True, 
    ):
        """[summary]

        Parameters
        ----------
        fraction : bool, optional
            [description], by default True
        at_l1 : [type], optional
            [description], by default None
        metric : str, optional
            [description], by default 'width'
        seed : [type], optional
            [description], by default None
        relative : bool, optional
            [description], by default True

        Returns
        -------
        [type]
            [description]
        """
        self._assert_registered()
        P = get_P_from_A(
            self.A, self.lb, self.ub, 
            K=(self.K if relative else None), 
            baseline=(self.baseline if relative else None), 
            bounded=True
        )
        if fraction:
            # dirac delta functions for perfect excitation
            signals = np.eye(self.filters.shape[-1])
            if relative:
                relative_to = self.relative_capture(signals)
            else:
                relative_to = self.capture(signals)
        else:
            relative_to = None
        
        return compute_gamut(
            P, at_l1=at_l1, relative_to=relative_to, 
            center_to_neutral=False, 
            center=True, 
            metric=metric, 
            seed=seed, 
        )
    
    @property
    def registered(self):
        return hasattr(self, 'A') and hasattr(self, 'Epsilon')
    
    def _assert_registered(self):
        assert self.registered, "No system has yet been registered."
    
    @property
    def underdetermined(self):
        return self.registered and (self.A.shape[0] < self.A.shape[1])
    
    def system_capture(self, X):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        self._assert_registered()
        X = np.asarray(X)
        return X @ self.A.T
    
    def system_relative_capture(self, X):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        self._assert_registered()
        B = self.system_capture(X)
        return self._relative_capture(B)
    
    def in_system(self, X):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]
        """
        self._assert_registered()
        X = np.asarray(X)
        return (X >= self.lb) & (X <= self.ub)
    
    def register_system_adaptation(self, x, add_baseline=True, add=False):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]
        """
        self._assert_registered()
        qb = self.system_capture(x)
        if add_baseline:
            qb = qb + self.baseline
        if add:
            self.K = self.K + 1/qb
        else:
            self.K = 1/qb
    
    def register_bounds(self, lb=None, ub=None):
        """[summary]

        Parameters
        ----------
        lb : [type], optional
            [description], by default None
        ub : [type], optional
            [description], by default None
        """
        self._assert_registered()
        # lb and ub
        lb_, ub_ = check_bounds(lb, ub, self.A.shape[1])
        if lb is not None:
            self.lb = lb_
        if ub is not None:
            self.ub = ub_
    
    def sample_in_hull(self, n=10, seed=None, engine=None):
        """[summary]

        Parameters
        ----------
        n : int, optional
            [description], by default 10
        seed : [type], optional
            [description], by default None
        engine : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        self._assert_registered()
        # check if bounded
        bounded = np.all(np.isfinite(self.ub))
        P = get_P_from_A(self.A, self.lb, self.ub, K=self.K, baseline=self.baseline, bounded=bounded)
        return sample_in_hull(P, n, seed=seed, engine=engine)
        
    ### fitting functions
    
    def register_targets(self, B, W=None):
        """[summary]

        Parameters
        ----------
        B : [type]
            [description]
        W : [type], optional
            [description], by default None
        """
        self._assert_registered()
        self.target_B = np.asarray(B)
        self.B = self.target_B.copy()
        if W is None:
            self.W = self.w
        else:
            self.W = np.asarray(W)
        return self
            
    def _assert_registered_targets(self):
        assert hasattr(self, 'B'), "No target has yet been registered for fitting."
        
    def _assert_fitted(self):
        assert hasattr(self, 'X'), "Target have not been fitted to system yet."
    
        ### methods that require registered system
    
    def in_hull(self, B=None, relative=True):
        """[summary]

        Parameters
        ----------
        B : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        self._assert_registered()
        if B is None:
            self._assert_registered_targets()
            B = self.B
        return in_hull_from_A(
            B, self.A, lb=self.lb, ub=self.ub, 
            K=(self.K if relative else None), 
            baseline=(self.baseline if relative else None)
        )
        
    def range_of_solutions(
        self, B=None, relative=True, error='raise', n=None, eps=1e-5
    ):
        """[summary]

        Parameters
        ----------
        B : [type], optional
            [description], by default None
        relative : bool, optional
            [description], by default True
        error : str, optional
            [description], by default 'raise'
        n : [type], optional
            [description], by default None
        eps : [type], optional
            [description], by default 1e-5

        Returns
        -------
        [type]
            [description]
        """
        self._assert_registered()
        if B is None:
            self._assert_registered_targets()
            B = self.B
        return range_of_solutions(
            B, self.A, lb=self.lb, ub=self.ub, 
            K=(self.K if relative else None), 
            baseline=(self.baseline if relative else None), 
            error=error, n=n, eps=eps
        )
    
    
    def fit(
        self, B=None, model='gaussian',
        batch_size=1, 
        verbose=0, 
        **opt_kwargs
    ):
        """[summary]

        Parameters
        ----------
        model : str, optional
            [description], by default 'gaussian'
        batch_size : int, optional
            [description], by default 1
        verbose : int, optional
            [description], by default 0

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        NameError
            [description]
        """
        self._assert_registered()
        if B is None:
            internal = True
            self._assert_registered_targets()
            B = self.B
        else:
            internal = False
        # linear (gaussian, poisson), excitation, nonlinear
        if callable(model):
            X, B, _ = lsq_nonlinear(
                self.A, B, 
                lb=self.lb, 
                ub=self.ub, 
                W=self.W, 
                K=self.K, 
                baseline=self.baseline, 
                nonlin=model, 
                batch_size=batch_size, 
                verbose=verbose, 
                return_pred=True, 
                **opt_kwargs
            )
            self.X = X
            self.B = B
            return self
        elif model in {'gaussian', 'poisson'}:
            X, B = lsq_linear(
                self.A, 
                B, 
                lb=self.lb, ub=self.ub, 
                W=self.W, K=self.K, 
                baseline=self.baseline, 
                batch_size=batch_size, 
                model=model, verbose=verbose, return_pred=True, 
                **opt_kwargs
            )
        elif model == 'excitation':
            X, B = lsq_linear_excitation(
                self.A, 
                B, 
                lb=self.lb, ub=self.ub, 
                W=self.W, K=self.K, 
                baseline=self.baseline, 
                batch_size=batch_size, 
                verbose=verbose, return_pred=True, 
                **opt_kwargs
            )
        else:
            raise NameError(f"Model name `{model}` unknown.")
            
        self.X = X
        self.B = B
        if internal:
            return self
        return X, B
    
    def fit_adaptive(
        self, 
        B=None,
        neutral_point=None, 
        delta_radius=1e-5, 
        delta_norm1=1e-5,
        adaptive_objective="unity", 
        verbose=0, 
        **opt_kwargs
    ):
        self._assert_registered()
        if B is None:
            internal = True
            self._assert_registered_targets()
            B = self.B
        else:
            internal = False
        X, scales, B = lsq_linear_adaptive(
            self.A, 
            B, 
            lb=self.lb, ub=self.ub, 
            W=self.W, K=self.K, 
            baseline=self.baseline, 
            verbose=verbose, return_pred=True, 
            neutral_point=neutral_point, 
            delta_norm1=delta_norm1, 
            delta_radius=delta_radius, 
            adaptive_objective=adaptive_objective, 
            **opt_kwargs
        )
        
        self.X = X
        self.B = B
        self.scales = scales
        if internal:
            return self
        return X, scales, B
    
    def fit_decomposition(
        self,
        B=None, 
        n_layers=None, 
        mask=None, 
        lbp=0, ubp=1, 
        max_iter=200, 
        init_iter=1000, 
        seed=None, 
        subsample='fast', 
        verbose=0, 
        equal_l1norm_constraint=True, 
        xtol=1e-8, 
        ftol=1e-8, 
        **opt_kwargs
    ):
        self._assert_registered()
        if B is None:
            internal = True
            self._assert_registered_targets()
            B = self.B
        else:
            internal = False
        X, P, B = lsq_linear_decomposition(
            self.A, 
            B, 
            n_layers=n_layers, 
            mask=mask, 
            lbp=lbp, ubp=ubp, 
            max_iter=max_iter, 
            init_iter=init_iter, 
            seed=seed, 
            subsample=subsample, 
            equal_l1norm_constraint=equal_l1norm_constraint, 
            xtol=xtol, ftol=ftol, 
            lb=self.lb, ub=self.ub, 
            W=self.W, K=self.K, 
            baseline=self.baseline, 
            verbose=verbose, return_pred=True, 
            **opt_kwargs 
        )
        
        self.X = X
        self.P = P
        self.B = B
        if internal:
            return self
        return X, P, B
    
    def fit_underdetermined(
        self, 
        B=None,
        underdetermined_opt=None,
        l2_eps=1e-5,
        batch_size=1, 
        verbose=0,
        **opt_kwargs
    ):
        self._assert_registered()
        if B is None:
            internal = True
            self._assert_registered_targets()
            B = self.B
        else:
            internal = False
        assert self.underdetermined, "System must be underdetermined."
        
        X, B = lsq_linear_underdetermined(
            self.A, 
            B, 
            lb=self.lb, ub=self.ub, 
            W=self.W, K=self.K, 
            baseline=self.baseline, 
            batch_size=batch_size, 
            verbose=verbose, return_pred=True, 
            underdetermined_opt=underdetermined_opt, 
            l2_eps=l2_eps,
            **opt_kwargs
        )
        
        self.X = X
        self.B = B
        if internal:
            return self
        return X, B
    
    def minimize_variance(
        self,
        B=None,
        Epsilon=None,
        batch_size=1, 
        verbose=0, 
        l2_eps=1e-5, 
        L1=None, 
        l1_eps=1e-5, 
        norm=None,
        **opt_kwargs
    ):
        self._assert_registered()
        if Epsilon is None:
            Epsilon = self.Epsilon

        if B is None:
            internal = True
            self._assert_registered_targets()
            B = self.B
        else:
            internal = False
        
        X, B, Bvar = lsq_linear_minimize(
            self.A, B, Epsilon, 
            lb=self.lb, ub=self.ub, 
            W=self.W, K=self.K, 
            baseline=self.baseline, 
            batch_size=batch_size, 
            verbose=verbose, return_pred=True, 
            l2_eps=l2_eps, 
            l1_eps=l1_eps, L1=L1, 
            norm=norm, 
            **opt_kwargs
        )
        
        self.X = X
        self.B = B
        self.Bvar = Bvar
        if internal:
            return self
        return X, B, Bvar
    
    # def substitute(self, C, **kwargs):
    #     self._assert_registered()
    #     pass
    
    ### scoring methods
    
    # r2-score, residuals, relative score, cosine similarity, etc.
        
    ### plotting functions
    
    def sources_plot(self, ax=None, colors=None, labels=None, **kwargs):
        self._assert_registered()
        return _simple_plotting_function(
            self.sources_domain, 
            self.sources, 
            labels=(self.sources_labels if labels is None else labels),
            colors=colors, ax=ax, **kwargs
        )
    
    def filter_plot(self, ax=None, colors=None, labels=None, **kwargs):
        return _simple_plotting_function(
            self.domain, self.filters, 
            labels=(self.labels if labels is None else labels), 
            colors=colors, ax=ax, **kwargs)
    
    # def simplex_plot(self):
    #     pass
    
    # def gamut_plot(self):
    

def _simple_plotting_function(x, ys, labels=None, colors=None, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
        
    if isinstance(x, Number):
        x = np.arange(ys.shape[-1])
        
    if colors is None:
        colors = sns.color_palette('rainbow', ys.shape[0])
        
    if labels is None:
        labels = np.arange(ys.shape[0])

    for label, y, color in zip(labels, ys, colors):
        kwargs['label'] = label
        kwargs['color'] = color
        ax.plot(x, y, **kwargs)
    
    return ax
        
    