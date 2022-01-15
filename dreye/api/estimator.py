"""Object-oriented Estimator class for all fitting procedures
"""

from itertools import combinations
from numbers import Number
import numpy as np
from scipy.special import comb
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from dreye.api.barycentric import barycentric_dim_reduction, cartesian_to_barycentric
from dreye.api.capture import calculate_capture
from dreye.api.convex import get_P_from_A, in_hull, in_hull_from_A, range_of_solutions
from dreye.api.domain import equalize_domains
from dreye.api.optimize.lsq_linear import lsq_linear, lsq_linear_adaptive, lsq_linear_decomposition, lsq_linear_excitation, lsq_linear_minimize, lsq_linear_underdetermined
from dreye.api.optimize.lsq_nonlinear import lsq_nonlinear
from dreye.api.plotting.basic import hull_outline, simple_plotting_function, vectors_plot
from dreye.api.plotting.simplex_plot import plot_simplex
from dreye.api.project import alpha_for_B_with_P
from dreye.api.sampling import sample_in_hull
from dreye.api.utils import check_bounds, l1norm, linear_transform
from dreye.api.metrics import compute_gamut


class ReceptorEstimator:
    """
    Core class in *drEye* for analyzing and fitting to capture values given 
    a set of receptor filters and a set of stimulation sources that comprise 
    an experimental system.
    
    Parameters
    ----------
    filters : ndarray of shape (n_filters, n_domain)
        Filter functions for each receptor type.
    domain : float or ndarray of shape (n_domain), optional
        Domain array specifying the domain covered by the filter function. 
        If float, it is assumed to be the step size in the domain (e.g. dx=1nm for wavelengths). 
        If array-like, it is assumed to be an ascending array where each element is the
        value in domain coordinates (e.g. [340, 350, ..., 670, 680]nm for wavelengths).
        By default 1.0.
    filters_uncertainty : ndarray of shape (n_filters, n_domain) or (n_samples, n_filters, n_domain), optional
        The standard deviation for each element in `filters`. 
        If the array has three dimensions, the array is assumed to correspond to 
        samples from a distribution of filters.
        By default None.
    w : float or ndarray of shape (n_filters), optional
        Importance weighting for each filter during fitting procedures, by default 1.0.
    labels : ndarray of shape (n_filters), optional
        The label names of each filter, by default None
    K : float or ndarray of shape (n_filters) or (n_filters, n_filters), optional
        The adaptational state of each receptor type, by default 1.0.
    baseline : float or ndarray of shape (n_filters), optional
        The baseline capture value of each receptor type, by default 0.0
    sources : ndarray of shape (n_sources, n_domain), optional
        The stimulation sources' normalized (excitation/spectral) distribution, by default None
    lb : float or ndarray of shape (n_sources), optional
        The lower bound value for each stimulation source, by default set to 0.
    ub : float or ndarray of shape (n_sources), optional
        The upper bound value for each stimulation source, by default set to inf
    sources_labels : ndarray of shape (n_sources), optional
        The label names of each source, by default None
    """
    
    def __init__(
        self, 
        filters,
        domain=1.0,
        filters_uncertainty=None,
        w=1.0,
        labels=None, 
        K=1.0, 
        baseline=0.0, 
        sources=None, 
        lb=None, ub=None, sources_labels=None
    ):
        self.filters = np.asarray(filters)
        self.domain = (domain if isinstance(domain, Number) else np.asarray(domain))
        self.filters_uncertainty = (
            filters_uncertainty if filters_uncertainty is None 
            else np.asarray(filters_uncertainty)
        )
        self.w = self.W = np.broadcast_to(w, (self.filters.shape[0],))  # initially set capital W to the same
        self.labels = (np.arange(self.filters.shape[0]) if labels is None else np.asarray(labels))
        
        self.register_adaptation(K)
        self.register_baseline(baseline)
        
        if sources is not None:
            self.register_system(sources, domain=domain, lb=lb, ub=ub, labels=sources_labels)
        
    def register_uncertainty(self, filters_uncertainty):
        """Register a new filter uncertainty function

        Parameters
        ----------
        filters_uncertainty : ndarray of shape (n_filters, n_domain), optional
            The standard deviation for each element in `filters`, by default None.
        """
        self.filters_uncertainty = (
            filters_uncertainty 
            if filters_uncertainty is None 
            else np.asarray(filters_uncertainty)
        )
    
    ### register state of adaptation and baseline
    
    def register_adaptation(self, K):
        """Register a new adaptational state.

        Parameters
        ----------
        K : float or ndarray of shape (n_filters) or (n_filters, n_filters), optional
            The adaptational state of each receptor type.
        """
        # K being one d or two d
        self.K = np.atleast_1d(K)
    
    def register_background_adaptation(self, background, domain=None, add_baseline=True, add=False):
        """Register a new adaptational state using a background excitation function.

        Parameters
        ----------
        background : ndarray of shape (n_domain)
            Background excitation function.
        domain : [type], optional
            If given, this is the domain for `background`, 
            and domains between `filters` and `background` will be equalized.
            By default None.
        add_baseline : bool, optional
            Whether to add the `baseline` value the adaptational state, by default True.
        add : bool, optional
            Whether to add this adaptational state to the current adaptational state, by default False.
        """
        qb = self.capture(background, domain=domain)
        if add_baseline:
            qb = qb + self.baseline
        if add:
            self.K = self.K + 1/qb
        else:
            self.K = 1/qb
    
    def register_baseline(self, baseline):
        """Register a new baseline value

        Parameters
        ----------
        baseline : float or ndarray of shape (n_filters)
            The baseline capture value of each receptor type.
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
        """Calculate the absolute excitation/light-induced capture.

        Parameters
        ----------
        signals : ndarray of shape (n_signals, n_domain)
            The signals exciting all photoreceptor types.
        domain : ndarray of shape (n_domain), optional
            If given, this is the domain for `signals`, 
            and domains between `filters` and `signals` will be equalized.
            By default None.

        Returns
        -------
        Q : ndarray of shape (n_filters, n_signals)
            The absolute light-induced capture.
            
        Notes
        -----
        The absolute excitation/light-induced capture is calculated 
        using the filter functions :math:`S(\lambda)` and the 
        signals :math:`I(\lambda)`:
         
        .. math:: 
        
            Q = \int_{\lambda} S(\lambda)I(\lambda) d\lambda
        """
        signals = np.asarray(signals)
        domain, filters, signals = self._check_domain(domain, signals)
        # filterdim x signaldim
        return calculate_capture(filters, signals, domain=domain)
    
    def uncertainty_capture(self, signals, domain=None):
        """Calculate the variance of the absolute excitation/light-induced capture.

        Parameters
        ----------
        signals : ndarray of shape (n_signals, n_domain)
            The signals exciting all photoreceptor types.
        domain : ndarray of shape (n_domain), optional
            If given, this is the domain for `signals`, 
            and domains between `filters` and `signals` will be equalized.
            By default None.

        Returns
        -------
        Epsilon : ndarray of shape (n_filters, n_signals)
            The variance of the absolute light-induced capture.

        Raises
        ------
        ValueError, AssertionError
            If the filters_uncertainty hasn't been previously defined or is not formatted correctly.
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
        """Calculate the relative capture given a signal

        Parameters
        ----------
        signals : ndarray of shape (n_signals, n_domain)
            The signals exciting all photoreceptor types.
        domain : ndarray of shape (n_domain), optional
            If given, this is the domain for `signals`, 
            and domains between `filters` and `signals` will be equalized.
            By default None.

        Returns
        -------
        B : ndarray of shape (n_filters, n_signals)
            The relative total capture.
            
        Notes
        -----
        The relative total capture is calculated from the absolute capture  
        :math:`Q` and with the adaptational state :math:`K` and 
        baseline capture :math:`baseline`:
         
        .. math:: 
        
            B = K (Q + baseline)
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
        P = self._get_P_from_A(relative=relative, bounded=True)
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
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        return hasattr(self, 'A') and hasattr(self, 'Epsilon')
    
    def _assert_registered(self):
        assert self.registered, "No system has yet been registered."
    
    @property
    def underdetermined(self):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
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
            
    def _get_P_from_A(self, relative=True, bounded=None, remove_zero=False):
        """[summary]

        Parameters
        ----------
        relative : bool, optional
            [description], by default True
        bounded : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        P = get_P_from_A(
            self.A, self.lb, self.ub, 
            K=(self.K if relative else None), 
            baseline=(self.baseline if relative else None), 
            bounded=bounded
        )
        if remove_zero:
            P = P[~np.all(P == 0, axis=-1)]
        return P
        
    def sample_in_gamut(self, *args, **kwargs):
        """Alias for `sample_in_hull`. 
        """
        return self.sample_in_gamut(*args, **kwargs)
    
    def sample_in_hull(self, n=10, seed=None, engine=None, l1=None, relative=True):
        """[summary]

        Parameters
        ----------
        n : int, optional
            [description], by default 10
        seed : [type], optional
            [description], by default None
        engine : [type], optional
            [description], by default None
        l1 : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        self._assert_registered()
        # check if bounded
        bounded = np.all(np.isfinite(self.ub))
        P = self._get_P_from_A(relative=relative, bounded=bounded)
        if l1 is None:
            return sample_in_hull(P, n, seed=seed, engine=engine)
        else:
            # sample within a simplex
            l1 = P.sum(axis=-1)
            # remove zero intensity
            P = P[l1 != 0]
            # reduce to barycentric coordinates and sample within that hull
            P = barycentric_dim_reduction(P)
            X = sample_in_hull(P, n, seed=seed, engine=engine)
            return cartesian_to_barycentric(X, L1=l1)
        
    def gamut_l1_scaling(self, *args, **kwargs):
        """Alias for `hull_l1_scaling`
        """
        return self.hull_l1_scaling(*args, **kwargs)
        
    def hull_l1_scaling(self, B, relative=True):
        """
        Scale `B` to fit within the hull of the system. 
        This is equivalent to intensity scaling of an image for color science.

        Parameters
        ----------
        B : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        if relative:
            A, baseline = linear_transform(self.A, self.K, self.baseline)
        else:
            A = self.A
            baseline = 0
        
        B = B - baseline
        # scale intensities to span gamut
        Amax = A * self.ub
        amax = np.min(np.max(Amax, axis=-1))
        bmax = np.max(B)
        return B * amax / bmax + baseline
    
    def gamut_dist_scaling(self, *args, **kwargs):
        """Alias for `hull_dist_scaling`
        """
        return self.hull_dist_scaling(*args, **kwargs)
    
    def hull_dist_scaling(self, B, neutral_point=None, relative=True):
        """
        Scale `B` within L1-normalized simplex plot to fit within hull of system.
        This is equivalent to saturation scaling of an image for color science.

        Parameters
        ----------
        B : [type]
            [description]
        neutral_point : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        if self.in_hull(B, normalized=True).all():
            return B.copy()
        
        if neutral_point is None:
            neutral_point = np.ones(self.filters.shape[0])
        neutral_point = np.atleast_2d(neutral_point)
        
        P = self._get_P_from_A(relative=relative, bounded=True, remove_zero=True)
        assert np.all(P >= 0)  # must be non-negative
        # replace zero point with neutral point
        B = B.copy()
        zero_rows = (B == 0).all(-1)
        B[zero_rows] = neutral_point
        L1 = l1norm(B, axis=-1) 

        center = barycentric_dim_reduction(neutral_point)
        
        baryP = barycentric_dim_reduction(P) - center
        baryB = barycentric_dim_reduction(B) - center
        
        # find alphas 
        if baryP.shape[1] == 1:
            assert np.any(baryP < 0) and np.any(baryP > 0)
            baryP_ = np.array([np.min(baryP), np.max(baryP)])
            alphas = baryP_[:, None, None] / baryB[None, ...]
            alphas[alphas <= 0] = np.nan
            alpha = np.nanmin(alphas)
        else:
            assert np.all(
                in_hull(baryP, np.zeros(baryP.shape[1]))
            ), "neutral point is not in hull."
            hull = ConvexHull(baryP)
            alphas = alpha_for_B_with_P(baryB, hull.equations)
            alpha = np.nanmin(alphas)

        baryB_scaled = baryB * alpha + center

        B = cartesian_to_barycentric(baryB_scaled, L1)
        B[zero_rows] = 0
        return B
        
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
    
    @property
    def registered_targets(self):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        return hasattr(self, 'B')

    def _assert_registered_targets(self):
        assert self.registered_targets, "No target has yet been registered for fitting."
        
    def _assert_fitted(self):
        assert hasattr(self, 'X'), "Target have not been fitted to system yet."
    
    ### methods that require registered system
        
    def in_gamut(self, *args, **kwargs):
        """Alias for `in_hull`.
        """
        return self.in_hull(*args, **kwargs)
    
    def in_hull(self, B=None, relative=True, normalized=False):
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
            
        # normalized does it within the l1-normalized simplex
        if normalized:
            P = self._get_P_from_A(relative=relative, bounded=True, remove_zero=True)
            P = barycentric_dim_reduction(P)
            B = barycentric_dim_reduction(B)
            return in_hull(P, B, bounded=True)
        
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
        """
        Fitting source intensities given relative capture values.

        Parameters
        ----------
        B : [type], optional
            [description], by default None
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
        delta_radius=1e-4, 
        delta_norm1=1e-4,
        adaptive_objective="unity", 
        verbose=0, 
        scale_w=1.0,
        **opt_kwargs
    ):
        """[summary]

        Parameters
        ----------
        B : [type], optional
            [description], by default None
        neutral_point : [type], optional
            [description], by default None
        delta_radius : [type], optional
            [description], by default 1e-5
        delta_norm1 : [type], optional
            [description], by default 1e-5
        adaptive_objective : str, optional
            [description], by default "unity"
        scaled_w : float or ndarray of shape (2)
            [description], by default 1.
        verbose : int, optional
            [description], by default 0

        Returns
        -------
        [type]
            [description]
        """
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
            scale_w=scale_w,
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
        """[summary]

        Parameters
        ----------
        B : [type], optional
            [description], by default None
        n_layers : [type], optional
            [description], by default None
        mask : [type], optional
            [description], by default None
        lbp : int, optional
            [description], by default 0
        ubp : int, optional
            [description], by default 1
        max_iter : int, optional
            [description], by default 200
        init_iter : int, optional
            [description], by default 1000
        seed : [type], optional
            [description], by default None
        subsample : str, optional
            [description], by default 'fast'
        verbose : int, optional
            [description], by default 0
        equal_l1norm_constraint : bool, optional
            [description], by default True
        xtol : [type], optional
            [description], by default 1e-8
        ftol : [type], optional
            [description], by default 1e-8

        Returns
        -------
        [type]
            [description]
        """
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
        l2_eps=1e-4,
        batch_size=1, 
        verbose=0,
        **opt_kwargs
    ):
        """[summary]

        Parameters
        ----------
        B : [type], optional
            [description], by default None
        underdetermined_opt : [type], optional
            [description], by default None
        l2_eps : [type], optional
            [description], by default 1e-5
        batch_size : int, optional
            [description], by default 1
        verbose : int, optional
            [description], by default 0

        Returns
        -------
        [type]
            [description]
        """
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
        l2_eps=1e-4, 
        L1=None, 
        l1_eps=1e-4, 
        norm=None,
        **opt_kwargs
    ):
        """[summary]

        Parameters
        ----------
        B : [type], optional
            [description], by default None
        Epsilon : [type], optional
            [description], by default None
        batch_size : int, optional
            [description], by default 1
        verbose : int, optional
            [description], by default 0
        l2_eps : [type], optional
            [description], by default 1e-5
        L1 : [type], optional
            [description], by default None
        l1_eps : [type], optional
            [description], by default 1e-5
        norm : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
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
    
    # TODO
    # def substitute(self, C, **kwargs):
    #     self._assert_registered()
    #     pass
    
    ### scoring methods
    
    # TODO r2-score, residuals, relative score, cosine similarity, etc.
        
    ### plotting functions
    
    def sources_plot(self, ax=None, colors=None, labels=None, **kwargs):
        """[summary]

        Parameters
        ----------
        ax : [type], optional
            [description], by default None
        colors : [type], optional
            [description], by default None
        labels : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        self._assert_registered()
        return simple_plotting_function(
            self.sources_domain, 
            self.sources, 
            labels=(self.sources_labels if labels is None else labels),
            colors=colors, ax=ax, **kwargs
        )
    
    def filter_plot(self, ax=None, colors=None, labels=None, **kwargs):
        """[summary]

        Parameters
        ----------
        ax : [type], optional
            [description], by default None
        colors : [type], optional
            [description], by default None
        labels : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        return simple_plotting_function(
            self.domain, self.filters, 
            labels=(self.labels if labels is None else labels), 
            colors=colors, ax=ax, **kwargs)
    
    def simplex_plot(
        self, 
        B=None,
        domain=None, 
        ax=None,
        cmap='rainbow',
        gradient_line_kws=None, 
        point_scatter_kws=None,
        hull_kws=None, 
        impure_lines=False, 
        add_center=True, 
        add_grid=False,
        add_hull=True,
        relative=True,
        domain_line=True,
        labels=None,
        label_size=16,
        **kwargs
    ):
        """
        Plot the simplex plot (aka chromaticity diagram) of all filter types. 
        This function only works for two, three, and four filter types (i.e. 1 < n_filters < 5). 

        Parameters
        ----------
        B : [type], optional
            [description], by default None
        domain : [type], optional
            [description], by default None
        ax : [type], optional
            [description], by default None
        cmap : str, optional
            [description], by default 'rainbow'
        gradient_line_kws : [type], optional
            [description], by default None
        point_scatter_kws : [type], optional
            [description], by default None
        hull_kws : [type], optional
            [description], by default None
        impure_lines : bool, optional
            [description], by default False
        add_center : bool, optional
            [description], by default True
        add_grid : bool, optional
            [description], by default False
        add_hull : bool, optional
            [description], by default True
        relative : bool, optional
            [description], by default True
        domain_line : bool, optional
            [description], by default True
        labels : [type], optional
            [description], by default None
        label_size : int, optional
            [description], by default 16

        Returns
        -------
        [type]
            [description]
        """
        gradient_line_kws = ({} if gradient_line_kws is None else gradient_line_kws)
        point_scatter_kws = ({} if point_scatter_kws is None else point_scatter_kws)
        hull_kws = ({} if hull_kws is None else hull_kws)
        labels = (self.labels if labels is None else labels)
        
        if B is None and self.registered_targets:
            B = self.B
        
        n = self.filters.shape[0]
        assert n in {2, 3, 4}, "Simplex plots only works for tri- or tetrachromatic animals."
        
        domain_ = self.domain
        if isinstance(domain_, Number):
            domain_ = np.arange(0, self.filters.shape[-1] * domain_, domain_)
        
        if domain is None:
            domain = 10  # default value
        # if numeric it indicates spacing
        if isinstance(domain, Number):
            dmin = np.min(domain_)
            dmax = np.max(domain_)
            # earliest round number
            dmin = np.ceil(dmin / domain) * domain
            domain = np.arange(dmin, dmax, domain)
            
        # dirac delta functions for perfect excitation
        signals = np.eye(self.filters.shape[-1])
        if relative:
            optimal = self.relative_capture(signals)
        else:
            optimal = self.capture(signals)
            
        ax = plot_simplex(
            n, ax=ax, labels=labels, label_size=label_size
        )
        
        if domain_line:
            points = optimal[
                np.argmin(
                    np.abs(
                        domain_[:, None] - domain
                    ), axis=0
                )
            ]
            
            gradient_line_kws['cmap'] = cmap
            
            ax = plot_simplex(
                n, 
                ax=ax,
                gradient_line=points, 
                gradient_color=domain, 
                lines=False,
                gradient_line_kws=gradient_line_kws, 
            )
            
            # add impure lines that connect non-adjacent sensitivities in the plot
            # also known as non-spectral lines in color science
            if impure_lines and (n != 2):
                qmaxs = barycentric_dim_reduction(
                    points[np.argmax(points, 0)]
                )
                for idx, jdx in combinations(range(n), 2):
                    # sensitivities next to each other
                    if abs(idx - jdx) == 1:
                        continue
                    _qs = qmaxs[[idx, jdx]].T
                    ax.plot(
                        *_qs,
                        color='black', 
                        linestyle='--', 
                        alpha=0.8
                    )

        if add_center:
            plot_simplex(
                n, 
                ax=ax, 
                points=np.ones((1, self.filters.shape[0])), 
                point_colors='gray', 
                point_scatter_kws=point_scatter_kws, 
                lines=False
            )

        if add_grid and (n != 2):
            # add lines that connect edges to center
            for i in range(n):
                x_ = np.zeros((2, n))
                x_[0, i] = 1
                x_[1] = 1
                x_[1:, i] = 0
                xs_ = barycentric_dim_reduction(x_)
                ax.plot(*xs_.T, color='gray', linestyle='--', alpha=0.5)
                
        if add_hull:
            
            P = self._get_P_from_A(relative=relative, bounded=True, remove_zero=True)
            default_kws = {
                'color':'lightgray', 
                'edgecolor': 'gray', 
                'linestyle':'--',
            }
            default_kws.update(hull_kws)
            hull_kws = default_kws
            
            if n <= 3:
                hull_kws.pop('edgecolor')
            
            ax = plot_simplex(
                n, 
                hull=P, 
                hull_kws=hull_kws, 
                ax=ax, 
                lines=False
            )
            
        if B is not None:
            ax = plot_simplex(
                n, 
                ax=ax, 
                points=B, 
                lines=False, 
                point_scatter_kws=kwargs
            )
        
        return ax
    
    def gamut_plot(self, *args, **kwargs):
        """Alias for `hull_plot`.
        """
        return self.hull_plot(*args, **kwargs)
    
    def hull_plot(
        self, B=None, 
        axes=None, labels=None, 
        sources_labels=None,
        colors=None, ncols=None, 
        fig_kws=None, relative=True, 
        sources_vectors=True,
        hull_kws=None,
        vectors_kws=None,
        **kwargs
    ):
        """Plot the hull of the system on all possible pairs of filter combinations (2D-plot).

        Parameters
        ----------
        B : [type], optional
            [description], by default None
        axes : [type], optional
            [description], by default None
        labels : [type], optional
            [description], by default None
        sources_labels : [type], optional
            [description], by default None
        colors : [type], optional
            [description], by default None
        ncols : [type], optional
            [description], by default None
        fig_kws : [type], optional
            [description], by default None
        relative : bool, optional
            [description], by default True
        sources_vectors : bool, optional
            [description], by default True
        hull_kws : [type], optional
            [description], by default None
        vectors_kws : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        self._assert_registered()
        
        if B is None and self.registered_targets:
            B = self.B
        
        nfilters = self.filters.shape[0]
        ncombos = comb(nfilters, 2, exact=True)
        
        if axes is None:
            fig_kws = ({} if fig_kws is None else fig_kws)
            kws = dict(
                # defaults
                sharex=False, 
                sharey=False, 
            )
            kws.update(fig_kws)
            
            fig, axes = plt.subplots(
                ncols=(1 if ncols is None else ncols), 
                nrows=(ncombos if ncols is None else int(np.ceil(ncombos/ncols))), 
                **kws  
            )
            axes = np.atleast_1d(axes)
            axes = axes.ravel()
        else:
            fig = plt.gcf()
            assert len(axes) == ncombos, f"the number of axes {len(axes)} is not equal to the number of combinations {ncombos}"
        
        labels = (self.labels if labels is None else labels)
        sources_labels = (self.sources_labels if sources_labels is None else sources_labels)
        
        P = self._get_P_from_A(relative=relative, bounded=True)
        
        # only one source active - and flip since get_P_from_A has sources in reversed order
        if relative:
            singleP = self.system_relative_capture(np.eye(self.sources.shape[0]) * self.ub)
            offsets = self.system_relative_capture(np.eye(self.sources.shape[0]) * self.lb)
        else:
            singleP = self.system_capture(np.eye(self.sources.shape[0]) * self.ub)
            offsets = self.system_capture(np.eye(self.sources.shape[0]) * self.lb)
        
        hull_kws = ({} if hull_kws is None else hull_kws)
        vectors_kws = ({} if vectors_kws is None else vectors_kws)
        hull_kws['zorder'] = hull_kws.get('zorder', 1)
        vectors_kws['zorder'] = vectors_kws.get('zorder', 1.5)
        kwargs['zorder'] = kwargs.get('zorder', 2)
        
        for idx, (xidx, yidx) in enumerate(combinations(range(nfilters), 2)):
            ax = axes[idx]
            # hull and vectors plot
            P_ = P[:, [xidx, yidx]]
            singleP_ = singleP[:, [xidx, yidx]]
            offsets_ = offsets[:, [xidx, yidx]]
            
            hull_outline(P_, ax=ax, **hull_kws)
            if sources_vectors:
                vectors_plot(
                    singleP_, 
                    offsets=offsets_,
                    ax=ax, colors=colors, labels=sources_labels, **vectors_kws
                )
                
            if B is not None:
                ax.scatter(B[:, xidx], B[:, yidx], **kwargs)
                
            ax.set_xlabel(labels[xidx])
            ax.set_ylabel(labels[yidx])

        return fig, axes
        
    