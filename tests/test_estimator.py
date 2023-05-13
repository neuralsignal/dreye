
from . import context

import os
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import dreye

from pytest import raises


# TODO implement unit tests for individual functions instead of testing the tutorials

 
def test_introduction():
    wls = np.arange(300, 700, 1)
    # peaks of the sensitivities
    peaks = np.array([420, 535])
    # opsin template
    # an empty axis is added to the peaks so that the array object has dimension (opsin_type x wavelengths)
    filters = dreye.govardovskii2000_template(wls, peaks[:, None])

    # %% [markdown]
    # Next, we define our LEDs. Here, we will use a simple Gaussian distribution to define our LEDs, but any spectral distribution is valid as long as it is formated as a `numpy.ndarray` object with dimension (led_type x wavelengths). Importantly, we normalize the array by the estimated integral so that the array is in units $1/nm$.

    # %%
    led_peaks = np.array([410, 480, 550])
    sources = norm.pdf(wls, loc=led_peaks[:, None], scale=20)
    sources = sources / dreye.integral(sources, wls, axis=-1, keepdims=True)
    assert isinstance(sources, np.ndarray)
    assert np.allclose(sources.sum(axis=-1), 1)

    # %% [markdown]
    # ## The `ReceptorEstimator` object
    # 
    # The main object used in *drEye* is the `ReceptorEstimator` object. This object stores the filters and sources, as well as other information and contains all the necessary methods to assess the gamut of the system and fit arbitrary spectral sensitivities. 
    # To initial this object, we need to supply the filters and wavelength domain array. We may also supply the LED spectra and the intensity bounds, or register them separately using the `register_system` method (see API reference for details). 

    # %%
    est = dreye.ReceptorEstimator(
        # filters array
        filters, 
        # wavelength array
        domain=wls, 
        # labels for each photoreceptor type (optional)
        labels=['S', 'L'], 
        # LED array, optional
        sources=sources, 
        # lower bound of each LED, optional
        lb=np.zeros(3), 
        # upper bound for each LED, optional - if None, then the upper bound is infinity
        ub=np.ones(3) * 0.1, 
        # labels for sources, optional
        sources_labels=['V', 'C', 'G']
    )

    # %% [markdown]
    # ### Plotting the sensitivity filters and LED sources

    # %%
    filters_colors = ['gray', 'black']
    sources_colors = ['violet', 'cyan', 'green']

    ax1 = est.filter_plot(colors=filters_colors)
    ax2 = plt.twinx(ax1)
    est.sources_plot(colors=sources_colors, ax=ax2)

    ax1.set_xlabel('wavelength (nm)')
    ax1.set_ylabel('relative sensitivity (a.u.)')
    ax2.set_ylabel('relative intensity (1/nm)')

    ax1.legend(loc=2)
    ax2.legend(loc=1)
    
    plt.close()

    # %% [markdown]
    # ### Plotting the gamut

    # %%
    fig, axes = est.gamut_plot(colors=sources_colors)
    fig.legend()
    plt.close()

    # %% [markdown]
    # ### Calculating capture
    # 
    # Here, we show how to calculate the capture of a photoreceptor or any sensory system that linearly integrates a stimulus according to its filtering properties.
    # 
    # The formula for calculating the absolute light-induced capture is as follows:
    # $$
    # Q = \int_{\lambda} S(\lambda)I(\lambda) d\lambda
    # $$
    # 
    # where $Q$ is the capture, $S(\lambda)$ is the sensory filter, and $I(\lambda)$ is the stimulus across wavelengths.
    # 
    # To calculate this capture, we use the `capture` method.

    # %%
    # random light distribution
    rng = np.random.default_rng(10)
    random_source = rng.random(wls.size) / wls.size * 0.5

    Q = est.capture(random_source)

    # %%
    # location of capture
    # relative should be set to False since the capture value is the absolute capture value and not the relative capture value (see later)
    fig, axes = est.gamut_plot(np.atleast_2d(Q), colors=sources_colors, color='black', relative=False)
    fig.legend()
    
    plt.close()

    # %%
    # point within the isoluminant plane
    ax = est.simplex_plot(np.atleast_2d(Q), color='black')
    plt.close()

    # %% [markdown]
    # ### Forms of adaptation and the baseline
    # 
    # Photoreceptor usually adapt to the background light. Here, we go over three different methods to define the adaptional state of a receptor and show how to define a baseline capture for the receptor (i.e. a bias/offset).
    # 
    # The relative capture described the capture after incorporating the adaptational state and baseline:
    # $$
    # q = K (Q + baseline)
    # $$
    # 
    # For a photoreceptor, background adaptation is calculated by first calculating the capture of the background and then adding the baseline:
    # $$
    # q = (Q + baseline) / (Q_b + baseline)
    # $$

    # %%
    assert np.allclose(est.K, 1)
    assert np.allclose(est.baseline, 0)
    # register a baseline capture
    est.register_baseline(1e-3)
    print(est.baseline)
    assert np.allclose(est.baseline, 1e-3)
    # setting K directly for each photoreceptor
    est.register_adaptation(np.array([1, 1.5]))
    print(est.K)
    assert np.allclose(est.K, np.array([1, 1.5]))
    
    # setting K according to a background signal (important is to set the baseline beforehand)
    background = norm.pdf(wls, loc=450, scale=100)  # flattish background
    est.register_background_adaptation(background)
    print(est.K)
    
    # setting K according to intensities of our sources
    source_bg = np.array([0.01, 0.015, 0.018])
    est.register_system_adaptation(source_bg)
    print(est.K)

    # %% [markdown]
    # So now we have our adaptational state set to specific intensities of our LEDs. What would be the relative capture if we show our background light, while the photoreceptors are adapted to our LED background intensities:

    # %%
    est.relative_capture(background)

    # %% [markdown]
    # We can also calculate the relative capture (or capture) given specific LED intensities using the `system_relative_capture` or `system_capture` method:

    # %%
    B = est.system_relative_capture(source_bg)
    print(B)  # this should be one
    assert np.allclose(B, 1)
    print(est.system_capture(source_bg))  # this is the absolute light-induced capture for the background intensities

    # let's sample various intensities and plot the captures in the gamut diagram
    # since we bound our samples by our intensity bounds, all samples should be within the gamut
    rng = np.random.default_rng(10)
    X = rng.random((100, 3)) * 0.1  # within the bounds of our intensities
    B = est.system_relative_capture(X)

    fig, axes = est.gamut_plot(B, colors=sources_colors, color='gray', alpha=0.5)
    plt.close()

    # %% [markdown]
    # ## Analyzing and fitting many capture values
    # 
    # * Is a capture value within the gamut of the system?
    # * Sampling values within the hull/gamut of the system (and at a specific intensity/l1-norm)
    # * Finding the optimal LED intensities for a desired spectral distribution (or target capture values)

    # %%
    # testing if a capture value is within the hull/gamut of the stimulation system
    print(est.in_hull(B).all())  # all previously sampled values should be within the hull
    assert est.in_hull(B).all()

    rng = np.random.default_rng(1)
    Bnew = rng.random((10, 2)) * 8
    inhull = est.in_hull(Bnew)
    print(inhull)  # not all are within the hull
    hull_colors = ['black' if x else 'tomato' for x in inhull]
    fig, axes = est.gamut_plot(Bnew, colors=sources_colors, c=hull_colors, alpha=0.5)
    plt.close()

    # %% [markdown]
    # ### Sampling within the hull
    # 
    # Unlike previously where we sampled uniformly within LED space, we can also sample within the hull/gamut directly using the `sample_in_hull` method.
    # This can help reduce the number of gaps and clumps in capture space (lower discrepency).

    # %%
    # sampling in LED space
    rng = np.random.default_rng(10)
    X = rng.random((50, 3)) * 0.1  # within the bounds of our intensities
    B1 = est.system_relative_capture(X)
    assert np.all(est.in_hull(B1))

    # using a QMC engine
    B2 = est.sample_in_hull(50, seed=10, engine='Halton')
    assert np.all(est.in_hull(B2))

    fig, axes = est.gamut_plot(B1, colors=sources_colors, c='cyan', alpha=0.5)
    fig, axes = est.gamut_plot(B2, colors=sources_colors, c='tomato', alpha=0.5, axes=axes)
    plt.close()

    # %% [markdown]
    # ### Fitting target capture values
    # 
    # Next, we will show the basics of finding the optimal intensity vectors for our LEDs for given capture values. More detail into different fitting procedures can be found in the API reference and other tutorials. The basic method for fitting is the `fit` method.

    # %%
    # Fitting points sampled in hull
    X, Bhat = est.fit(B2, verbose=1)

    # all points can be fit perfectly
    print(np.allclose(B2, Bhat))
    print(np.abs(B2 - Bhat))
    assert np.all(np.abs(B2 - Bhat) < 1e-4)

    # %%
    # Fitting with some points outside the hull
    rng = np.random.default_rng(1)
    Bnew = rng.random((10, 2)) * 8
    inhull = est.in_hull(Bnew)
    print(inhull)  # not all are within the hull
    colors = sns.color_palette('tab10', len(Bnew))

    # Fitting points
    X, Bnewhat1 = est.fit(Bnew, verbose=1)
    
    # important test
    assert np.allclose(Bnewhat1, est.system_relative_capture(X)), 'either system relative capture is inaccurate or predicted capture calculation is wrong'
    
    # Using different models - gives a different result for out-of-gamut points - see API for details
    X, Bnewhat2 = est.fit(Bnew, model='poisson', verbose=1)
    X, Bnewhat3 = est.fit(Bnew, model='excitation', verbose=1, solver='ECOS')

    fig, axes = est.gamut_plot(Bnew, colors=sources_colors, c=colors, alpha=0.3)
    fig, axes = est.gamut_plot(Bnewhat1, colors=sources_colors, c=colors, marker='x', alpha=1, axes=axes)
    fig, axes = est.gamut_plot(Bnewhat2, colors=sources_colors, c=colors, marker='s', alpha=1, axes=axes)
    fig, axes = est.gamut_plot(Bnewhat3, colors=sources_colors, c=colors, marker='+', alpha=1, axes=axes)
    plt.close()


def test_uncertainty():
    # %%
    filters_colors = ['gray', 'black']
    sources_colors = ['violet', 'cyan', 'green']

    # wavelength range
    wls = np.arange(300, 700, 1)
    # peaks of the sensitivities
    peaks = np.array([420, 535])
    # opsin template
    # an empty axis is added to the peaks so that the array object has dimension (opsin_type x wavelengths)
    filters = dreye.govardovskii2000_template(wls, peaks[:, None])

    led_peaks = np.array([410, 480, 550])
    sources = norm.pdf(wls, loc=led_peaks[:, None], scale=20)
    sources = sources / dreye.integral(sources, wls, axis=-1, keepdims=True)

    est = dreye.ReceptorEstimator(
        # filters array
        filters, 
        ### ADDING FILTER UNCERTAINTY
        ### In this case it scales with the value of the filter function (heteroscedastic)
        filters_uncertainty=filters * 0.1,
        # wavelength array
        domain=wls, 
        # labels for each photoreceptor type (optional)
        labels=['S', 'L'], 
        # LED array, optional
        sources=sources, 
        # lower bound of each LED, optional
        lb=np.zeros(3), 
        # upper bound for each LED, optional - if None, then the upper bound is infinity
        ub=np.ones(3) * 0.1, 
        # labels for sources, optional
        sources_labels=['V', 'C', 'G'], 
        baseline=1e-3
    )

    # setting K according to intensities of our sources
    source_bg = np.array([0.01, 0.015, 0.018])
    est.register_system_adaptation(source_bg)

    # %% [markdown]
    # The attribute `Epsilon` defines the variance in capture for each source and filter. `A` defines the mean light-induced capture for each source and filter.

    # %%
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.heatmap(
        est.Epsilon, ax=ax1, 
        vmin=0, cmap='viridis'
    )
    sns.heatmap(
        est.A, ax=ax2,
        vmin=0, cmap='viridis' 
    )
    plt.close()

    # %% [markdown]
    # ## Underdetermined System
    # 
    # Since this is an underdetermined system (more sources than filters), there can exist multiple solutions for fitting to capture values that are within the gamut of the system. Let's first sample points within the gamut of the system.

    # %%
    # Fitting with some points outside the hull
    B = est.sample_in_hull(4, seed=2)

    # Fitting points
    Xhat, Bhat = est.fit(B, verbose=1)

    fig, axes = est.gamut_plot(B, colors=sources_colors, c='gray', alpha=0.3)
    plt.close()

    # %% [markdown]
    # Using the `range_of_solutions` method we can find the range of solutions that fit each of these five examples.

    # %%
    xmin, xmax = est.range_of_solutions(B)

    # if we specificy a number of points that we want within that range the function also returns those intensities for our LEDs
    xmin, xmax, X = est.range_of_solutions(B, n=10)
    X = np.stack(X)

    # for each of our five samples we have ten intensity combinations that are evenly space along the n
    print(X.shape)

    # %% [markdown]
    # For each of our five samples we have ten intensity combinations that are evenly spaced along the line of possible solutions. But which intensity set should we choose for each of our five samples? In this case, we may wish to choose a set of intensities that minimizes the expected variance in the capture values given our uncertainty of our filter functions. To do this, we can use the `minimize_variance` method. NB: This method generally works well if the system is underdetermined and the points lie within the gamut of the system. Otherwise, it does not make much sense to use this method.

    # %%
    # the last object returned is the variance in the capture, which we will ignore for now.
    Xhat2, Bhat2, _ = est.minimize_variance(B)

    ### -- plotting --

    # can compare the intensities found to our previous result by plotting both cases
    from dreye.api.plotting.simplex_plot import plot_simplex

    ax = None
    # gradient line showing all possible solutions with the color indicating the overall intensity
    for idx, X_ in enumerate(X):
        ax = plot_simplex(
            n=3, 
            gradient_line=X_, 
            ax=ax, lines=False, 
            gradient_line_kws={'add_colorbar': False, 'vmin': 0, 'zorder': 2}
        )

    # with variance minization
    ax = plot_simplex(
        n=3, 
        points=Xhat2, 
        point_scatter_kws={'marker': 'x', 'color': 'tomato', 'zorder': 2.6}, 
        labels=est.sources_labels, ax=ax
    )
    # standard approach - no variance minimization
    ax = plot_simplex(
        n=3, 
        points=Xhat, 
        point_scatter_kws={'marker': 'o', 'color': 'cyan', 'zorder': 2.5}, 
        ax=ax, lines=False
    )
    
    plt.close()

    # %% [markdown]
    # As we can see in the plot above, sometimes the standard approach gives the same result, but other times the variance minimization approach differs siicantly. In our case, we defined our uncertainty according to the observed heteroscedasticity in photoreceptor noise. In this case, the variance minimization approach will ensure that the samples drawn in capture space have similar distances to each other when compared to the resulting fitted instensity combinations in LED space. There are other options to consider in the `minimize_variance` method that are detailed in the API reference.

    # %% [markdown]
    # ## Other underdetermined objectives
    # 
    # We can also think of other objectives to achieve in the case that our system is underdetermined. For example we may wish that the fit is as close as possible to a certain overall intensity of the LEDs. To do this we can use the `fit_underdetermined` method. This method has many options that are detailed in the API reference.

    # %%
    # the goal is to get the overall intensity as low as possible
    Xhat3, Bhat3 = est.fit_underdetermined(B, underdetermined_opt='min')
    # the goal is to get the overall intensity as close as possible to 0.1
    Xhat3, Bhat3 = est.fit_underdetermined(B, underdetermined_opt=0.1)

    ### -- plotting --

    # can compare the intensities found to our previous result by plotting both cases
    from dreye.api.plotting.simplex_plot import plot_simplex

    ax = None
    # gradient line showing all possible solutions with the color indicating the overall intensity
    for idx, X_ in enumerate(X):
        ax = plot_simplex(
            n=3, 
            gradient_line=X_, 
            ax=ax, lines=False, 
            gradient_line_kws={'add_colorbar': False, 'vmin': 0, 'zorder': 2}
        )

    # with variance minization
    ax = plot_simplex(
        n=3, 
        points=Xhat2, 
        point_scatter_kws={'marker': 'x', 'color': 'tomato', 'zorder': 2.6}, 
        labels=est.sources_labels, ax=ax
    )
    # standard approach - no variance minimization
    ax = plot_simplex(
        n=3, 
        points=Xhat, 
        point_scatter_kws={'marker': 'o', 'color': 'cyan', 'zorder': 2.5}, 
        ax=ax, lines=False
    )
    # with an overall intensity objective
    ax = plot_simplex(
        n=3, 
        points=Xhat3, 
        point_scatter_kws={'marker': 's', 'color': 'green', 'zorder': 2.6}, 
        labels=est.sources_labels, ax=ax
    )
    
    plt.close()
    
    
def test_gamut_corrections():
        
    # %%
    filters_colors = ['lightgray', 'gray', 'black']
    sources_colors = ['violet', 'cyan', 'green']

    # wavelength range
    wls = np.arange(300, 700, 1)
    # peaks of the sensitivities
    peaks = np.array([380, 445, 535])
    # opsin template
    # an empty axis is added to the peaks so that the array object has dimension (opsin_type x wavelengths)
    filters = dreye.govardovskii2000_template(wls, peaks[:, None])

    led_peaks = np.array([400, 465, 550])
    sources = norm.pdf(wls, loc=led_peaks[:, None], scale=20)
    sources = sources / dreye.integral(sources, wls, axis=-1, keepdims=True)

    est = dreye.ReceptorEstimator(
        # filters array
        filters, 
        ### ADDING FILTER UNCERTAINTY
        ### In this case it scales with the value of the filter function (heteroscedastic)
        filters_uncertainty=filters * 0.1,
        # wavelength array
        domain=wls, 
        # labels for each photoreceptor type (optional)
        labels=['S', 'M', 'L'], 
        # LED array, optional
        sources=sources, 
        # lower bound of each LED, optional
        lb=np.zeros(3), 
        # upper bound for each LED, optional - if None, then the upper bound is infinity
        ub=np.ones(3) * 0.1, 
        # labels for sources, optional
        sources_labels=['V', 'C', 'G'], 
        baseline=1e-3
    )

    ax1 = est.filter_plot(colors=filters_colors)
    ax2 = plt.twinx(ax1)
    est.sources_plot(colors=sources_colors, ax=ax2)

    ax1.set_xlabel('wavelength (nm)')
    ax1.set_ylabel('relative sensitivity (a.u.)')
    ax2.set_ylabel('relative intensity (1/nm)')

    ax1.legend(loc=2)
    ax2.legend(loc=1)
    
    plt.close()

    # %% [markdown]
    # Load hyperspectral images and interpolate to wavelength points defined above:

    # %%
    image = np.load(os.path.join(os.path.dirname(__file__), 'data', 'flower_image.npy'))
    image_wls = np.load(os.path.join(os.path.dirname(__file__), 'data', 'wls_image.npy'))
    # convert hyperspectral image in units of W/m2 to photonflux units (uE)
    image = dreye.irr2flux(image, image_wls, prefix='micro')
    print(image.shape, image_wls.shape)

    im = interp1d(image_wls, image, axis=-1, bounds_error=False, fill_value=0)(wls)
    print(im.shape)
    im_shape = im.shape[:2]
    # reshape to 2D
    im = im.reshape(-1, wls.size)
    print(im.shape)

    # %% [markdown]
    # The photoreceptors will be adapted to the mean spectrum of the image

    # %%
    est.register_background_adaptation(im.mean(0))
    print(est.K)

    # %% [markdown]
    # Calculate the relative capture for all points in the image

    # %%
    B = est.relative_capture(im)

    # %% [markdown]
    # Plot capture image

    # %%
    # rescale value for better contrast, and convert to uint8

    def scale_reformat_image(B, im_shape):
        capture_image = (B - np.min(B, 0)) / (np.max(B, 0) - np.min(B, 0))
        capture_image = capture_image.reshape(*im_shape, -1)
        capture_image = (capture_image * 255).astype(np.uint8)
        return capture_image

    capture_image = scale_reformat_image(B, im_shape)
    print(capture_image.min((0, 1)), capture_image.max((0, 1)))

    plt.imshow(capture_image.reshape(*im_shape, -1))
    plt.close()

    # %% [markdown]
    # Let's plot a pairwise gamut plot to see if points are within the gamut

    # %%
    est.gamut_plot(B, colors=sources_colors, ncols=3, fig_kws={'figsize': (15, 5)}, c='gray', alpha=0.5)
    plt.close()

    # %% [markdown]
    # As can be seen in these plots, all points are highly outside of the gamut of the system because the intensity range of the system is limited.
    # 
    # Let's first try to scale the overall intensity of the image (i.e the L1 norm) using the `gamut_l1_scaling` method:

    # %%
    Bscaled = est.gamut_l1_scaling(B)

    est.gamut_plot(Bscaled, colors=sources_colors, ncols=3, fig_kws={'figsize': (15, 5)}, c='gray', alpha=0.5, vectors_kws={'width': 0.0001})
    plt.close()
    # %% [markdown]
    # We can now see that the intensity range of the image is more properly scaled for the image, but various values still remain outside of the gamut. For this we can try to scale the angles in the L1-normalized simplex (chromaticity diagram) to adjust the animal's ``saturation'' value for each pixel. To do this we have the `gamut_dist_scaling` method:

    # %%
    Bscaled2 = est.gamut_dist_scaling(Bscaled)

    est.gamut_plot(Bscaled2, colors=sources_colors, ncols=3, fig_kws={'figsize': (15, 5)}, c='gray', alpha=0.5, vectors_kws={'width': 0.0001})
    plt.close()
    # %% [markdown]
    # This method does not guarantee that all points will be within the hull, but aims to get as many points as possible within the hull. The method can also give weird results if a lot of points are outside the hull, and the projection into the hull would still leave most points outside the hull. 

    # %%
    B_in = est.in_gamut(B)
    Bscaled_in = est.in_gamut(Bscaled)
    Bscaled2_in = est.in_gamut(Bscaled2)

    # fraction within the hull
    print(np.mean(B_in))
    print(np.mean(Bscaled_in))
    print(np.mean(Bscaled2_in))

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5))

    ax1.imshow(scale_reformat_image(B, im_shape).reshape(*im_shape, -1))
    ax2.imshow(scale_reformat_image(Bscaled, im_shape).reshape(*im_shape, -1))
    ax3.imshow(scale_reformat_image(Bscaled2, im_shape).reshape(*im_shape, -1))

    ax1.set_title('original targets')
    ax2.set_title('intensity scaled targets')
    ax3.set_title('saturation scaled targets')

    plt.close()

    # %% [markdown]
    # For the gamut corrected images, we can fit our targets as in the introduction:

    # %%
    # we will ignore the fitted intensities here since we care more about how much the image gets
    # burned after fitting
    # We will batch across pixels (i.e. samples), so that the fitting procedure is faster
    _, Bscaled_hat = est.fit(Bscaled, batch_size=100)
    _, Bscaled2_hat = est.fit(Bscaled2, batch_size=100)

    # %%
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5))

    ax1.imshow(scale_reformat_image(B, im_shape).reshape(*im_shape, -1))
    ax2.imshow(scale_reformat_image(Bscaled_hat, im_shape).reshape(*im_shape, -1))
    ax3.imshow(scale_reformat_image(Bscaled2_hat, im_shape).reshape(*im_shape, -1))

    ax1.set_title('original targets')
    ax2.set_title('fitted intensity scaled targets')
    ax3.set_title('fitted saturation scaled targets')
    
    plt.close()

    # %% [markdown]
    # For this case, it appears that intensity scaling alone is probably sufficient and that saturation scaling changes the image color too much (as the stimulation system does not have a wide gamut for S vs. L/M - see previous gamut plots)

    # %% [markdown]
    # ## Scaling intensity and saturation simultaneously
    # 
    # We have also implemented an algorithm that scales saturation and intensity simultaneously, that allows us to fine-tune the tradeoff between the two. To do this, we need to use the method `fit_adaptive`:

    # %%
    _, scales, Bscaled_fit = est.fit_adaptive(
        B, 
        # maximum difference between scaled intensity and achieved intensity for single sample 
        delta_norm1=1e-5,
        # maximum difference between scaled chromatic value and achieved chromatic value (i.e. l1-normalized sample) for single sample
        delta_radius=1e-4,  # we want to preserve the chromatic values well
        # the two parameters above are highly dependent on the range of values of B
        # here the objectives details are set
        adaptive_objective='max',  # try to obtain the maximum relative intensity and relative chromatic values possible
        scale_w=np.array([0.001, 10]),  # trade-off between intensity and chromatic value for max objective
    )

    # how intensity and saturation are scaled
    print(scales)

    est.gamut_plot(Bscaled_fit, colors=sources_colors, ncols=3, fig_kws={'figsize': (15, 5)}, c='gray', alpha=0.5, vectors_kws={'width': 0.0001})
    plt.close()
    # %%
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(30, 5))

    ax1.imshow(scale_reformat_image(B, im_shape).reshape(*im_shape, -1))
    ax2.imshow(scale_reformat_image(Bscaled_hat, im_shape).reshape(*im_shape, -1))
    ax3.imshow(scale_reformat_image(Bscaled2_hat, im_shape).reshape(*im_shape, -1))
    ax4.imshow(scale_reformat_image(Bscaled_fit, im_shape).reshape(*im_shape, -1))

    ax1.set_title('original targets')
    ax2.set_title('fitted intensity scaled targets')
    ax3.set_title('fitted saturation scaled targets')
    ax4.set_title('fitting of intensity and scale')
    plt.close()

    # %% [markdown]
    # More details on the methods can be found in the API reference.


def test_patterned():
        
    # load previous model
    filters_colors = ['lightgray', 'gray', 'black']
    sources_colors = ['violet', 'cyan', 'green']

    # wavelength range
    wls = np.arange(300, 700, 1)
    # peaks of the sensitivities
    peaks = np.array([380, 445, 535])
    # opsin template
    # an empty axis is added to the peaks so that the array object has dimension (opsin_type x wavelengths)
    filters = dreye.govardovskii2000_template(wls, peaks[:, None])

    led_peaks = np.array([400, 465, 550])
    sources = norm.pdf(wls, loc=led_peaks[:, None], scale=20)
    sources = sources / dreye.integral(sources, wls, axis=-1, keepdims=True)

    est = dreye.ReceptorEstimator(
        # filters array
        filters, 
        ### ADDING FILTER UNCERTAINTY
        ### In this case it scales with the value of the filter function (heteroscedastic)
        filters_uncertainty=filters * 0.1,
        # wavelength array
        domain=wls, 
        # labels for each photoreceptor type (optional)
        labels=['S', 'M', 'L'], 
        # LED array, optional
        sources=sources, 
        # lower bound of each LED, optional
        lb=np.zeros(3), 
        # upper bound for each LED, optional - if None, then the upper bound is infinity
        ub=np.ones(3) * 0.1, 
        # labels for sources, optional
        sources_labels=['V', 'C', 'G'], 
        baseline=1e-3
    )

    ax1 = est.filter_plot(colors=filters_colors)
    ax2 = plt.twinx(ax1)
    est.sources_plot(colors=sources_colors, ax=ax2)

    ax1.set_xlabel('wavelength (nm)')
    ax1.set_ylabel('relative sensitivity (a.u.)')
    ax2.set_ylabel('relative intensity (1/nm)')

    ax1.legend(loc=2)
    ax2.legend(loc=1)
    plt.close()

    # %%
    # load previous image
    image = np.load(os.path.join(os.path.dirname(__file__), 'data', 'flower_image.npy'))
    image_wls = np.load(os.path.join(os.path.dirname(__file__), 'data', 'wls_image.npy'))
    # convert hyperspectral image in units of W/m2 to photonflux units (uE)
    image = dreye.irr2flux(image, image_wls, prefix='micro')
    print(image.shape, image_wls.shape)

    im = interp1d(image_wls, image, axis=-1, bounds_error=False, fill_value=0)(wls)
    print(im.shape)
    im_shape = im.shape[:2]
    # reshape to 2D
    im = im.reshape(-1, wls.size)
    print(im.shape)

    # register adaptation
    est.register_background_adaptation(im.mean(0))
    print(est.K)

    # target capture
    B = est.relative_capture(im)

    # rescale value for better contrast, and convert to uint8

    def scale_reformat_image(B, im_shape):
        capture_image = (B - np.min(B, 0)) / (np.max(B, 0) - np.min(B, 0))
        capture_image = capture_image.reshape(*im_shape, -1)
        capture_image = (capture_image * 255).astype(np.uint8)
        return capture_image

    capture_image = scale_reformat_image(B, im_shape)
    print(capture_image.min((0, 1)), capture_image.max((0, 1)))

    plt.imshow(capture_image.reshape(*im_shape, -1))
    plt.close()

    # %% [markdown]
    # Let's first rescale the capture values accordingly to fit them all within the gamut (see `gamut_corrections.ipynb` tutorial for details). This is a useful preprocessing step for the following fitting procedure especially when the intensity scaling of the system is different from that of the image.

    # %%
    _, scales, Bscaled = est.fit_adaptive(
        B, 
        # maximum difference between scaled intensity and achieved intensity for single sample 
        delta_norm1=1e-5,
        # maximum difference between scaled chromatic value and achieved chromatic value (i.e. l1-normalized sample) for single sample
        delta_radius=1e-4,  # we want to preserve the chromatic values well
        # the two parameters above are highly dependent on the range of values of B
        # here the objectives details are set
        adaptive_objective='max',  # try to obtain the maximum relative intensity and relative chromatic values possible
        scale_w=np.array([0.001, 10]),  # trade-off between intensity and chromatic value for max objective
    )

    # how intensity and saturation are scaled
    print(scales)

    est.gamut_plot(Bscaled, colors=sources_colors, ncols=3, fig_kws={'figsize': (15, 5)}, c='gray', alpha=0.5, vectors_kws={'width': 0.0001})
    plt.close()
    
    # %% [markdown]
    # ## The problem of fewer subframes than LED sources
    # 
    # Modern custom projector systems allow for the independent control of the subframe number and how each LED is assigned to individual subframe (i.e. they can be combined arbitrarily for each subframe). Here we will go assume that we only have two subframe but that we can arbitrarily combine LEDs in each subframe. To do this we will use the method `fit_decomposition`. See the API reference for more details:

    # %%
    X, P, Bfit = est.fit_decomposition(
        Bscaled, 
        # the number of subframes
        n_layers=2, 
        seed=1,
    )
    # X are the intensities of each LED for each subframe
    # P is the opacity or intensity of each Pixel and subframe
    # Thus we get:
    # est.K * (P @ X @ est.A.T + baseline) = Bfit
    print(X.shape, P.shape, Bfit.shape)

    # %%
    for i in range(2):
        plt.bar(np.arange(0, 6, 2)+i, X[i], width=0.8)
    plt.xticks(np.arange(0.5, 6.5, 2), est.sources_labels)
    plt.title("Intensities of LEDs for subframe 1 and 2 (blue and orange)")
    plt.close()

    # %%
    # goodness-of-fit for the whole image - even with fewer subframes than number of filters
    # but with a flexible source assignment for each subframe
    from sklearn.metrics import r2_score
    r2 = r2_score(Bscaled, Bfit)
    
    assert (r2 > 0.9), f"r2 score too low {r2}"

    # %% [markdown]
    # See the API reference for more detail.



        

