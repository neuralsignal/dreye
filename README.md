# DrEye: Exploiting Receptor Space Geometry for Stimulus Design across Animals

[![DOI](https://zenodo.org/badge/243093421.svg)](https://zenodo.org/badge/latestdoi/243093421)

*drEye* is a package that implements various approaches to design stimuli for sensory receptors. The main focus of the package is geared towards designing color stimuli for any animal under investigation, where the photoreceptor spectral sensitivities are known. The hardware-agnostic approach incorporates photoreceptor models within the framework of the principle of univariance. This enables experimenters to identify the most effective way to combine multiple light sources to create desired distributions of light, and thus easily construct relevant stimuli for mapping the color space of an organism. The methods support broad applications in color vision science and provide a framework for uniform stimulus designs across experimental systems. Many of the methods described can be used more generally to design stimuli for other sensory organs or used more broadly where a set of linear filters define the input to a system. 

## Documentationa and tutorials

Documentation and tutorials can be found here <https://dreye.readthedocs.io/en/latest/>.

## Web application

To test stimulus creation, check out the corresponding web applitcation:
<https://share.streamlit.io/gucky92/dreyeapp/main/app.py>.

## Paper

Our paper that explains the purpose of the package *drEye* and goes through key concepts:

*"Exploiting colour space geometry for visual stimulus design across animals"*;
Matthias P. Christenson, S. Navid Mousavi, Elie Oriol, Sarah L. Heath and Rudy Behnia;
Philosophical Transactions of the Royal Society B: Biological Sciences;
<https://royalsocietypublishing.org/doi/10.1098/rstb.2021.0280>.

Please reference this paper when using *drEye*.

## Installation

```bash
pip install dreye
```

In order to use the non-linear fitting procedures, JAX should be installed separately:

```bash
pip install jax[cpu]
```

## Common Issues

* Running jax on the new Macbook Pro chips can run into problems. Make sure to install the versions that work with the M1 chip. For my purposes jaxlib==0.1.60 and jax==0.2.10 currently work (14/01/21).


## Old code

The nonlinear fitting procedures for the variance minimization, underdetermined, and decomposition algorithms and the silent substitution algorithm described in the paper have yet to be refactored into the new API. 
The old API can be found at <https://github.com/gucky92/dreye_ext>.
The linear versions (faster and convex) are already available in the new API.


## Development

If you are interested in contributing to the project, please email at gucky@gucky.eu.
We would also love any type of general feedback or contributions to the code and methods implemented.
