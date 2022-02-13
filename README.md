# DrEye: Exploiting Receptor Space Geometry for Stimulus Design across Animals

*drEye* is a package that implements various approaches to design stimuli for sensory receptors. The main focus of the package is geared towards designing color stimuli for any animal under investigation, where the photoreceptor spectral sensitivities are known. The hardware-agnostic approach incorporates photoreceptor models within the framework of the principle of univariance. This enables experimenters to identify the most effective way to combine multiple light sources to create desired distributions of light, and thus easily construct relevant stimuli for mapping the color space of an organism. The methods support broad applications in color vision science and provide a framework for uniform stimulus designs across experimental systems. Many of the methods described can be used more generally to design stimuli for other sensory organs or used more broadly where a set of linear filters define the input to a system. 

## Documentationa and tutorials

Documentation and tutorials can be found here <https://dreye.readthedocs.io/en/latest/>.

## Preprint

Our preprint that explains the purpose of the package *drEye* and goes through key concepts is available on bioRxiv (<https://www.biorxiv.org/content/10.1101/2022.01.17.476640v1>).
Please reference this preprint when using *drEye*.

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

## Development

If you are interested in contributing to the project, please email at gucky@gucky.eu.
We would also love any type of general feedback or contributions to the code and methods implemented.
