{{ header }}

.. _user_guide:

==========
User Guide
==========

Installation
------------

`DrEye` can be installed via pip from `PyPI <https://pypi.org/project/dreye>`__:


.. code-block:: bash

   pip install dreye

You can also clone the git repository and install the package from source.

In order to use the non-linear fitting procedures, JAX should be installed separately:

.. code-block:: bash

   pip install jax[cpu]


Tutorials
---------

The different tutorials are listed below:

* `Introduction to dreye <introduction.ipynb>`_.
   .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/gucky92/dreye/HEAD?labpath=tutorials%2Fintroduction.ipynb
* `Dealing with filter uncertainty and underdetermined systems <uncertainty.ipynb>`_.
   .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/gucky92/dreye/HEAD?labpath=tutorials%2Funcertainty.ipynb
* `Gamut-corrective approaches <gamut_corrections.ipynb>`_.
   .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/gucky92/dreye/HEAD?labpath=tutorials%2Fgamut_corrections.ipynb
* `Patterned stimuli <patterned.ipynb>`_.
   .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/gucky92/dreye/HEAD?labpath=tutorials%2Fpatterned.ipynb

.. toctree::
    :maxdepth: 2
    :glob:
    :hidden:

    philosophy
    Introduction to *drEye* <introduction.nblink>
    Dealing with filter uncertainty and underdetermined systems <uncertainty.nblink>
    Gamut-corrective approaches <gamut_corrections.nblink>
    Patterned stimuli <patterned.nblink>
    Spherical coordinates <spherical_coordinates.rst>

