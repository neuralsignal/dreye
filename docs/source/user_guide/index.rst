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
* `Dealing with filter uncertainty and underdetermined systems <uncertainty.ipynb>`_.
* `Gamut-corrective approaches <gamut_corrections.ipynb>`_.
* `Patterned stimuli <patterned.ipynb>`_.

.. toctree::
    :maxdepth: 2
    :glob:
    :hidden:

    philosophy
    Introduction to *drEye* <introduction.nblink>
    Dealing with filter uncertainty and underdetermined systems <uncertainty.nblink>
    Gamut-corrective approaches <gamut_corrections.nblink>
    Patterned stimuli <patterned.nblink>

