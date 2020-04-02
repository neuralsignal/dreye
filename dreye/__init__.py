"""
dreye
=====

*dreye* (drosophila eye) is a *Python* photoreceptor and (color) vision
package implementing various photoreceptor and (color) vision models and
algorithms, mainly for drosophila.
Part of the code was modified from the Colour GitHub project
(https://github.com/colour-science/colour; BSD-3-Clause license),
which is a package implementing color vision models and algorithms for human
observers.
The format for this package was inspired by the Colour GitHub project
(<https://github.com/colour-science/colour>), which is a package implementing
color vision models and algorithms for human observers.

Sub-packages
------------
-   adaptation: Chromatic adaptation models and transformations.
-   algebra: Algebra utilities.
-   constants
-   continuous
-   difference?
-   examples
-   errors
-   io
-   photoreceptor
-   plotting
-   recovery?
-   spectral: SpectralDistribution, Filter, Opsin
"""

# import all core elements and constants
from dreye.core import *
from dreye.constants import *
