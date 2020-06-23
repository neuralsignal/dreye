"""Filter class
"""

import numpy as np
from scipy import signal

from dreye.utilities import asarray, is_integer


class Filter1D:
    """
    Create a one-dimensional filter for an array.

    Parameters
    ----------
    method : str
        Window method from `scipy.signal.windows`.
    M : int
        The length of the window in indices.
    method_args : kwargs
        Keyword arguments passed to the window method.

    See Also
    --------
    scipy.signal.windows
    """

    def __init__(self, method, M, **method_args):
        assert (
            hasattr(signal.windows, method)
            and (method not in ['hanning', 'get_window'])
        ), f"window method {method} does not exist"

        assert is_integer(M), (f"M is not of type int, but {type(M)}.")

        self._method = method
        self._M = M
        self._method_args = method_args

    @property
    def M(self):
        """
        The length of the window.
        """
        return self._M

    @property
    def method(self):
        """
        The window method.
        """
        return self._method

    @property
    def method_args(self):
        """
        Arguments passed to the window method
        """
        return self._method_args

    @property
    def window(self):
        """
        The window array.
        """
        return getattr(signal.windows, self.method)(
            self.M, **self.method_args)

    def __call__(self, arr, axis=None, mode='same'):
        """
        Apply the window to an array along a specific axis.

        Parameters
        ----------
        arr : array-like
            Any dimensional array.
        axis : int, optional
            Axis along which to apply the window
        mode : str {'full', 'valid', 'same'}, optional
            A string indicating the size of the output:

            ``full``
               The output is the full discrete linear convolution
               of the inputs. (Default)
            ``valid``
               The output consists only of those elements that do not
               rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
               must be at least as large as the other in every dimension.
            ``same``
               The output is the same size as `in1`, centered
               with respect to the 'full' output.

        Returns
        -------
        out : array
            An N-dimensional array containing a subset of discrete linear
            convolution of `arr` with the `window`.

        See Also
        --------
        scipy.signal.fftconvolve
        scipy.signal.convolve
        """

        arr = asarray(arr)
        window = self.window

        if axis is None:
            assert window.ndim == arr.ndim
        elif window.ndim == arr.ndim:
            pass
        else:
            slices = tuple(
                slice(None, None, None) if i == axis
                else None
                for i in range(arr.ndim)
            )
            window = window[slices]

        window /= np.sum(window, axis=axis, keepdims=True)
        # dealing with borders - extend borders (interpolation or zeropad)
        return signal.fftconvolve(
            arr, window, mode=mode, axes=axis
        )
