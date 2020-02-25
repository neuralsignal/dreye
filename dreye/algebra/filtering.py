"""Filter class
"""

import numpy as np
from scipy import signal


class Filter1D:
    """filter class
    """

    def __init__(self, method, M, **method_args):
        """
        """

        assert (
            hasattr(signal.windows, method)
            and (method not in ['hanning', 'get_window'])
        ), f"window method {method} does not exist"

        assert isinstance(M, int), (f"M is not of type int, but {type(M)}.")

        self._method = method
        self._M = M
        self._method_args = method_args

    @property
    def M(self):
        """
        """

        return self._M

    @property
    def method(self):
        """
        """

        return self._method

    @property
    def method_args(self):
        """
        """

        return self._method_args

    @property
    def window(self):
        """
        """

        return getattr(signal.windows, self.method)(
            self.M, **self.method_args)

    def __call__(self, arr, axis=None, mode='same'):
        """
        """

        arr = np.asarray(arr)

        window = self.window

        if axis is None:
            assert window.ndim == arr.ndim
        elif window.ndim == arr.ndim:
            pass
        else:
            assert arr.ndim < 3
            window = np.expand_dims(window, (axis+1) % 2)

        window /= np.sum(window, axis=axis, keepdims=True)
        # dealing with borders - extend borders (interpolation or zeropad)
        return signal.fftconvolve(
            arr, window, mode=mode, axes=axis
        )
