"""
Dependent Excitation Models
"""

import warnings
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import least_squares

from dreye.utilities.abstract import inherit_docstrings
from dreye.estimators.excitation_models import IndependentExcitationFit


@inherit_docstrings
class DependentExcitationFit(IndependentExcitationFit):

    def __init__(
        self,
        *,
        independent_layers=None,  # int
        layer_assignments=None,  # list of lists or array-like
        bit_depth=1,
        photoreceptor_model=None,  # dict or Photoreceptor class
        fit_weights=None,
        background=None,  # dict or Spectrum instance or array-like
        measured_spectra=None,  # MeasuredSpectraContainer, numpy.ndarray
        max_iter=None,
        unidirectional=False,
        bg_ints=None,
        fit_only_uniques=False,
        ignore_bounds=None,
        lsq_kwargs=None,
        background_external=None, 
        intensity_bounds=None, 
        wavelengths=None, 
        seed=None,
        n_epochs=None, 
        epoch_iter=None,
        verbose=False, 
        n_jobs=None
    ):
        super().__init__(
            photoreceptor_model=photoreceptor_model,
            measured_spectra=measured_spectra,
            background=background,
            max_iter=max_iter,
            unidirectional=unidirectional,
            fit_weights=fit_weights,
            fit_only_uniques=fit_only_uniques,
            lsq_kwargs=lsq_kwargs,
            ignore_bounds=ignore_bounds,
            bg_ints=bg_ints,
            background_external=background_external, 
            intensity_bounds=intensity_bounds, 
            wavelengths=wavelengths, 
            verbose=verbose, 
            n_jobs=n_jobs
        )
        self.independent_layers = independent_layers
        self.layer_assignments = layer_assignments
        self.bit_depth = bit_depth
        self.seed = seed
        self.n_epochs = n_epochs
        self.epoch_iter = epoch_iter

    def _fit(self, X):
        if self.independent_layers is None and self.layer_assignments is None:
            self._independent_layers_ = len(self.measured_spectra_)
            self._layer_assignments_ = [[idx] for idx in range(len(self.measured_spectra_))]
        elif self.independent_layers is None:
            self._independent_layers_ = len(self.layer_assignments)
            self._layer_assignments_ = self.layer_assignments
            # TODO assert right type and indices exist in measured_spectra
        elif self.layer_assignments is None:
            self._independent_layers_ = self.independent_layers
            self._layer_assignments_ = [
                list(range(len(self.measured_spectra_))) 
                for _ in range(self._independent_layers_)
            ]
        else:
            assert len(self.layer_assignments) == self.independent_layers
            self._independent_layers_ = self.independent_layers
            self._layer_assignments_ = self.layer_assignments
            # TODO assert right type and indices exist in measured_spectra

        row_idcs = []  # leds
        col_idcs = []  # layers
        for idx, source_idcs in enumerate(self._layer_assignments_):
            row_idcs.extend(list(source_idcs))
            col_idcs.extend([idx]*len(source_idcs))
        self._row_idcs_ = np.array(row_idcs)
        self._col_idcs_ = np.array(col_idcs)
        self._offset_ = len(row_idcs)

        # overwrite this method when subclassing
        self.capture_X_, self.excite_X_ = self._process_X(X)

        # if only fit uniques used different iterator
        if self.fit_only_uniques:
            # get uniques
            _, xidcs, xinverse = np.unique(
                self.capture_X_, axis=0, return_index=True, return_inverse=True
            )
            fitted_intensities, layer_intensities, pixel_strength = self._fit_sample(
                self.capture_X_[xidcs], self.excite_X_[xidcs]
            )
            fitted_intensities = fitted_intensities[xinverse]
            pixel_strength = pixel_strength[xinverse]
        else:
            fitted_intensities, layer_intensities, pixel_strength = self._fit_sample(
                self.capture_X_, self.excite_X_
            )

        # len(measured_spectra) x independent_layers, len(X) x independent_layers
        self.layer_intensities_, self.pixel_strength_, self.fitted_intensities_ = (
            layer_intensities, pixel_strength, fitted_intensities
        )
        self.fitted_excite_X_ = self.get_excitation(self.fitted_intensities_.T)
        self.fitted_capture_X_ = self.photoreceptor_model_.inv_excitefunc(
            self.fitted_excite_X_
        )

        return self

    def _reformat_intensities(self, w=None, **kwargs):
        # len(measured_spectra) x independent_layers, len(X) x independent_layers
        ws, pixel_strength = self._format_intensities(w=w, **kwargs)
        # len(X) x len(measured_spectra)
        return pixel_strength @ ws.T

    def _format_intensities(self, w=None, ws=None, pixel_strength=None):
        offset = 0
        if ws is None:
            ws = np.zeros((len(self.measured_spectra_), self._independent_layers_))
            ws[self._row_idcs_, self._col_idcs_] = w
            offset = self._offset_
            # for idx, source_idcs in enumerate(self._layer_assignments_):
            #     ws[source_idcs, idx] = w[offset:offset+len(source_idcs)]
            #     offset += len(source_idcs)
        
        if pixel_strength is None:
            pixel_strength = w[offset:].reshape(-1, self._independent_layers_)
        return ws, pixel_strength

    def _fit_sample(self, capture_x, excite_x):
        # capture_x.shape == excite_x.shape - numpy.ndarray (n_pixels x n_opsins)
        np.random.seed(self.seed)
        # adjust bounds if necessary
        bounds = list(self.intensity_bounds_)
        # two element list of numpy arrays with the lower and upper bound
        # ([l_led1, l_led2], [u_led1, u_led2])
        if self._unidirectional_:
            if np.all(capture_x >= self.capture_border_):
                bounds[0] = self.bg_ints_  # self.bg_ints_ - numpy.ndarray (n_leds)
            elif np.all(capture_x <= self.capture_border_):
                bounds[1] = self.bg_ints_
        # find initial w0 using linear least squares by using the mean capture across all pixels
        w0 = self._init_sample(capture_x.mean(0), bounds)
        # w0: np.ndarray (n_leds)

        # init all parameters

        # add independent layer dimensions        
        w0s = []
        bounds0 = []
        bounds1 = []
        # layer_assignments: list of lists 
        # e.g. [[0, 1, 2], [3, 4], [0, 2, 4]]
        for source_idcs in self._layer_assignments_:
            w0s.append(w0[source_idcs])
            # [np.array([2.3, 4.5, .3]), np.array([6, 3]), np.array([2.3, 0.3, 3])]
            bounds0.append(bounds[0][source_idcs])
            bounds1.append(bounds[1][source_idcs])

        # reformatted w0 and bounds
        w0 = np.concatenate(w0s)
        bounds = (
            np.concatenate(bounds0), 
            np.concatenate(bounds1)
        )

        # pixel strength values
        n_pixels = len(capture_x)
        p0 = np.random.random(
            n_pixels*self._independent_layers_
        ).reshape(-1, self._independent_layers_)
        # proper rounding or integer mapping for p0
        pbounds = (0, 1)

        epoch_iter = (10 if self.epoch_iter is None else self.epoch_iter)
        n_epochs = (10 if self.n_epochs is None else self.n_epochs)

        if self.verbose:
            iterator = tqdm(range(n_epochs), desc='epochs')
        else:
            iterator = range(n_epochs)

        for _ in iterator:
            # step 1
            result = least_squares(
                self._objective,
                x0=w0.ravel(),
                args=(excite_x,),
                kwargs={'pixel_strength': p0},
                jac=self._w0_derivative,
                bounds=bounds,
                max_nfev=epoch_iter,
                **({} if self.lsq_kwargs is None else self.lsq_kwargs)
            )
            w0, p0 = self._format_intensities(result.x, pixel_strength=p0)
            # step 2
            if self.n_jobs is None:
                p0_ = np.zeros(p0.shape)
                if self.verbose:
                    pixel_iter = tqdm(range(n_pixels), desc='pixel fitting', leave=False)
                else:
                    pixel_iter = range(n_pixels)
                
                for idx in pixel_iter:
                    result = least_squares(
                        self._ppoint_objective,
                        x0=p0[idx],
                        args=(excite_x[idx],),
                        kwargs={'ws': w0},
                        jac=self._ppoint_derivative,
                        bounds=pbounds,
                        max_nfev=epoch_iter,
                        **({} if self.lsq_kwargs is None else self.lsq_kwargs)
                    )
                    p0_[idx] = result.x
                p0 = p0_
            else:
                p0 = Parallel(n_jobs=self.n_jobs, verbose=int(self.verbose))(
                        delayed(least_squares)(
                            self._ppoint_objective,
                            x0=p0[idx],
                            args=(excite_x[idx],),
                            kwargs={'ws': w0},
                            jac=self._ppoint_derivative,
                            bounds=pbounds,
                            max_nfev=epoch_iter,
                            **({} if self.lsq_kwargs is None else self.lsq_kwargs)
                        ) for idx in range(n_pixels)
                )
                p0 = np.array([p00.x for p00 in p0])
            # result = least_squares(
            #     self._objective,
            #     x0=p0.ravel(),
            #     args=(excite_x,),
            #     kwargs={'ws': w0},
            #     jac=self._p0_derivative,
            #     bounds=pbounds,
            #     max_nfev=epoch_iter,
            #     **({} if self.lsq_kwargs is None else self.lsq_kwargs)
            # )
            # w0, p0 = self._format_intensities(result.x, ws=w0)
            p0 = p0 / np.max(p0)
            p0 = (np.ceil(p0 * 2**self.bit_depth) - 1) / (2**self.bit_depth - 1)

        result = least_squares(
            self._objective,
            x0=w0.ravel(),
            args=(excite_x,),
            kwargs={'pixel_strength': p0},
            jac=self._w0_derivative,
            bounds=bounds,
            max_nfev=self.max_iter,
            **({} if self.lsq_kwargs is None else self.lsq_kwargs)
        )
        w0, p0 = self._format_intensities(result.x, pixel_strength=p0)

        layer_intensities, pixel_strength = w0, p0
        fitted_intensities = self._reformat_intensities(ws=w0, pixel_strength=p0)
        
        return fitted_intensities, layer_intensities, pixel_strength

    def _objective(self, w, excite_x, **kwargs):
        w = self._reformat_intensities(w, **kwargs).T  # n_leds x n_pixels
        # get excitation values given intensities
        x_pred = self.get_excitation(w)
        return (self.fit_weights_ * (excite_x - x_pred)).ravel()  # residuals

    def _ppoint_objective(self, w, excite_x, ws):
        w = ws @ w
        # get excitation values given intensities
        x_pred = self.get_excitation(w)
        return (self.fit_weights_ * (excite_x - x_pred)).ravel()  # residuals

    def _w0_derivative(self, w, excite_x, pixel_strength):
        nvars = w.size
        w = self._reformat_intensities(w, pixel_strength=pixel_strength).T  # n_leds x n_pixels
        # samples x opsins x leds x independent_layers
        x_pred_deriv = self._excite_derivative(w)[..., None] * pixel_strength[:, None, None, :]
        x_pred_deriv = x_pred_deriv[..., self._row_idcs_, self._col_idcs_]

        # get only changeable variables
        # x_pred_deriv = np.zeros(fprime.shape[:2]+(nvars,))
        # offset = 0
        # for idx, source_idcs in enumerate(self._layer_assignments_):
        #     x_pred_deriv[..., offset:offset+len(source_idcs)] = fprime[..., source_idcs, idx]
        #     offset += len(source_idcs)
        
        # (samples x opsins) x leds
        return (self.fit_weights_[..., None] * -x_pred_deriv).reshape(-1, nvars)

    def _p0_derivative(self, w, excite_x, ws):
        nvars = w.size
        w = self._reformat_intensities(w, ws=ws).T  # n_leds x n_pixels
        # samples x opsins x (leds->summed) x independent_layers
        fprime = (self._excite_derivative(w)[..., None] * ws[None, None, :, :]).sum(axis=-2)
        # samples x opsins x (samples x independent_layers)
        x_pred_deriv = np.apply_along_axis(
            np.diag, 0, fprime
        ).transpose((0, 2, 1, 3)).reshape(*fprime.shape[:-1], -1)

        # (samples x opsins) x pixels
        return (self.fit_weights_[..., None] * -x_pred_deriv).reshape(-1, nvars)

    def _ppoint_derivative(self, w, excite_x, ws):
        w = ws @ w
        # opsins x (leds->summed) x independent_layers
        x_pred_deriv = (self._excite_derivative(w)[..., None] * ws[None, :, :]).sum(axis=-2)
        # opsins x pixels
        return (self.fit_weights_[..., None] * -x_pred_deriv)