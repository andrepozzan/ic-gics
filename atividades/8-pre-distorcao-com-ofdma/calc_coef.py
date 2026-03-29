import numpy as np
from scipy.optimize import least_squares
from dataclasses import dataclass


from complex_coef import unpackComplexCoefficients, packComplexCoefficients


@dataclass(frozen=True)
class DPDConfig:
    n_lut_lines: int
    n_lut_columns: int
    initial_complex_coef: np.ndarray
    interpolation_in: np.ndarray

def estimatedValueWithLUT(x_data, lut_out_matrix, cfg):
    length_of_x_data = len(x_data)
    result = np.zeros(length_of_x_data, dtype=complex)
    
    for line in range(cfg.n_lut_lines):
        # Move the elements of an array in a circular pattern.
        delayed = np.roll(x_data, line)
        
        # Zero out the first 'line' elements to create a memory effect
        delayed[:line] = 0

        x_abs = np.abs(delayed)
        real_interpolation = np.interp(x_abs, cfg.interpolation_in, lut_out_matrix[line].real)
        imag_interpolation = np.interp(x_abs, cfg.interpolation_in, lut_out_matrix[line].imag)
        result += delayed * (real_interpolation + 1j * imag_interpolation)
    return result


def residuals(lut_out_real, x_data, y_data, cfg):
    lut_out_complex = unpackComplexCoefficients(lut_out_real, cfg.n_lut_lines, cfg.n_lut_columns)
    y_estimated = estimatedValueWithLUT(x_data, lut_out_complex, cfg)
    residual = y_data - y_estimated
    res_vector = np.concatenate([residual.real, residual.imag])
    return res_vector

def calcCoef(in_data, out_data, cfg):
    initial_real_coef = packComplexCoefficients(cfg.initial_complex_coef)
    result = least_squares(residuals, initial_real_coef, args=(in_data, out_data, cfg), verbose=2)
    return unpackComplexCoefficients(result.x, cfg.n_lut_lines, cfg.n_lut_columns)

def dpdTraining(in_training, out_training, cfg):
    dpd_coef = calcCoef(out_training, in_training, cfg)
    return dpd_coef