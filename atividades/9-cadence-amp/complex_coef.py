import numpy as np


def unpackComplexCoefficients(real_coef, n_lut_lines, n_lut_columns):
    matrix_size = n_lut_lines * n_lut_columns
    real_parts = real_coef[:matrix_size].reshape(n_lut_lines, n_lut_columns)
    imag_parts = real_coef[matrix_size:].reshape(n_lut_lines, n_lut_columns)
    return real_parts + 1j * imag_parts


def packComplexCoefficients(complex_coef):
    real_parts = complex_coef.real.flatten()
    imag_parts = complex_coef.imag.flatten()
    return np.concatenate([real_parts, imag_parts])