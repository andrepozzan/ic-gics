import numpy as np

def unpackComplexCoefficients(real_coef, N_LUT_LINES, N_LUT_COLUMNS):
    MATRIZ_SIZE = N_LUT_LINES * N_LUT_COLUMNS
    real_parts = real_coef[:MATRIZ_SIZE].reshape(N_LUT_LINES, N_LUT_COLUMNS)
    imag_parts = real_coef[MATRIZ_SIZE:].reshape(N_LUT_LINES, N_LUT_COLUMNS)
    complex_coef = real_parts + 1j * imag_parts
    return complex_coef

def packComplexCoefficients(complex_coef):
    real_parts = complex_coef.real.flatten()
    imag_parts = complex_coef.imag.flatten()
    real_coef = np.concatenate([real_parts, imag_parts])
    return real_coef
