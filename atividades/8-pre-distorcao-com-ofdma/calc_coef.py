import numpy as np
from scipy.optimize import least_squares


from complex_coef import unpackComplexCoefficients, packComplexCoefficients

def estimatedValueWithLUT(x_data, lut_out_matrix, interpolation_in, N_LUT_LINES):
    length_of_x_data = len(x_data)
    result = np.zeros(length_of_x_data, dtype=complex)
    
    for line in range(N_LUT_LINES):
        # Move the elements of an array in a circular pattern.
        delayed = np.roll(x_data, line)
        
        # Zero out the first 'line' elements to create a memory effect
        delayed[:line] = 0

        x_abs = np.abs(delayed)
        real_interpolation = np.interp(x_abs, interpolation_in, lut_out_matrix[line].real)
        imag_interpolation = np.interp(x_abs, interpolation_in, lut_out_matrix[line].imag)
        result += delayed * (real_interpolation + 1j * imag_interpolation)
    return result


def residuals(lut_out_real, x_data, y_data, N_LUT_LINES, N_LUT_COLUMNS, interpolation_in):
    lut_out_complex = unpackComplexCoefficients(lut_out_real, N_LUT_LINES, N_LUT_COLUMNS)
    y_estimated = estimatedValueWithLUT(x_data, lut_out_complex, interpolation_in, N_LUT_LINES)
    residual = y_data - y_estimated
    res_vector = np.concatenate([residual.real, residual.imag])
    return res_vector

def calcCoef(in_data, out_data, N_LUT_LINES, N_LUT_COLUMNS, initial_complex_coef, interpolation_in):
    initial_real_coef = packComplexCoefficients(initial_complex_coef)
    result = least_squares(residuals, initial_real_coef, args=(in_data, out_data, N_LUT_LINES, N_LUT_COLUMNS, interpolation_in), verbose=2)
    return unpackComplexCoefficients(result.x, N_LUT_LINES, N_LUT_COLUMNS)

def dpdTraining(in_training, out_training, N_LUT_LINES, N_LUT_COLUMNS, initial_complex_coef, interpolation_in):
    dpd_coef = calcCoef(out_training, in_training, N_LUT_LINES, N_LUT_COLUMNS, initial_complex_coef, interpolation_in)
    return dpd_coef