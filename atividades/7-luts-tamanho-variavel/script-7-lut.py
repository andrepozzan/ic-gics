from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

mat = loadmat('in_out_SBRT2_direto.mat')
#Flatten para garantir que são vetores de uma dimensão
in_training = mat['in_extraction'].flatten()
out_training = mat['out_extraction'].flatten()
in_validation = mat['in_validation'].flatten()
out_validation = mat['out_validation'].flatten()

n_in_points = 80
lut_in = np.linspace(0.0, np.max(np.abs(in_training)), n_in_points)
lut_out = out_training

LUTS_SIZE = 2

def unpackComplexCoefficients(real_coef):
    # Para matriz LUTS_SIZE x n_in_points
    n = LUTS_SIZE * n_in_points
    real_parts = real_coef[:n].reshape(LUTS_SIZE, n_in_points)
    imag_parts = real_coef[n:].reshape(LUTS_SIZE, n_in_points)
    complex_coef = real_parts + 1j * imag_parts
    return complex_coef

def packComplexCoefficients(complex_coef):
    # Para matriz LUTS_SIZE x n_in_points
    real_parts = complex_coef.real.flatten()
    imag_parts = complex_coef.imag.flatten()
    real_coef = np.concatenate([real_parts, imag_parts])
    return real_coef

def estimatedValueWithLUTS_SIZE(x_data, lut_out_matrix):
    n = len(x_data)
    result = np.zeros(n, dtype=complex)
    
    for M in range(LUTS_SIZE):
        delayed = np.roll(x_data, M)
        delayed[:M] = 0  # zera início (sem histórico real)
        # print("DELAYED: ",delayed)

        x_abs = np.abs(delayed)
        real_interp = np.interp(x_abs, lut_in, lut_out_matrix[M].real)
        imag_interp = np.interp(x_abs, lut_in, lut_out_matrix[M].imag)
        result += delayed * (real_interp + 1j * imag_interp)
    return result

def residuals(lut_out_real, x_data, y_data):
    lut_out_complex = unpackComplexCoefficients(lut_out_real)
    y_est = estimatedValueWithLUTS_SIZE(x_data, lut_out_complex)
    res = y_data - y_est
    res_vec = np.concatenate([res.real, res.imag])
    return res_vec

def calcCoef(in_data, out_data, n_in_points):
    initial_complex_coef = (np.random.randn(LUTS_SIZE, n_in_points) +
                            1j * np.random.randn(LUTS_SIZE, n_in_points))
    
    initial_real_coef = packComplexCoefficients(initial_complex_coef)
    result = least_squares(residuals, initial_real_coef, args=(in_data, out_data), verbose=2)
    return unpackComplexCoefficients(result.x) 

def calculate_nmse(out_validation, saida_estimada):
    erro = out_validation - saida_estimada
    nmse = 10 * np.log10(np.sum(np.abs(erro)**2) / np.sum(np.abs(out_validation)**2))
    return nmse

print("Tamanho dos dados de treinamento:",len(in_training),"x", len(out_training))

optimized_coef = calcCoef(in_training, out_training, n_in_points)

out_estimated = estimatedValueWithLUTS_SIZE(in_validation, optimized_coef)

nmse = calculate_nmse(out_validation, out_estimated)
print(f"NMSE: {nmse:.6f} dB")

plt.scatter(in_validation.real, out_validation.real, label='Dados Originais', color='blue', s=100)
plt.scatter(in_validation.real, out_estimated.real, color='orange', label='Ajuste', alpha=0.8, s=100)
plt.xlabel('in_validation (parte real)', fontsize=30)
plt.ylabel('out_validation (parte real)', fontsize=30)
plt.title('Dados Originais e Estimados com Erros', fontsize=36)
plt.legend(fontsize=30, markerscale=2)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=25)
plt.show()
