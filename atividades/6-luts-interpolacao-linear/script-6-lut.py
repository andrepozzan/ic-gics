from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

mat = loadmat('in_out_SBRT2_direto.mat')
in_training = mat['in_extraction'].flatten()
out_training = mat['out_extraction'].flatten()
in_validation = mat['in_validation'].flatten()
out_validation = mat['out_validation'].flatten()

n_in_points = 80
lut_in = np.linspace(0.0, np.max(np.abs(in_training)), n_in_points)
lut_out = out_training

def unpackComplexCoefficients(real_coef):
    # Metade do vetor são os componentes reais e a outra metade são os imaginários
    real_parts = real_coef[:len(real_coef)//2]
    imag_parts = real_coef[len(real_coef)//2:]
    complex_coef = real_parts + 1j * imag_parts
    return complex_coef  # Retorna vetor 1D de coeficientes complexos

def packComplexCoefficients(complex_coef):
    # Metade do vetor são os componentes reais e a outra metade são os imaginários
    real_parts = complex_coef.real
    imag_parts = complex_coef.imag
    real_coef = np.concatenate([real_parts, imag_parts])
    return real_coef

def estimatedValueWithLUTOptimized(x_data, lut_out):
    x_abs = np.abs(x_data)
    real_interp = np.interp(x_abs, lut_in, lut_out.real)
    imag_interp = np.interp(x_abs, lut_in, lut_out.imag)
    result = x_data * (real_interp + 1j * imag_interp)
    return result

def residuals(lut_out_real, x_data, y_data):
    lut_out_complex = unpackComplexCoefficients(lut_out_real)
    y_est = estimatedValueWithLUTOptimized(x_data, lut_out_complex)
    res = y_data - y_est
    res_vec = np.concatenate([res.real, res.imag])
    return res_vec

def calcCoef(in_data, out_data, n_in_points):
    initial_complex_coef = np.random.randn(n_in_points) + 1j * np.random.randn(n_in_points)
    initial_real_coef = packComplexCoefficients(initial_complex_coef)
    result = least_squares(residuals, initial_real_coef, args=(in_data, out_data), verbose=2)
    return unpackComplexCoefficients(result.x) 

def calcOutOptimized(in_data, coef):
    # Interpolação separada para parte real e imaginária
    coef_real_interp = np.interp(np.abs(in_data), lut_in, coef.real)
    coef_imag_interp = np.interp(np.abs(in_data), lut_in, coef.imag)
    coef_interp = coef_real_interp + 1j * coef_imag_interp

    # Calcula saída
    calcResult = in_data * coef_interp

    # Calcula coeficientes polares (apenas para debug)
    coefComplexA = np.abs(coef_interp)
    coefComplexTeta = np.angle(coef_interp)
    polarCoefs = list(zip(coefComplexA, coefComplexTeta))

    print("Representação polar do primeiro coeficiente:", polarCoefs[0], "...\n")    
    print("Calculando saída otimizada com", len(in_data), "amostras e LUT de", len(coef), "pontos")
    
    return calcResult

def calculate_nmse(out_validation, saida_estimada):
    erro = out_validation - saida_estimada
    nmse = 10 * np.log10(np.sum(np.abs(erro)**2) / np.sum(np.abs(out_validation)**2))
    return nmse

print("Tamanho dos dados de treinamento:",len(in_training),"x", len(out_training))

optimized_coef = calcCoef(in_training, out_training, n_in_points)

out_estimated = calcOutOptimized(in_validation, optimized_coef)

nmse = calculate_nmse(out_validation, out_estimated)
print(f"NMSE: {nmse:.6f} dB")

plt.scatter(in_validation.real, out_validation.real, label='Dados Originais', color='blue')
plt.scatter(in_validation.real, out_estimated.real, color='orange', label='Ajuste', alpha=0.8)
plt.xlabel('in_validation (parte real)')
plt.ylabel('out_validation (parte real)')
plt.title('Dados Originais e Estimados com Erros')
plt.legend()
plt.grid()
plt.show()
