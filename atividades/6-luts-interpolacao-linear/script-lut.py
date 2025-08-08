from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

mat = loadmat('in_out_SBRT2_direto.mat')
in_training = mat['in_extraction'].flatten()
out_training = mat['out_extraction'].flatten()
in_validation = mat['in_validation'].flatten()
out_validation = mat['out_validation'].flatten()

lut_in = np.abs(in_training)
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
    sort_idx = np.argsort(lut_in)
    return np.interp(x_abs, lut_in[sort_idx], lut_out[sort_idx])

def residuals(lut_out_real, x_data, y_data):
    lut_out_complex = unpackComplexCoefficients(lut_out_real)
    y_est = estimatedValueWithLUTOptimized(x_data, lut_out_complex)
    return (y_data - y_est).ravel()    


def calcCoef(in_data, out_data, n_in_points):
    initial_complex_coef = np.random.randn(n_in_points) + 1j * np.random.randn(n_in_points)
    initial_real_coef = packComplexCoefficients(initial_complex_coef)

    result = least_squares(residuals, initial_real_coef, args=(in_data, out_data), verbose=2)
    
    return unpackComplexCoefficients(result.x) 

def calcOutOptimized(in_data, coef):
    calcResult = []
    polarCoefs = []
    
    for i in range(len(in_data)):
        calcResult.append(in_data[i] * coef[i])
        
        coefComplexA = np.sqrt(coef[i].real**2 + coef[i].imag**2)  
        coefComplexTeta = np.arctan(coef[i].imag / coef[i].real)
        polarCoefs.append([(coefComplexA, coefComplexTeta)])
    
    
    print("Representação polar dos coeficientes:", polarCoefs[0], "...\n")    
    print("Calculando saída otimizada com coeficientes:", coef, "\n Len:", len(in_data),"x", len(coef))
    return np.array(calcResult)

print("Lut in length:", len(lut_in), "\nTraining len data:",len(in_training),"x", len(out_training))

optimized_coef = calcCoef(in_training, out_training, len(in_training))

out_estimated = calcOutOptimized(in_validation, optimized_coef)

plt.scatter(in_validation.real, out_validation.real, label='Dados Originais', color='blue')
plt.scatter(in_validation.real, out_estimated.real, color='orange', label='Ajuste',alpha=0.8)
plt.xlabel('in_validation (parte real)')
plt.ylabel('out_validation (parte real)')
plt.title('Dados Originais e Estimados com Erros')
plt.legend()
plt.grid()
plt.show()
