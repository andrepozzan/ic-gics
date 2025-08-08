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

def estimatedValueWithLUT(x_data):
    """Estima valores usando interpolação linear com LUT"""
    y_est = []

    for n in range(len(x_data)):
        x = np.abs(x_data[n])
        distances = np.abs(lut_in - x)
        
        idxs = np.argsort(distances)[:2]
        x0, x1 = lut_in[idxs[0]], lut_in[idxs[1]]
        y0, y1 = lut_out[idxs[0]], lut_out[idxs[1]]
        
        # Evita divisão por zero
        if np.isclose(x1, x0):
            estimated_value = y0
        else:
            estimated_value = (y1 - y0) / (x1 - x0) * (x - x0) + y0

        y_est.append(estimated_value)

    return np.array(y_est)

def estimatedValueWithLUTOptimized(x_data, lut_values):
    """Estima valores usando interpolação linear com LUT otimizada"""
    y_est = []
    
    # Cria valores de entrada da LUT baseados na magnitude dos dados de treinamento
    lut_input_values = np.linspace(np.min(np.abs(lut_in)), np.max(np.abs(lut_in)), len(lut_values))
    
    for n in range(len(x_data)):
        x = np.abs(x_data[n])
        
        # Encontra os dois pontos mais próximos na LUT
        distances = np.abs(lut_input_values - x)
        idxs = np.argsort(distances)[:2]
        
        x0, x1 = lut_input_values[idxs[0]], lut_input_values[idxs[1]]
        y0, y1 = lut_values[idxs[0]], lut_values[idxs[1]]
        
        # Evita divisão por zero
        if np.isclose(x1, x0):
            estimated_value = y0
        else:
            estimated_value = (y1 - y0) / (x1 - x0) * (x - x0) + y0

        y_est.append(estimated_value)

    return np.array(y_est)

def lutResiduals(lut_params, x_data, y_data):
    # Usa a função unpackComplexCoefficients para desmontar os números complexos
    n_lut = len(lut_params) // 2
    lut_values = unpackComplexCoefficients(lut_params, n_lut)
    
    # Estima usando interpolação linear da LUT
    y_est = estimatedValueWithLUTOptimized(x_data, lut_values)
    
    resid = y_data - y_est
    return np.concatenate((resid.real, resid.imag))

def calcLUTByLeastSquares(in_data, out_data, n_lut_points=100):
    # Inicialização da LUT com interpolação dos dados de treinamento
    inital_complex = np.interp(
        np.linspace(np.min(np.abs(in_data)), np.max(np.abs(in_data)), n_lut_points),
        np.abs(in_data), 
        out_data
    )
    
    print("Estimativa inicial dos valores coef", inital_complex)

    # Vetor real concatenando parte real e imaginária da LUT
    inital_real = np.concatenate((inital_complex.real, inital_complex.imag))
    
    result = least_squares(lutResiduals, inital_real, args=(in_data, out_data), verbose=2)
    
    # Reconstrói a LUT otimizada
    n_lut = len(result.x) // 2
    lut_real = result.x[:n_lut]
    lut_imag = result.x[n_lut:]
    final_lut = lut_real + 1j * lut_imag
    
    print("\nLUT otimizada concluída")
    print(f"Número de pontos na LUT: {len(final_lut)}")

    return final_lut

def calcNMSE(out_estimated, out_validation):
    # Para números complexos, calculamos a magnitude quadrada
    error_power = np.sum(np.abs(out_validation - out_estimated)**2)
    signal_power = np.sum(np.abs(out_validation)**2)
    nmse = 10*np.log10(error_power / signal_power)
    return nmse

def calcVectorError(out_estimated, out_data):
    """Calcula o erro elemento a elemento"""
    vector_error = out_data - out_estimated
    return vector_error
    
def checkError(vector_error, in_data, out_data, out_estimated_data):
    """Exibe métricas de erro e retorna os pontos de maior e menor erro, tanto real quanto estimado."""
    
    # Calcula NMSE
    nmse = calcNMSE(out_estimated_data, out_data)
    
    mean_error = np.mean(np.abs(vector_error))
    max_abs_error = np.max(np.abs(vector_error))
    min_abs_error = np.min(np.abs(vector_error))
    
    print(f"\nNMSE: {nmse:.6f}")
    print("Erro médio:", mean_error)
    print("Erro máximo em módulo:", max_abs_error)
    print("Erro mínimo em módulo:", min_abs_error)
    
    max_idx = np.argmax(np.abs(vector_error))
    min_idx = np.argmin(np.abs(vector_error))

    # Pontos de erro máximo
    max_error_point_original = (in_data[max_idx], out_data[max_idx])
    max_error_point_estimated = (in_data[max_idx], out_estimated_data[max_idx])
    
    # Pontos de erro mínimo
    min_error_point_original = (in_data[min_idx], out_data[min_idx])
    min_error_point_estimated = (in_data[min_idx], out_estimated_data[min_idx])
    
    return (
        max_error_point_original,
        max_error_point_estimated,
        min_error_point_original,
        min_error_point_estimated
    )
    
def plotErrorPair(p1, p2, color, label_base, marker1="x", marker2="o", zorder=5):
    # Para dados complexos, plotamos apenas a parte real
    x1, y1 = p1[0].real, p1[1].real
    x2, y2 = p2[0].real, p2[1].real

    plt.scatter(x1, y1, color=color, label=f'{label_base} Original', marker=marker1, s=150, zorder=zorder)
    plt.scatter(x2, y2, color=color, label=f'{label_base} Estimado', marker=marker2, s=80, zorder=zorder)

    plt.plot([x1, x2], [y1, y2], color=color, linestyle='--', linewidth=1.5, label=f'Linha de {label_base}', zorder=zorder-1)


# Otimiza a LUT usando least squares
lut_optimized = calcLUTByLeastSquares(in_training, out_training)

# Estimação usando interpolação linear da LUT otimizada
print("\nEstimando valores com interpolação linear da LUT otimizada...")
out_estimated = estimatedValueWithLUTOptimized(in_validation, lut_optimized)



# Exibição dos resultados
vector_error = calcVectorError(out_estimated, out_validation)
max_error_point_original, max_error_point_estimated, min_error_point_original, min_error_point_estimated = checkError(vector_error, in_validation, out_validation, out_estimated)

plotErrorPair(max_error_point_original, max_error_point_estimated, color='green', label_base='Erro Máximo')
plotErrorPair(min_error_point_original, min_error_point_estimated, color='red', label_base='Erro Mínimo')

plt.scatter(in_validation.real, out_validation.real, label='Dados Originais', color='blue')
plt.scatter(in_validation.real, out_estimated.real, color='orange', label='Ajuste',alpha=0.8)
plt.xlabel('in_validation (parte real)')
plt.ylabel('out_validation (parte real)')
plt.title('Dados Originais e Estimados com Erros')
plt.legend()
plt.grid()
plt.show()

def unpackComplexCoefficients(x_real, n_lut_points):
    # Metade do vetor são os componentes reais e a outra metade são os imaginários
    real_parts = x_real[:len(x_real)//2]
    imag_parts = x_real[len(x_real)//2:]
    
    complex_coef = real_parts + 1j * imag_parts
    return complex_coef  # Retorna vetor 1D de coeficientes complexos
