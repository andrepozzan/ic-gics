from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

loaded_data = loadmat('in_out_SBRT2_direto.mat')
in_training = loaded_data['in_extraction']
out_training = loaded_data['out_extraction']

in_validation = loaded_data['in_validation']
out_validation = loaded_data['out_validation']


#Definições dos parâmetros do modelo
# P - Ordem do polinômio
# M - Profundidade de memória
P, M = 5, 3

# Coeficientes iniciais complexos (ex: aleatórios)

# initial_complex = np.zeros(P*(M+1)) + 1j * np.zeros(P*(M+1))
# initial_complex = np.ones(P*(M+1)) + 1j * np.ones(P*(M+1))
initial_complex = np.random.randn(P*(M+1)) + 1j * np.random.randn(P*(M+1))

# Vetor real concatenando parte real e imaginária
initial_real = np.concatenate((initial_complex.real, initial_complex.imag))

print(f"P: {P}, M: {M}  \ninitial_estimated: ", np.array(initial_real), "\n")

def estimatedValueWithComplex(x_data, coef_matrix, P, M):
    amplitude_entrada = np.abs(x_data[dataIndex])
    y_est = []

    for n in range(len(x_data)):
        estimated_value = 0.0 + 0.0j
        for p in range(1, P + 1):
            for m in range(M + 1):
                dataIndex = max(0, n - m)
                power = 2*p - 2
                
                term = (np.abs(x_data[dataIndex])**power) * x_data[dataIndex]
                estimated_value += term * coef_matrix[p-1, m]
        y_est.append(estimated_value)

    return np.array(y_est)

def unpackComplexCoefficients(x_real, P, M):
    # Metade do vetor são os componentes reais e a outra metade são os imaginários
    real_parts = x_real[:len(x_real)//2]
    imag_parts = x_real[len(x_real)//2:]
    
    complex_coef = real_parts + 1j * imag_parts
    return complex_coef.reshape((P, M+1))


def mpResiduals(x_real, x_data, y_data, P, M):
    coef_matrix = unpackComplexCoefficients(x_real, P, M)

    y_est = estimatedValueWithComplex(x_data, coef_matrix, P, M)
  
    resid = y_data.ravel() - y_est

    return np.concatenate((resid.real, resid.imag))

def calcCoefByLeastSquares(in_data, out_data):
    result = least_squares(mpResiduals, initial_real, args=(in_data.ravel(), out_data.ravel(), P, M),verbose=2)
    final_coef = unpackComplexCoefficients(result.x, P, M)
    
    print("\nResultado da otimização:", final_coef)

    return final_coef

def calcVectorError(out_estimated, out_data):
    vector_error = []
    for i in range(len(out_data)):
        error = out_estimated[i] - out_data[i]
        vector_error.append(error)
        
    return np.abs(vector_error)
    
def checkError(vector_error, in_data, out_data, out_estimated_data):
    """Exibe métricas de erro e retorna os pontos de maior e menor erro, tanto real quanto estimado."""
    
    mean_error = np.mean(vector_error)
    max_abs_error = np.max(np.abs(vector_error))
    min_abs_error = np.min(np.abs(vector_error))
    
    print("\nErro médio:", mean_error)
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
    x1, y1 = p1[0].item(), p1[1].item()
    x2, y2 = p2[0].item(), p2[1].item()

    plt.scatter(x1, y1, color=color, label=f'{label_base} Original', marker=marker1, s=150, zorder=zorder)
    plt.scatter(x2, y2, color=color, label=f'{label_base} Estimado', marker=marker2, s=80, zorder=zorder)

    plt.plot([x1, x2], [y1, y2], color=color, linestyle='--', linewidth=1.5, label=f'Linha de {label_base}', zorder=zorder-1)


coef = calcCoefByLeastSquares(in_training, out_training)

out_estimated = estimatedValueWithComplex(in_validation.ravel(), coef, P, M)

vector_error = calcVectorError(out_estimated, out_validation)

max_error_point_original, max_error_point_estimated, min_error_point_original, min_error_point_estimated = checkError(vector_error, in_validation, out_validation, out_estimated)


# Exibição

plotErrorPair(max_error_point_original, max_error_point_estimated, color='green', label_base='Erro Máximo')
plotErrorPair(min_error_point_original, min_error_point_estimated, color='red', label_base='Erro Mínimo')

plt.scatter(in_validation, out_validation, label='Dados Originais', color='blue')
plt.scatter(in_validation, out_estimated, color='orange', label='Ajuste',alpha=0.8)
plt.xlabel('in_validation')
plt.ylabel('out_validation')
plt.title('Dados Originais e Estimados com Erros')
plt.legend()
plt.grid()
plt.show()
