from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

loaded_data = loadmat('IN_OUT_PA.mat')
in_training = loaded_data['in']
out_training = loaded_data['out']


#Definições dos parâmetros do modelo
# P - Ordem do polinômio
# M - Profundidade de memória
P, M = 2, 1 

# Estimativas iniciais para os coeficientes:

# initial_estimated = np.zeros(P*(M+1))
# initial_estimated  = np.ones (P*(M+1))
initial_estimated = np.random.randn(P*(M+1))

print(f"P: {P}, M: {M}  \ninitial_estimated: ", np.array(initial_estimated), "\n")

def createRegressionMatrix(in_data, P, M):
    regression_matrix = []

    for n in range(len(in_data)):
        line_of_regression_matrix = []
        
        for p in range(1, P + 1):  # Incluir termos até a ordem P
            for m in range(M + 1):  # Considerar atrasos até M
                dataIndex = n - m
                if dataIndex >= 0:  # Verificar se o índice é válido
                    line_of_regression_matrix.append(in_data[dataIndex, 0] ** p)
                else:
                    line_of_regression_matrix.append(0)  # Adicionar zero se o índice for inválido
        regression_matrix.append(line_of_regression_matrix)
        
    return np.array(regression_matrix);

def mp_residuals(coef, x, y, P, M):
    """
    coef: vetor de coeficientes de tamanho P*(M+1)
    x: sinal de entrada (array 1D)
    y: sinal de saída medido (array 1D)
    P: ordem polinomial
    M: profundidade de memória
    retorna: vetor de resíduos y_est - y
    """
    # Reconstruir h_p(m) em matriz P x (M+1)
    coef_matrix = coef.reshape(P, M + 1)
    y_est = []

    for n in range(len(x)):
        estimated_value = 0.0

        for p in range(1, P + 1):  # Incluir termos até a ordem P
            for m in range(M + 1):  # Considerar atrasos até M
                dataIndex = n - m
                if dataIndex >= 0:  # Verificar se o índice é válido
                    estimated_value += coef_matrix[p - 1, m] * (x[dataIndex] ** p) # Equação de Volterra

        y_est.append(estimated_value)

    # Assim a função de resíduo ajuda o modelo a convergir com base na equação de Volterra.
    return np.array(y_est) - y

def calcCoefByLeastSquares():
    result = least_squares(mp_residuals, initial_estimated, args=(in_training.ravel(), out_training.ravel(), P, M),verbose=2)
    print("\nResultado da otimização:", result)
    coef = result.x
    print("\nCoeficientes resultantes: a =", coef)

    return coef

def calcVectorError(out_estimated):
    vector_error = []
    for i in range(len(in_training)):
        error = out_estimated[i] - out_training[i]
        vector_error.append(error)
        
    return np.array(vector_error)
    
def checkError(vector_error):    
    print("\nErro médio:", np.mean(vector_error), "\nErro máximo em módulo:", np.max(np.abs(vector_error)), "\nErro mínimo em módulo:", np.min(np.abs(vector_error)))
    max_error_index = np.argmax(np.abs(vector_error))
    min_error_index = np.argmin(np.abs(vector_error))
    
    # Pontos de maior e menor erro, para análise visual do modelo
    max_error_point_original = (in_training[max_error_index], out_training[max_error_index])
    max_error_point_estimated = (in_training[max_error_index], out_estimated_volterra[max_error_index])
    min_error_point_original = (in_training[min_error_index], out_training[min_error_index])
    min_error_point_estimated = (in_training[min_error_index], out_estimated_volterra[min_error_index])
    
    return max_error_point_original, max_error_point_estimated, min_error_point_original, min_error_point_estimated

def plotErrorPair(p1, p2, color, label_base, marker1="x", marker2="o", zorder=5):
    x1, y1 = p1[0].item(), p1[1].item()
    x2, y2 = p2[0].item(), p2[1].item()

    plt.scatter(x1, y1, color=color, label=f'{label_base} Original', marker=marker1, s=150, zorder=zorder)
    plt.scatter(x2, y2, color=color, label=f'{label_base} Estimado', marker=marker2, s=80, zorder=zorder)

    plt.plot([x1, x2], [y1, y2], color=color, linestyle='--', linewidth=1.5, label=f'Linha de {label_base}', zorder=zorder-1)

coef = calcCoefByLeastSquares()
regression_matrix = createRegressionMatrix(in_training, P, M)
out_estimated_volterra = np.dot(regression_matrix, coef)

vector_error = calcVectorError(out_estimated_volterra)
max_error_point_original, max_error_point_estimated, min_error_point_original, min_error_point_estimated = checkError(vector_error)

plotErrorPair(max_error_point_original, max_error_point_estimated, color='green', label_base='Erro Máximo')
plotErrorPair(min_error_point_original, min_error_point_estimated, color='red', label_base='Erro Mínimo')

plt.scatter(in_training, out_training, label='Dados Originais', color='blue')
plt.xlabel('in_training')
plt.ylabel('out_training')
plt.title('Dados Originais e Estimados com Erros')
plt.scatter(in_training, out_estimated_volterra, color='orange', label='Ajuste')
plt.legend()
plt.grid()
plt.show()
