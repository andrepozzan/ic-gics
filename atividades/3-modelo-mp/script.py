from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

data = loadmat('in_out_SBRT2_direto.mat')

in_extraction = data['in_extraction']
out_extraction = data['out_extraction']

in_validation = data['in_validation']
out_validation = data['out_validation']

P_VOLTERRA = 5
M_VOLTERRA = 1

in_extraction = np.array(in_extraction)

def criaMatrizDeRegressao(in_data, P_VOLTERRA, M_VOLTERRA):
    matrizDeRegressao = []

    for n in range(len(in_data)):
        linhaDaMatrizDeRegressao = []
        
        for p in range(1, P_VOLTERRA + 1):  
            for m in range(M_VOLTERRA + 1): 
                dataIndex = max(0, n - m)
                power = 2*p-2
                
                numeroCalculado = (abs(in_data[dataIndex]**power)) * in_data[dataIndex]
                numeroCalculado = numeroCalculado.item()
                                        
                linhaDaMatrizDeRegressao.append(numeroCalculado)
        matrizDeRegressao.append(linhaDaMatrizDeRegressao)

    return np.array(matrizDeRegressao);



matrizDeRegressaoExtraction = criaMatrizDeRegressao(in_extraction, P_VOLTERRA, M_VOLTERRA)
resultado = np.linalg.lstsq(matrizDeRegressaoExtraction, out_extraction, rcond=None)
coeficientes = resultado[0]
print("Coeficientes:",coeficientes)



matrizDeRegressaoValidation = criaMatrizDeRegressao(in_validation, P_VOLTERRA, M_VOLTERRA)
saidaEstimadaPolinomio = np.dot(matrizDeRegressaoValidation, coeficientes)


nmse = 10*np.log10(np.sum((out_validation - saidaEstimadaPolinomio)**2)/np.sum(out_validation**2))
print(f"NMSE: {nmse} dB")

# Primeiro gráfico
plt.plot(in_validation, label='Entrada (in_validation)')
plt.plot(out_validation, label='Saída (out_validation)')
plt.plot(saidaEstimadaPolinomio, label='Estimado (saidaEstimadaPolinomio)', linestyle='--')
plt.xlabel('Amostras')
plt.ylabel('Amplitude')
plt.title(f"Comparação dos Dados de Entrada, Saída e Estimado para P = {P_VOLTERRA} M = {M_VOLTERRA}")
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()

plt.scatter(np.abs(in_validation), np.abs(out_validation), label="Dados Medidos")
plt.scatter(np.abs(in_validation), np.abs(saidaEstimadaPolinomio), label="Estimado", alpha=0.7)
plt.xlabel("Amplitude de Entrada")
plt.ylabel("Amplitude de Saída")
plt.title("Gráfico AM-AM")
plt.legend()
plt.grid()
plt.show()


#Variação da fase de entrada para a de saída
fase_out = np.angle(out_validation)
fase_in = np.angle(in_validation)
delta_fase = np.degrees(fase_out - fase_in)  # Converte para graus

fase_out_estimada = np.angle(saidaEstimadaPolinomio)
delta_fase_estimada = np.degrees(fase_out_estimada - fase_in)  # Converte para graus

plt.scatter(np.abs(in_validation), delta_fase, label="Diferença de Fase (AM-PM)")
plt.scatter(np.abs(in_validation), delta_fase_estimada, label="Estimado", alpha=0.7)
plt.xlabel("Amplitude de Entrada")
plt.ylabel("Diferença de Fase (graus)")
plt.title("Gráfico AM-PM")
plt.legend()
plt.grid()
plt.show()
