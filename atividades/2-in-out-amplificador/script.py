from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

data = loadmat('IN_OUT_PA.mat')

in_data = data['in']
out_data = data['out']

P_VOLTERRA = 10
M_VOLTERRA = 2

in_data = np.array(in_data)

def criaMatrizDeRegressao(in_data, P_VOLTERRA, M_VOLTERRA):
    matrizDeRegressao = []

    for n in range(len(in_data)):
        linhaDaMatrizDeRegressao = []
        
        for p in range(1, P_VOLTERRA + 1):  # Incluir termos até a ordem P_VOLTERRA
            for m in range(M_VOLTERRA + 1):  # Considerar atrasos até M_VOLTERRA
                dataIndex = n - m
                if dataIndex >= 0:  # Verificar se o índice é válido
                    linhaDaMatrizDeRegressao.append(in_data[dataIndex, 0] ** p)
                else:
                    linhaDaMatrizDeRegressao.append(0)  # Adicionar zero se o índice for inválido
        matrizDeRegressao.append(linhaDaMatrizDeRegressao)
        
    return np.array(matrizDeRegressao);



matrizDeRegressao = criaMatrizDeRegressao(in_data, P_VOLTERRA, M_VOLTERRA)


resultado = np.linalg.lstsq(matrizDeRegressao, out_data)

coeficientes = resultado[0]

print("Coeficientes:",coeficientes)

def gerarOndaDeEntrada(fs,time, freq1, freq2, freq3):
    t = np.linspace(0, time, fs) 

    # Onda composta por múltiplas frequências
    ondaDeEntrada =  0.7*np.sin(2 * np.pi * freq1 * t) +0.5* np.sin(2 * np.pi * freq2 * t) + 0.3* np.sin(2 * np.pi * freq3 * t)

    # Normalizando o sinal entre -1 e 1
    ondaDeEntrada = ondaDeEntrada / np.max(np.abs(ondaDeEntrada))
    ondaDeEntrada = ondaDeEntrada.reshape(-1, 1)

    return ondaDeEntrada


#Algumas ondas de exemplo
ondaDeEntrada = gerarOndaDeEntrada(100,0.065,5,10,200) 
#ondaDeEntrada = gerarOndaDeEntrada(100,0.082,5,20,500)
#ondaDeEntrada = gerarOndaDeEntrada(100, 0.04, 10, 15, 100)
#ondaDeEntrada = gerarOndaDeEntrada(100,0.04,10,80,150)
#ondaDeEntrada = gerarOndaDeEntrada(100, 0.053, 7, 30, 250)


matrizDeRegressaoTeste = criaMatrizDeRegressao(ondaDeEntrada, P_VOLTERRA, M_VOLTERRA);

saidaEstimadaDaOnda = np.dot(matrizDeRegressaoTeste, coeficientes);

saidaEstimadaVolterra = np.dot(matrizDeRegressao, coeficientes)


fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Primeiro gráfico
axs[0].plot(in_data, label='Entrada (in_data)')
axs[0].plot(out_data, label='Saída (out_data)')
axs[0].plot(saidaEstimadaVolterra, label='Estimado (saidaEstimadaVolterra)', linestyle='--')
axs[0].set_xlabel('Amostras')
axs[0].set_ylabel('Amplitude')
axs[0].set_title(f"Comparação dos Dados de Entrada, Saída e Estimado para P = {P_VOLTERRA} M = {M_VOLTERRA}")
axs[0].legend()
axs[0].grid(True)


# Segundo gráfico
axs[1].plot(ondaDeEntrada, label='Entrada (ondaDeEntrada)')
axs[1].plot(saidaEstimadaDaOnda, label='Estimado Teste (saidaEstimadaDaOnda)', linestyle='--')
axs[1].set_xlabel('Amostras')
axs[1].set_ylabel('Amplitude')
axs[1].set_title(f"Comparação da onda simulada para a entrada e o comportamento de saída, para P = {P_VOLTERRA} M = {M_VOLTERRA}")
axs[1].legend()
axs[1].grid(True)


plt.tight_layout()
plt.show()
