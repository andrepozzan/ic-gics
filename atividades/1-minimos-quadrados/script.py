import matplotlib.pyplot as plt
import numpy as np
from math import sqrt



vetorEntradasAplicadas = np.array([1.4, 10.4, 19.4, 28.4]);
vetorSaidasDesejadas = np.array([33.62, 95.95, 178.42, 236.70]);


# Criando a matriz de regressão
matrizDeRegressao = np.vstack([vetorEntradasAplicadas, np.ones(len(vetorEntradasAplicadas))]).T

a, b = np.linalg.lstsq(matrizDeRegressao, vetorSaidasDesejadas, rcond=None)[0]

# Estimativa dos valores de y
yEstimado = a * vetorEntradasAplicadas + b

print("Coeficientes da regressão linear:")
print(f"a (inclinação): {a:.5f}".replace(".", ","))
print(f"b (intercepto): {b:.5f}".replace(".", ","))


# Criar o gráfico
fig, ax = plt.subplots()
ax.scatter(vetorEntradasAplicadas, vetorSaidasDesejadas, color='blue', label='Valores reais (yi)', marker='o')
ax.plot(vetorEntradasAplicadas, yEstimado, color='red', label=f'Reta ajustada: y={a:.2f}x + {b:.2f}', linestyle='-')

# Obter os limites dos eixos
x_min, x_max = min(vetorEntradasAplicadas), max(vetorEntradasAplicadas)
y_min, y_max = min(vetorSaidasDesejadas), max(vetorSaidasDesejadas)

# Definir um espaçamento (padding) proporcional
padding_x = (x_max - x_min) * 0.2  # 10% do intervalo do eixo X
padding_y = (y_max - y_min) * 0.2  # 10% do intervalo do eixo Y

# Aplicar os novos limites com padding
ax.set_xlim(x_min - padding_x, x_max + padding_x)
ax.set_ylim(y_min - padding_y, y_max + padding_y)

# Calcular a razão de escala
escala = (y_max - y_min) / (x_max - x_min)

# Exibir a razão de escala no terminal
print(f'Razão de escala (Y/X): {escala:.4f}')

# Adicionar a razão de escala no subtítulo
plt.suptitle(f'Coeficientes: a={a:.2f}, b={b:.2f} | Equação: y={a:.2f}x + {b:.2f} | Razão de escala (Y/X): {escala:.4f}', 
             fontsize=10, y=0.95)

# Títulos e legendas
plt.title('Regressão Linear - Comparação entre valores reais e estimados')
plt.xlabel('xi')
plt.ylabel('yi')
plt.legend()
plt.grid(True)

# Mostrar o gráfico
plt.show()
