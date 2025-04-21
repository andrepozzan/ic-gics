import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


vetorEntradasAplicadas = np.array([1.4, 10.4, 19.4, 28.4]);
vetorSaidasDesejadas = np.array([33.62, 95.95, 178.42, 236.70]);


x_media = np.mean(vetorEntradasAplicadas)
y_media = np.mean(vetorSaidasDesejadas)

delta_x = vetorEntradasAplicadas - x_media
delta_y = vetorSaidasDesejadas - y_media
delta_xy = delta_x * delta_y
delta_x2 = delta_x**2

soma_delta_xy = np.sum(delta_xy)
soma_delta_x2 = np.sum(delta_x2)

a = soma_delta_xy / soma_delta_x2
b = y_media - a * x_media

tabela_mmq = pd.DataFrame({
    "x": vetorEntradasAplicadas,
    "y": vetorSaidasDesejadas,
    "x̄": [x_media] + [''] * (len(vetorEntradasAplicadas) - 1),
    "ȳ": [y_media] + [''] * (len(vetorEntradasAplicadas) - 1),
    "Σ(x)": [np.sum(vetorEntradasAplicadas)] + [''] * (len(vetorEntradasAplicadas) - 1),
    "Σ(y)": [np.sum(vetorSaidasDesejadas)] + [''] * (len(vetorEntradasAplicadas) - 1),
    "δxi = (xi - x̄)": delta_x,
    "δyi = (yi - ȳ)": delta_y,
    "δxi²": delta_x2,
    "Σ(δxi²)": [soma_delta_x2] + [''] * (len(vetorEntradasAplicadas) - 1),
    "δxi * δyi": delta_xy,
    "Σ(δxi * δyi)": [soma_delta_xy] + [''] * (len(vetorEntradasAplicadas) - 1),
    "Coeficiente a": [a] + [''] * (len(vetorEntradasAplicadas) - 1),
    "Coeficiente b": [b] + [''] * (len(vetorEntradasAplicadas) - 1)
})

yEstimado = a * vetorEntradasAplicadas + b


fig, ax = plt.subplots()
ax.scatter(vetorEntradasAplicadas, vetorSaidasDesejadas, color='blue', label='Valores reais (yi)', marker='o')
ax.plot(vetorEntradasAplicadas, yEstimado, color='red', label='Valores estimados (ŷi)', linewidth=2)

x_min, x_max = min(vetorEntradasAplicadas), max(vetorEntradasAplicadas)
y_min, y_max = min(vetorSaidasDesejadas), max(vetorSaidasDesejadas)
padding_x = (x_max - x_min) * 0.2
padding_y = (y_max - y_min) * 0.2
ax.set_xlim(x_min - padding_x, x_max + padding_x)
ax.set_ylim(y_min - padding_y, y_max + padding_y)

escala = (y_max - y_min) / (x_max - x_min)

plt.suptitle(f'Coeficientes: a={a:.2f}, b={b:.2f} | Equação: y={a:.2f}x + {b:.2f} | Razão de escala: {escala:.4f}', 
             fontsize=10, y=0.95)

with open("mmq-tabela-saida.txt", "w", encoding="utf-8") as f:
    f.write(tabela_mmq.to_string(float_format=lambda x: f"{x:.5f}"))

tabela_mmq.to_csv('mmq-tabela-saida.csv', index=False, encoding='utf-8')


plt.title('Regressão Linear - Comparação entre valores reais e estimados')
plt.xlabel('xi (cm³)')
plt.ylabel('yi (g)')
plt.legend()
plt.grid(True)

plt.show()

