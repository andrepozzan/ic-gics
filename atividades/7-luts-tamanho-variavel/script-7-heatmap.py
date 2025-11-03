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


# Parâmetros para varrer
n_in_points_list = list(range(5, 125, 5))  # Exemplo: 20, 40, ..., 120
LUTS_SIZE_list = list(range(2, 10))         # 1 a 10

# Matriz para armazenar NMSE
nmse_matrix = np.zeros((len(n_in_points_list), len(LUTS_SIZE_list)))


# --- Funções adaptadas para aceitar LUTS_SIZE como argumento ---
def unpackComplexCoefficients(real_coef, LUTS_SIZE):
    n = LUTS_SIZE * n_in_points
    real_parts = real_coef[:n].reshape(LUTS_SIZE, n_in_points)
    imag_parts = real_coef[n:].reshape(LUTS_SIZE, n_in_points)
    complex_coef = real_parts + 1j * imag_parts
    return complex_coef

def packComplexCoefficients(complex_coef):
    real_parts = complex_coef.real.flatten()
    imag_parts = complex_coef.imag.flatten()
    real_coef = np.concatenate([real_parts, imag_parts])
    return real_coef

def estimatedValueWithLUTS_SIZE(x_data, lut_out_matrix, LUTS_SIZE):
    n = len(x_data)
    result = np.zeros(n, dtype=complex)
    for M in range(LUTS_SIZE):
        delayed = np.roll(x_data, M)
        delayed[:M] = 0  # zera início (sem histórico real)
        x_abs = np.abs(delayed)
        real_interp = np.interp(x_abs, lut_in, lut_out_matrix[M].real)
        imag_interp = np.interp(x_abs, lut_in, lut_out_matrix[M].imag)
        result += delayed * (real_interp + 1j * imag_interp)
    return result

def residuals(lut_out_real, x_data, y_data, LUTS_SIZE):
    lut_out_complex = unpackComplexCoefficients(lut_out_real, LUTS_SIZE)
    y_est = estimatedValueWithLUTS_SIZE(x_data, lut_out_complex, LUTS_SIZE)
    res = y_data - y_est
    res_vec = np.concatenate([res.real, res.imag])
    return res_vec

def calcCoef(in_data, out_data, n_in_points, LUTS_SIZE):
    initial_complex_coef = (np.random.randn(LUTS_SIZE, n_in_points) +
                            1j * np.random.randn(LUTS_SIZE, n_in_points))
    initial_real_coef = packComplexCoefficients(initial_complex_coef)
    result = least_squares(residuals, initial_real_coef, args=(in_data, out_data, LUTS_SIZE), verbose=0)
    return unpackComplexCoefficients(result.x, LUTS_SIZE)

def calculate_nmse(out_validation, saida_estimada):
    erro = out_validation - saida_estimada
    nmse = 10 * np.log10(np.sum(np.abs(erro)**2) / np.sum(np.abs(out_validation)**2))
    return nmse


print("Tamanho dos dados de treinamento:", len(in_training), "x", len(out_training))

for i_n, n_in_points in enumerate(n_in_points_list):
    lut_in = np.linspace(0.0, np.max(np.abs(in_training)), n_in_points)
    for j_l, LUTS_SIZE in enumerate(LUTS_SIZE_list):
        print(f"Calculando para n_in_points = {n_in_points}, LUTS_SIZE = {LUTS_SIZE}...")
        optimized_coef = calcCoef(in_training, out_training, n_in_points, LUTS_SIZE)
        out_estimated = estimatedValueWithLUTS_SIZE(in_validation, optimized_coef, LUTS_SIZE)
        nmse = calculate_nmse(out_validation, out_estimated)
        print(f"NMSE (n_in_points={n_in_points}, LUTS_SIZE={LUTS_SIZE}): {nmse:.6f} dB")
        nmse_matrix[i_n, j_l] = nmse

# Para o gráfico de dispersão, salva o último resultado
last_out_estimated = out_estimated


# --- Heatmap 2D ---
plt.figure(figsize=(10, 6))
# Usar colormap 'plasma' para maior contraste
im = plt.imshow(nmse_matrix, aspect='auto', cmap='plasma',
               origin='lower',
               extent=[min(LUTS_SIZE_list)-0.5, max(LUTS_SIZE_list)+0.5, min(n_in_points_list)-0.5, max(n_in_points_list)+0.5])
plt.colorbar(im, label='NMSE (dB)')
plt.yticks(n_in_points_list)
plt.xticks(LUTS_SIZE_list)
plt.xlabel('LUTS_SIZE', fontsize=16)
plt.ylabel('n_in_points', fontsize=16)
plt.title('Heatmap NMSE vs LUTS_SIZE e n_in_points', fontsize=18)
# Adiciona valores no heatmap com cor adaptativa para contraste
import matplotlib
colormap = plt.get_cmap('plasma')
for i in range(len(n_in_points_list)):
    for j in range(len(LUTS_SIZE_list)):
        val = nmse_matrix[i, j]
        # Obtém cor RGBA do pixel
        norm_val = (val - nmse_matrix.min()) / (nmse_matrix.max() - nmse_matrix.min() + 1e-12)
        rgba = colormap(norm_val)
        # Calcula luminância para decidir cor do texto
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text_color = 'black' if luminance > 0.5 else 'white'
        plt.text(LUTS_SIZE_list[j], n_in_points_list[i], f"{val:.2f}",
                 ha='center', va='center', color=text_color, fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()


# --- (Opcional) Gráfico de dispersão para o último caso testado ---
plt.figure()
plt.scatter(in_validation.real, out_validation.real, label='Dados Originais', color='blue', s=100)
plt.scatter(in_validation.real, last_out_estimated.real, color='orange', label='Ajuste', alpha=0.8, s=100)
plt.xlabel('in_validation (parte real)', fontsize=30)
plt.ylabel('out_validation (parte real)', fontsize=30)
plt.title(f'Dados Originais e Estimados\n(n_in_points={n_in_points_list[-1]}, LUTS_SIZE={LUTS_SIZE_list[-1]})', fontsize=22)
plt.legend(fontsize=20, markerscale=2)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=18)
plt.show()
