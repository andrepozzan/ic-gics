from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


from qam import qam_modulate_passband, qam_demodulate_passband
from ofdma import generate_OFDMA_signal, extract_bits_from_ofdma

from calc_coef import calcCoef, estimatedValueWithLUT, dpdTraining

#Uses flatten to be sure that the data is in 1D format
mat = loadmat('in_out_SBRT2_direto.mat')
in_training = mat['in_extraction'].flatten()
out_training = mat['out_extraction'].flatten()
in_validation = mat['in_validation'].flatten()
out_validation = mat['out_validation'].flatten()


N_USERS = 2
SUB_PER_USER = 4
MOD_ORDER = 16
NFFT = 2048

N_LUT_LINES = 4 #LUT matrix lines
N_LUT_COLUMNS = 4#LUT matrix columns
MATRIZ_SIZE = N_LUT_LINES * N_LUT_COLUMNS

FS = 1000e6  # Taxa de amostragem: 1 MHz
FC = 50e6  # Frequência da portadora: 100 kHz

# Techniques for initializing the LUT coefficients:
# initial_complex_coef = (np.zeros((N_LUT_LINES, N_LUT_COLUMNS)) + 1j * np.zeros((N_LUT_LINES, N_LUT_COLUMNS)))
# initial_complex_coef = (np.ones((N_LUT_LINES, N_LUT_COLUMNS)) + 1j * np.ones((N_LUT_LINES, N_LUT_COLUMNS)))
initial_complex_coef = (np.random.randn(N_LUT_LINES, N_LUT_COLUMNS) + 1j * np.random.randn(N_LUT_LINES, N_LUT_COLUMNS))


interpolation_in = np.linspace(0.0, np.max(np.abs(in_training)), N_LUT_COLUMNS)


#Normalized Mean Square Error
def calculate_nmse(out_validation, saida_estimada):
    erro = out_validation - saida_estimada
    nmse = 10 * np.log10(np.sum(np.abs(erro)**2) / np.sum(np.abs(out_validation)**2))
    return nmse



   

ofdma_signal, bits_enviados = generate_OFDMA_signal(NFFT, N_USERS, SUB_PER_USER, MOD_ORDER)

# Normalizar OFDMA
ofdma_signal = ofdma_signal / np.max(np.abs(ofdma_signal))

# Ajustar ganho para região de compressão do PA
G = 0.7 * np.max(np.abs(in_training))
ofdma_signal = ofdma_signal * G

theoric_signal, t = qam_modulate_passband(ofdma_signal, FC, FS)


optimized_coef = calcCoef(in_training, out_training, N_LUT_LINES, N_LUT_COLUMNS, initial_complex_coef, interpolation_in )


dpd_coef = dpdTraining(in_training, out_training, N_LUT_LINES, N_LUT_COLUMNS, initial_complex_coef, interpolation_in)
out_dpd = estimatedValueWithLUT(ofdma_signal, dpd_coef, interpolation_in, N_LUT_LINES)


out_of_amp = estimatedValueWithLUT(out_dpd, optimized_coef, interpolation_in, N_LUT_LINES)

out_of_antena, _ = qam_modulate_passband(out_of_amp, FC, FS)
demodulate_antena = qam_demodulate_passband(out_of_antena, FC, FS)



bits_recebidos = extract_bits_from_ofdma(demodulate_antena, NFFT, N_USERS, SUB_PER_USER, MOD_ORDER)


print("Bits Enviados por Usuário:")
for i, bits in enumerate(bits_enviados):
    print(f"Usuário {i}: {bits}")
    
print("\nBits Recebidos por Usuário:")
for i, bits in enumerate(bits_recebidos):
    print(f"Usuário {i}: {bits}")

#NMSE
out_estimated = estimatedValueWithLUT(in_validation, optimized_coef, interpolation_in, N_LUT_LINES)
nmse = calculate_nmse(out_validation, out_estimated)

print(f"NMSE: {nmse:.6f} dB")


plt.plot(t, theoric_signal, label='Original (Referência)', color='blue', linewidth=2, alpha=0.8)
plt.plot(t, out_of_antena, label='Saída do PA com DPD', color='red', linestyle='--', linewidth=2)
plt.title("Validação da Cascata: Original vs Saída do Sistema")

# plt.figure()
# plt.hist(np.abs(in_training), bins=N_LUT_COLUMNS, color='skyblue', edgecolor='black')
# plt.axhline(y=N_LUT_LINES * 10, color='red', linestyle='--', label='Mínimo Recomendado')
# plt.title(f"Distribuição de Amostras por Coluna da LUT ({N_LUT_COLUMNS} colunas)")
# plt.xlabel("Amplitude |x|")
# plt.ylabel("Número de Amostras")
# plt.legend()


amp_input = np.abs(ofdma_signal)
amp_output_dpd = np.abs(out_of_amp)
out_no_dpd = estimatedValueWithLUT(ofdma_signal, optimized_coef, interpolation_in, N_LUT_LINES)
amp_output_no_dpd = np.abs(out_no_dpd)
plt.figure(figsize=(10, 7))
plt.scatter(amp_input, amp_output_no_dpd, s=80, label='PA sem DPD', alpha=0.8)
plt.scatter(amp_input, amp_output_dpd, s=80, label='PA com DPD (Cascata)', alpha=0.8)
lim = np.max(amp_input)
plt.plot([0, lim], [0, lim], 'r--', label='Referência Linear', linewidth=2)
plt.xlabel("Amplitude de Entrada")
plt.ylabel("Amplitude de Saı́da")
plt.title("Gráfico AM-AM")




plt.legend()
plt.grid()
plt.show()

