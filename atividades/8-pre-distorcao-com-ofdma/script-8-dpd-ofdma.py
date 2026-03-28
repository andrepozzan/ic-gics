from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

#Uses flatten to be sure that the data is in 1D format
mat = loadmat('in_out_SBRT2_direto.mat')
in_training = mat['in_extraction'].flatten()
out_training = mat['out_extraction'].flatten()
in_validation = mat['in_validation'].flatten()
out_validation = mat['out_validation'].flatten()

# ==== START Initial definition os Parameters ====
N_USERS = 8
SUB_PER_USER = 8
MOD_ORDER = 256
NFFT = 2048

N_LUT_LINES = 3 #LUT matrix lines
N_LUT_COLUMNS = 3#LUT matrix columns
MATRIZ_SIZE = N_LUT_LINES * N_LUT_COLUMNS

# Parâmetros de modulação
fs = 1e6  # Taxa de amostragem: 1 MHz
fc = 10e3  # Frequência da portadora: 100 kHz

# Techniques for initializing the LUT coefficients:

# initial_complex_coef = (np.zeros((N_LUT_LINES, N_LUT_COLUMNS)) + 1j * np.zeros((N_LUT_LINES, N_LUT_COLUMNS)))
# initial_complex_coef = (np.ones((N_LUT_LINES, N_LUT_COLUMNS)) + 1j * np.ones((N_LUT_LINES, N_LUT_COLUMNS)))
initial_complex_coef = (np.random.randn(N_LUT_LINES, N_LUT_COLUMNS) + 1j * np.random.randn(N_LUT_LINES, N_LUT_COLUMNS))

# ==== END Initial definition os Parameters ====

# Get the maximum absolute value from the training input data to define the interpolation range
interpolation_in = np.linspace(0.0, np.max(np.abs(in_training)), N_LUT_COLUMNS)



# Complex number tools:
def unpackComplexCoefficients(real_coef):
    real_parts = real_coef[:MATRIZ_SIZE].reshape(N_LUT_LINES, N_LUT_COLUMNS)
    imag_parts = real_coef[MATRIZ_SIZE:].reshape(N_LUT_LINES, N_LUT_COLUMNS)
    complex_coef = real_parts + 1j * imag_parts
    return complex_coef

def packComplexCoefficients(complex_coef):
    real_parts = complex_coef.real.flatten()
    imag_parts = complex_coef.imag.flatten()
    real_coef = np.concatenate([real_parts, imag_parts])
    return real_coef



def estimatedValueWithLUT(x_data, lut_out_matrix):
    length_of_x_data = len(x_data)
    result = np.zeros(length_of_x_data, dtype=complex)
    
    for line in range(N_LUT_LINES):
        # Move the elements of an array in a circular pattern.
        delayed = np.roll(x_data, line)
        
        # Zero out the first 'line' elements to create a memory effect
        delayed[:line] = 0

        x_abs = np.abs(delayed)
        real_interpolation = np.interp(x_abs, interpolation_in, lut_out_matrix[line].real)
        imag_interpolation = np.interp(x_abs, interpolation_in, lut_out_matrix[line].imag)
        result += delayed * (real_interpolation + 1j * imag_interpolation)
    return result


def residuals(lut_out_real, x_data, y_data):
    lut_out_complex = unpackComplexCoefficients(lut_out_real)
    y_estimated = estimatedValueWithLUT(x_data, lut_out_complex)
    residual = y_data - y_estimated
    res_vector = np.concatenate([residual.real, residual.imag])
    return res_vector

def calcCoef(in_data, out_data):
    initial_real_coef = packComplexCoefficients(initial_complex_coef)
    result = least_squares(residuals, initial_real_coef, args=(in_data, out_data), verbose=2)
    return unpackComplexCoefficients(result.x) 

#Normalized Mean Square Error
def calculate_nmse(out_validation, saida_estimada):
    erro = out_validation - saida_estimada
    nmse = 10 * np.log10(np.sum(np.abs(erro)**2) / np.sum(np.abs(out_validation)**2))
    return nmse





# ===== START OFDMA SECTION ======

def qam_mod(bits, modulation_order):
    
    k = int(np.log2(modulation_order)) 


    k_half = k // 2
    n_levels = 2**k_half
    
    # Adjust the bit string to be a multiple of k (bits per symbol)
    bits = list(bits)
    if len(bits) % k != 0:
        bits += ['0'] * (k - (len(bits) % k))

    def binary_to_gray(n):
        return n ^ (n >> 1)

    levels = np.arange(n_levels)
    gray_levels = binary_to_gray(levels)
    
    mapping_i = np.zeros(n_levels)
    pam_levels = 2 * np.arange(n_levels) - (n_levels - 1)
    for i, g in enumerate(gray_levels):
        mapping_i[g] = pam_levels[i]

    mapping_q = np.zeros(n_levels)
    for i, g in enumerate(gray_levels):
        mapping_q[g] = pam_levels[n_levels - 1 - i]

    # 5. Processamento dos bits
    symbols = []
    for i in range(0, len(bits), k):
        bit_group = bits[i:i+k]
        
        # Divide o grupo: metade para I, metade para Q
        bits_i = "".join(bit_group[:k_half])
        bits_q = "".join(bit_group[k_half:])
        
        # Converte para decimal
        idx_i = int(bits_i, 2)
        idx_q = int(bits_q, 2)
        
        # Mapeia para a constelação usando o mapeamento de Gray
        comp_i = mapping_i[idx_i]
        comp_q = mapping_q[idx_q]
        
        symbols.append(complex(comp_i, comp_q))

    return np.array(symbols)

def qam_demod(symbols, modulation_order):
    k = int(np.log2(modulation_order))
    k_half = k // 2
    n_levels = 2 ** k_half

    pam_levels = 2 * np.arange(n_levels) - (n_levels - 1)

    # Inverse maps from detected PAM value to Gray index for each axis.
    inv_map_i = {}
    inv_map_q = {}
    
    for b in range(n_levels):
        g = b ^ (b >> 1)
        inv_map_i[pam_levels[b]] = g
        inv_map_q[pam_levels[n_levels - 1 - b]] = g

    # Simple AGC: match received average symbol power to the ideal M-QAM power.
    ideal_symbol_power = (2.0 / 3.0) * (n_levels**2 - 1)
    
    measured_power = np.mean(np.abs(symbols) ** 2)
    if measured_power > 0:
        gain = np.sqrt(ideal_symbol_power / measured_power)
        symbols = symbols * gain

    I = np.real(symbols)
    Q = np.imag(symbols)

    idx_i = np.argmin(np.abs(I[:, None] - pam_levels[None, :]), axis=1)
    idx_q = np.argmin(np.abs(Q[:, None] - pam_levels[None, :]), axis=1)
    i_levels = pam_levels[idx_i]
    q_levels = pam_levels[idx_q]

    bits = []
    for i_level, q_level in zip(i_levels, q_levels):
        gray_i = int(inv_map_i[i_level])
        gray_q = int(inv_map_q[q_level])
        bits.append(format(gray_i, f'0{k_half}b') + format(gray_q, f'0{k_half}b'))

    return ''.join(bits)


def generate_OFDMA_signal(Nfft, number_of_users, subcarriers_per_user, modulation_order):
    # Complex subcarrier grid (size Nfft).
    
    complete_band = np.zeros(Nfft, dtype=complex)

    # Store original user bit streams for optional BER checks.
    all_original_bits = []

    for user in range(number_of_users):
        start = user * subcarriers_per_user
        end = start + subcarriers_per_user

        # Generate random user bits and map them to QAM symbols.
        bits = ''.join(np.random.choice(['0','1'], int(np.log2(modulation_order))*subcarriers_per_user))
        
        
        symbols = qam_mod(bits, modulation_order)
        print("USER", user, "BITS:", bits)

        # Assign user symbols to contiguous subcarriers.
        complete_band[start:end] = symbols
        
        # Keep the original bit stream.
        all_original_bits.append(bits)

    # Convert frequency-domain grid to time-domain signal.
    band_ifft = np.fft.ifft(complete_band) * np.sqrt(Nfft)

    return band_ifft, all_original_bits

def extract_bits_from_ofdma(received_signal, Nfft, number_of_users, subcarriers_per_user, modulation_order):
    recovered_band = np.fft.fft(received_signal) / np.sqrt(Nfft) # Ajuste de escala comum em OFDM
    
    
    all_recovered_bits = []
    for user in range(number_of_users):
        start = user * subcarriers_per_user
        end = start + subcarriers_per_user
        user_symbols = recovered_band[start:end]
        
        # 3. Demodula
        user_bits = qam_demod(user_symbols, modulation_order)
        all_recovered_bits.append(user_bits)
        
    return all_recovered_bits 

# Modulation QAM: s(t) = I(t)·cos(2πfc·t) − Q(t)·sin(2πfc·t)
def qam_modulate_passband(baseband_signal, fc, fs):
    N = len(baseband_signal)
    t = np.arange(N) / fs  # vetor de tempo
    
    # Extrai componentes I e Q
    I = np.real(baseband_signal)
    Q = np.imag(baseband_signal)
    
    carrier_cos = np.cos(2 * np.pi * fc * t)
    carrier_sin = np.sin(2 * np.pi * fc * t)
    
    s_t = I * carrier_cos - Q * carrier_sin
    
    return s_t, t

def qam_demodulate_passband(s_t, fc, fs):
    N = len(s_t)
    t = np.arange(N) / fs
    # Downconversion (Multiplica por portadora e filtra/pega envelope)
    # Em simulação, podemos usar a quadratura direta:
    I_rec = s_t * np.cos(2 * np.pi * fc * t) * 2
    Q_rec = -s_t * np.sin(2 * np.pi * fc * t) * 2
    
    # Em um sistema real, aqui iria um filtro passa-baixas (LPF)
    # Como estamos em ambiente controlado, o sinal complexo original é:
    return I_rec + 1j * Q_rec


# ===== END OFDMA SECTION ======

def dpdTraining(in_training, out_training):
    dpd_coef = calcCoef(out_training, in_training)     
    return dpd_coef
   

ofdma_signal, bits_enviados = generate_OFDMA_signal(NFFT, N_USERS, SUB_PER_USER, MOD_ORDER)

# Normalizar OFDMA
ofdma_signal = ofdma_signal / np.max(np.abs(ofdma_signal))

# Ajustar ganho para região de compressão do PA
G = 0.7 * np.max(np.abs(in_training))
ofdma_signal = ofdma_signal * G

theoric_signal, t = qam_modulate_passband(ofdma_signal, fc, fs)


optimized_coef = calcCoef(in_training, out_training)


dpd_coef = dpdTraining(in_training, out_training)
out_dpd = estimatedValueWithLUT(ofdma_signal, dpd_coef)


out_of_amp = estimatedValueWithLUT(out_dpd, optimized_coef)

out_of_antena, _ = qam_modulate_passband(out_of_amp, fc, fs)
demodulate_antena = qam_demodulate_passband(out_of_antena, fc, fs)



bits_recebidos = extract_bits_from_ofdma(demodulate_antena, NFFT, N_USERS, SUB_PER_USER, MOD_ORDER)


print("Bits Enviados por Usuário:")
for i, bits in enumerate(bits_enviados):
    print(f"Usuário {i}: {bits}")
    
print("\nBits Recebidos por Usuário:")
for i, bits in enumerate(bits_recebidos):
    print(f"Usuário {i}: {bits}")

#NMSE
out_estimated = estimatedValueWithLUT(in_validation, optimized_coef)
nmse = calculate_nmse(out_validation, out_estimated)

print(f"NMSE: {nmse:.6f} dB")


plt.plot(t, theoric_signal, label='Original (Referência)', color='blue', linewidth=2, alpha=0.8)
plt.plot(t, out_of_antena, label='Saída do PA com DPD', color='red', linestyle='--', linewidth=2)
plt.title("Validação da Cascata: Original vs Saída do Sistema")

# Verificação de densidade de dados
# plt.figure()
# plt.hist(np.abs(in_training), bins=N_LUT_COLUMNS, color='skyblue', edgecolor='black')
# plt.axhline(y=N_LUT_LINES * 10, color='red', linestyle='--', label='Mínimo Recomendado')
# plt.title(f"Distribuição de Amostras por Coluna da LUT ({N_LUT_COLUMNS} colunas)")
# plt.xlabel("Amplitude |x|")
# plt.ylabel("Número de Amostras")
# plt.legend()


# AM-AM GRAPH -------------
# amp_input = np.abs(ofdma_signal)
# amp_output_dpd = np.abs(out_of_amp)
# out_no_dpd = estimatedValueWithLUT(ofdma_signal, optimized_coef)
# amp_output_no_dpd = np.abs(out_no_dpd)
# plt.figure(figsize=(10, 7))
# plt.scatter(amp_input, amp_output_no_dpd, s=80, label='PA sem DPD', alpha=0.8)
# plt.scatter(amp_input, amp_output_dpd, s=80, label='PA com DPD (Cascata)', alpha=0.8)
# lim = np.max(amp_input)
# plt.plot([0, lim], [0, lim], 'r--', label='Referência Linear', linewidth=2)
# plt.xlabel("Amplitude de Entrada")
# plt.ylabel("Amplitude de Saı́da")
# plt.title("Gráfico AM-AM")




plt.legend()
plt.grid()
plt.show()

