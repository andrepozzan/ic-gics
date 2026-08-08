from scipy.io import loadmat
from scipy.optimize import least_squares
from numba import njit, prange
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ofdma import generate_OFDMA_signal, extract_bits_from_ofdma

# --- PARÂMETROS GERAIS ---
N_USERS = 16
SUB_PER_USER = 26
MOD_ORDER = 1024
NFFT = 2048

FS = 1e9
FC = 3.84e6

P_MAX = 10
M_MAX = 10
FIXED_SNR_DB = 30 # Valor fixo de ruído para a simulação

OFDMA_SIGNAL = None
BITS_SEND = None

# --- CARREGAMENTO DE DADOS ---
loaded_data = loadmat('in_out_SBRT2_direto.mat')
in_training = loaded_data['in_extraction'].ravel()
out_training = loaded_data['out_extraction'].ravel()
in_validation = loaded_data['in_validation'].ravel()
out_validation = loaded_data['out_validation'].ravel()


# --- FUNÇÕES DE MODELAGEM MATEMÁTICA ---
@njit(parallel=True)
def estimatedValueWithComplex(x_data, coef_matrix, P, M):
    N = len(x_data)
    y_est = np.empty(N, dtype=np.complex128)

    for n in prange(N):
        estimated_value = 0.0 + 0.0j
        for p in range(1, P + 1):
            for m in range(M + 1):
                # Mantém a causalidade do sistema: se n-m < 0, o termo é nulo
                if n - m < 0:
                    term = 0.0 + 0.0j
                else:
                    idx = n - m
                    power = 2 * p - 2
                    term = (np.abs(x_data[idx]) ** power) * x_data[idx]
                estimated_value += term * coef_matrix[p - 1, m]
        y_est[n] = estimated_value

    return y_est


def unpackComplexCoefficients(x_real, P, M):
    half = len(x_real) // 2
    real_parts = x_real[:half]
    imag_parts = x_real[half:]
    complex_coef = real_parts + 1j * imag_parts
    return complex_coef.reshape((P, M + 1))


def mpResiduals(x_real, x_data, y_data, P, M):
    coef_matrix = unpackComplexCoefficients(x_real, P, M)
    y_est = estimatedValueWithComplex(x_data, coef_matrix, P, M)
    resid = y_data - y_est
    return np.concatenate((resid.real, resid.imag))


def generateInitialComplex(P, M, mean, std, seed):
    rng = np.random.default_rng(seed)
    shape = P * (M + 1)
    real_part = rng.normal(loc=mean.real, scale=std, size=shape)
    imag_part = rng.normal(loc=mean.imag, scale=std, size=shape)
    return real_part + 1j * imag_part


def fit_model_for_pm(P, M):
    seed = 1000 + P * 100 + M
    initial_complex = generateInitialComplex(
        P, M, mean=np.mean(in_training), std=np.std(in_training), seed=seed)
    initial_real = np.concatenate((initial_complex.real, initial_complex.imag))

    result = least_squares(
        mpResiduals,
        initial_real,
        args=(in_training, out_training, P, M),
        verbose=0,
    )

    coef = unpackComplexCoefficients(result.x, P, M)
    return coef


def fit_dpd_for_pm(P, M):
    seed = 2000 + P * 100 + M
    initial_complex = generateInitialComplex(
        P, M, mean=np.mean(out_training), std=np.std(out_training), seed=seed)
    initial_real = np.concatenate((initial_complex.real, initial_complex.imag))

    result = least_squares(
        mpResiduals,
        initial_real,
        args=(out_training, in_training, P, M),
        verbose=0,
    )

    coef = unpackComplexCoefficients(result.x, P, M)
    return coef


def fit_models_for_pm(P, M):
    return fit_model_for_pm(P, M), fit_dpd_for_pm(P, M)


# --- ADIÇÃO DE RUÍDO E CÁLCULO DE ERRO ---
def add_awgn(signal, snr_db, seed):
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(np.abs(signal) ** 2))
    if signal_power == 0:
        return signal.copy()

    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    if np.iscomplexobj(signal):
        noise_scale = np.sqrt(noise_power / 2.0)
        noise = noise_scale * (
            rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape)
        )
    else:
        noise = rng.normal(scale=np.sqrt(noise_power), size=signal.shape)

    return signal + noise


def calculate_ber(tx_bits, rx_bits):
    n_bits = min(len(tx_bits), len(rx_bits))
    if n_bits == 0:
        return 0.0, 0, 0

    n_errors = sum(tb != rb for tb, rb in zip(tx_bits[:n_bits], rx_bits[:n_bits]))
    ber = n_errors / n_bits
    return ber, n_errors, n_bits


def evaluate_pm_snr(P, M, snr_db, pa_coef, dpd_coef):
    if OFDMA_SIGNAL is None or BITS_SEND is None:
        raise RuntimeError('OFDMA frame was not initialized before evaluation.')

    out_dpd = estimatedValueWithComplex(OFDMA_SIGNAL, dpd_coef, P, M)
    modeled_baseband = estimatedValueWithComplex(out_dpd, pa_coef, P, M)

    recovered_baseband = add_awgn(
        modeled_baseband,
        snr_db,
        seed=10_000 + P * 1_000 + M * 100 + int(round(snr_db * 10)),
    )
    bits_received = extract_bits_from_ofdma(
        recovered_baseband, NFFT, N_USERS, SUB_PER_USER, MOD_ORDER)

    user_bers = []
    for tx_bits, rx_bits in zip(BITS_SEND, bits_received):
        ber, _, _ = calculate_ber(tx_bits, rx_bits)
        user_bers.append(ber)

    ber_mean = float(np.mean(user_bers)) if user_bers else 0.0
    return P, M, snr_db, ber_mean


# --- AVALIAÇÃO E PLOTAGEM ---
def run_3d_evaluation(p_max, m_max, snr_db):
    results = []
    model_cache = {}
    
    # Extração dos modelos DPD e PA
    for p in range(1, p_max + 1):
        for m in range(1, m_max + 1):
            model_cache[(p, m)] = fit_models_for_pm(p, m)

    # Criação dos parâmetros a serem iterados
    params = [(p, m) for p in range(1, p_max + 1)
                     for m in range(1, m_max + 1)]

    # Avaliação fixa para a SNR definida
    with tqdm(total=len(params), desc='Progresso P×M') as pbar:
        for p, m in params:
            pa_coef, dpd_coef = model_cache[(p, m)]
            _, _, _, ber_mean = evaluate_pm_snr(p, m, float(snr_db), pa_coef, dpd_coef)
            results.append((p, m, ber_mean))
            pbar.update(1)

    return results


def plot_results(results, snr_db):
    # Extrai os dados em formato de arrays
    p_values = np.array([item[0] for item in results], dtype=float)
    m_values = np.array([item[1] for item in results], dtype=float)
    ber_values = np.array([item[2] for item in results], dtype=float)

    if len(results) == 0:
        print('Nenhum resultado disponível para plotagem.')
        return

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot de superfície 3D (malha colorida baseada na BER)
    surf = ax.plot_trisurf(
        p_values, 
        m_values, 
        ber_values, 
        cmap='viridis_r',  # _r inverte as cores (menor BER = mais quente/claro)
        edgecolor='none', 
        alpha=0.85
    )
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='BER')

    ax.set_xlabel('P (Polynomial degree)')
    ax.set_ylabel('M (Memory)')
    ax.set_zlabel('BER')
    ax.set_title(f'P and M (SNR fixed {snr_db} dB)')
    ax.set_xticks(range(1, P_MAX + 1))
    ax.set_yticks(range(1, M_MAX + 1))
    
    plt.tight_layout()
    plt.show()


def main():
    global OFDMA_SIGNAL, BITS_SEND

    # Geração do sinal OFDMA
    OFDMA_SIGNAL, BITS_SEND = generate_OFDMA_signal(
        NFFT, N_USERS, SUB_PER_USER, MOD_ORDER)
    OFDMA_SIGNAL = OFDMA_SIGNAL / np.max(np.abs(OFDMA_SIGNAL))

    # Executa a simulação
    results = run_3d_evaluation(P_MAX, M_MAX, FIXED_SNR_DB)

    print(f'\nResults for P and M (SNR = {FIXED_SNR_DB} dB):')
    for p, m, ber_mean in results:
        print(f'P={p}, M={m} -> Average BER = {ber_mean:.6f}')

    # Plota a superfície 3D
    plot_results(results, FIXED_SNR_DB)


if __name__ == '__main__':
    main()