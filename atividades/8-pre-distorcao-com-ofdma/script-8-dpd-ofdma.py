
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
from scipy.signal import hilbert

from qam import qam_mod, qam_modulate_passband, qam_demodulate_passband
from ofdma import generate_OFDMA_signal, extract_bits_from_ofdma

from calc_coef import DPDConfig, calcCoef, estimatedValueWithLUT, dpdTraining
from plotting import (
    plot_psd_comparison,
    plot_constellation_comparison,
    plot_time_domain_comparison,
    plot_histogram_amplitude,
    plot_am_am,
)

# Uses flatten to be sure that the data is in 1D format
mat = loadmat('in_out_SBRT2_direto.mat')
in_training = mat['in_extraction'].flatten()
out_training = mat['out_extraction'].flatten()
in_validation = mat['in_validation'].flatten()
out_validation = mat['out_validation'].flatten()

print("in_training shape:", in_training)


# csv_data = np.genfromtxt('dados-para-TREINO.csv', delimiter=',', skip_header=1)

# v_in_full = csv_data[:, 1]   # /IN_AMP Y
# v_out_full = csv_data[:, 3]  # /OUT_AMP Y

# # Divide os dados coletados do Cadence em Extração (Treinamento) e Validação (50% / 50%)
# split_idx = len(v_in_full) // 2

# in_training = v_in_full[:split_idx]
# out_training = v_out_full[:split_idx]
# in_validation = v_in_full[split_idx:]
# out_validation = v_out_full[split_idx:]



N_USERS = 2
SUB_PER_USER = 48
MOD_ORDER = 256
NFFT = 2048

N_LUT_LINES = 4  # LUT matrix lines
N_LUT_COLUMNS = 4  # LUT matrix columns

FC = 26e9  # Carrier frequency centered at 26 GHz for Cadence Virtuoso
FS = 104e9  # Sampling rate high enough to represent a 26 GHz passband signal

initial_complex_coef = np.zeros((N_LUT_LINES, N_LUT_COLUMNS), dtype=complex)


interpolation_in = np.linspace(
    0.0, np.max(np.abs(in_training)), N_LUT_COLUMNS)

dpd_cfg = DPDConfig(
    n_lut_lines=N_LUT_LINES,
    n_lut_columns=N_LUT_COLUMNS,
    initial_complex_coef=initial_complex_coef,
    interpolation_in=interpolation_in,
)


# Normalized Mean Square Error
def calculate_nmse(out_validation, out_estimated):
    erro = out_validation - out_estimated
    nmse = 10 * np.log10(np.sum(np.abs(erro)**2) /
                         np.sum(np.abs(out_validation)**2))
    return nmse


def export_pwl_signal(signal, fs, output_path):
    """Export a real-valued sampled signal as Spectre-friendly PWL pairs."""
    time_vector = np.arange(len(signal)) / fs
    pwl_data = np.column_stack((time_vector, np.real(signal)))
    np.savetxt(output_path, pwl_data, fmt='%.12e %.12e')
    return output_path

ofdma_signal, bits_send = generate_OFDMA_signal(
    NFFT, N_USERS, SUB_PER_USER, MOD_ORDER)

ofdma_signal = ofdma_signal / np.max(np.abs(ofdma_signal))

ideal_theoric_signal, t = qam_modulate_passband(
    ofdma_signal, FC, FS)


optimized_coef = calcCoef(
    in_training, out_training, dpd_cfg)

dpd_coef = dpdTraining(in_training, out_training, dpd_cfg)
out_dpd = estimatedValueWithLUT(
    ofdma_signal, dpd_coef, dpd_cfg)


out_of_amp = estimatedValueWithLUT(
    out_dpd, optimized_coef, dpd_cfg)

out_of_antena, t_out = qam_modulate_passband(out_of_amp, FC, FS)

pwl_output_path = Path(__file__).with_name('out_of_antena.pwl')
export_pwl_signal(out_of_antena, FS, pwl_output_path)
print(f"PWL exportado em: {pwl_output_path}")


# Em vez de exportar 'out_of_antena', você exporta o sinal ideal gerado pela sua portadora:
pwl_output_path = Path(__file__).with_name('sinal_entrada_cadence.pwl')

# Certifique-se de aplicar o fator de atenuação se necessário para manter a amplitude linear:
fator_atenuacao = 0.05  # Ajuste conforme testamos anteriormente para não saturar
sinal_para_cadence = ideal_theoric_signal * fator_atenuacao

export_pwl_signal(sinal_para_cadence, FS, pwl_output_path)
print(f"PWL de treinamento exportado em: {pwl_output_path}")


demodulate_antena = qam_demodulate_passband(
    out_of_antena, FC, FS)


bits_received = extract_bits_from_ofdma(
    demodulate_antena, NFFT, N_USERS, SUB_PER_USER, MOD_ORDER)


def group_bits(bit_string, group_size=4):
    return ' '.join(bit_string[i:i + group_size] for i in range(0, len(bit_string), group_size))


def calculate_ber(tx_bits, rx_bits):
    n_bits = min(len(tx_bits), len(rx_bits))

    # avoid div by zero
    if n_bits == 0:
        return 0.0, 0, 0

    # zip create a pair of bits (tx, rx) and sum the number of pairs that are different
    n_errors = sum(tb != rb for tb, rb in zip(
        tx_bits[:n_bits], rx_bits[:n_bits]))
    ber = n_errors / n_bits
    return ber, n_errors, n_bits


def split_bits_by_symbol(bit_string, bits_per_symbol):
    return [bit_string[i:i + bits_per_symbol] for i in range(0, len(bit_string), bits_per_symbol)]


print("Bits send per User:")
for i, bits in enumerate(bits_send):
    print(f"User {i}: {group_bits(bits)}")

print("\nBits received per User:")
for i, bits in enumerate(bits_received):
    print(f"User {i}: {group_bits(bits)}")

print("\nBER per User:")
for i, (tx_bits, rx_bits) in enumerate(zip(bits_send, bits_received)):
    ber, n_errors, n_bits = calculate_ber(tx_bits, rx_bits)
    ber_percent = ber * 100
    print(f"User {i}: BER = {ber_percent:.2f}% ({n_errors}/{n_bits})")


# Constellation diagrams for original and demodulated bit streams.
plot_constellation_comparison(bits_send, bits_received, MOD_ORDER)

# NMSE
out_estimated = estimatedValueWithLUT(
    in_validation, optimized_coef, dpd_cfg)
nmse = calculate_nmse(out_validation, out_estimated)

print(f"NMSE: {nmse:.6f} dB")

# plot_time_domain_comparison(ideal_theoric_signal, out_of_antena)
plot_histogram_amplitude(in_training, N_LUT_LINES, N_LUT_COLUMNS)


amp_input = np.abs(ofdma_signal)
amp_output_dpd_block = np.abs(out_dpd)
amp_output_dpd = np.abs(out_of_amp)
out_no_dpd = estimatedValueWithLUT(
    ofdma_signal, optimized_coef, dpd_cfg)
amp_output_no_dpd = np.abs(out_no_dpd)


plot_am_am(amp_input, amp_output_dpd_block, amp_output_no_dpd, amp_output_dpd)

# PSD comparison before and after DPD.
plot_psd_comparison(ofdma_signal, out_no_dpd, out_of_amp, FS)
