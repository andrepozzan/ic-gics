
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
from scipy.signal import hilbert, correlate

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

FC = 26e9  # Carrier frequency centered at 26 GHz for Cadence Virtuoso
FS = 104e9  # Sampling rate high enough to represent a 26 GHz passband signal




csv_data = np.genfromtxt('dados-para-TREINO.csv', delimiter=',', skip_header=1)

IN_AMP_X = csv_data[:, 0]   # /IN_AMP X
IN_AMP_Y = csv_data[:, 1]   # /IN_AMP Y
OUT_AMP_X = csv_data[:, 2]  # /OUT_AMP X
OUT_AMP_Y = csv_data[:, 3]  # /OUT_AMP Y


def estimate_sampling_rate(time_vector):
    dt = np.diff(time_vector)
    dt_mean = np.mean(dt)
    if dt_mean <= 0:
        raise ValueError('Invalid time vector in Cadence CSV: non-positive sampling interval')
    return 1.0 / dt_mean


def passband_to_complex_baseband(passband_signal, time_vector, fc):
    analytic_signal = hilbert(passband_signal)
    return analytic_signal * np.exp(-1j * 2 * np.pi * fc * time_vector)


def estimate_integer_delay(reference_signal, target_signal):
    corr = correlate(target_signal, reference_signal, mode='full', method='fft')
    lags = np.arange(-len(reference_signal) + 1, len(target_signal))
    return int(lags[np.argmax(np.abs(corr))])


FS_CADENCE = estimate_sampling_rate(IN_AMP_X)

in_baseband = passband_to_complex_baseband(IN_AMP_Y, IN_AMP_X, FC)
out_baseband = passband_to_complex_baseband(OUT_AMP_Y, OUT_AMP_X, FC)

# Align Cadence output to input before splitting to training/validation.
cadence_delay = estimate_integer_delay(in_baseband, out_baseband)
out_baseband = np.roll(out_baseband, -cadence_delay)
print(f"Cadence estimated delay (samples): {cadence_delay}")

# Divide os dados coletados do Cadence em Extração (Treinamento) e Validação (50% / 50%)
split_idx = len(in_baseband) // 2

in_training = in_baseband[:split_idx]
out_training = out_baseband[:split_idx]

in_validation = in_baseband[split_idx:]
out_validation = out_baseband[split_idx:]

print("IN_TRAINING", in_training,"\n\n OUT_TRAINING", out_training)


N_USERS = 3
SUB_PER_USER = 12
MOD_ORDER = 256
NFFT = 2048

N_LUT_LINES = 4  # LUT matrix lines
N_LUT_COLUMNS = 4  # LUT matrix columns

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

# Match OFDMA drive amplitude to the Cadence training range used by the LUT model.
training_drive_ref = np.percentile(np.abs(in_training), 99)
if training_drive_ref <= 0:
    training_drive_ref = np.max(np.abs(in_training))
if training_drive_ref <= 0:
    training_drive_ref = 1.0

ofdma_signal = ofdma_signal * training_drive_ref

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


pwl_output_path = Path(__file__).with_name('sinal_entrada_cadence.pwl')

fator_atenuacao = 1.0
sinal_para_cadence = ideal_theoric_signal * fator_atenuacao

export_pwl_signal(sinal_para_cadence, FS, pwl_output_path)
print(f"PWL de treinamento exportado em: {pwl_output_path}")


demodulate_antena = qam_demodulate_passband(
    out_of_antena, FC, FS)

ofdma_delay = estimate_integer_delay(ofdma_signal, out_of_amp)
out_of_amp_aligned = np.roll(out_of_amp, -ofdma_delay)
print(f"OFDMA estimated delay (samples): {ofdma_delay}")

bits_received = extract_bits_from_ofdma(
    out_of_amp_aligned, NFFT, N_USERS, SUB_PER_USER, MOD_ORDER)


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


# plot_constellation_comparison(bits_send, bits_received, MOD_ORDER)

# NMSE
out_estimated = estimatedValueWithLUT(
    in_validation, optimized_coef, dpd_cfg)
nmse = calculate_nmse(out_validation, out_estimated)

print(f"NMSE: {nmse:.6f} dB")

# plot_histogram_amplitude(in_training, N_LUT_LINES, N_LUT_COLUMNS)


# AM-AM only from training data (independent from OFDM user count).
amp_in_train = np.abs(in_training)
amp_out_train = np.abs(out_training)


def build_binned_amam_curve(x, y, n_bins=80, min_samples=12):
    if len(x) == 0:
        return np.array([]), np.array([])

    x_min = np.min(x)
    x_max = np.max(x)
    if x_max <= x_min:
        return np.array([x_min]), np.array([np.median(y)])

    edges = np.linspace(x_min, x_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.digitize(x, edges) - 1

    curve_x = []
    curve_y = []
    for b in range(n_bins):
        mask = bin_idx == b
        if np.count_nonzero(mask) >= min_samples:
            curve_x.append(centers[b])
            curve_y.append(np.median(y[mask]))

    if len(curve_x) == 0:
        return np.array([]), np.array([])

    curve_y = np.maximum.accumulate(np.asarray(curve_y))
    return np.asarray(curve_x), curve_y


train_curve_x, train_curve_y = build_binned_amam_curve(amp_in_train, amp_out_train)

plt.figure(figsize=(10, 7))
plt.scatter(amp_in_train, amp_out_train, s=10, alpha=0.12, color='purple', label='Training samples')
if len(train_curve_x) > 0:
    plt.plot(train_curve_x, train_curve_y, color='blue', linewidth=2.8, label='Training static AM-AM (median)')
lim_train = np.max(amp_in_train) if len(amp_in_train) else 1.0
plt.plot([0, lim_train], [0, lim_train], 'r--', linewidth=2, label='Linear Reference')
plt.title('AM-AM Training Only', fontsize=18)
plt.xlabel('Input Amplitude |x_train|', fontsize=14)
plt.ylabel('Output Amplitude |y_train|', fontsize=14)
plt.grid(True, alpha=0.4)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


amp_input = np.abs(ofdma_signal)
out_no_dpd = estimatedValueWithLUT(
    ofdma_signal, optimized_coef, dpd_cfg)

# Align branches before AM-AM plotting; otherwise x[n] and y[n] mismatch creates artificial loops.
delay_dpd_block = estimate_integer_delay(ofdma_signal, out_dpd)
delay_no_dpd = estimate_integer_delay(ofdma_signal, out_no_dpd)
delay_with_dpd = estimate_integer_delay(ofdma_signal, out_of_amp)

out_dpd_aligned = np.roll(out_dpd, -delay_dpd_block)
out_no_dpd_aligned = np.roll(out_no_dpd, -delay_no_dpd)
out_with_dpd_aligned = np.roll(out_of_amp, -delay_with_dpd)

guard = max(N_LUT_LINES, abs(delay_dpd_block), abs(delay_no_dpd), abs(delay_with_dpd)) + 2
if 2 * guard < len(ofdma_signal):
    valid_slice = slice(guard, -guard)
else:
    valid_slice = slice(0, len(ofdma_signal))

amp_input = np.abs(ofdma_signal[valid_slice])
amp_output_dpd_block = np.abs(out_dpd_aligned[valid_slice])
amp_output_no_dpd = np.abs(out_no_dpd_aligned[valid_slice])
amp_output_dpd = np.abs(out_with_dpd_aligned[valid_slice])


plot_am_am(amp_input, amp_output_dpd_block, amp_output_no_dpd, amp_output_dpd)

plot_psd_comparison(ofdma_signal, out_no_dpd, out_of_amp, FS)
