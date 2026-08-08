
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import itertools


from qam import qam_mod, qam_modulate_passband, qam_demodulate_passband
from ofdma import generate_OFDMA_signal, extract_bits_from_ofdma

from calc_coef import DPDConfig, calcCoef, estimatedValueWithLUT, dpdTraining

# Uses flatten to be sure that the data is in 1D format
mat = loadmat('in_out_SBRT2_direto.mat')
in_training = mat['in_extraction'].flatten()
out_training = mat['out_extraction'].flatten()
in_validation = mat['in_validation'].flatten()
out_validation = mat['out_validation'].flatten()


N_USERS = 8
SUB_PER_USER = 26
MOD_ORDER = 1024
NFFT = 2048

N_LUT_LINES = 4  # LUT matrix lines
N_LUT_COLUMNS = 4  # LUT matrix columns

FS = 1e9  # Sampling rate: 1 MHz
FC = 3.84e6  # Carrier frequency: 3.84 MHz (typical for LTE)

# Techniques for initializing the LUT coefficients:
# initial_complex_coef = (np.zeros((N_LUT_LINES, N_LUT_COLUMNS)) + 1j * np.zeros((N_LUT_LINES, N_LUT_COLUMNS)))
# initial_complex_coef = (np.ones((N_LUT_LINES, N_LUT_COLUMNS)) + 1j * np.ones((N_LUT_LINES, N_LUT_COLUMNS)))
initial_complex_coef = (np.random.randn(
    N_LUT_LINES, N_LUT_COLUMNS) + 1j * np.random.randn(N_LUT_LINES, N_LUT_COLUMNS))


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

out_of_antena, _ = qam_modulate_passband(out_of_amp, FC, FS)

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


def annotate_constellation_labels(ax, symbols, labels, prefix, max_labels=8):
    # Place TX labels above the point and RX labels below.
    # Stagger offsets by index to reduce overlap between nearby labels.
    # By default (max_labels=None) annotate all symbols.
    base_dy = 14 if prefix.upper().startswith('TX') else -14

    symbols_list = list(symbols)
    labels_list = list(labels)
    total = len(symbols_list)
    if max_labels is not None:
        total = min(total, int(max_labels))

    for idx in range(total):
        sym = symbols_list[idx]
        label = labels_list[idx] if idx < len(
            labels_list) else None
        # fallback label when bits-string is missing
        if label is None or label == '':
            label = f"idx{idx}"
        # horizontal offset: ALWAYS place label to the LEFT of the symbol
        # stagger slightly by index so nearby labels don't fully overlap
        stagger_x = (idx % 4) * 6
        dx = -int(14 + stagger_x)

        # vertical offset: TX above, RX below, plus small row stagger
        row_stagger = ((idx // 4) % 3) * 5
        dy = int(
            base_dy + (row_stagger if base_dy > 0 else -row_stagger))

        # horizontal alignment: right (since label is left of symbol)
        ha = 'right'
        va = 'bottom' if base_dy > 0 else 'top'

        ax.annotate(
            f"{prefix}:{label}",
            xy=(sym.real, sym.imag),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=9,
            ha=ha,
            va=va,
            bbox=dict(boxstyle='round,pad=0.15',
                      facecolor='white', alpha=0.85, edgecolor='none'),
            arrowprops=dict(
                arrowstyle='-', color='gray', alpha=0.25, lw=0.6),
        )


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
    print(f"User {i}: BER = {
          ber_percent:.2f}% ({n_errors}/{n_bits})")


# Constellation diagrams for original and demodulated bit streams.
# bits_per_symbol = int(np.log2(MOD_ORDER))
# for user_idx, (tx_bits, rx_bits) in enumerate(zip(bits_send, bits_received)):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     tx_symbols = qam_mod(tx_bits, MOD_ORDER)
#     rx_symbols = qam_mod(rx_bits, MOD_ORDER)

#     ax.scatter(tx_symbols.real, tx_symbols.imag, s=300, alpha=1,
#                marker='o', color='blue', label='Original bits')
#     ax.scatter(rx_symbols.real, rx_symbols.imag, s=300, alpha=1,
#                marker='x', color='red', label='Demodulated bits')

#     tx_labels = split_bits_by_symbol(
#         tx_bits, bits_per_symbol)
#     rx_labels = split_bits_by_symbol(
#         rx_bits, bits_per_symbol)

#     annotate_constellation_labels(
#         ax, tx_symbols, tx_labels, 'TX')
#     annotate_constellation_labels(
#         ax, rx_symbols, rx_labels, 'RX')

#     ax.axhline(0, color='gray', linewidth=0.8)
#     ax.axvline(0, color='gray', linewidth=0.8)
#     ax.grid(True, alpha=0.3)

#     user_points = np.concatenate([tx_symbols, rx_symbols])
#     axis_limit = 1.2 * \
#         np.max(np.abs(np.concatenate(
#             [user_points.real, user_points.imag])))
#     if axis_limit == 0:
#         axis_limit = 1.0

#     ax.set_aspect('equal', adjustable='box')
#     ax.set_xlim(-axis_limit, axis_limit)
#     ax.set_ylim(-axis_limit, axis_limit)
#     ax.set_xlabel("I", fontsize=30)
#     ax.set_ylabel("Q", fontsize=30)
#     ax.legend(fontsize=20)
#     ax.tick_params(axis='both', which='major', labelsize=18)
#     ax.set_title(
#         f"User {user_idx}: Original vs Demodulated", fontsize=22)
#     plt.tight_layout()
#     plt.show()

# NMSE
out_estimated = estimatedValueWithLUT(
    in_validation, optimized_coef, dpd_cfg)
nmse = calculate_nmse(out_validation, out_estimated)

print(f"NMSE: {nmse:.6f} dB")

# plt.figure(figsize=(24, 4.2))
# plt.plot(t, ideal_theoric_signal, label='Original (Reference)',
#         color='blue', linewidth=2, alpha=0.8)
# plt.plot(t, out_of_antena, label='PA Output with DPD',
#         color='red', linestyle='--', linewidth=2)
# plt.title(
#    "Cascade Validation: Original vs System Output", fontsize=16)
# plt.xlabel("Time (s)", fontsize=24, labelpad=10)
# plt.ylabel("Amplitude (a.u.)", fontsize=24, labelpad=10)
# plt.legend(fontsize=16)
# plt.grid()
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.22)

# plt.figure()


# plt.hist(np.abs(in_training), bins=N_LUT_COLUMNS *
#          N_LUT_LINES, color='skyblue', edgecolor='black')
# plt.axhline(y=N_LUT_LINES * 10, color='red',
#             linestyle='--', label='Recommended Minimum')
# plt.title(f"Sample Distribution per LUT Column ({N_LUT_COLUMNS} columns)", fontsize=22)
# plt.xlabel("Amplitude |x|", fontsize=30)
# plt.ylabel("Number of Samples", fontsize=30)
# plt.legend(fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=18)


amp_input = np.abs(ofdma_signal)
amp_output_dpd_block = np.abs(out_dpd)
amp_output_dpd = np.abs(out_of_amp)
out_no_dpd = estimatedValueWithLUT(
    ofdma_signal, optimized_coef, dpd_cfg)
amp_output_no_dpd = np.abs(out_no_dpd)


plt.figure(figsize=(20, 9))
plt.scatter(amp_input, amp_output_dpd_block,
            s=80, label='DPD output', alpha=0.8)
plt.scatter(amp_input, amp_output_no_dpd, s=80,
            label='PA without DPD', alpha=0.8)
plt.scatter(amp_input, amp_output_dpd, s=80,
            label='PA with DPD (Cascade)', alpha=0.8)
lim = np.max(amp_input)
plt.plot([0, lim], [0, lim], 'r--',
         label='Linear Reference', linewidth=2)
plt.xlabel("Input Amplitude", fontsize=30)
plt.ylabel("Output Amplitude", fontsize=30)
plt.title("AM-AM Plot", fontsize=22)


plt.legend(fontsize=20)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=18)
plt.show()
