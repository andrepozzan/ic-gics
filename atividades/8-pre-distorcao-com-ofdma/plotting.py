import numpy as np
import matplotlib.pyplot as plt

from qam import qam_mod

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


def calculate_psd(signal, fs):
    n_samples = len(signal)
    if n_samples == 0:
        return np.array([]), np.array([])

    window = np.hanning(n_samples)
    windowed_signal = signal * window
    spectrum = np.fft.fftshift(np.fft.fft(windowed_signal))
    freq = np.fft.fftshift(np.fft.fftfreq(n_samples, d=1 / fs))

    power_density = (np.abs(spectrum) ** 2) / (np.sum(window ** 2) * fs)
    psd_db = 10 * np.log10(power_density + 1e-20)

    return freq, psd_db


def plot_psd_comparison(ofdma_signal, out_no_dpd, out_with_dpd, fs):
    freq_in, psd_in = calculate_psd(ofdma_signal, fs)
    freq_no_dpd, psd_no_dpd = calculate_psd(out_no_dpd, fs)
    freq_with_dpd, psd_with_dpd = calculate_psd(out_with_dpd, fs)

    psd_reference = np.max([
        np.max(psd_in) if psd_in.size else -np.inf,
        np.max(psd_no_dpd) if psd_no_dpd.size else -np.inf,
        np.max(psd_with_dpd) if psd_with_dpd.size else -np.inf,
    ])

    plt.figure(figsize=(10, 7))
    plt.plot(freq_in / 1e6, psd_in - psd_reference,
             color='black', label='Input signal', linewidth=2.5, zorder=1)
    plt.plot(freq_with_dpd / 1e6, psd_with_dpd - psd_reference,
             color='red', label='Output with DPD', linewidth=2.5, zorder=2)
    plt.plot(freq_no_dpd / 1e6, psd_no_dpd - psd_reference,
             color='lime', label='Output without DPD', linewidth=2.5, zorder=3)

    plt.xlabel('Frequency (MHz)', fontsize=12)
    plt.ylabel('Power Spectral Density (dB/MHz)', fontsize=12)
    plt.grid(True, which='both', color='gray', alpha=0.5, linestyle='-')

    legend = plt.legend(loc='upper right', fontsize=10, frameon=True)
    legend.get_frame().set_edgecolor('black')

    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.show()


def split_bits_by_symbol(bit_string, bits_per_symbol):
    return [bit_string[i:i + bits_per_symbol] for i in range(0, len(bit_string), bits_per_symbol)]


def annotate_constellation_labels(ax, symbols, labels, prefix, max_labels=8):
    base_dy = 14 if prefix.upper().startswith('TX') else -14

    symbols_list = list(symbols)
    labels_list = list(labels)
    total = len(symbols_list)
    if max_labels is not None:
        total = min(total, int(max_labels))

    for idx in range(total):
        sym = symbols_list[idx]
        label = labels_list[idx] if idx < len(labels_list) else None
        if label is None or label == '':
            label = f"idx{idx}"
        stagger_x = (idx % 4) * 6
        dx = -int(14 + stagger_x)
        row_stagger = ((idx // 4) % 3) * 5
        dy = int(base_dy + (row_stagger if base_dy > 0 else -row_stagger))
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
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85, edgecolor='none'),
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.25, lw=0.6),
        )


def plot_constellation_comparison(bits_send, bits_received, mod_order, max_labels=8):
    bits_per_symbol = int(np.log2(mod_order))

    for user_idx, (tx_bits, rx_bits) in enumerate(zip(bits_send, bits_received)):
        fig, ax = plt.subplots(figsize=(10, 10))
        tx_symbols = qam_mod(tx_bits, mod_order)
        rx_symbols = qam_mod(rx_bits, mod_order)

        ax.scatter(tx_symbols.real, tx_symbols.imag, s=300, alpha=1,
                   marker='o', color='blue', label='Original bits')
        ax.scatter(rx_symbols.real, rx_symbols.imag, s=300, alpha=1,
                   marker='x', color='red', label='Demodulated bits')

        tx_labels = split_bits_by_symbol(tx_bits, bits_per_symbol)
        rx_labels = split_bits_by_symbol(rx_bits, bits_per_symbol)

        annotate_constellation_labels(ax, tx_symbols, tx_labels, 'TX', max_labels=max_labels)
        annotate_constellation_labels(ax, rx_symbols, rx_labels, 'RX', max_labels=max_labels)

        ax.axhline(0, color='gray', linewidth=0.8)
        ax.axvline(0, color='gray', linewidth=0.8)
        ax.grid(True, alpha=0.3)

        user_points = np.concatenate([tx_symbols, rx_symbols])
        axis_limit = 1.2 * np.max(np.abs(np.concatenate([user_points.real, user_points.imag])))
        if axis_limit == 0:
            axis_limit = 1.0

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_xlabel('I', fontsize=30)
        ax.set_ylabel('Q', fontsize=30)
        ax.legend(fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_title(f'User {user_idx}: Original vs Demodulated', fontsize=22)
        plt.tight_layout()
        plt.show()


def plot_time_domain_comparison(reference_signal, signal_with_dpd):
    plt.figure(figsize=(24, 4.2))
    plt.plot(reference_signal, label='Original (Reference)', color='blue', linewidth=2, alpha=0.8)
    plt.plot(signal_with_dpd, label='PA Output with DPD', color='red', linestyle='--', linewidth=2)
    plt.title('Cascade Validation: Original vs System Output', fontsize=16)
    plt.xlabel('Time (s)', fontsize=24, labelpad=10)
    plt.ylabel('Amplitude (a.u.)', fontsize=24, labelpad=10)
    plt.legend(fontsize=16)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.22)
    plt.show()


def plot_histogram_amplitude(samples, n_lut_lines, n_lut_columns):
    plt.figure()
    plt.hist(np.abs(samples), bins=n_lut_columns * n_lut_lines, color='skyblue', edgecolor='black')
    plt.axhline(y=n_lut_lines * 10, color='red', linestyle='--', label='Recommended Minimum')
    plt.title(f'Sample Distribution per LUT Column ({n_lut_columns} columns)', fontsize=22)
    plt.xlabel('Amplitude |x|', fontsize=30)
    plt.ylabel('Number of Samples', fontsize=30)
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()


def plot_am_am(amp_input, amp_output_dpd_block, amp_output_no_dpd, amp_output_dpd):
    plt.figure(figsize=(20, 9))
    plt.scatter(amp_input, amp_output_dpd_block, s=80, label='DPD output', alpha=0.8)
    plt.scatter(amp_input, amp_output_no_dpd, s=80, label='PA without DPD', alpha=0.8)
    plt.scatter(amp_input, amp_output_dpd, s=80, label='PA with DPD (Cascade)', alpha=0.8)
    lim = np.max(amp_input)
    plt.plot([0, lim], [0, lim], 'r--', label='Linear Reference', linewidth=2)
    plt.xlabel('Input Amplitude', fontsize=30)
    plt.ylabel('Output Amplitude', fontsize=30)
    plt.title('AM-AM Plot', fontsize=22)
    plt.legend(fontsize=20)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()
