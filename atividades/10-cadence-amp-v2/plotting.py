import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

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
        label_to_right = sym.real < 0
        dx = int(14 + stagger_x) if label_to_right else -int(14 + stagger_x)

        # vertical offset: TX above, RX below, plus small row stagger
        row_stagger = ((idx // 4) % 3) * 5
        dy = int(
            base_dy + (row_stagger if base_dy > 0 else -row_stagger))

        # horizontal alignment: right (since label is left of symbol)
        ha = 'left' if label_to_right else 'right'
        va = 'bottom' if base_dy > 0 else 'top'

        ax.annotate(
            f"{prefix}:{label}",
            xy=(sym.real, sym.imag),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=22,
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


def _set_data_limits(axis, *arrays, dimension='x', include_zero=False):
    values = np.concatenate([
        np.asarray(array, dtype=float).reshape(-1)
        for array in arrays
    ])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if include_zero:
        minimum = min(minimum, 0.0)
        maximum = max(maximum, 0.0)
    span = maximum - minimum
    margin = 0.05 * span if span > 0 else max(abs(maximum), 1.0) * 0.05
    setter = axis.set_xlim if dimension == 'x' else axis.set_ylim
    setter(minimum - margin, maximum + margin)


def _occupied_psd_region(*psd_arrays):
    """Return the run-dependent frequency interval containing the signal."""
    stacked = np.vstack([np.asarray(psd, dtype=float) for psd in psd_arrays])
    envelope = np.max(stacked, axis=0)
    finite = np.isfinite(envelope)
    if not np.any(finite):
        return np.ones(envelope.shape, dtype=bool)

    # Keep the complete occupied band, including lower-power edge subcarriers.
    threshold = np.max(envelope[finite]) - 30.0
    mask = finite & (envelope >= threshold)
    if not np.any(mask):
        mask = finite
    return mask


def plot_psd_comparison(ofdma_signal, out_no_dpd, out_with_dpd, fs, output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else None
    freq_in, psd_in = calculate_psd(ofdma_signal, fs)
    freq_no_dpd, psd_no_dpd = calculate_psd(out_no_dpd, fs)
    freq_with_dpd, psd_with_dpd = calculate_psd(out_with_dpd, fs)

    # Referência global baseada no pico máximo entre todos os sinais
    psd_reference = np.max([
        np.max(psd_in) if psd_in.size else -np.inf,
        np.max(psd_no_dpd) if psd_no_dpd.size else -np.inf,
        np.max(psd_with_dpd) if psd_with_dpd.size else -np.inf,
    ])
    
    # Mantém o eixo de frequência completo (convertido para MHz) para exibir as bordas
    plot_frequency = freq_in / 1e6
    plot_psd_in = psd_in - psd_reference
    plot_psd_no_dpd = psd_no_dpd - psd_reference
    plot_psd_with_dpd = psd_with_dpd - psd_reference

    figure = plt.figure(figsize=(14, 10))
    
    # Ordem de plotagem: Input por baixo, Sem DPD (mostrando o espalhamento) e Com DPD por cima
    plt.plot(plot_frequency, plot_psd_in,
             color='black', label='Input signal', linewidth=2.0, zorder=1)
    plt.plot(plot_frequency, plot_psd_no_dpd,
             color='lime', label='Output without DPD', linewidth=2.0, zorder=2, alpha=0.8)
    plt.plot(plot_frequency, plot_psd_with_dpd,
             color='red', label='Output with DPD', linewidth=2.0, zorder=3)

    plt.xlabel('Frequency (MHz)', fontsize=30)
    plt.ylabel('Power Spectral Density (dB/MHz)', fontsize=30)
    
    # Configuração opcional de zoom nas bordas: 
    # Se quiser ver uma faixa específica ao redor da portadora (ex: ± largura de banda do sinal * 2.5)
    # Calcule o span central com base na máscara ocupada se desejar, mas deixe margem ampla:
    occupied_mask = _occupied_psd_region(psd_in, psd_no_dpd, psd_with_dpd)
    occupied_freq = plot_frequency[occupied_mask]
    if occupied_freq.size:
        f_center = np.mean(occupied_freq)
        f_span = (np.max(occupied_freq) - np.min(occupied_freq)) * 2.0  # Amplia o span para mostrar as bordas
        plt.xlim(f_center - f_span, f_center + f_span)

    plt.ylim(-80, 5) # Limite padrão em dB para enxergar o piso de ruído e o espalhamento lateral
    plt.grid(True, which='both', color='gray', alpha=0.5, linestyle='-')

    legend = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                        fontsize=30, frameon=True)
    legend.get_frame().set_edgecolor('black')

    plt.tick_params(axis='both', which='major', labelsize=30)
    
    plt.tight_layout()
    if output_dir is not None:
        np.savetxt(
            output_dir / 'psd_comparison.csv',
            np.column_stack((freq_in, plot_psd_in,
                             plot_psd_no_dpd, plot_psd_with_dpd)),
            delimiter=',',
            header='frequency_hz,input_db,no_dpd_db,with_dpd_db',
            comments='')
        figure.savefig(output_dir / 'psd_comparison.png',
                       dpi=300, bbox_inches='tight')
    plt.close(figure)

def split_bits_by_symbol(bit_string, bits_per_symbol):
    return [bit_string[i:i + bits_per_symbol] for i in range(0, len(bit_string), bits_per_symbol)]


def annotate_constellation_labels(ax, symbols, labels, prefix, max_labels=8):
    base_dy = 24 if prefix.upper().startswith('TX') else -24

    symbols_list = list(symbols)
    labels_list = list(labels)
    total = len(symbols_list)
    if max_labels is not None:
        total = min(total, int(max_labels))

    placed_boxes = []
    renderer = None
    for idx in range(total):
        sym = symbols_list[idx]
        label = labels_list[idx] if idx < len(labels_list) else None
        if label is None or label == '':
            label = f"idx{idx}"
        horizontal_offsets = (0, 28, -28, 56, -56)
        vertical_offsets = (base_dy, base_dy * 2, base_dy * 3,
                            base_dy * 4, base_dy * 5)
        candidates = [
            (horizontal, vertical)
            for vertical in vertical_offsets
            for horizontal in horizontal_offsets
        ]
        if renderer is None:
            ax.figure.canvas.draw()
            renderer = ax.figure.canvas.get_renderer()

        annotation = None
        for dx, dy in candidates:
            candidate = ax.annotate(
                f"{prefix}:{label}",
                xy=(sym.real, sym.imag),
                xytext=(dx, dy),
                textcoords='offset points',
                fontsize=16,
                ha='center',
                va='bottom' if dy > 0 else 'top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.9, edgecolor='none'),
                arrowprops=dict(arrowstyle='-', color='gray', alpha=0.35,
                                lw=0.7),
            )
            ax.figure.canvas.draw()
            box = candidate.get_window_extent(renderer).expanded(1.03, 1.08)
            if not any(box.overlaps(previous) for previous in placed_boxes):
                annotation = candidate
                placed_boxes.append(box)
                break
            candidate.remove()
        if annotation is None:
            annotation = ax.annotate(
                f"{prefix}:{label}", xy=(sym.real, sym.imag),
                xytext=(0, base_dy), textcoords='offset points',
                fontsize=16, ha='center',
                va='bottom' if base_dy > 0 else 'top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.9, edgecolor='none'),
                arrowprops=dict(arrowstyle='-', color='gray', alpha=0.35,
                                lw=0.7),
            )


def plot_constellation_comparison(bits_send, bits_received, mod_order,
                                  max_labels=8, output_dir=None):
    bits_per_symbol = int(np.log2(mod_order))
    output_dir = Path(output_dir) if output_dir is not None else None

    for user_idx, (tx_bits, rx_bits) in enumerate(zip(bits_send, bits_received)):
        fig, ax = plt.subplots(figsize=(10, 10))
        tx_symbols = qam_mod(tx_bits, mod_order)
        rx_symbols = qam_mod(rx_bits, mod_order)

        ax.scatter(tx_symbols.real, tx_symbols.imag, s=500, alpha=1,
                   marker='o', color='blue', label='Original bits')
        ax.scatter(rx_symbols.real, rx_symbols.imag, s=500, alpha=1,
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
        ax.legend(fontsize=20, markerscale=1, handlelength=1.8)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_title(f'User {user_idx}: Original vs Demodulated', fontsize=30)
        plt.tight_layout()
        if output_dir is not None:
            fig.savefig(output_dir / f'constellation_user_{user_idx}.png',
                dpi=300, bbox_inches='tight')
        plt.close(fig)
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
    input_normalized = _normalize_amplitude(amp_input)
    dpd_block_normalized = _normalize_amplitude(amp_output_dpd_block)
    no_dpd_normalized = _normalize_amplitude(amp_output_no_dpd)
    dpd_normalized = _normalize_amplitude(amp_output_dpd)

    plt.figure(figsize=(20, 9))
    plt.scatter(input_normalized, dpd_block_normalized, s=45,
                label='DPD output', alpha=0.55)
    plt.scatter(input_normalized, no_dpd_normalized, s=45,
                label='PA without DPD', alpha=0.55)
    plt.scatter(input_normalized, dpd_normalized, s=45,
                label='PA with DPD (Cascade)', alpha=0.55)
    plt.plot([0, 1], [0, 1], 'r--', label='Linear reference', linewidth=2)
    plt.xlabel('Normalized input amplitude', fontsize=24)
    plt.ylabel('Normalized output amplitude', fontsize=24)
    plt.title('Normalized AM-AM response', fontsize=22)
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.legend(fontsize=20)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()


def _normalize_amplitude(amplitude):
    maximum = np.max(amplitude) if len(amplitude) else 0.0
    if maximum <= 0:
        return amplitude
    return amplitude / maximum


def plot_time_domain_comparison(reference_signal, signal_with_dpd,
                                sampling_rate=None, output_dir=None,
                                sample_count=240):
    output_dir = Path(output_dir) if output_dir is not None else None
    sample_count = min(sample_count, len(reference_signal), len(signal_with_dpd))
    sample_slice = slice(0, sample_count)
    time_scale = 1e9 / sampling_rate if sampling_rate else 1.0
    time_label = 'Time (ns)' if sampling_rate else 'Sample index'
    time = (np.arange(sample_count) * time_scale if sampling_rate
            else np.arange(sample_count))

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(time, np.real(reference_signal[sample_slice]),
                 label='Original reference', color='black', linewidth=1.4)
    axes[0].plot(time, np.real(signal_with_dpd[sample_slice]),
                 label='PA output with DPD', color='tab:red',
                 linestyle='--', linewidth=2.8)
    axes[0].set_ylabel('Real amplitude', fontsize=30)
    axes[0].set_title('Time-domain comparison', fontsize=36)
    axes[0].legend(fontsize=30, markerscale=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, np.abs(reference_signal[sample_slice]),
                 label='Reference envelope', color='tab:blue', linewidth=2.8)
    axes[1].plot(time, np.abs(signal_with_dpd[sample_slice]),
                 label='PA output envelope', color='tab:orange',
                 linestyle='--', linewidth=2.8)
    axes[1].set_xlabel(time_label, fontsize=30)
    axes[1].set_ylabel('Envelope', fontsize=30)
    axes[1].legend(fontsize=30, markerscale=2)
    for axis in axes:
        axis.tick_params(axis='both', labelsize=25)
    _set_data_limits(
        axes[0],
        np.real(reference_signal[sample_slice]),
        np.real(signal_with_dpd[sample_slice]),
        dimension='y',
    )
    _set_data_limits(
        axes[1],
        np.abs(reference_signal[sample_slice]),
        np.abs(signal_with_dpd[sample_slice]),
        dimension='y',
    )
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    if output_dir is not None:
        figure.savefig(output_dir / 'time_domain_comparison.png',
                       dpi=300, bbox_inches='tight')
    plt.close(figure)
