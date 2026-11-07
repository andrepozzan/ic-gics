import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch

from evaluation import build_binned_am_am_curve
from qam import qam_mod

def calculate_psd(signal, fs):
    """Calculate a centered PSD using SciPy."""
    if len(signal) == 0:
        return np.array([]), np.array([])

    segment = min(1024, len(signal))
    frequency, power_density = welch(
        np.asarray(signal),
        fs=float(fs),
        window='hann',
        nperseg=segment,
        return_onesided=False,
        scaling='density',
    )
    order = np.argsort(frequency)
    frequency = frequency[order]
    power_density = power_density[order]
    psd_db = 10 * np.log10(np.maximum(power_density, np.finfo(float).tiny))
    return frequency, psd_db


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
                          alpha=1.0, edgecolor='none'),
                arrowprops=dict(arrowstyle='-', color='#444444', alpha=1.0,
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
                          alpha=1.0, edgecolor='none'),
                arrowprops=dict(arrowstyle='-', color='#444444', alpha=1.0,
                                lw=0.7),
            )


def _split_symbols_by_user(symbols, bits_by_user, mod_order):
    bits_per_symbol = int(np.log2(mod_order))
    result = []
    offset = 0
    for bits in bits_by_user:
        count = len(bits) // bits_per_symbol
        result.append(np.asarray(symbols[offset:offset + count], dtype=complex))
        offset += count
    return result


def _constellation_axis_limit(mod_order):
    """Return one symmetric axis limit for every constellation plot."""
    levels_per_axis = int(np.sqrt(mod_order))
    maximum_level = levels_per_axis - 1
    return 1.2 * maximum_level if maximum_level > 0 else 1.0


def plot_constellation_comparison(bits_send, bits_received, mod_order,
                                  measured_symbols=None, max_labels=8,
                                  output_dir=None):
    bits_per_symbol = int(np.log2(mod_order))
    output_dir = Path(output_dir) if output_dir is not None else None

    for user_idx, (tx_bits, rx_bits) in enumerate(zip(bits_send, bits_received)):
        fig, ax = plt.subplots(figsize=(10, 10))
        tx_symbols = qam_mod(tx_bits, mod_order)
        # Plot the decisions made by the demodulator. Raw measured symbols
        # can still include the channel gain and look collapsed at the origin.
        rx_symbols = qam_mod(rx_bits, mod_order)

        ax.scatter(tx_symbols.real, tx_symbols.imag, s=260, alpha=1,
                   marker='o', facecolors='none', edgecolors='#1769aa',
                   linewidths=1.5, label='Pontos ideais (TX)', zorder=3)
        ax.scatter(rx_symbols.real, rx_symbols.imag, s=70, alpha=1,
                   marker='o', color='#e53935', edgecolors='white',
                   linewidths=0.35, label='Pontos demodulados (RX)', zorder=2)
        tx_labels = split_bits_by_symbol(tx_bits, bits_per_symbol)
        rx_labels = split_bits_by_symbol(rx_bits, bits_per_symbol)
        annotate_constellation_labels(
            ax, tx_symbols, tx_labels, 'TX', max_labels=max_labels)
        annotate_constellation_labels(
            ax, rx_symbols, rx_labels, 'RX', max_labels=max_labels)
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.axvline(0, color='gray', linewidth=0.8)
        ax.grid(True, alpha=0.3)

        ax.set_aspect('equal', adjustable='box')
        axis_limit = _constellation_axis_limit(mod_order)
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_xlabel('I', fontsize=30)
        ax.set_ylabel('Q', fontsize=30)
        ax.legend(fontsize=20, markerscale=1, handlelength=1.8)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_title(
            f'Usuário {user_idx}: símbolos demodulados e ideais',
            fontsize=22,
        )
        plt.tight_layout()
        if output_dir is not None:
            fig.savefig(output_dir / f'constellation_user_{user_idx}.png',
                dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_constellation_all_users(bits_send, bits_received, mod_order,
                                 measured_symbols=None, output_dir=None):
    """Plot the same ideal and demodulated symbols joined across users."""
    output_dir = Path(output_dir) if output_dir is not None else None
    figure, axis = plt.subplots(figsize=(11, 11))
    all_transmitted = [qam_mod(bits, mod_order) for bits in bits_send]
    all_received = [qam_mod(bits, mod_order) for bits in bits_received]
    transmitted_symbols = (
        np.concatenate(all_transmitted)
        if all_transmitted else np.array([], dtype=complex)
    )
    received_symbols = (
        np.concatenate(all_received)
        if all_received else np.array([], dtype=complex)
    )
    if not len(transmitted_symbols) and not len(received_symbols):
        return

    if len(transmitted_symbols):
        axis.scatter(
            transmitted_symbols.real, transmitted_symbols.imag, s=110,
            alpha=0.85, marker='o', facecolors='none',
            edgecolors='#1769aa', linewidths=1.2,
            label='Pontos ideais (TX)', zorder=3,
        )
    if len(received_symbols):
        axis.scatter(
            received_symbols.real, received_symbols.imag, s=55, alpha=0.9,
            marker='o', color='#e53935', edgecolors='white',
            linewidths=0.35, label='Pontos demodulados (RX)', zorder=2,
        )
    axis_limit = _constellation_axis_limit(mod_order)
    axis.set_xlim(-axis_limit, axis_limit)
    axis.set_ylim(-axis_limit, axis_limit)
    axis.set_aspect('equal', adjustable='box')
    axis.axhline(0, color='#777777', linewidth=0.8)
    axis.axvline(0, color='#777777', linewidth=0.8)
    axis.grid(True, color='#bdbdbd', linewidth=0.7, alpha=0.75)
    axis.set_xlabel('I', fontsize=20)
    axis.set_ylabel('Q', fontsize=20)
    axis.set_title(f'Constelação geral: símbolos demodulados ({len(all_received)})',
                   fontsize=24,
                   fontweight='bold')
    axis.legend(fontsize=12, loc='best', frameon=True)
    axis.tick_params(axis='both', which='major', labelsize=14)
    figure.tight_layout()
    if output_dir is not None:
        figure.savefig(output_dir / 'constellation_all_users.png',
                       dpi=300, bbox_inches='tight')
    plt.close(figure)
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


def normalize_amplitude(amplitude):
    maximum = np.max(amplitude) if len(amplitude) else 0.0
    if maximum <= 0:
        return amplitude
    return amplitude / maximum


def _padded_limits(*arrays, include_zero=False):
    values = np.concatenate([
        np.asarray(array, dtype=float).reshape(-1)
        for array in arrays
    ])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if include_zero:
        minimum = min(minimum, 0.0)
        maximum = max(maximum, 0.0)
    span = maximum - minimum
    margin = 0.05 * span if span > 0 else max(abs(maximum), 1.0) * 0.05
    return minimum - margin, maximum + margin


def _save_figure(figure, output_path):
    if output_path is not None:
        figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)


def plot_training_am_am(input_signal, output_signal, output_dir=None):
    input_amplitude = normalize_amplitude(np.abs(input_signal))
    output_amplitude = normalize_amplitude(np.abs(output_signal))
    curve_input, curve_output = build_binned_am_am_curve(
        input_amplitude, output_amplitude)

    output_dir = Path(output_dir) if output_dir is not None else None
    figure = plt.figure(figsize=(14, 8))
    plt.scatter(input_amplitude, output_amplitude, s=50, alpha=0.28,
                color='purple', label='Training samples')
    if len(curve_input) > 0:
        plt.plot(curve_input, curve_output, color='blue', linewidth=5.0,
                 label='Training static AM-AM (median)')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=3.0,
             label='Linear reference')
    plt.title('Normalized AM-AM training response', fontsize=36)
    plt.xlabel('Normalized input amplitude |x_train|', fontsize=30)
    plt.ylabel('Normalized output amplitude |y_train|', fontsize=30)
    plt.xlim(*_padded_limits(input_amplitude, curve_input, include_zero=True))
    plt.ylim(*_padded_limits(output_amplitude, curve_output, include_zero=True))
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=30, markerscale=2, frameon=True)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tight_layout()
    if output_dir is not None:
        np.savetxt(output_dir / 'training_am_am.csv',
                   np.column_stack((input_amplitude, output_amplitude)),
                   delimiter=',', header='input_amplitude,output_amplitude',
                   comments='')
    _save_figure(figure, output_dir / 'training_am_am.png'
                 if output_dir is not None else None)


def plot_validation_real_comparison(in_validation, out_validation,
                                    out_estimated, output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else None
    figure = plt.figure(figsize=(14, 8))
    plt.scatter(in_validation.real, out_validation.real,
                label='Original data', color='blue', s=100)
    plt.scatter(in_validation.real, out_estimated.real,
                color='orange', label='Fit', alpha=0.8, s=100)
    limits = _padded_limits(
        in_validation.real, out_validation.real, out_estimated.real,
        include_zero=True,
    )
    plt.plot(limits, limits, 'k--', linewidth=2.5,
             label='Linear reference')
    plt.xlabel('in_validation (real part)', fontsize=30)
    plt.ylabel('out_validation (real part)', fontsize=30)
    plt.title('Original and Estimated Data with Errors', fontsize=36)
    plt.legend(fontsize=30, markerscale=2)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.xlim(*limits)
    plt.ylim(*limits)
    plt.tight_layout()
    _save_figure(
        figure,
        output_dir / 'validation_real_comparison.png'
        if output_dir is not None else None,
    )


def plot_raw_am_am(input_signal, output_signal, output_dir=None,
                   tail_percentile=99.0):
    input_amplitude = normalize_amplitude(np.abs(input_signal))
    output_amplitude = normalize_amplitude(np.abs(output_signal))
    cutoff = np.percentile(input_amplitude, tail_percentile)
    visible = input_amplitude <= cutoff

    output_dir = Path(output_dir) if output_dir is not None else None
    figure = plt.figure(figsize=(14, 8))
    plt.scatter(input_amplitude[visible], output_amplitude[visible],
                s=10, alpha=0.16, color='tab:blue',
                label='Training samples')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2,
             label='Linear reference')
    plt.title('Normalized AM-AM response without statistical curve',
              fontsize=36)
    plt.xlabel('Normalized input amplitude |x_train|', fontsize=30)
    plt.ylabel('Normalized output amplitude |y_train|', fontsize=30)
    plt.xlim(*_padded_limits(input_amplitude[visible], include_zero=True))
    plt.ylim(*_padded_limits(output_amplitude[visible], include_zero=True))
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=30, markerscale=2)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tight_layout()
    if output_dir is not None:
        np.savetxt(
            output_dir / 'training_am_am_raw.csv',
            np.column_stack((input_amplitude[visible], output_amplitude[visible])),
            delimiter=',', header='input_amplitude,output_amplitude',
            comments='',
        )
    _save_figure(figure, output_dir / 'training_am_am_raw.png'
                 if output_dir is not None else None)


def plot_static_am_am(pa_model, dpd_model, maximum_input_amplitude,
                      memory_line_count, sample_count=120, output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else None
    amplitudes = np.linspace(
        0.01 * maximum_input_amplitude,
        maximum_input_amplitude,
        sample_count,
    )
    settling_length = max(memory_line_count + 2, 8)
    pa_output = []
    cascade_output = []

    for amplitude in amplitudes:
        constant_signal = np.full(settling_length, amplitude, dtype=complex)
        pa_output.append(np.abs(pa_model.predict(constant_signal)[-1]))

    pa_output = np.asarray(pa_output)
    low_amplitude_count = max(3, sample_count // 10)
    linear_gain = np.sum(
        amplitudes[:low_amplitude_count] * pa_output[:low_amplitude_count]
    ) / np.sum(amplitudes[:low_amplitude_count] ** 2)
    if linear_gain <= 0:
        linear_gain = 1.0

    for amplitude in amplitudes:
        predistortion_target = np.full(
            settling_length, linear_gain * amplitude, dtype=complex)
        predistorted_signal = dpd_model.predict(predistortion_target)
        predistorted_amplitude = np.abs(predistorted_signal)
        limit_mask = predistorted_amplitude > maximum_input_amplitude
        predistorted_signal[limit_mask] *= (
            maximum_input_amplitude / predistorted_amplitude[limit_mask])
        cascade_output.append(
            np.abs(pa_model.predict(predistorted_signal)[-1]))

    cascade_output = np.asarray(cascade_output)

    normalized_input = amplitudes / maximum_input_amplitude
    normalized_pa = pa_output / (linear_gain * maximum_input_amplitude)
    normalized_cascade = cascade_output / (linear_gain * maximum_input_amplitude)

    figure = plt.figure(figsize=(14, 8))
    
    # Curvas com cores definidas, marcadores diferenciados e zorder para visibilidade perfeita
    plt.plot(normalized_input, normalized_pa, 'o-', color='tab:blue', markersize=10,
             linewidth=1.5, label='PA without DPD', zorder=2)
    plt.plot(normalized_input, normalized_cascade, 's-', color='tab:red', markersize=10,
             linewidth=1.5, label='PA with DPD', zorder=3)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2,
             label='Ideal linear response', zorder=1)
             
    plt.xlabel('Normalized input amplitude', fontsize=30)
    plt.ylabel('Output / small-signal linear gain', fontsize=30)
    plt.title('Static AM-AM response with memory settling', fontsize=36)
    plt.xlim(*_padded_limits(normalized_input, include_zero=True))
    plt.ylim(*_padded_limits(normalized_pa, normalized_cascade,
                             include_zero=True))
                             
    plt.grid(True, alpha=0.35, linestyle='--', zorder=0)
    plt.legend(fontsize=30, markerscale=2, frameon=True)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tight_layout()
    
    if output_dir is not None:
        np.savetxt(
            output_dir / 'static_am_am.csv',
            np.column_stack((normalized_input, normalized_pa,
                             normalized_cascade)),
            delimiter=',',
            header='normalized_input,pa_without_dpd,pa_with_dpd',
            comments='')
            
    _save_figure(figure, output_dir / 'static_am_am.png'
                 if output_dir is not None else None)

def plot_raw_static_am_am(input_signal, no_dpd_output, dpd_input,
                          dpd_output, output_dir=None, tail_percentile=99.0):
    input_signal = np.asarray(input_signal, dtype=complex)
    no_dpd_output = np.asarray(no_dpd_output, dtype=complex)
    dpd_input = np.asarray(dpd_input, dtype=complex)
    dpd_output = np.asarray(dpd_output, dtype=complex)

    input_scale = max(
        np.percentile(np.abs(input_signal), tail_percentile),
        np.percentile(np.abs(dpd_input), tail_percentile),
        np.finfo(float).eps,
    )
    low_amplitude = np.abs(input_signal) <= np.percentile(
        np.abs(input_signal), 20.0)
    if np.count_nonzero(low_amplitude) < 3:
        low_amplitude = np.ones(len(input_signal), dtype=bool)
    linear_gain = np.vdot(
        input_signal[low_amplitude], no_dpd_output[low_amplitude])
    linear_gain /= np.vdot(input_signal[low_amplitude],
                           input_signal[low_amplitude])
    output_scale = max(
        abs(linear_gain) * input_scale,
        np.finfo(float).eps,
    )

    no_dpd_input_amplitude = np.abs(input_signal) / input_scale
    no_dpd_output_amplitude = np.abs(no_dpd_output) / output_scale
    dpd_input_amplitude = np.abs(dpd_input) / input_scale
    dpd_output_amplitude = np.abs(dpd_output) / output_scale

    no_dpd_visible = no_dpd_input_amplitude <= np.percentile(
        no_dpd_input_amplitude, tail_percentile)
    dpd_visible = dpd_input_amplitude <= np.percentile(
        dpd_input_amplitude, tail_percentile)

    output_dir = Path(output_dir) if output_dir is not None else None
    figure = plt.figure(figsize=(14, 8))
    plt.scatter(
        no_dpd_input_amplitude[no_dpd_visible],
        no_dpd_output_amplitude[no_dpd_visible],
        s=10, alpha=0.16, color='tab:blue', label='PA without DPD',
    )
    plt.scatter(
        dpd_input_amplitude[dpd_visible],
        dpd_output_amplitude[dpd_visible],
        s=10, alpha=0.16, color='tab:orange', label='PA with DPD',
    )
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2,
             label='Linear reference')
    plt.xlabel('Normalized input amplitude', fontsize=30)
    plt.ylabel('Normalized output amplitude', fontsize=30)
    plt.title('Raw AM-AM response from OFDMA samples', fontsize=36)
    plt.xlim(0, 1.05)
    plt.ylim(*_padded_limits(
        no_dpd_output_amplitude[no_dpd_visible],
        dpd_output_amplitude[dpd_visible], include_zero=True))
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=30, markerscale=2)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tight_layout()
    if output_dir is not None:
        np.savetxt(
            output_dir / 'static_am_am_raw.csv',
            np.column_stack((
                no_dpd_input_amplitude[no_dpd_visible],
                no_dpd_output_amplitude[no_dpd_visible],
                dpd_input_amplitude[dpd_visible],
                dpd_output_amplitude[dpd_visible],
            )),
            delimiter=',',
            header='no_dpd_input,no_dpd_output,dpd_input,dpd_output',
            comments='',
        )
    _save_figure(figure, output_dir / 'static_am_am_raw.png'
                 if output_dir is not None else None)
