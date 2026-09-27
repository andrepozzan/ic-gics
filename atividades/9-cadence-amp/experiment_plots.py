import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from evaluation import build_binned_am_am_curve


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
    figure = plt.figure(figsize=(10, 7))
    plt.scatter(input_amplitude, output_amplitude, s=10, alpha=0.12,
                color='purple', label='Training samples')
    if len(curve_input) > 0:
        plt.plot(curve_input, curve_output, color='blue', linewidth=2.8,
                 label='Training static AM-AM (median)')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2,
             label='Linear reference')
    plt.title('Normalized AM-AM training response', fontsize=18)
    plt.xlabel('Normalized input amplitude |x_train|', fontsize=14)
    plt.ylabel('Normalized output amplitude |y_train|', fontsize=14)
    plt.xlim(*_padded_limits(input_amplitude, curve_input, include_zero=True))
    plt.ylim(*_padded_limits(output_amplitude, curve_output, include_zero=True))
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if output_dir is not None:
        np.savetxt(output_dir / 'training_am_am.csv',
                   np.column_stack((input_amplitude, output_amplitude)),
                   delimiter=',', header='input_amplitude,output_amplitude',
                   comments='')
    _save_figure(figure, output_dir / 'training_am_am.png'
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

    figure = plt.figure(figsize=(10, 7))
    plt.plot(normalized_input, normalized_pa, 'o-', markersize=3,
             linewidth=1.5, label='PA without DPD')
    plt.plot(normalized_input, normalized_cascade, 'o-', markersize=3,
             linewidth=1.5, label='PA with DPD')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2,
             label='Ideal linear response')
    plt.xlabel('Normalized input amplitude', fontsize=14)
    plt.ylabel('Output / small-signal linear gain', fontsize=14)
    plt.title('Static AM-AM response with memory settling', fontsize=18)
    plt.xlim(*_padded_limits(normalized_input, include_zero=True))
    plt.ylim(*_padded_limits(normalized_pa, normalized_cascade,
                             include_zero=True))
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=11)
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
