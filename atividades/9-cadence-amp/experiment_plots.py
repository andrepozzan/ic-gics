import matplotlib.pyplot as plt
import numpy as np

from evaluation import build_binned_am_am_curve


def normalize_amplitude(amplitude):
    maximum = np.max(amplitude) if len(amplitude) else 0.0
    if maximum <= 0:
        return amplitude
    return amplitude / maximum


def plot_training_am_am(input_signal, output_signal):
    input_amplitude = normalize_amplitude(np.abs(input_signal))
    output_amplitude = normalize_amplitude(np.abs(output_signal))
    curve_input, curve_output = build_binned_am_am_curve(
        input_amplitude, output_amplitude)

    plt.figure(figsize=(10, 7))
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
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_static_am_am(pa_model, dpd_model, maximum_input_amplitude,
                      memory_line_count, sample_count=120):
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
        cascade_output.append(
            np.abs(pa_model.predict(predistorted_signal)[-1]))

    cascade_output = np.asarray(cascade_output)

    normalized_input = amplitudes / maximum_input_amplitude
    normalized_pa = pa_output / (linear_gain * maximum_input_amplitude)
    normalized_cascade = cascade_output / (linear_gain * maximum_input_amplitude)

    plt.figure(figsize=(10, 7))
    plt.plot(normalized_input, normalized_pa, 'o-', markersize=3,
             linewidth=1.5, label='PA without DPD')
    plt.plot(normalized_input, normalized_cascade, 'o-', markersize=3,
             linewidth=1.5, label='PA with DPD')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2,
             label='Ideal linear response')
    plt.xlabel('Normalized input amplitude', fontsize=14)
    plt.ylabel('Output / small-signal linear gain', fontsize=14)
    plt.title('Static AM-AM response with memory settling', fontsize=18)
    plt.xlim(0, 1.02)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
