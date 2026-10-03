import matplotlib
matplotlib.use('Agg')
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
