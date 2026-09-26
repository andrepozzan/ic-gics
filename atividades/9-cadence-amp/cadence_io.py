from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import correlate, hilbert


@dataclass(frozen=True)
class TrainingValidationData:
    input_training: np.ndarray
    output_training: np.ndarray
    input_validation: np.ndarray
    output_validation: np.ndarray
    sampling_rate: float


def estimate_sampling_rate(time_vector):
    intervals = np.diff(time_vector)
    mean_interval = np.mean(intervals)
    if mean_interval <= 0:
        raise ValueError('Invalid Cadence time vector')
    return 1.0 / mean_interval


def passband_to_complex_baseband(passband_signal, time_vector, carrier_frequency):
    analytic_signal = hilbert(passband_signal)
    return analytic_signal * np.exp(
        -1j * 2 * np.pi * carrier_frequency * time_vector)


def estimate_exact_delay(reference_signal, target_signal):
    correlation = correlate(
        target_signal, reference_signal, mode='full', method='fft')
    lags = np.arange(-len(reference_signal) + 1, len(target_signal))
    peak_index = np.argmax(np.abs(correlation))

    fractional_offset = 0.0
    if 0 < peak_index < len(correlation) - 1:
        alpha = np.abs(correlation[peak_index - 1])
        beta = np.abs(correlation[peak_index])
        gamma = np.abs(correlation[peak_index + 1])
        denominator = alpha - 2 * beta + gamma
        if denominator != 0:
            fractional_offset = 0.5 * (alpha - gamma) / denominator

    return lags[peak_index] + fractional_offset


def apply_fractional_delay(signal, delay_fraction):
    frequencies = np.fft.fftfreq(len(signal))
    phase_shift = np.exp(-1j * 2 * np.pi * frequencies * delay_fraction)
    return np.fft.ifft(np.fft.fft(signal) * phase_shift)


def align_cadence_signals(input_signal, output_signal):
    exact_delay = estimate_exact_delay(input_signal, output_signal)
    integer_delay = int(np.round(exact_delay))
    fractional_delay = exact_delay - integer_delay

    if integer_delay > 0:
        aligned_input = input_signal[:-integer_delay]
        aligned_output = output_signal[integer_delay:]
    elif integer_delay < 0:
        aligned_input = input_signal[-integer_delay:]
        aligned_output = output_signal[:integer_delay]
    else:
        aligned_input = input_signal.copy()
        aligned_output = output_signal.copy()

    aligned_output = apply_fractional_delay(aligned_output, -fractional_delay)
    trim_length = int(0.05 * len(aligned_input))
    if trim_length > 0 and 2 * trim_length < len(aligned_input):
        aligned_input = aligned_input[trim_length:-trim_length]
        aligned_output = aligned_output[trim_length:-trim_length]

    print(f'Cadence estimated delay: {exact_delay:.4f} samples')
    print(f'Integer part: {integer_delay}, fractional part: {fractional_delay:.4f}')
    return aligned_input, aligned_output


def load_cadence_dataset(csv_path, carrier_frequency):
    csv_data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    input_time = csv_data[:, 0]
    input_passband = csv_data[:, 1]
    output_time = csv_data[:, 2]
    output_passband = csv_data[:, 3]

    input_baseband = passband_to_complex_baseband(
        input_passband, input_time, carrier_frequency)
    output_baseband = passband_to_complex_baseband(
        output_passband, output_time, carrier_frequency)
    input_baseband, output_baseband = align_cadence_signals(
        input_baseband, output_baseband)

    split_index = len(input_baseband) // 2
    return TrainingValidationData(
        input_training=input_baseband[:split_index],
        output_training=output_baseband[:split_index],
        input_validation=input_baseband[split_index:],
        output_validation=output_baseband[split_index:],
        sampling_rate=estimate_sampling_rate(input_time),
    )


def load_mat_dataset(extraction_path, validation_path):
    from scipy.io import loadmat

    extraction_data = loadmat(extraction_path)
    validation_data = loadmat(validation_path)
    required_fields = ('x', 'y')
    for data, path in ((extraction_data, extraction_path),
                       (validation_data, validation_path)):
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(
                f'Missing fields {missing_fields} in MAT file: {path}')

    return TrainingValidationData(
        input_training=np.asarray(extraction_data['x']).reshape(-1),
        output_training=np.asarray(extraction_data['y']).reshape(-1),
        input_validation=np.asarray(validation_data['x']).reshape(-1),
        output_validation=np.asarray(validation_data['y']).reshape(-1),
        sampling_rate=np.nan,
    )


def load_dataset(source, cadence_csv_path, extraction_mat_path,
                 validation_mat_path, carrier_frequency):
    if source == 'cadence':
        return load_cadence_dataset(cadence_csv_path, carrier_frequency)
    if source == 'mat':
        return load_mat_dataset(extraction_mat_path, validation_mat_path)
    raise ValueError(f'Unsupported data source: {source}')


def export_pwl_signal(signal, sampling_rate, output_path):
    time_vector = np.arange(len(signal)) / sampling_rate
    pwl_data = np.column_stack((time_vector, np.real(signal)))
    np.savetxt(output_path, pwl_data, fmt='%.12e %.12e')
    return Path(output_path)
