import numpy as np
from scipy.signal import correlate
from scipy.optimize import minimize_scalar


def calculate_nmse(reference_signal, estimated_signal):
    error = reference_signal - estimated_signal
    return 10 * np.log10(
        np.sum(np.abs(error) ** 2) / np.sum(np.abs(reference_signal) ** 2))


def estimate_integer_delay(reference_signal, target_signal):
    correlation = correlate(
        target_signal, reference_signal, mode='full', method='fft')
    lags = np.arange(-len(reference_signal) + 1, len(target_signal))
    return int(lags[np.argmax(np.abs(correlation))])


def estimate_fractional_delay(reference_signal, target_signal):
    correlation = correlate(
        target_signal, reference_signal, mode='full', method='fft')
    lags = np.arange(-len(reference_signal) + 1, len(target_signal))
    peak_index = int(np.argmax(np.abs(correlation)))
    integer_delay = float(lags[peak_index])
    if len(reference_signal) != len(target_signal):
        return integer_delay

    def negative_correlation(delay):
        aligned = fractional_shift(target_signal, -delay)
        return -abs(np.vdot(reference_signal, aligned))

    result = minimize_scalar(
        negative_correlation,
        bounds=(integer_delay - 0.5, integer_delay + 0.5),
        method='bounded',
        options={'xatol': 1e-3},
    )
    return float(result.x)


def fractional_shift(signal, shift):
    frequencies = np.fft.fftfreq(len(signal))
    spectrum = np.fft.fft(signal)
    return np.fft.ifft(spectrum * np.exp(-2j * np.pi * frequencies * shift))


def align_by_delay(reference_signal, target_signal):
    delay = estimate_fractional_delay(reference_signal, target_signal)
    return fractional_shift(target_signal, -delay), delay


def compensate_complex_gain(reference_signal, target_signal):
    denominator = np.vdot(reference_signal, reference_signal)
    if np.abs(denominator) == 0:
        return target_signal, 1.0 + 0.0j
    gain = np.vdot(reference_signal, target_signal) / denominator
    if np.abs(gain) == 0:
        return target_signal, gain
    return target_signal / gain, gain


def calculate_evm(reference_symbols, measured_symbols):
    reference_symbols = np.asarray(reference_symbols, dtype=complex)
    measured_symbols = np.asarray(measured_symbols, dtype=complex)
    sample_count = min(len(reference_symbols), len(measured_symbols))
    if sample_count == 0:
        raise ValueError('Cannot calculate EVM without symbols')
    reference_symbols = reference_symbols[:sample_count]
    measured_symbols = measured_symbols[:sample_count]
    corrected_symbols, gain = compensate_complex_gain(
        reference_symbols, measured_symbols)
    error = corrected_symbols - reference_symbols
    reference_rms = np.sqrt(np.mean(np.abs(reference_symbols) ** 2))
    if reference_rms == 0:
        raise ValueError('Cannot calculate EVM for zero reference power')
    evm_ratio = np.sqrt(np.mean(np.abs(error) ** 2)) / reference_rms
    evm_percent = 100.0 * evm_ratio
    evm_db = 20.0 * np.log10(max(evm_ratio, np.finfo(float).tiny))
    return evm_ratio, evm_percent, evm_db, gain


def calculate_ber(transmitted_bits, received_bits):
    bit_count = min(len(transmitted_bits), len(received_bits))
    if bit_count == 0:
        return 0.0, 0, 0
    error_count = sum(
        transmitted != received
        for transmitted, received in zip(
            transmitted_bits[:bit_count], received_bits[:bit_count]))
    return error_count / bit_count, error_count, bit_count


def group_bits(bit_string, group_size=4):
    return ' '.join(
        bit_string[index:index + group_size]
        for index in range(0, len(bit_string), group_size))


def build_binned_am_am_curve(input_amplitude, output_amplitude,
                             bin_count=80, minimum_samples=12):
    if len(input_amplitude) == 0:
        return np.array([]), np.array([])

    minimum = np.min(input_amplitude)
    maximum = np.max(input_amplitude)
    if maximum <= minimum:
        return np.array([minimum]), np.array([np.median(output_amplitude)])

    edges = np.linspace(minimum, maximum, bin_count + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_indices = np.digitize(input_amplitude, edges) - 1
    curve_input = []
    curve_output = []

    for bin_index in range(bin_count):
        mask = bin_indices == bin_index
        if np.count_nonzero(mask) >= minimum_samples:
            curve_input.append(centers[bin_index])
            curve_output.append(np.median(output_amplitude[mask]))

    if not curve_input:
        return np.array([]), np.array([])
    return np.asarray(curve_input), np.maximum.accumulate(curve_output)
