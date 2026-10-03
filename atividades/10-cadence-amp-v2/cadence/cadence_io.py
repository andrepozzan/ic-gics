from dataclasses import dataclass
import logging
from pathlib import Path
import re

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


LOGGER = logging.getLogger('cadence_pa.io')


@dataclass(frozen=True)
class TrainingValidationData:
    input_training: np.ndarray
    output_training: np.ndarray
    input_validation: np.ndarray
    output_validation: np.ndarray
    sampling_rate: float


def passband_to_complex_baseband(passband_signal, time_vector,
                                 carrier_frequency, sampling_rate,
                                 bandwidth=None):
    analytic_signal = hilbert(passband_signal)
    baseband = analytic_signal * np.exp(
        -1j * 2 * np.pi * carrier_frequency * time_vector)
    if bandwidth is None:
        return baseband
    normalized_cutoff = bandwidth / (sampling_rate / 2)
    sos = butter(6, normalized_cutoff, btype='lowpass', output='sos')
    return sosfiltfilt(sos, baseband)
    
def resample_trace(time_vector, signal, sampling_rate, sample_count=None):
    if sample_count is None:
        sample_count = int(np.floor(time_vector[-1] * sampling_rate)) + 1
    target_time = np.arange(sample_count) / sampling_rate
    resampled_signal = np.interp(
        target_time, time_vector, signal,
        left=signal[0], right=signal[-1])
    return target_time, resampled_signal


def read_psfascii_transient(results_directory, input_trace='n_in',
                            output_trace='n_out'):
    result_files = sorted(
        path for path in Path(results_directory).rglob('*')
        if path.is_file() and path.name == 'tran.tran.tran'
    ) + sorted(
        path for path in Path(results_directory).rglob('*')
        if path.is_file() and path.name == 'envlp.td.envlp'
    ) + sorted(
        path for path in Path(results_directory).rglob('*')
        if path.is_file() and path.name not in ('tran.tran.tran', 'envlp.td.envlp')
        and path.suffix not in ('.log', '.out'))

    number_re = re.compile(
        r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')
    name_value_re = re.compile(
        r'"([^"]+)"\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)')

    for result_file in result_files:
        try:
            text = result_file.read_text(errors='ignore')
        except OSError:
            continue
        value_section = text.split('VALUE', 1)
        if len(value_section) != 2:
            continue

        rows = []
        current_time = None
        current_values = {}
        for line in value_section[1].splitlines():
            line = line.strip()
            if not line or line == 'END':
                continue
            match = name_value_re.match(line)
            if not match:
                continue
            name, value_str = match.group(1), float(match.group(2))
            if name == 'time':
                # Flush previous timestep
                if current_time is not None and \
                        input_trace in current_values and \
                        output_trace in current_values:
                    rows.append((current_time,
                                 current_values[input_trace],
                                 current_values[output_trace]))
                current_time = value_str
                current_values = {}
            else:
                current_values[name] = value_str

        # Flush the last timestep
        if current_time is not None and \
                input_trace in current_values and \
                output_trace in current_values:
            rows.append((current_time,
                         current_values[input_trace],
                         current_values[output_trace]))

        if rows:
            LOGGER.info('Reading Spectre result: %s', result_file)
            return np.asarray(rows, dtype=float).T

    raise FileNotFoundError(
        f'Could not find PSF ASCII traces {input_trace} and {output_trace} '
        f'in {results_directory}')


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


def export_pwl_signal(signal, sampling_rate, output_path):
    time_vector = np.arange(len(signal)) / sampling_rate
    pwl_data = np.column_stack((time_vector, np.real(signal)))
    np.savetxt(output_path, pwl_data, fmt='%.12e %.12e')
    return Path(output_path)
