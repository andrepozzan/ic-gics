from pathlib import Path
import argparse

import numpy as np

from cadence_io import export_pwl_signal, load_dataset
from evaluation import (
    align_by_delay,
    calculate_ber,
    calculate_nmse,
    group_bits,
)
from experiment_config import ExperimentConfig
from experiment_plots import plot_static_am_am
from lut_model import LUTLineSelector, VariableLUTModel, create_lut_config
from ofdma import extract_bits_from_ofdma, generate_OFDMA_signal
from plotting import plot_psd_comparison
from qam import qam_modulate_passband


BASE_PATH = Path(__file__).parent
DEFAULT_CONFIG = ExperimentConfig()


def scale_to_training_range(signal, training_signal):
    reference_amplitude = np.percentile(np.abs(training_signal), 99)
    if reference_amplitude <= 0:
        reference_amplitude = np.max(np.abs(training_signal))
    if reference_amplitude <= 0:
        reference_amplitude = 1.0
    return signal / np.max(np.abs(signal)) * reference_amplitude


def print_ber_report(transmitted_bits, received_bits):
    print('Bits sent per user:')
    for user_index, bits in enumerate(transmitted_bits):
        print(f'User {user_index}: {group_bits(bits)}')

    print('\nBits received per user:')
    for user_index, bits in enumerate(received_bits):
        print(f'User {user_index}: {group_bits(bits)}')

    print('\nBER per user:')
    for user_index, (sent, received) in enumerate(
            zip(transmitted_bits, received_bits)):
        ber, errors, bit_count = calculate_ber(sent, received)
        print(f'User {user_index}: BER = {ber * 100:.2f}% '
              f'({errors}/{bit_count})')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the Cadence PA experiment')
    parser.add_argument(
        '--data-source', choices=('cadence', 'mat'), default=DEFAULT_CONFIG.data_source,
        help='Use the Cadence CSV or the extraction/validation MAT files')
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    config = ExperimentConfig(data_source=arguments.data_source)
    dataset = load_dataset(
        config.data_source,
        BASE_PATH / 'dados-para-TREINO.csv',
        BASE_PATH / 'dados' / 'pa_sync_extraction.mat',
        BASE_PATH / 'dados' / 'pa_sync_validation.mat',
        config.carrier_frequency,
    )
    input_training = dataset.input_training
    output_training = dataset.output_training
    input_validation = dataset.input_validation
    output_validation = dataset.output_validation

    maximum_amplitude = np.percentile(np.abs(input_training), 99.9)
    lut_config = create_lut_config(
        maximum_amplitude, config.lut_columns_per_line)
    selector = LUTLineSelector(
        lut_config, config.selected_memory_line_count, verbose=2)
    selector.select(input_training, output_training)
    selected_model = selector.create_selected_model().fit(
        input_training, output_training)

    print('LUT contributions:', selector.contributions)
    print('Selected memory lines:', selector.selected_lines)
    print('LUT coefficients:', lut_config.coefficient_count, '->',
          selected_model.config.coefficient_count)

    dpd_maximum_amplitude = np.percentile(np.abs(output_training), 99.9)
    dpd_lut_config = create_lut_config(
        dpd_maximum_amplitude,
        config.lut_columns_per_line,
        active_lines=selector.selected_lines,
    )
    dpd_model = VariableLUTModel(dpd_lut_config, verbose=2).fit(
        output_training, input_training)
    ofdma_signal, transmitted_bits = generate_OFDMA_signal(
        config.fft_size,
        config.number_of_users,
        config.subcarriers_per_user,
        config.modulation_order,
    )
    ofdma_signal = scale_to_training_range(ofdma_signal, input_training)

    ideal_passband_signal, _ = qam_modulate_passband(
        ofdma_signal, config.carrier_frequency, config.sampling_rate)
    dpd_output = dpd_model.predict(ofdma_signal)
    amplifier_output = selected_model.predict(dpd_output)
    passband_output, _ = qam_modulate_passband(
        amplifier_output, config.carrier_frequency, config.sampling_rate)

    export_pwl_signal(
        passband_output, config.sampling_rate,
        BASE_PATH / 'out_of_antena.pwl')
    export_pwl_signal(
        ideal_passband_signal, config.sampling_rate,
        BASE_PATH / 'sinal_entrada_cadence.pwl')

    aligned_output, delay = align_by_delay(ofdma_signal, amplifier_output)
    print(f'OFDMA estimated delay (samples): {delay}')
    received_bits = extract_bits_from_ofdma(
        aligned_output,
        config.fft_size,
        config.number_of_users,
        config.subcarriers_per_user,
        config.modulation_order,
    )
    print_ber_report(transmitted_bits, received_bits)

    validation_output = selected_model.predict(input_validation)
    print(f'NMSE: {calculate_nmse(output_validation, validation_output):.6f} dB')

    plot_static_am_am(
        selected_model,
        dpd_model,
        maximum_amplitude,
        selected_model.config.line_count,
    )
    no_dpd_output = selected_model.predict(ofdma_signal)
    plot_psd_comparison(
        ofdma_signal, no_dpd_output, amplifier_output, config.sampling_rate)


if __name__ == '__main__':
    main()
