from pathlib import Path
import argparse
import logging
import shutil

import numpy as np

BASE_PATH = Path(__file__).parent
NETLIST_PATH = BASE_PATH / 'cadence' / 'Sim_Doherty_1.scs'
NPORT_NETLIST_PATH = BASE_PATH / 'cadence' / 'Sim_Nport_1.scs'
DOHERTY_SUBCKT_PATH = BASE_PATH / 'cadence' / 'doherty-amp.scs'
LAST_EXECUTION_DIR = BASE_PATH / 'cadence' / 'ultima_execucao'

from cadence.cadence_runner import run_spectre_simulation
from cadence.cadence_io import (
    export_pwl_signal,
    load_mat_dataset,
    passband_to_complex_baseband,
    read_psfascii_transient,
    resample_trace,
)
from evaluation import (
    align_by_delay,
    calculate_ber,
    calculate_nmse,
    compensate_complex_gain,
)
from experiment_config import ExperimentConfig
from experiment_plots import plot_static_am_am, plot_training_am_am
from lut_model import LUTLineSelector, VariableLUTModel, create_lut_config
from ofdma import extract_bits_from_ofdma, generate_OFDMA_signal
from plotting import plot_psd_comparison
from qam import qam_modulate_passband


DEFAULT_CONFIG = ExperimentConfig()
LOGGER = logging.getLogger('cadence_pa')


def log_progress(message):
    LOGGER.info(message)


def scale_to_training_range(signal, training_signal):
    reference_amplitude = np.percentile(np.abs(training_signal), 99)
    if reference_amplitude <= 0:
        reference_amplitude = np.max(np.abs(training_signal))
    if reference_amplitude <= 0:
        reference_amplitude = 1.0
    return signal / np.max(np.abs(signal)) * reference_amplitude


def estimate_small_signal_gain(input_signal, output_signal):
    amplitude_limit = np.percentile(np.abs(input_signal), 10)
    low_amplitude = np.abs(input_signal) <= amplitude_limit
    input_subset = input_signal[low_amplitude]
    output_subset = output_signal[low_amplitude]
    denominator = np.vdot(input_subset, input_subset)
    if np.abs(denominator) == 0:
        return 1.0 + 0.0j
    return np.vdot(input_subset, output_subset) / denominator


def limit_signal_amplitude(signal, maximum_amplitude):
    amplitude = np.abs(signal)
    scale = np.ones_like(amplitude, dtype=float)
    nonzero = amplitude > maximum_amplitude
    scale[nonzero] = maximum_amplitude / amplitude[nonzero]
    return signal * scale


def add_guard_prefix(signal, config):
    settling_samples = int(np.ceil(
        config.startup_settling_time * config.sampling_rate))
    if settling_samples <= 0:
        return signal, 0
    prefix = signal[-settling_samples:]
    return np.concatenate((prefix, signal)), settling_samples


def remove_guard_prefix(time_vector, signal, settling_samples):
    if settling_samples <= 0:
        return time_vector, signal
    return time_vector[settling_samples:], signal[settling_samples:]


def print_ber_report(transmitted_bits, received_bits):
    LOGGER.info('BER summary (%d users, %d bits/user)',
                len(transmitted_bits), len(transmitted_bits[0]))
    total_errors = 0
    total_bits = 0
    for user_index, (sent, received) in enumerate(
            zip(transmitted_bits, received_bits)):
        ber, errors, bit_count = calculate_ber(sent, received)
        total_errors += errors
        total_bits += bit_count
        LOGGER.info('  User %02d: %.2f%% (%d/%d errors)',
                    user_index, ber * 100, errors, bit_count)
    overall_ber = total_errors / total_bits if total_bits else 0.0
    LOGGER.info('  Overall: %.2f%% (%d/%d errors)',
                overall_ber * 100, total_errors, total_bits)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the Cadence PA experiment')
    parser.add_argument(
        '--data-source', choices=('mat',), default=DEFAULT_CONFIG.data_source,
        help='Use the extraction/validation MAT files')
    parser.add_argument(
        '--run-spectre', action='store_true',
        help='Run the Spectre netlist after exporting the input PWL signal')
    parser.add_argument(
        '--cadence-validation', action='store_true',
        help='Validate the DPD pipeline with the local behavioral N-port PA')
    parser.add_argument(
        '--spectre-executable', default=None,
        help='Spectre executable path; defaults to SPECTRE or spectre')
    parser.add_argument(
        '--ssh-host', default=DEFAULT_CONFIG.cadence_ssh_host or None,
        help='Remote server host; defaults to ExperimentConfig')
    parser.add_argument(
        '--ssh-user', default=DEFAULT_CONFIG.cadence_ssh_user or None,
        help='SSH user; defaults to ExperimentConfig')
    parser.add_argument(
        '--ssh-key', default=DEFAULT_CONFIG.cadence_ssh_key or None,
        help='Private SSH key path; avoids password prompts')
    parser.add_argument(
        '--remote-directory', default=DEFAULT_CONFIG.cadence_remote_directory,
        help='Remote working directory for the Spectre simulation')
    parser.add_argument(
        '--remote-setup-command', default=None,
        help='Remote setup command; defaults to source ~/cadence/gpdk045/cds')
    arguments = parser.parse_args()
    if (arguments.run_spectre or arguments.cadence_validation) and not arguments.ssh_host:
        parser.error('--run-spectre/--cadence-validation requires --ssh-host or a configured host')
    if arguments.run_spectre and arguments.cadence_validation:
        parser.error('choose either --run-spectre or --cadence-validation')
    return arguments


def main():
    arguments = parse_arguments()
    if LAST_EXECUTION_DIR.exists():
        shutil.rmtree(LAST_EXECUTION_DIR)
    LAST_EXECUTION_DIR.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LAST_EXECUTION_DIR / 'execution.log',
                                mode='w', encoding='utf-8'),
        ],
        force=True,
    )
    config = ExperimentConfig(data_source=arguments.data_source)
    cadence_validation = arguments.cadence_validation
    cadence_netlist = NPORT_NETLIST_PATH if cadence_validation else NETLIST_PATH
    cadence_remote_netlist = None
    cadence_input_trace = 'n_in'
    cadence_output_trace = 'n_out'
    log_progress('Starting experiment')

    if arguments.run_spectre or cadence_validation:
        log_progress('Generating extraction stimulus')
        np.random.seed(7)
        extraction_signal, _ = generate_OFDMA_signal(
            config.fft_size,
            config.number_of_users,
            config.subcarriers_per_user,
            config.modulation_order,
        )
        extraction_signal = (
            extraction_signal / np.max(np.abs(extraction_signal)) * 0.04)
        extraction_signal_tx, extraction_guard_samples = add_guard_prefix(
            extraction_signal, config)
        extraction_passband, extraction_time = qam_modulate_passband(
            extraction_signal_tx, config.carrier_frequency, config.sampling_rate)
        extraction_pwl = BASE_PATH / 'cadence' / 'training_input.pwl'
        export_pwl_signal(
            extraction_passband, config.sampling_rate, extraction_pwl)
        log_progress('Running first Spectre simulation')
        extraction_results = run_spectre_simulation(
            cadence_netlist,
            output_directory=BASE_PATH / 'cadence' / 'spectre_run',
            spectre_executable=arguments.spectre_executable,
            ssh_host=arguments.ssh_host,
            ssh_user=arguments.ssh_user,
            remote_directory=arguments.remote_directory,
            remote_setup_command=arguments.remote_setup_command,
            input_pwl_path=extraction_pwl,
            ssh_key=arguments.ssh_key,
            ssh_port=config.cadence_ssh_port,
            ssh_control_path=config.cadence_ssh_control_path,
            remote_include_directories=config.cadence_remote_include_directories,
            remote_netlist_path=cadence_remote_netlist,
            remote_input_directory=config.cadence_remote_input_directory,
            max_step=config.spectre_max_step,
            stop_time=len(extraction_passband) / config.sampling_rate,
            strobe_period=1 / config.sampling_rate,
            save_traces=(cadence_input_trace, cadence_output_trace),
            local_include_files=(
                (DOHERTY_SUBCKT_PATH,) if not cadence_validation else ())
        )
        log_progress('Reading first Spectre results')

        raw_result_time, result_input, result_output = read_psfascii_transient(
            extraction_results, input_trace=cadence_input_trace,
            output_trace=cadence_output_trace)
        result_time, result_input = resample_trace(
            raw_result_time, result_input, config.sampling_rate,
            sample_count=len(extraction_passband))
        _, result_output = resample_trace(
            raw_result_time, result_output, config.sampling_rate,
            sample_count=len(extraction_passband))
        input_baseband = passband_to_complex_baseband(
            result_input, result_time, config.carrier_frequency,
            config.sampling_rate, config.baseband_bandwidth)
        output_baseband = passband_to_complex_baseband(
            result_output, result_time, config.carrier_frequency,
            config.sampling_rate, config.baseband_bandwidth)
        if extraction_guard_samples > 0:
            input_baseband = input_baseband[extraction_guard_samples:]
            output_baseband = output_baseband[extraction_guard_samples:]
        output_baseband, extraction_delay = align_by_delay(
            input_baseband, output_baseband)
        output_baseband, extraction_gain = compensate_complex_gain(
            input_baseband, output_baseband)
        LOGGER.info('Training alignment: delay %.3f samples; gain %s',
                    extraction_delay, extraction_gain)
        split_index = len(input_baseband) // 2
        input_training = input_baseband[:split_index]
        output_training = output_baseband[:split_index]
        input_validation = input_baseband[split_index:]
        output_validation = output_baseband[split_index:]
        log_progress(f'Loaded {len(input_training)} extraction samples')
    else:
        dataset = load_mat_dataset(
            BASE_PATH / 'dados' / 'pa_sync_extraction.mat',
            BASE_PATH / 'dados' / 'pa_sync_validation.mat',
        )
        input_training = dataset.input_training
        output_training = dataset.output_training
        input_validation = dataset.input_validation
        output_validation = dataset.output_validation
        output_training, mat_delay = align_by_delay(
            input_training, output_training)
        output_training, mat_gain = compensate_complex_gain(
            input_training, output_training)
        output_validation = output_validation / mat_gain
        LOGGER.info('MAT alignment: delay %.3f samples; gain %s',
                    mat_delay, mat_gain)

    log_progress('Training variable LUT model')
    maximum_amplitude = np.percentile(np.abs(input_training), 99.9)
    lut_config = create_lut_config(
        maximum_amplitude, config.lut_columns_per_line)
    selector = LUTLineSelector(
        lut_config, config.selected_memory_line_count, verbose=0)
    selector.select(input_training, output_training)
    selected_model = selector.create_selected_model().fit(
        input_training, output_training)
    log_progress('PA model training completed')

    LOGGER.info('Selected memory lines: %s; coefficients: %d -> %d',
                selector.selected_lines, lut_config.coefficient_count,
                selected_model.config.coefficient_count)

    dpd_maximum_amplitude = np.percentile(np.abs(output_training), 99.9)
    dpd_lut_config = create_lut_config(
        dpd_maximum_amplitude,
        config.lut_columns_per_line,
        active_lines=selector.selected_lines,
    )

    dpd_model = VariableLUTModel(dpd_lut_config, verbose=0).fit(
        output_training, input_training)
    
    log_progress('DPD model training completed')
    log_progress('Generating test OFDMA signal')

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
    predicted_amplifier_output = selected_model.predict(dpd_output)
    dpd_output_tx, test_guard_samples = add_guard_prefix(
        dpd_output, config)
    test_passband_signal, _ = qam_modulate_passband(
        dpd_output_tx, config.carrier_frequency, config.sampling_rate)

    export_pwl_signal(
        test_passband_signal, config.sampling_rate,
        BASE_PATH / 'cadence' / 'out_of_antena.pwl')
    export_pwl_signal(
        test_passband_signal, config.sampling_rate,
        BASE_PATH / 'cadence' / 'sinal_entrada_cadence.pwl')

    if arguments.run_spectre or cadence_validation:
        log_progress('Running second Spectre simulation')
        test_results = run_spectre_simulation(
            cadence_netlist,
            output_directory=BASE_PATH / 'cadence' / 'spectre_run',
            spectre_executable=arguments.spectre_executable,
            ssh_host=arguments.ssh_host,
            ssh_user=arguments.ssh_user,
            remote_directory=arguments.remote_directory,
            remote_setup_command=arguments.remote_setup_command,
            input_pwl_path=BASE_PATH / 'cadence' / 'out_of_antena.pwl',
            ssh_key=arguments.ssh_key,
            ssh_port=config.cadence_ssh_port,
            ssh_control_path=config.cadence_ssh_control_path,
            remote_include_directories=config.cadence_remote_include_directories,
            remote_netlist_path=cadence_remote_netlist,
            remote_input_directory=config.cadence_remote_input_directory,
            max_step=config.spectre_max_step,
            stop_time=len(test_passband_signal) / config.sampling_rate,
            strobe_period=1 / config.sampling_rate,
            save_traces=(cadence_input_trace, cadence_output_trace),
            local_include_files=(
                (DOHERTY_SUBCKT_PATH,) if not cadence_validation else ())
        )
        log_progress('Reading second Spectre results')
        raw_test_time, test_input, test_output = read_psfascii_transient(
            test_results, input_trace=cadence_input_trace,
            output_trace=cadence_output_trace)
        test_time, test_input = resample_trace(
            raw_test_time, test_input, config.sampling_rate,
            sample_count=len(test_passband_signal))
        _, test_output = resample_trace(
            raw_test_time, test_output, config.sampling_rate,
            sample_count=len(test_passband_signal))
        amplifier_output = passband_to_complex_baseband(
            test_output, test_time, config.carrier_frequency,
            config.sampling_rate, config.baseband_bandwidth)
        if test_guard_samples > 0:
            amplifier_output = amplifier_output[test_guard_samples:]
        log_progress('Real Spectre output loaded')
    else:
        amplifier_output = predicted_amplifier_output

    log_progress('Calculating BER')
    aligned_output, delay = align_by_delay(ofdma_signal, amplifier_output)
    aligned_output, complex_gain = compensate_complex_gain(
        ofdma_signal, aligned_output)
    LOGGER.info('Compensation gain: %s; estimated delay: %.3f samples',
                complex_gain, delay)
    received_bits = extract_bits_from_ofdma(
        aligned_output,
        config.fft_size,
        config.number_of_users,
        config.subcarriers_per_user,
        config.modulation_order,
    )
    print_ber_report(transmitted_bits, received_bits)

    log_progress('Calculating NMSE and generating plots')
    validation_output = selected_model.predict(input_validation)
    LOGGER.info('Validation NMSE: %.6f dB',
                calculate_nmse(output_validation, validation_output))

    plot_training_am_am(input_training, output_training,
                        output_dir=LAST_EXECUTION_DIR)

    plot_static_am_am(
        selected_model,
        dpd_model,
        maximum_amplitude,
        selected_model.config.line_count,
        output_dir=LAST_EXECUTION_DIR,
    )
    no_dpd_output = selected_model.predict(ofdma_signal)
    plot_psd_comparison(
        ofdma_signal, no_dpd_output, amplifier_output, config.sampling_rate,
        output_dir=LAST_EXECUTION_DIR)


if __name__ == '__main__':
    main()
