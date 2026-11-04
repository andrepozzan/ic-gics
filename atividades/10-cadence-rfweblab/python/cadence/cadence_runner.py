import os
import logging
from pathlib import Path
import shlex
import shutil
import subprocess


LOGGER = logging.getLogger('cadence_pa.spectre')


def _validate_spectre_executable(executable):
    executable_name = Path(executable).name.lower()
    if executable_name in {'virtuoso', 'adexl', 'maestro', 'ade'}:
        raise ValueError(
            f'Graphical Cadence executable is not supported: {executable}. '
            'Use the command-line spectre executable.')


def _remote_expression(path):
    if path.startswith('~/'):
        return '$HOME/' + shlex.quote(path[2:])
    return shlex.quote(path)


def run_spectre_simulation(netlist_path, spectre_executable=None,
                           output_directory=None, ssh_host=None,
                           ssh_user=None, remote_directory=None,
                           remote_setup_command=None, input_pwl_path=None,
                           ssh_key=None, ssh_port=22,
                           ssh_control_path='~/.ssh/sockets/%r@%h-%p',
                           remote_include_directories=None,
                           remote_netlist_path=None,
                           remote_input_directory=None, max_step=1e-12,
                           stop_time=None, save_traces=('n_in', 'n_out'),
                           local_include_files=(), strobe_period=None):
    netlist_path = Path(netlist_path).resolve()
    output_directory = Path(
        output_directory or netlist_path.parent / 'spectre_run')
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    executable = spectre_executable or os.environ.get('SPECTRE', 'spectre')
    _validate_spectre_executable(executable)
    if not ssh_host:
        raise ValueError('SSH host is required for remote Spectre execution')

    return _run_remote_spectre(
        netlist_path,
        output_directory,
        executable,
        ssh_host,
        ssh_user,
        remote_directory or '~/simulation',
        remote_setup_command or 'source ~/cadence/gpdk045/cds',
        input_pwl_path or netlist_path.with_name('sinal_entrada_cadence.pwl'),
        ssh_key,
        ssh_port,
        ssh_control_path,
        remote_include_directories or ('~/simulation',),
        remote_netlist_path,
        remote_input_directory,
        max_step,
        stop_time,
        save_traces,
        local_include_files,
        strobe_period,
    )


def _run_remote_spectre(netlist_path, output_directory, executable,
                        ssh_host, ssh_user, remote_directory,
                        setup_command, input_pwl_path, ssh_key, ssh_port,
                        ssh_control_path, include_directories,
                        remote_netlist_path, remote_input_directory,
                        max_step, stop_time, save_traces,
                        local_include_files, strobe_period):
    host = f'{ssh_user}@{ssh_host}' if ssh_user else ssh_host
    remote_dir = _remote_expression(remote_directory)
    input_directory = remote_input_directory or remote_directory

    if remote_netlist_path:
        remote_netlist_dir, remote_netlist_name = remote_netlist_path.rsplit('/', 1)
        netlist_workdir = _remote_expression(remote_netlist_dir)
        simulation_netlist_name = 'Sim_Doherty_transient.scs'
    else:
        remote_netlist_dir = remote_directory
        remote_netlist_name = netlist_path.name
        netlist_workdir = remote_dir
        simulation_netlist_name = remote_netlist_name

    remote_output = f'{remote_dir}/spectre_run'
    remote_output_scp = f'{host}:{remote_directory}/spectre_run'
    remote_target = f'{host}:{remote_directory}/'
    local_input = Path(input_pwl_path)
    if not local_input.is_file():
        raise FileNotFoundError(f'PWL file not found: {local_input}')

    auxiliary_include = netlist_path.with_name('ade_e.scs')
    expanded_key = os.path.expanduser(ssh_key) if ssh_key else None
    control_path = os.path.expanduser(ssh_control_path)
    Path(control_path).parent.mkdir(parents=True, exist_ok=True)
    common_options = [
        '-o', f'Port={ssh_port}',
        '-o', 'ConnectTimeout=15',
        '-o', 'ServerAliveInterval=15',
        '-o', 'ServerAliveCountMax=3',
        '-o', 'ControlMaster=auto',
        '-o', f'ControlPath={control_path}',
        '-o', 'ControlPersist=1h',
        '-o', 'ForwardX11=no',
        '-o', 'ForwardAgent=no',
        '-o', 'Compression=yes',
    ]
    ssh_options = list(common_options)
    scp_options = list(common_options)
    if expanded_key:
        ssh_options += ['-i', expanded_key]
        scp_options += ['-i', expanded_key]

    LOGGER.info('Connecting to %s', host)
    prepare_command = (
        f'mkdir -p {remote_dir} {_remote_expression(remote_netlist_dir)} '
        f'{_remote_expression(input_directory)} && '
        f'rm -rf {remote_dir}/spectre_run')
    if remote_netlist_path:
        stop_expression = f'{stop_time:g}' if stop_time is not None else '20n'
        source_netlist = f'{netlist_workdir}/{shlex.quote(remote_netlist_name)}'
        transient_netlist = f'{netlist_workdir}/{simulation_netlist_name}'
        prepare_command += (
            f' && cp {source_netlist} {transient_netlist}'
            f' && printf "\\ntran tran stop={stop_expression} '
            f'maxstep={max_step:g}\\nsave {save_traces[0]} '
            f'{save_traces[1]}\\n" '
            f' >> {transient_netlist}')
    try:
        subprocess.run(
            ['ssh', *ssh_options, host, prepare_command],
            check=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f'SSH preparation on {host} timed out after 120 seconds. '
            'Check the host, VPN/network access, and SSH control socket.') from error

    if not remote_netlist_path:
        subprocess.run(
            ['scp', *scp_options, str(netlist_path), remote_target],
            check=True, text=True)

    for include_file in local_include_files:
        include_path = Path(include_file).resolve()
        if not include_path.is_file():
            raise FileNotFoundError(
                f'Local include file not found: {include_path}')
        subprocess.run(
            ['scp', *scp_options, str(include_path),
             f'{host}:{netlist_workdir}/'],
            check=True, text=True)

    upload_targets = {
        f'{host}:{netlist_workdir}/sinal_entrada_cadence.pwl',
        f'{host}:{remote_dir}/sinal_entrada_cadence.pwl',
        f'{host}:{input_directory}/sinal_entrada_cadence.pwl',
    }
    for target in sorted(upload_targets):
        subprocess.run(
            ['scp', *scp_options, str(local_input), target],
            check=True, text=True)

    if auxiliary_include.is_file():
        subprocess.run(
            ['scp', *scp_options, str(auxiliary_include),
             f'{host}:{remote_netlist_dir}/ade_e.scs'],
            check=True, text=True)

    if not remote_netlist_path:
        stop_expression = f'{stop_time:g}' if stop_time is not None else '20n'
        strobe_expression = (
            f' strobeperiod={strobe_period:g}' if strobe_period else '')
        override_command = (
            f"sed -i -E "
            f"'s/^tran[[:space:]]+tran[[:space:]]+stop=[^[:space:]]+"
            f"[[:space:]]+maxstep=[^[:space:]]+/"
            f"tran tran stop={stop_expression} maxstep={max_step:g}"
            f"{strobe_expression}/' "
            f"{netlist_workdir}/{simulation_netlist_name}")
        subprocess.run(
            ['ssh', *ssh_options, host, override_command],
            check=True, text=True)
    LOGGER.info('Input files uploaded; starting Spectre in batch mode (no GUI)')

    include_options = []
    for directory in include_directories:
        include_options.append(f'-I{_remote_expression(directory)}')
    remote_command = (
        f'cd {netlist_workdir} && {setup_command} && '
        f'{shlex.quote(executable)} {shlex.quote(simulation_netlist_name)} '
        f'{" ".join(include_options)} '
        f'-raw {remote_output} -format psfascii')
    completed = subprocess.run(
        ['ssh', *ssh_options, host, remote_command],
        check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            'Remote Spectre simulation failed with exit code '
            f'{completed.returncode}.\n{completed.stdout}\n{completed.stderr}')

    LOGGER.info('Remote Spectre completed; downloading results')
    subprocess.run(
        ['scp', *scp_options, '-r', remote_output_scp,
         str(output_directory.parent)],
        check=True, text=True)
    LOGGER.info('Spectre results downloaded: %s', output_directory)
    return output_directory
