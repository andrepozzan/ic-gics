import os
import logging
from pathlib import Path
import shlex
import shutil
import subprocess


LOGGER = logging.getLogger('cadence_pa.spectre')


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
                           remote_input_directory=None):
    netlist_path = Path(netlist_path).resolve()
    output_directory = Path(
        output_directory or netlist_path.parent / 'spectre_run')
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    executable = spectre_executable or os.environ.get('SPECTRE', 'spectre')
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
    )


def _run_remote_spectre(netlist_path, output_directory, executable,
                        ssh_host, ssh_user, remote_directory,
                        setup_command, input_pwl_path, ssh_key, ssh_port,
                        ssh_control_path, include_directories,
                        remote_netlist_path, remote_input_directory):
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
        '-o', 'ForwardX11=yes',
        '-o', 'ForwardX11Trusted=yes',
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
        source_netlist = f'{netlist_workdir}/{shlex.quote(remote_netlist_name)}'
        transient_netlist = f'{netlist_workdir}/{simulation_netlist_name}'
        prepare_command += (
            f' && cp {source_netlist} {transient_netlist}'
            f' && printf "\\ntran tran stop=20n maxstep=10p\\nsave IN_AMP OUT_AMP\\n" '
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

    subprocess.run(
        ['scp', *scp_options, str(local_input),
         f'{host}:{input_directory}/sinal_entrada_cadence.pwl'],
        check=True, text=True)

    if auxiliary_include.is_file():
        subprocess.run(
            ['scp', *scp_options, str(auxiliary_include),
             f'{host}:{remote_netlist_dir}/ade_e.scs'],
            check=True, text=True)
    LOGGER.info('Input files uploaded; starting remote Spectre')

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
