from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    data_source: str = 'mat'
    cadence_ssh_host: str = 'analog2'
    cadence_ssh_user: str = 'GRR20243424'
    cadence_ssh_key: str = '~/.ssh/id_ed25519'
    cadence_ssh_port: int = 22
    cadence_ssh_control_path: str = '~/.ssh/sockets/%r@%h-%p'
    cadence_remote_directory: str = '~/simulation'
    cadence_remote_netlist: str | None = None
    cadence_remote_input_directory: str = '~/cadence/gpdk045'
    cadence_remote_include_directories: tuple[str, ...] = (
        '~/simulation',
        '~/cadence/gpdk045',
        '~/cadence/gpdk045/simulacao',
    )
    carrier_frequency: float = 26e9
    sampling_rate: float = 104e9
    spectre_max_step: float = 1e-12
    baseband_bandwidth: float = 3e9
    startup_settling_time: float = 1e-9
    input_backoff_db: float = 6.0
    number_of_users: int = 4
    subcarriers_per_user: int = 12
    modulation_order: int = 16
    fft_size: int = 2048
    memory_line_count: int = 5
    selected_memory_line_count: int = 3
    lut_columns_per_line: tuple[int, ...] = (16, 12, 8, 4, 2)

    def __post_init__(self):
        if self.data_source != 'mat':
            raise ValueError("data_source must be 'mat'")
        if len(self.lut_columns_per_line) != self.memory_line_count:
            raise ValueError('One LUT size is required per memory line')
        if self.spectre_max_step <= 0:
            raise ValueError('spectre_max_step must be positive')
        if not 0 < self.baseband_bandwidth < self.sampling_rate / 2:
            raise ValueError('baseband_bandwidth must be below Nyquist')
        if self.startup_settling_time < 0:
            raise ValueError('startup_settling_time must not be negative')
        if self.input_backoff_db < 0:
            raise ValueError('input_backoff_db must not be negative')
