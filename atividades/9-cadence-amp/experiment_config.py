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
    cadence_remote_netlist: str = '~/cadence/gpdk045/simulacao/doherty-amp.scs'
    cadence_remote_input_directory: str = '~/cadence/gpdk045'
    cadence_remote_include_directories: tuple[str, ...] = (
        '~/simulation',
        '~/cadence/gpdk045',
        '~/cadence/gpdk045/simulacao',
    )
    carrier_frequency: float = 26e9
    sampling_rate: float = 104e9
    number_of_users: int = 10
    subcarriers_per_user: int = 48
    modulation_order: int = 256
    fft_size: int = 2048
    memory_line_count: int = 5
    selected_memory_line_count: int = 3
    lut_columns_per_line: tuple[int, ...] = (16, 12, 8, 4, 2)

    def __post_init__(self):
        if self.data_source != 'mat':
            raise ValueError("data_source must be 'mat'")
        if len(self.lut_columns_per_line) != self.memory_line_count:
            raise ValueError('One LUT size is required per memory line')
