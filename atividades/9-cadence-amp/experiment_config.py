from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    data_source: str = 'cadence'
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
        if self.data_source not in ('cadence', 'mat'):
            raise ValueError("data_source must be 'cadence' or 'mat'")
        if len(self.lut_columns_per_line) != self.memory_line_count:
            raise ValueError('One LUT size is required per memory line')
