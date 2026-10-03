import numpy as np
from scipy.optimize import least_squares
from dataclasses import dataclass


@dataclass(frozen=True)
class DPDConfig:
    n_lut_lines: int
    lut_columns_per_line: tuple[int, ...]
    initial_complex_coef: tuple[np.ndarray, ...]
    interpolation_in_per_line: tuple[np.ndarray, ...]
    active_lut_lines: tuple[int, ...] | None = None

    def __post_init__(self):
        if len(self.lut_columns_per_line) != self.n_lut_lines:
            raise ValueError('One LUT size is required for each memory line')
        if any(current < following for current, following in zip(
                self.lut_columns_per_line, self.lut_columns_per_line[1:])):
            raise ValueError('LUT sizes must not increase for older memory lines')
        if len(self.initial_complex_coef) != self.n_lut_lines:
            raise ValueError('One initial coefficient vector is required for each memory line')
        if len(self.interpolation_in_per_line) != self.n_lut_lines:
            raise ValueError('One interpolation grid is required for each memory line')
        if self.active_lut_lines is None:
            object.__setattr__(self, 'active_lut_lines', tuple(range(self.n_lut_lines)))

        for line in self.active_lut_lines:
            if line < 0 or line >= self.n_lut_lines:
                raise ValueError(f'Invalid active LUT line: {line}')

    @property
    def n_lut_columns(self):
        return max(self.lut_columns_per_line)

    def with_active_lines(self, active_lut_lines):
        return DPDConfig(
            n_lut_lines=self.n_lut_lines,
            lut_columns_per_line=self.lut_columns_per_line,
            initial_complex_coef=self.initial_complex_coef,
            interpolation_in_per_line=self.interpolation_in_per_line,
            active_lut_lines=tuple(sorted(active_lut_lines)),
        )


def _pack_complex_coefficients(coefficient_lines, cfg):
    return np.concatenate([
        np.concatenate([coefficient_lines[line].real for line in cfg.active_lut_lines]),
        np.concatenate([coefficient_lines[line].imag for line in cfg.active_lut_lines]),
    ])


def _unpack_complex_coefficients(real_coef, cfg):
    n_active_coefficients = sum(
        cfg.lut_columns_per_line[line] for line in cfg.active_lut_lines)
    real_part = real_coef[:n_active_coefficients]
    imag_part = real_coef[n_active_coefficients:]
    coefficient_lines = [
        np.zeros(columns, dtype=complex)
        for columns in cfg.lut_columns_per_line
    ]

    offset = 0
    for line in cfg.active_lut_lines:
        columns = cfg.lut_columns_per_line[line]
        coefficient_lines[line] = (
            real_part[offset:offset + columns]
            + 1j * imag_part[offset:offset + columns]
        )
        offset += columns
    return coefficient_lines

def estimatedValueWithLUT(x_data, lut_out_matrix, cfg):
    length_of_x_data = len(x_data)
    result = np.zeros(length_of_x_data, dtype=complex)
    
    for line in cfg.active_lut_lines:
        # Move the elements of an array in a circular pattern.
        delayed = np.roll(x_data, line)
        
        # Zero out the first 'line' elements to create a memory effect
        delayed[:line] = 0

        x_abs = np.abs(delayed)
        interpolation_in = cfg.interpolation_in_per_line[line]
        real_interpolation = np.interp(x_abs, interpolation_in, lut_out_matrix[line].real)
        imag_interpolation = np.interp(x_abs, interpolation_in, lut_out_matrix[line].imag)
        result += delayed * (real_interpolation + 1j * imag_interpolation)
    return result


def residuals(lut_out_real, x_data, y_data, cfg):
    lut_out_complex = _unpack_complex_coefficients(lut_out_real, cfg)
    y_estimated = estimatedValueWithLUT(x_data, lut_out_complex, cfg)
    residual = y_data - y_estimated
    res_vector = np.concatenate([residual.real, residual.imag])
    return res_vector

def calcCoef(in_data, out_data, cfg):
    initial_real_coef = _pack_complex_coefficients(cfg.initial_complex_coef, cfg)
    result = least_squares(
        residuals,
        initial_real_coef,
        args=(in_data, out_data, cfg),
        verbose=0,
    )
    return _unpack_complex_coefficients(result.x, cfg)

def dpdTraining(in_training, out_training, cfg):
    dpd_coef = calcCoef(out_training, in_training, cfg)
    return dpd_coef