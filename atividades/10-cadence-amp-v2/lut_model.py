from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class LUTConfig:
    columns_per_line: tuple[int, ...]
    interpolation_grids: tuple[np.ndarray, ...]
    active_lines: tuple[int, ...] | None = None

    def __post_init__(self):
        line_count = len(self.columns_per_line)
        if len(self.interpolation_grids) != line_count:
            raise ValueError('One interpolation grid is required per memory line')
        if any(current < following for current, following in zip(
                self.columns_per_line, self.columns_per_line[1:])):
            raise ValueError('LUT sizes must not increase for older memory lines')
        active_lines = self.active_lines
        if active_lines is None:
            active_lines = tuple(range(line_count))
            object.__setattr__(self, 'active_lines', active_lines)
        if not active_lines:
            raise ValueError('At least one LUT line must be active')
        if any(line < 0 or line >= line_count for line in active_lines):
            raise ValueError('Active LUT line is outside the configured range')

    @property
    def line_count(self):
        return len(self.columns_per_line)

    @property
    def coefficient_count(self):
        return 2 * sum(self.columns_per_line[line] for line in self.active_lines)

    def with_active_lines(self, active_lines):
        return LUTConfig(
            columns_per_line=self.columns_per_line,
            interpolation_grids=self.interpolation_grids,
            active_lines=tuple(sorted(active_lines)),
        )


class VariableLUTModel:
    def __init__(self, config, verbose=0):
        self.config = config
        self.verbose = verbose
        self.coefficients = self._create_zero_coefficients()

    def _create_zero_coefficients(self):
        return [
            np.zeros(columns, dtype=complex)
            for columns in self.config.columns_per_line
        ]

    def _pack_coefficients(self, coefficients):
        real_part = np.concatenate([
            coefficients[line].real for line in self.config.active_lines])
        imaginary_part = np.concatenate([
            coefficients[line].imag for line in self.config.active_lines])
        return np.concatenate((real_part, imaginary_part))

    def _unpack_coefficients(self, packed_coefficients):
        active_columns = sum(
            self.config.columns_per_line[line] for line in self.config.active_lines)
        real_part = packed_coefficients[:active_columns]
        imaginary_part = packed_coefficients[active_columns:]
        coefficients = self._create_zero_coefficients()

        offset = 0
        for line in self.config.active_lines:
            column_count = self.config.columns_per_line[line]
            coefficients[line] = (
                real_part[offset:offset + column_count]
                + 1j * imaginary_part[offset:offset + column_count]
            )
            offset += column_count
        return coefficients

    def predict(self, input_signal, coefficients=None):
        coefficients = self.coefficients if coefficients is None else coefficients
        result = np.zeros(len(input_signal), dtype=complex)

        for line in self.config.active_lines:
            delayed_signal = np.roll(input_signal, line)
            delayed_signal[:line] = 0
            amplitude = np.abs(delayed_signal)
            grid = self.config.interpolation_grids[line]
            real_gain = np.interp(amplitude, grid, coefficients[line].real)
            imaginary_gain = np.interp(amplitude, grid, coefficients[line].imag)
            result += delayed_signal * (real_gain + 1j * imaginary_gain)
        return result

    def _residuals(self, packed_coefficients, input_signal, target_signal):
        coefficients = self._unpack_coefficients(packed_coefficients)
        residual = target_signal - self.predict(input_signal, coefficients)
        return np.concatenate((residual.real, residual.imag))

    def fit(self, input_signal, target_signal):
        initial_coefficients = self._pack_coefficients(self.coefficients)
        result = least_squares(
            self._residuals,
            initial_coefficients,
            args=(input_signal, target_signal),
            verbose=self.verbose,
        )
        self.coefficients = self._unpack_coefficients(result.x)
        return self


class LUTLineSelector:
    def __init__(self, full_config, selected_line_count, verbose=0):
        if selected_line_count > full_config.line_count:
            raise ValueError('Selected line count exceeds configured line count')
        self.full_config = full_config
        self.selected_line_count = selected_line_count
        self.verbose = verbose
        self.contributions = {}
        self.selected_lines = ()

    def select(self, input_signal, target_signal):
        full_model = VariableLUTModel(self.full_config, self.verbose)
        full_model.fit(input_signal, target_signal)
        full_error = np.sum(np.abs(
            target_signal - full_model.predict(input_signal)) ** 2)

        for line in range(self.full_config.line_count):
            reduced_config = self.full_config.with_active_lines(
                other_line for other_line in range(self.full_config.line_count)
                if other_line != line)
            reduced_model = VariableLUTModel(reduced_config, self.verbose)
            reduced_error = np.sum(np.abs(
                target_signal - reduced_model.predict(
                    input_signal, full_model.coefficients)) ** 2)
            self.contributions[line] = reduced_error - full_error

        ranked_lines = sorted(
            self.contributions,
            key=self.contributions.get,
            reverse=True,
        )
        self.selected_lines = tuple(sorted(ranked_lines[:self.selected_line_count]))
        return full_model

    def create_selected_model(self):
        selected_config = self.full_config.with_active_lines(self.selected_lines)
        return VariableLUTModel(selected_config, self.verbose)


def create_lut_config(maximum_amplitude, columns_per_line, active_lines=None):
    interpolation_grids = tuple(
        np.linspace(0.0, maximum_amplitude, columns)
        for columns in columns_per_line
    )
    return LUTConfig(
        columns_per_line=tuple(columns_per_line),
        interpolation_grids=interpolation_grids,
        active_lines=active_lines,
    )
