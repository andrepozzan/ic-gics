function plot_am_pm_error(analysis, metrics, output_dir, keep_open)
  phase_count = min([numel(analysis.input_validation), ...
                     metrics.sample_count]);
  input_signal = analysis.input_validation(1:phase_count);
  measured_signal = analysis.output_validation(1:phase_count);
  estimated_signal = analysis.validation_output_estimated(1:phase_count);
  input_amplitude = abs(input_signal);
  valid_samples = input_amplitude > max(input_amplitude) * 1e-6;
  input_amplitude = input_amplitude(valid_samples);
  measured_signal = measured_signal(valid_samples);
  estimated_signal = estimated_signal(valid_samples);
  input_signal = input_signal(valid_samples);

  % AM/PM e a fase relativa entre saida e entrada, nao a fase absoluta
  % do sinal IQ. O resultado e mantido no intervalo [-180, 180] graus.
  measured_phase = angle(measured_signal .* conj(input_signal)) * 180 / pi;
  estimated_phase = angle(estimated_signal .* conj(input_signal)) * ...
                   180 / pi;

  figure_handle = figure('Color', 'w');
  plot(abs(input_signal), measured_phase, '.', ...
       'Color', [0.12, 0.55, 0.32], 'MarkerSize', 10);
  hold on;
  plot(abs(input_signal), estimated_phase, '.', ...
       'Color', [0.88, 0.35, 0.08], 'MarkerSize', 10);
  grid on; box on;
  xlabel('|x[n]|', 'FontSize', 22);
  ylabel('Fase (graus)', 'FontSize', 22);
  title('Caracteristica AM/PM - validacao', 'FontSize', 26);
  legend('PA medido', 'Modelo estimado', 'Location', 'northeast', ...
         'FontSize', 18);
  set(gca, 'FontSize', 18, 'LineWidth', 1.5);
  save_report_figure(figure_handle, output_dir, '03_am_pm', keep_open);

  figure_handle = figure('Color', 'w');
  plot(abs(input_signal), abs(metrics.validation_error(1:phase_count)), ...
       '.', 'Color', [0.55, 0.16, 0.55], 'MarkerSize', 10);
  grid on; box on;
  xlabel('|x[n]|', 'FontSize', 22);
  ylabel('|erro|', 'FontSize', 22);
  title(sprintf('Erro absoluto (NMSE = %.2f dB)', metrics.nmse_db), ...
        'FontSize', 26);
  set(gca, 'FontSize', 18, 'LineWidth', 1.5);
  save_report_figure(figure_handle, output_dir, '04_erro_validacao', keep_open);
end
