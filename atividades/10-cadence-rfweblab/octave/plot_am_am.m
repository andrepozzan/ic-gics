function plot_am_am(analysis, metrics, output_dir, keep_open)
  figure_handle = figure('Color', 'w');
  plot(abs(analysis.input_training), abs(analysis.output_training), '.', ...
       'Color', [0.12, 0.35, 0.78], 'MarkerSize', 10);
  grid on; box on;
  xlabel('|x[n]| - entrada', 'FontSize', 22);
  ylabel('|y[n]| - saida', 'FontSize', 22);
  title(sprintf('Extracao (%d amostras)', ...
                numel(analysis.input_training)), 'FontSize', 26);
  set(gca, 'FontSize', 18, 'LineWidth', 1.5);
  save_report_figure(figure_handle, output_dir, '01_am_am_extracao', ...
                     keep_open);

  figure_handle = figure('Color', 'w');
  plot(abs(analysis.input_validation), abs(analysis.output_validation), '.', ...
       'Color', [0.12, 0.55, 0.32], 'MarkerSize', 10);
  hold on;
  plot(abs(analysis.input_validation(1:metrics.sample_count)), ...
       abs(analysis.validation_output_estimated), '.', ...
       'Color', [0.88, 0.35, 0.08], 'MarkerSize', 10);
  grid on; box on;
  xlabel('|x[n]| - entrada', 'FontSize', 22);
  ylabel('|y[n]| - saida', 'FontSize', 22);
  title(sprintf('Validacao e modelo (NMSE = %.2f dB)', metrics.nmse_db), ...
        'FontSize', 26);
  legend('PA medido', 'Modelo estimado', 'Location', 'northeast', ...
         'FontSize', 18);
  set(gca, 'FontSize', 18, 'LineWidth', 1.5);
  save_report_figure(figure_handle, output_dir, '02_am_am_validacao', ...
                     keep_open);
end
