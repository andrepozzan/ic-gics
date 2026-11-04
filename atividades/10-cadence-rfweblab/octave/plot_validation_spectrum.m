function plot_validation_spectrum(analysis, output_dir, keep_open)
  signal_count = min([numel(analysis.input_validation), ...
                      numel(analysis.output_validation), ...
                      numel(analysis.validation_output_estimated)]);
  nfft = 2 ^ nextpow2(signal_count);
  frequency = (-nfft / 2:nfft / 2 - 1)' * analysis.sampling_rate / ...
              nfft / 1e9;

  input_spectrum = fftshift(abs(fft( ...
      analysis.input_validation(1:signal_count), nfft)));
  measured_spectrum = fftshift(abs(fft( ...
      analysis.output_validation(1:signal_count), nfft)));
  estimated_spectrum = fftshift(abs(fft( ...
      analysis.validation_output_estimated(1:signal_count), nfft)));
  reference_level = max([input_spectrum; measured_spectrum; ...
                         estimated_spectrum]);

  figure_handle = figure('Color', 'w');
  plot(frequency, 20 * log10(max(input_spectrum / reference_level, eps)), ...
       'Color', [0.12, 0.35, 0.78], 'LineWidth', 1.2);
  hold on;
  plot(frequency, 20 * log10(max(measured_spectrum / reference_level, eps)), ...
       'Color', [0.12, 0.55, 0.32], 'LineWidth', 1.2);
  plot(frequency, 20 * log10(max(estimated_spectrum / reference_level, eps)), ...
       '--', 'Color', [0.88, 0.35, 0.08], 'LineWidth', 1.2);
  grid on; box on;
  xlabel('Frequencia (GHz)', 'FontSize', 22);
  ylabel('Magnitude normalizada (dB)', 'FontSize', 22);
  title('Espectro da validacao', 'FontSize', 26);
  legend('Entrada', 'PA medido', 'Modelo estimado', 'Location', 'northeast');
  set(gca, 'FontSize', 18, 'LineWidth', 1.5);

  save_report_figure(figure_handle, output_dir, '03_espectro', keep_open);
end
