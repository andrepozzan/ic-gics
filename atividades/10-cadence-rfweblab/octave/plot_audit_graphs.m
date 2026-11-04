function plot_audit_graphs(analysis, output_dir, keep_open)
  required = {'ofdma_input', 'ofdma_output_no_dpd', 'dpd_output', ...
              'aligned_amplifier_output', 'reference_symbols', ...
              'measured_symbols', 'sampling_rate'};
  for index = 1:numel(required)
    if ~isfield(analysis, required{index})
      warning(['Campo ausente; graficos de auditoria ignorados: ', ...
               required{index}]);
      return;
    end
  end

  input_signal = analysis.ofdma_input(:);
  no_dpd = analysis.ofdma_output_no_dpd(:);
  dpd = analysis.dpd_output(:);
  aligned = analysis.aligned_amplifier_output(:);
  sample_count = min([numel(input_signal), numel(no_dpd), numel(dpd), ...
                      numel(aligned)]);
  input_signal = input_signal(1:sample_count);
  no_dpd = no_dpd(1:sample_count);
  dpd = dpd(1:sample_count);
  aligned = aligned(1:sample_count);
  time_ns = (0:sample_count - 1)' / analysis.sampling_rate * 1e9;

  figure_handle = figure('Color', 'w');
  plot(abs(input_signal), abs(no_dpd), '.', 'Color', [0.12, 0.35, 0.78], ...
       'MarkerSize', 14);
  hold on;
  plot(abs(input_signal), abs(dpd), '.', 'Color', [0.85, 0.33, 0.10], ...
       'MarkerSize', 14);
  plot(abs(input_signal), abs(aligned), '.', 'Color', [0.12, 0.55, 0.32], ...
       'MarkerSize', 14);
  grid on; box on;
  xlabel('|x|', 'FontSize', 22); ylabel('|y|', 'FontSize', 22);
  title('AM-AM: sem DPD, DPD e PA', 'FontSize', 26);
  legend('PA sem DPD', 'Entrada DPD', 'PA com DPD', ...
         'Location', 'northeast', 'FontSize', 18);
  set(gca, 'FontSize', 18, 'LineWidth', 1.5);
  save_report_figure(figure_handle, output_dir, '07_auditoria_am_am', keep_open);

  figure_handle = figure('Color', 'w');
  plot(time_ns, real(input_signal), 'Color', [0.12, 0.35, 0.78], ...
       'LineWidth', 1.8);
  hold on;
  plot(time_ns, real(no_dpd), 'Color', [0.85, 0.33, 0.10], 'LineWidth', 1.8);
  plot(time_ns, real(aligned), 'Color', [0.12, 0.55, 0.32], ...
       'LineWidth', 1.8);
  grid on; box on;
  xlabel('Tempo (ns)', 'FontSize', 22); ylabel('Amplitude', 'FontSize', 22);
  title('Dominio do tempo auditado', 'FontSize', 26);
  legend('Entrada', 'PA sem DPD', 'PA com DPD', ...
         'Location', 'northeast', 'FontSize', 18);
  set(gca, 'FontSize', 18, 'LineWidth', 1.5);
  save_report_figure(figure_handle, output_dir, '08_auditoria_tempo', keep_open);

  nfft = 2 ^ nextpow2(sample_count);
  frequency = (-nfft / 2:nfft / 2 - 1)' * ...
              analysis.sampling_rate / nfft / 1e6;
  spectra = [fftshift(abs(fft(input_signal, nfft))), ...
             fftshift(abs(fft(no_dpd, nfft))), ...
             fftshift(abs(fft(aligned, nfft)))];
  spectra_db = 20 * log10(max(spectra ./ max(spectra(:)), eps));
  figure_handle = figure('Color', 'w');
  plot(frequency, spectra_db(:, 1), 'Color', [0.12, 0.35, 0.78], ...
       'LineWidth', 1.8);
  hold on;
  plot(frequency, spectra_db(:, 2), 'Color', [0.85, 0.33, 0.10], ...
       'LineWidth', 1.8);
  plot(frequency, spectra_db(:, 3), 'Color', [0.12, 0.55, 0.32], ...
       'LineWidth', 1.8);
  grid on; box on;
  xlabel('Frequencia (MHz)', 'FontSize', 22);
  ylabel('Magnitude (dB)', 'FontSize', 22);
  title('Espectro auditado', 'FontSize', 26);
  ylim([-80, 5]);
  legend('Entrada', 'PA sem DPD', 'PA com DPD', ...
         'Location', 'northeast', 'FontSize', 18);
  set(gca, 'FontSize', 18, 'LineWidth', 1.5);
  save_report_figure(figure_handle, output_dir, '09_auditoria_espectro', keep_open);

  reference = analysis.reference_symbols(:);
  measured = analysis.measured_symbols(:);
  symbol_count = min(numel(reference), numel(measured));
  figure_handle = figure('Color', 'w');
  plot(real(reference(1:symbol_count)), imag(reference(1:symbol_count)), ...
       '.', 'Color', [0.55, 0.16, 0.55], 'MarkerSize', 14);
  hold on;
  plot(real(measured(1:symbol_count)), imag(measured(1:symbol_count)), ...
       '.', 'Color', [0.88, 0.35, 0.08], 'MarkerSize', 14);
  grid on; box on; axis equal;
  xlabel('I', 'FontSize', 22); ylabel('Q', 'FontSize', 22);
  title('Constelacao auditada', 'FontSize', 26);
  legend('Referencia', 'Medida', 'Location', 'northeast', 'FontSize', 18);
  set(gca, 'FontSize', 18, 'LineWidth', 1.5);
  save_report_figure(figure_handle, output_dir, '10_auditoria_constelacao', keep_open);

  figure_handle = figure('Color', 'w');
  plot(abs(analysis.input_training), abs(analysis.output_training), '.', ...
       'Color', [0.12, 0.35, 0.78], 'MarkerSize', 14);
  grid on; box on;
  xlabel('|x_{train}|', 'FontSize', 22);
  ylabel('|y_{train}|', 'FontSize', 22);
  title('AM-AM bruto de treinamento', 'FontSize', 26);
  set(gca, 'FontSize', 18, 'LineWidth', 1.5);
  save_report_figure(figure_handle, output_dir, '11_dados_treinamento', keep_open);
end
