function metrics = calculate_validation_metrics(analysis)
  sample_count = min([numel(analysis.output_validation), ...
                      numel(analysis.validation_output_estimated)]);
  if sample_count == 0
    error('Nao ha amostras suficientes para calcular a validacao.');
  end

  metrics.sample_count = sample_count;
  metrics.validation_error = ...
      analysis.output_validation(1:sample_count) - ...
      analysis.validation_output_estimated(1:sample_count);
  reference = analysis.output_validation(1:sample_count);
  metrics.nmse_db = 10 * log10(sum(abs(metrics.validation_error).^2) / ...
                                sum(abs(reference).^2));
end
