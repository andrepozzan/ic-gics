function analysis = load_analysis_data(analysis_path, output_dir)
  if exist(analysis_path, 'file') ~= 2
    error(['Arquivo de analise nao encontrado: ', analysis_path, ...
           '. Execute primeiro script-10-cadence-rfweblab.py.']);
  end
  if exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
  end

  analysis = load(analysis_path);
  required_fields = {'input_training', 'output_training', ...
                     'input_validation', 'output_validation', ...
                     'validation_output_estimated', 'sampling_rate'};
  for index = 1:numel(required_fields)
    if ~isfield(analysis, required_fields{index})
      error(['Campo ausente no arquivo de analise: ', ...
             required_fields{index}]);
    end
  end

  analysis.input_training = analysis.input_training(:);
  analysis.output_training = analysis.output_training(:);
  analysis.input_validation = analysis.input_validation(:);
  analysis.output_validation = analysis.output_validation(:);
  analysis.validation_output_estimated = ...
      analysis.validation_output_estimated(:);
end
