% AUDITAR_PA Auditoria independente dos resultados da atividade 10.
%
% O Python executa o treinamento, DPD, calculos e simulacoes. Ao terminar,
% ele exporta todos os sinais e metricas para pa_analysis.mat. Este script
% nao executa novamente o treinamento nem altera os resultados: ele apenas
% carrega o MAT e reproduz as analises no Octave/MATLAB para auditoria.
%
% Uso no Octave, a partir da raiz da atividade:
%   octave --quiet --path octave --eval \
%       "auditar_pa('', true)"
%
% No MATLAB, adicione a pasta octave ao caminho e execute:
%   addpath('octave');
%   auditar_pa('resultados/AAAAmmdd_HHMMSS', true);

function auditar_pa(execution_dir, interactive)
  script_dir = fileparts(mfilename('fullpath'));
  project_dir = fileparts(script_dir);
  if nargin < 1 || isempty(execution_dir)
    execution_dir = '';
  end
  if nargin < 2
    interactive = false;
  end
  if isempty(execution_dir)
    executions_dir = fullfile(project_dir, 'resultados');
    entries = dir(fullfile(executions_dir, '20*'));
    entries = entries([entries.isdir]);
    if isempty(entries)
      error(['Nenhuma pasta de execucao encontrada em: ', executions_dir]);
    end
    [~, newest] = max([entries.datenum]);
    execution_dir = fullfile(executions_dir, entries(newest).name);
  elseif exist(execution_dir, 'dir') ~= 7
    error(['Pasta de execucao nao encontrada: ', execution_dir]);
  end

  analysis_path = fullfile(execution_dir, 'pa_analysis.mat');
  output_dir = fullfile(execution_dir, 'octave_report');

  analysis = load_analysis_data(analysis_path, output_dir);
  metrics = calculate_validation_metrics(analysis);

  plot_am_am(analysis, metrics, output_dir, interactive);
  plot_am_pm_error(analysis, metrics, output_dir, interactive);
  plot_validation_spectrum(analysis, output_dir, interactive);
  plot_audit_graphs(analysis, output_dir, interactive);

  fprintf('Relatorio Octave salvo em: %s\n', output_dir);
  fprintf('NMSE de validacao calculado no Octave: %.6f dB\n', ...
          metrics.nmse_db);
  if isfield(analysis, 'validation_nmse_db')
    fprintf('NMSE exportado pelo Python: %.6f dB\n', ...
            analysis.validation_nmse_db);
  end
  if interactive
    fprintf('Feche as janelas do Octave para finalizar.\n');
    while ~isempty(findall(0, 'type', 'figure'))
      pause(0.25);
    end
  end
end
