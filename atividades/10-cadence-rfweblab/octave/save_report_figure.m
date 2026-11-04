function save_report_figure(figure_handle, output_dir, base_name, keep_open)
  set(figure_handle, 'Units', 'pixels', 'Position', [80, 80, 1400, 900]);
  axes_handles = findall(figure_handle, 'Type', 'axes');
  for index = 1:numel(axes_handles)
    set(axes_handles(index), 'Units', 'normalized', ...
        'Position', [0.12, 0.14, 0.80, 0.76]);
  end
  print(figure_handle, fullfile(output_dir, [base_name, '.png']), ...
        '-dpng', '-r220');
  if nargin < 4 || ~keep_open
    close(figure_handle);
  end
end
