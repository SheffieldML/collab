function collabDisplay(model, spaceNum)
  
% COLLABDISPLAY Displays the provided collaborative filter model.
% FORMAT
% DESC displays the collaborative model as provided.
% ARG model : the model to display.
% ARG spaceNum : number of spaces to indent display.
%
% SEEALSO : modelDisplay
%
% COPYRIGHT : Neil D. Lawrence, 2008

% COLLAB

  if nargin > 1
    spacing = repmat(32, 1, spaceNum);
  else
    spaceNum = 0;
    spacing = [];
  end
  spacing = char(spacing);
  fprintf(spacing);
  fprintf('Collaborative filter GPLVM:\n')
  fprintf(spacing);
  fprintf('  Number of data points: %d\n', model.N);
  fprintf(spacing);
  fprintf('  Input dimension: %d\n', model.q);
  fprintf(spacing);
  fprintf('  Number of processes: %d\n', model.d);
  if model.heteroNoise
    fprintf(spacing);
    fprintf('  Heteroschodastic noise model, mean %2.4f, sd %2.4f\n', mean(model.diagvar), sqrt(var(model.diagvar)))  
  end
  if model.M > 1
    fprintf(spacing);
    fprintf('  Mixture model with %d components.\n', model.M)
    fprintf(spacing);
    fprintf('    Output variance %2.4f.\n', model.sigma2)
  end
  fprintf(spacing);
  fprintf('  Kernel:\n')

  kernDisplay(model.kern, spaceNum+2)
end