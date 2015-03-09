function [params, names] = collabExtractParam(model)

% COLLABEXTRACTPARAM Extract a parameter vector from a COLLAB model.
% FORMAT
% DESC extracts the model parameters from a structure containing
% the information about a Gaussian process.
% ARG model : the model structure containing the information about
% the model.
% RETURN params : a vector of parameters from the model.
%
% DESC does the same as above, but also returns parameter names.
% ARG model : the model structure containing the information about
% the model.
% RETURN params : a vector of parameters from the model.
% RETURN names : cell array of parameter names.
%
% SEEALSO : collabCreate, collabExpandParam, modelExtractParam
%
% COPYRIGHT : Neil D. Lawrence, 2009

% COLLAB

if nargout > 1
  returnNames = true;
else
  returnNames = false;
end

if returnNames
  [params, names] = kernExtractParam(model.kern);
  for i = 1:length(names)
    names{i} = ['Kernel, ' names{i}];
  end
else
  params = kernExtractParam(model.kern);
end
params = [model.X(:)' params];
if returnNames
  for i = 1:size(model.X, 1)
    for j = 1:size(model.X, 2)
      Xnames{i, j} = ['X(' num2str(i) ', ' num2str(j) ')'];
    end
  end
  names = {Xnames{:}, names{:}};
end
fhandle = str2func([model.noiseTransform 'Transform']);
if model.heteroNoise
  params = [params fhandle(model.diagvar, 'xtoa')'];
  if returnNames
    for i = 1:model.N
      sigNames{i} = ['Sigma2(' num2str(i) ')'];
    end
    names = {names{:}, sigNames{:}};
  end
elseif model.M > 1
  params = [params fhandle(model.sigma2, 'xtoa')];
  if returnNames
    names = {names{:}, 'Sigma2'};
  end
end