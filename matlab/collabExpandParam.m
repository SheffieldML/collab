function model = collabExpandParam(model, params)

% COLLABEXPANDPARAM Expand a parameter vector into a COLLAB model.
% FORMAT
% DESC takes the given vector of parameters and places them in the
% model structure, it then updates any stored representations that
% are dependent on those parameters, for example kernel matrices
% etc..
% ARG model : the model structure for which parameters are to be
% updated.
% ARG params : a vector of parameters for placing in the model
% structure.
% RETURN model : a returned model structure containing the updated
% parameters.
% 
% SEEALSO : collabCreate, collabExtractParam, modelExtractParam
%
% COPYRIGHT : Neil D. Lawrence, 2009

% COLLAB


  startVal = 1;
  endVal = model.N*model.q;
  model.X = reshape(params(startVal:endVal), model.N, model.q);
  startVal = endVal +1;
  endVal = endVal + model.kern.nParams;
  model.kern = kernExpandParam(model.kern, params(startVal:endVal));
  
  fhandle = str2func([model.noiseTransform 'Transform']);
  if isfield(model, 'heteroNoise') && model.heteroNoise
    startVal = endVal + 1;
    endVal = endVal + model.N;
    model.diagvar = fhandle(params(startVal:endVal), 'atox')';
  elseif model.M>1
    startVal = endVal + 1;
    endVal = endVal + 1;
    model.sigma2 = fhandle(params(startVal:endVal), 'atox');
  end
  model = collabUpdateKernels(model);
end