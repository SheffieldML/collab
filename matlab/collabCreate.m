function model = collabCreate(q, d, y, options);

% COLLABCREATE Create a COLLAB model with inducing varibles/pseudo-inputs.
% FORMAT
% DESC creates a collaborative filter structure with a latent space of q.
% ARG q : input data dimension.
% ARG d : the number of processes (i.e. output data dimension).
% ARG Y : the data.
% ARG options : options structure as defined by collabOptions.m.
% RETURN model : model structure containing the GP collaborative filter.
%
% SEEALSO : collabOptions, modelCreate
%
% COPYRIGHT : Neil D. Lawrence, 2008

% COLLAB

  
  model.type = 'collab';
  
  model.q = q;
  model.d = d;
  model.N = size(y, 1);
  model.y = y;
  model.mu = zeros(model.N, 1);
  model.sd = ones(model.N, 1);
  model.currentOut = 1;
  model.m = collabComputeM(model);
  model.numParams = model.N*model.q;
  model.kern = kernCreate(q, options.kern);
  model.numParams = model.numParams + model.kern.nParams;
  model.X = randn(model.N, q)*0.001;
  model.change = zeros(size(model.X));
  model.changeParam = zeros(1, model.kern.nParams);
  % This forces kernel computation.
  %model = collabExpandParam(model, initParams);
  model.heteroNoise = options.heteroNoise; % Whether or not to have diagonal
                                           % noise variance.
  model.noiseTransform = optimiDefaultConstraint('positive');
  model.M = options.numComps;
  if model.M > 1
    model.pi = repmat(1/model.M, 1, model.M);
    model.sigma2 = exp(-2);
    model.lnsigma2Change = 0;
    ind = find(model.m);
    model = collabInitS(model);
    model.numParams = model.numParams + 1;
  end
  if model.heteroNoise
    model.diagvar = repmat(exp(-2), model.N, 1);
    model.lndiagChange = zeros(model.N, 1);
    model.numParams = model.numParams + model.N;
  end
  initParams = collabExtractParam(model);
  model = collabExpandParam(model, initParams);
end
