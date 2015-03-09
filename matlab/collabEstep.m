function model = collabEstep(model, maxIters)
  
% COLLABESTEP Do E step updates and compute resulting Kinv for each component.
% FORMAT
% DESC computes the means and variances of each component of the mixture
% model.
% ARG model : the model for which the means and variances are to be
% computed.
% ARG K : the computed covariance matrix.
% ARG y : the target values.
% RETURN mu : the mean for each component (as a cell array).
% RETURN varsigma : the variance for each component (as a cell array).
%
% SEEALSO : collabLogLikeGradient
%
% COPYRIGHT : Neil D. Lawrence, 2009

% COLLAB

  if nargin < 2
    maxIters = 100;
  end
  ind = find(model.m);
  
  [model.expectation.f, model.expectation.varf] = collabComponentPosteriorMeanVar(model);
  for i = 1:maxIters
    model.expectation.s{model.currentOut} = collabComputeS(model);
    model = collabUpdateKernels(model);
    [model.expectation.f, model.expectation.varf] = collabComponentPosteriorMeanVar(model);
  end

end
