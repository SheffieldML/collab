function [mu, varsigma, secondMoment] = collabComponentPosteriorMeanVar(model, X)
% COLLABCOMPONENTPOSTERIORMEANVAR Compute the posterior mean and variance for each component.
% FORMAT
% DESC computes the posterior mean and variance asssociated with each
% component of the mixture model.
% ARG model : the model for which means and variances are computed.
% ARG x : optional input argment where means and variances are to be
% computed. If not provided model.X is used.
% RETURN mu : the mean associated with each of the components as a cell
% array.
% RETURN varsigma : the variance associated with each of the components
% as a cell array.
% RETURN secondMoment : the second moment associated with each of the
% components as a cell array.
%
% SEEALSO : collabCreate
% 
% COPYRIGHT : Neil D. Lawrence, 2009
  
% COLLAB

  
% Work out component means and variances.
  ind = find(model.m);
  if nargin > 1
    Kx = kernCompute(model.kern, model.X(ind, :), X);
    diagK = kernDiagCompute(model.kern, X);
  else 
    Kx = model.K;
    diagK = diag(model.K);
  end
  ind = find(model.m);
  for m = 1:model.M
    Kinvk = model.invK{m}*Kx;
    mu{m} = Kinvk'*model.m(ind);
    varsigma{m} = diagK - sum(Kx.*Kinvk, 1)';
    if nargout > 2
      secondMoment{m} = varsigma{m} + mu{m}.*mu{m};
    end
  end
end
  