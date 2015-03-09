function [mu, varsig] = collabPosteriorMeanVarCell(model, ind, y, X)

% COLLABPOSTERIORMEANVARCELL Mean and variances of the posterior at points given by X.
% FORMAT
% DESC returns the posterior mean and variance for a given set of
% points.
% ARG model : the model for which the posterior will be computed.
% ARG x : the input positions for which the posterior will be
% computed.
% ARG ind : the indices of the train data for that user
% RETURN mu : the mean of the posterior distribution.
% RETURN sigma : the variances of the posterior distributions.
%
% SEEALSO : collabCreate
%
% COPYRIGHT : Raquel Urtasun, 2009

% COLLAB

  mu = zeros(size(X, 1), size(y, 2));
  % Compute kernel for new point.
  for i = 1:size(y, 2)
    %ind = find(y(:, i));
    KX_star = kernCompute(model.kern, model.X(ind, :), X);  
    K = kernCompute(model.kern, model.X(ind, :));
    invK = pdinv(K);
    %yind = y(ind, i);
    mu(:, i) =KX_star'*invK*y;
    % Compute if variances required.
  end
  if nargout > 1
    diagK = kernDiagCompute(model.kern, X);
    Kinvk = invK*KX_star;
    varsig = diagK - sum(KX_star.*Kinvk, 1)';
    varsig = repmat(varsig, 1, size(y, 2));
  end
end
