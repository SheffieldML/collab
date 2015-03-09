function [mu, varsig] = collabPosteriorMeanVar(model, y, X);

% COLLABPOSTERIORMEANVAR Mean and variances of the posterior at points given by X.
% FORMAT
% DESC returns the posterior mean and variance for a given set of
% points.
% ARG model : the model for which the posterior will be computed.
% ARG x : the input positions for which the posterior will be
% computed.
% RETURN mu : the mean of the posterior distribution.
% RETURN sigma : the variances of the posterior distributions.
%
% SEEALSO : collabCreate
%
% COPYRIGHT : Neil D. Lawrence, 2008

% COLLAB

  if nargout > 1
    diagK = kernDiagCompute(model.kern, X);
    varsig = zeros(size(X, 1), size(y, 2));
    sndMoment = zeros(size(X, 1), size(y, 2));
    
  end

  mu = zeros(size(X, 1), size(y, 2));
  % Compute kernel for new point.
  for i = 1:size(y, 2)
    ind = find(y(:, i));
    model.m = y(:, i);
    yind = y(ind, i);
    if model.M > 1
      model = collabInitS(model);
    end  
    model = collabUpdateKernels(model);
    KX_star = kernCompute(model.kern, model.X(ind, :), X);  
    if model.M > 1
      model = collabEstep(model);
      for m = 1:model.M
        mum{m} = KX_star'*model.invK{m}*yind;
        mu(:, i) = mu(:,i) + model.pi(m)*mum{m};
      end
    else
      mu(:, i) =KX_star'*model.invK*yind;
    end
    % Compute if variances required.
    if model.M > 1
      for m = 1:model.M
        Kinvk = model.invK{m}*KX_star;
        varsigm = diagK - sum(KX_star.*Kinvk, 1)';
        sndMoment(:, i) = sndMoment(:, i) + model.pi(m)*(mum{m}.*mum{m} + varsigm);
      end
      varsig(:, i) = sndMoment(:, i) - mu(:, i).*mu(:, i);
    end
  end
  % Compute if variances required.
  if nargout > 1 && model.M == 1
    Kinvk = model.invK*KX_star;
    varsig = diagK - sum(KX_star.*Kinvk, 1)';
    varsig = repmat(varsig, 1, size(y, 2));
  end
end
