function [s, numer] = collabComputeS(model)

% COLLABCOMPUTES Compute the responsibilities for the mixture model.
% FORMAT
% DESC computes the responsibilities for the mixture model.
% ARG model : the model for which the responsibilities are required.
% RETURN s : the responsibilities associated with the components and the
% data.
% RETURN numer : the numerator when the expectations are computed.
%
% SEEALSO : collabCreate, collabEstep
%
% COPYRIGHT : Neil D. Lawrence, 2009

% COLLAB
  
  % update the expected value of the components.
    ind = find(model.m);
    lognumer = zeros(size(ind, 1), model.M);
    for m = 1:model.M
      yhat = (model.m(ind) - model.expectation.f{m});
      y2 = yhat.*yhat + model.expectation.varf{m};
      if model.heteroNoise
        % Log of numerator of s.
        lognumer(:, m) = log(model.pi(m)) + (-.5*y2./model.diagvar(ind));
      else
        % Log of numerator of s.
        lognumer(:, m) = log(model.pi(m)) + (-.5*y2/model.sigma2);
      end
      % subtract maximum value from log numerator to keep numerically stable.
      numer = exp(lognumer - repmat(max(lognumer, [], 2), 1, model.M)); 
      numer = numer + 1e-6;
      s = spalloc(model.N, model.M, length(ind)*model.M);
      % normalize to obtain the expectations.
      s(ind, :) = (numer)./repmat(sum(numer, 2), 1, model.M);
    end
  end
