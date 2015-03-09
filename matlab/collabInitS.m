function model = collabInitS(model)
  
% COLLABINITS Initialize the expectations of S for the collaborative filter.
% FORMAT
% DESC initilizes the expectations of S for the collaborative filter
% model.
% ARG model : the model structure for which expectations are being
% initialized.
% RETURN model : the model structure with the expectations initalized.
%
% SEEALSO : collabExpandParam, collabCreate
%
% COPYRIGHT : Neil D. Lawrence, 2009

% COLLAB

  
  ind = find(model.m);
  model.expectation.s{model.currentOut} = spalloc(model.N, model.M, length(ind));
  lognumer = repmat(log(model.pi), length(ind), 1) ...
      + randn(length(ind), model.M)*0.001;
  numer = exp(lognumer - repmat(max(lognumer, [], 2), 1, model.M));
  
  model.expectation.s{model.currentOut}(ind, :) = numer./repmat(sum(numer, 2), 1, model.M);
end