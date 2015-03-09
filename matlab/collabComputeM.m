function m = collabComputeM(model)
  
% COLLABCOMPUTEM Computes target values inside model.
% FORMAT
% DESC takes in a model and an output user and computes the target values
% for that user.
% ARG model : the model for which the values of m are to be computed. The
% field currentOut should be set to which user is to be taken from the data.
%
% SEEALSO : collabCreate
%
% COPYRIGHT : Neil D. Lawrence, 2009

% COLLAB
  
  if iscell(model.y)
    m = spalloc(model.N, 1, length(model.y{model.currentOut, 1}));
    m(model.y{model.currentOut, 1}, :) = double(y{model.currentOut, 2});
  else
    m = model.y(:, model.currentOut);
  end
  ind = find(m);
  m(ind) = m(ind) - model.mu(ind);
  m(ind) = m(ind)./model.sd(ind);

end