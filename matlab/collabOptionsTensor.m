function options = collabOptionsTensor(approx);

% COLLABOPTIONSTENSOR Return default options for COLLAB model with a tensor
% FORMAT
% DESC returns the default options in a structure for a COLLAB model.
% RETURN options : structure containing the default options for the
% given approximation type.
%
% SEEALSO : collabCreateTensor
%
% COPYRIGHT : Raquel Urtasun, 2008

% COLLAB


  options.kern = {'cmpnd', {'tensor', 'rbf', 'rbf'}, 'bias', 'white'};
  options.numActive = 0;
  options.beta = [];

end
