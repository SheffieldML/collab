function options = collabOptions(approx);

% COLLABOPTIONS Return default options for COLLAB model.
% FORMAT
% DESC returns the default options in a structure for a COLLAB model.
% RETURN options : structure containing the default options for the
% given approximation type.
%
% SEEALSO : collabCreate
%
% COPYRIGHT : Neil D. Lawrence, 2008

% COLLAB

  options.kern = {'rbf', 'bias', 'white'};
  options.numActive = 0;
  options.beta = [];
  options.heteroNoise = false;
  options.numComps = 1;
  
end
