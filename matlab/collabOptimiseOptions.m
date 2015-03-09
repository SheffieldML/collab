function options = collabOptimiseOptions

% COLLABOPTIMISEOPTIONS returns default options for collaborative filter optimisation.
% FORMAT
% DESC returns default options for the optimization of the collaborative
% filter.
% RETURN options : the default options structure.
%
% SEEALSO : collabOptimise, collabCreate
%
% COPYRIGHT : Neil D. Lawrence, 2008

% COLLAB
  
  options.momentum = 0.5;
  options.learnRate = 0.0001;
  options.paramMomentum = 0.5;
  options.paramLearnRate = 0.0001;
  options.noiseMomentum = 0.5;
  options.noiseLearnRate = 0.0001;
  options.optimiseParam = true;
  options.showEvery = 100;
  options.saveEvery = 10000;
  options.showLikelihood = false;
  options.numIters = 50;
  options.saveName = 'save';
 end
