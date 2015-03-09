function kern = rbfadditionalKernParamInit(kern)

% RBFADDITIONALKERNPARAMINIT RBF kernel with side information.
% FORMAT
% The radial basis function kernel (RBF) is sometimes also known as the
% squared exponential kernel. It is a very smooth non-linear kernel and is a
% popular choice for generic use.
%	
%	k(x_i, x_j) = sigma2 * exp(-gamma/2 *(additional(x_i) - additional(x_j))'*(additional(x_i) - additional(x_j)))
%	
% The parameters are sigma2, the process variance (kern.variance) and gamma,
% the inverse width (kern.inverseWidth). The inverse width controls how wide
% the basis functions are, the larger gamma, the smaller the basis functions
% are.
%r
% DESC computes the RBF kernel with the side information for
% collaborative filtering.
% RETURN kern : the kernel structure with the default parameters placed in.
% ARG kern : the kernel structure which requires initialisation.
%	
% SEEALSO : rbfkernParamInit, kernCreate, kernParamInit
%
% COPYRIGHT : Raquel Urtasun, 2009
%
% COPYRIGHT : Neil D. Lawrence, 2004, 2005, 2006

% COLLAB

kern.inverseWidth = 1;
kern.variance = 1;
kern.nParams = 2;

% Constrains parameters positive for optimisation.
kern.transforms.index = [1 2];
kern.transforms.type = optimiDefaultConstraint('positive');
kern.isStationary = true;

% it requires a field with the additional information
