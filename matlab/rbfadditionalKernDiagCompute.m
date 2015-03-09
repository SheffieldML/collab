function k = rbfadditionalKernDiagCompute(kern, x)

% RBFADDITIONALKERNDIAGCOMPUTE Compute diagonal of RBF side information kernel.
% FORMAT
% DESC computes the diagonal of the kernel
%	matrix for the radial basis function kernel given a design matrix of
%	inputs.
% RETURN k : a vector containing the diagonal of the kernel matrix computed
%	   at the given points.
% ARG kern : the kernel structure for which the matrix is computed.
% ARG i - input data indices.
%	
% SEEALSO : rbfadditionalKernParamInit, kernDiagCompute, kernCreate,
% rbfaddtiionalKernCompute
%
% COPYRIGHT : Raquel Urtasun 2009
%
% COPYRIGHT : Neil D. Lawrence, 2004, 2005, 2006

% COLLAB

k = repmat(kern.variance, size(x, 1), 1);
