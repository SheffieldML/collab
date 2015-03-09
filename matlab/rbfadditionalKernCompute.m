function [k, sk, n2] = rbfadditionalKernCompute(kern, x, x2)

% RBFADDITIONALKERNCOMPUTE Compute the RBF kernel given the parameters and X.
% FORMAT
% DESC computes the kernel parameters for the radial basis function kernel
% given inputs associated with rows and columns.
% RETURN K : the kernel matrix computed at the given points.
% ARG kern : the kernel structure for which the matrix is computed.
% ARG i : the index of the input matrix associated with the rows of the kernel.
% ARG i2 : the index of the input matrix associated with the columns of the kernel.
%
% DESC computes the kernel matrix for the
%	radial basis function kernel given a design matrix of inputs.
% RETURN k : the kernel matrix computed at the given points.
% ARG kern : the kernel structure for which the matrix is computed.
% ARG i : the index of the input data matrix in the form of a design matrix.
%	
% SEEALSO : rbfadditionalKernParamInit, kernCompute, kernCreate,
% rbfadditionalKernDiagCompute
% 
% COPYRIGHT : Raquel Urtasun, 2009
%
% COPYRIGHT : Neil D. Lawrence, 2004, 2005, 2006

% COLLAB
  
if nargin < 3
  n2 = dist2(kern.additional(x,:), kern.additional(x,:));
  wi2 = (.5 .* kern.inverseWidth);
  sk = exp(-n2*wi2);
else
  n2 = dist2(kern.additional(x,:), kern.additional(x2,:));
  wi2 = (.5 .* kern.inverseWidth);
  sk = exp(-n2*wi2);
end
k = sk*kern.variance;