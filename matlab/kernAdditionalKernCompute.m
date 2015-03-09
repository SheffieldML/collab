function [k, n2] = kernAdditionalKernCompute(kern, x, x2)

% KERNADDITIONALKERNCOMPUTE Compute the RBF kernel given the parameters and X.
%
%	Description:
%
%	K = KERNADDITIONALKERNCOMPUTE(KERN, X, X2) computes the kernel parameters for
%	the radial basis function kernel given inputs associated with rows
%	and columns.
%	 Returns:
%	  K - the kernel matrix computed at the given points.
%	 Arguments:
%	  KERN - the kernel structure for which the matrix is computed.
%	  X - the index of the input matrix associated with the rows of the kernel.
%	  X2 - the index of the input matrix associated with the columns of the kernel.
%
%	K = KERNADDITIONALKERNCOMPUTE(KERN, X) computes the kernel matrix for the
%	radial basis function kernel given a design matrix of inputs.
%	 Returns:
%	  K - the kernel matrix computed at the given points.
%	 Arguments:
%	  KERN - the kernel structure for which the matrix is computed.
%	  X - the index of the input data matrix in the form of a design matrix.
%	
%
%	See also
%	RBFADDITIONALKERNPARAMINIT, KERNCOMPUTE, KERNCREATE, RBFADDITIONALKERNDIAGCOMPUTE


%	Copyright (c) 2009 Raquel Urtasun
% 	rbfKernCompute.m version 1.0


if nargin < 3
n2 = dist2(additional(x,:), additional(x,:));
  wi2 = (.5 .* kern.inverseWidth);
  k = kern.variance*exp(-n2*wi2);
else
  n2 = dist2(additional(x,:), additional(x2,:));
  wi2 = (.5 .* kern.inverseWidth);
  k = kern.variance*exp(-n2*wi2);
end
