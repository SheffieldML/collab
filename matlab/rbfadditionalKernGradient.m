function g = rbfadditionalKernGradient(kern, x, varargin)

% RBFADDITIONALKERNGRADIENT Gradient of RBF with side information kernel's parameters.
% FORMAT
% DESC computes the gradient of functions with respect to the
% radial basis function with side information
% kernel's parameters. As well as the kernel structure and the
% input positions, the user provides a matrix PARTIAL which gives
% the partial derivatives of the function with respect to the
% relevant elements of the kernel matrix. 
% ARG kern : the kernel structure for which the gradients are being
% computed.
% ARG i : the input indices for which the gradients are being
% computed. 
% ARG partial : matrix of partial derivatives of the function of
% interest with respect to the kernel matrix. The argument takes
% the form of a square matrix of dimension  numData, where numData is
% the number of rows in I.
% RETURN g : gradients of the function of interest with respect to
% the kernel parameters. The ordering of the vector should match
% that provided by the function kernExtractParam.
%
% FORMAT
% DESC computes the derivatives as above, but input locations are
% now provided in two vectors associated with rows and columns of
% the kernel matrix. 
% ARG kern : the kernel structure for which the gradients are being
% computed.
% ARG i1 : the input indices associated with the rows of the
% kernel matrix.
% ARG i2 : the input indices associated with the columns of the
% kernel matrix.
% ARG partial : matrix of partial derivatives of the function of
% interest with respect to the kernel matrix. The matrix should
% have the same number of rows as I1 and the same number of columns
% as I2 has rows.
% RETURN g : gradients of the function of interest with respect to
% the kernel parameters.
%
% SEEALSO rbfadditionalKernParamInit, kernGradient, rbfadditionalKernDiagGradient, kernGradX
%
% COPYRIGHT : Neil D. Lawrence, 2004, 2005, 2006, 2009
%
% COPYRIGHT : Raquel Urtasun, 2009

% COLLAB
  
% The last argument is covGrad
if nargin < 4
  [k, sk, dist2xx] = rbfadditionalKernCompute(kern, x);
else
  [k, sk, dist2xx] = rbfadditionalKernCompute(kern, x, varargin{1});
end
g(1) = - .5*sum(sum(varargin{end}.*k.*dist2xx));
g(2) =  sum(sum(varargin{end}.*sk));
