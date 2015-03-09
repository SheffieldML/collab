function gX = rbfadditionalKernGradX(kern, X, X2)

% RBFADDITIONALKERNGRADX Gradient of RBF kernel with respect to input locations.
% FORMAT
% DESC computes the gradident of the radial basis function
% kernel with respect to the input positions where both the row
% positions and column positions are provided separately.
% ARG kern : kernel structure for which gradients are being
% computed.
% ARG i1 : row locations against which gradients are being computed.
% ARG i2 : column locations against which gradients are being computed.
% RETURN g : the returned gradients. The gradients are returned in
% a matrix which is numData2 x numInputs x numData1. Where numData1 is
% the number of data points in I1, numData2 is the number of data
% points in I2 and numInputs is the number of input
% dimensions in X.
%
% SEEALSO : rbfKernParamInit, kernGradX, rbfKernDiagGradX
%
% COPYRIGHT : Neil D. Lawrence, 2004, 2005, 2006
% 
% COPYRIGHT : Raquel Urtasun, 2009
  
% COLLAB

gX = zeros(size(X2, 1), size(X2, 2), size(X, 1));
