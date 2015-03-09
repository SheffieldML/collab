function model = collabCreateTensor(q, d, Y, options);

% COLLABCREATE Create a COLLAB model with inducing varibles/pseudo-inputs.
% FORMAT
% DESC creates a collaborative filter structure with a latent space of q.
% ARG q : input data dimension.
% ARG d : the number of processes (i.e. output data dimension).
% ARG options : options structure as defined by collabOptions.m.
% RETURN model : model structure containing the GP collaborative filter.
%
% SEEALSO : collabOptions, modelCreate
%
% COPYRIGHT : Raquel Urtasun, 2008

% COLLAB


model.type = 'collab';

model.q = q;
model.d = d;
if size(Y, 1) == 1 && size(Y, 2) == 1
  model.N = Y;
else
  model.N = size(Y, 1);
end
%keyboard;
model.kern = kernCreate(q, options.kern);
%initParams = collabExtractParam(model);
model.X = randn(model.N, q)*0.001;
% add the labels
model.X(:,end) = [[1:model.N]'];
model.change = zeros(size(model.X));
%keyboard;
model.changeParam = zeros(1, model.kern.nParams);
model.mu = zeros(model.N, 1);
model.sd = ones(model.N, 1);
% This forces kernel computation.
%model = collabExpandParam(model, initParams);
