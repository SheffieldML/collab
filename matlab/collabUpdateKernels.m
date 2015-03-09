function model = collabUpdateKernels(model)

% COLLABUPDATEKERNELS Update the kernels that are needed.
% FORMAT
% DESC updates any representations of the kernel in the model
% structure, such as invK, logDetK or K.
% ARG model : the model structure for which kernels are being
% updated.
% RETURN model : the model structure with the kernels updated.
%
% SEEALSO : collabExpandParam, collabCreate
%
% COPYRIGHT : Neil D. Lawrence, 2009

% COLLAB
  
  ind = find(model.m);
  n = length(ind);
  model.K = kernCompute(model.kern, model.X(ind, :));
  s = model.expectation.s{model.currentOut};
  if model.M > 1
    % mixture model.
    for m = 1:model.M
      if model.heteroNoise
        Binv = diag(model.diagvar(ind)./s(ind, m));
      else
        Binv = diag(model.sigma2./s(ind, m));
      end
      Kadd = model.K + Binv;
      [model.invK{m}, U] = pdinv(Kadd);
      model.logDetK(m) = logdet(model.K, U);
    end
  elseif model.heteroNoise
    n = length(ind);
    [model.invK, U] = pdinv(model.K + spdiags(model.diagvar(ind, :), 0, n, n));
    model.logDetK = logdet(model.K, U);
  else
    [model.invK, U] = pdinv(model.K);
    model.logDetK = logdet(model.K, U);
  end

end