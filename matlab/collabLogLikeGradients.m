function [g, g_param, g_noise] = collabLogLikeGradients(model)
  
% COLLABLOGLIKEGRADIENTS Gradient of the latent points.
% FORMAT 
% DESC computes the gradient of the latent points given ratings as a
% sparse matrix.
% ARG model : the model of the data.
% ARG y : the ratings for an individual.
%
% SEEALSO : collabLogLikelihood
%
% COPYRIGHT : Neil D. Lawrence, 2008, 2009
  
% COLLAB

  g_param = zeros(1, model.kern.nParams);
  fullInd = find(model.m);

  g = spalloc(size(model.X, 1), size(model.X, 2), length(fullInd)*model.q);
  if model.heteroNoise
    g_noise = spalloc(size(model.X, 1), 1, length(fullInd));
  elseif model.M > 1
    g_noise = 0;
  else
    g_noise = [];
  end
  g_param = zeros(1, model.kern.nParams);
  % For large inputs, split them into blocks of maximum 1000.
  maxBlock = ceil(length(fullInd)/ceil(length(fullInd)/1000));
  span = 0:maxBlock:length(fullInd);
  if rem(length(fullInd), maxBlock)
    span = [span length(fullInd)];
  end
  
  for block = 2:length(span)
    ind = fullInd(span(block-1)+1:span(block));
    m = model.m(ind, 1);

    X = model.X(ind, :);
    N = length(ind);
    if ~isfield(model, 'noise') || isempty(model.noise)
      if model.M > 1
        n = length(ind);
        % mixture model.
        gK = zeros(n);
        for i = 1:model.M
          invKy = model.invK{i}*m;
          gKm{i} =  0.5*(invKy*invKy'- model.invK{i});
          gK = gK + gKm{i};
        end
      else
        invKy = model.invK*m;
        gK = -model.invK + invKy*invKy';
        gK = gK * 0.5;
      end
      %%% Prepare to Compute Gradients with respect to X %%%
      gKX = kernGradX(model.kern, X, X);
      gKX = gKX*2;
      dgKX = kernDiagGradX(model.kern, X);
      for i = 1:length(ind)
        gKX(i, :, i) = dgKX(i, :);
      end
      gX = zeros(N, model.q);
      
      counter = 0;
      for i = 1:N
        counter = counter + 1;
        for j = 1:model.q
          gX(i, j) = gX(i, j) + gKX(:, j, i)'*gK(:, counter);
        end
      end
      g(ind, :) = gX;
      g_param = g_param + kernGradient(model.kern, X, gK);

      fhandle = str2func([model.noiseTransform 'Transform']);
      if model.heteroNoise 
        if model.M>1
          % Mixture model.
          for i = 1:model.M
            fact = fhandle(model.diagvar(ind), 'gradfact');
            g_noise(ind, :) = g_noise(ind, :) ...
                      + diag(gKm{i})./model.expectation.s{model.currentOut}(ind, i).*fact;
          end
        else
          g_noise(ind, :) = diag(gK);
          fact = fhandle(model.diagvar(ind), 'gradfact');
          g_noise(ind, :) = g_noise(ind, :).*fact;
        end
      elseif model.M > 1 
        % Mixture model.
        for i = 1:model.M
          fact = fhandle(model.sigma2, 'gradfact');
          g_noise = g_noise ...
                    + sum(diag(gKm{i})./model.expectation.s{model.currentOut}(ind, i))*fact;
        end
      end
      
    else
      muse = muse-1; % make muse start from zero.
      % Create an IVM model and update site parameters.
      options = ivmOptions;
      options.kern = model.kern;
      options.noise = model.noise;
      options.selectionCriterion = model.selectionCriterion;
      options.numActive = min(model.numActive, N);
      imodel = ivmCreate(model.q, 1, X, muse, options);
      imodel = ivmOptimiseIVM(imodel, options.display);
      gX = gplvmApproxLogLikeActiveSetGrad(imodel);
      gX = reshape(gX, length(imodel.I), size(imodel.X, 2));
      g(ind(imodel.I), :) = gX;
      g_param = g_param + ivmApproxLogLikeKernGrad(imodel);
    end
  end
  if nargout < 2
    g = [g(:)' g_param g_noise'];
  end
end
