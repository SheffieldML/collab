function ll = collabLogLikelihood(model)

% COLLABLOGLIKELIHOOD Compute the log likelihood of a COLLAB.
% FORMAT
% DESC computes the log likelihood of a data set given a COLLAB model.
% ARG model : the COLLAB model for which log likelihood is to be
% computed.
% RETURN ll : the log likelihood of the data in the COLLAB model.
%
% SEEALSO : collabCreate, collabLogLikeGradients, modelLogLikelihood
%
% COPYRIGHT : Neil D. Lawrence, 2009

% COLLAB

  ll = 0;
  
  if iscell(model.y)
    total = size(model.y, 1);
  else
    total = size(model.y, 2);
  end

  for i = 1:total
    model.currentOut = i;
    model.m = collabComputeM(model, i);
    if model.M > 1
      model = collabInitS(model);
    end
    model = collabUpdateKernels(model);
    if model.M > 1
      model = collabEstep(model);
    end
    %/~
    % This code was for splitting large data into blocks.
    %   maxBlock = ceil(length(fullInd)/ceil(length(fullInd)/1000));
    %   span = 0:maxBlock:length(fullInd);
    %   if rem(length(fullInd), maxBlock)
    %     span = [span length(fullInd)];
    %   end
    
    %   for block = 2:length(span)
    %     ind = fullInd(span(block-1)+1:span(block));
    %     if iscell(y)
    %       yuse = double(y{1, 2}(span(block-1)+1:span(block)));
    %     else
    %       yuse = y(ind, 1);
    %     end
    
    %     N = length(ind);
    %~/
    if ~isfield(model, 'noise') || isempty(model.noise)
      
      ind = find(model.m);
      muse = model.m(ind);
      if model.M> 1      
        for i = 1:model.M
          ll = ll - 0.5*model.logDetK(i) - 0.5*muse'*model.invK{i}*muse;
        end
      else
        ll = ll - 0.5*model.logDetK - 0.5*muse'*model.invK*muse;
      end
    end
  end
end
