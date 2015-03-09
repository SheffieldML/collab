function model = collabOptimise(model, Y, options)  
  
% COLLABOPTIMISE Optimise the collaborative filter.
% FORMAT 
% DESC optimises the collaborative filter model using stochastic gradient
% descent.
% ARG model : the model to optimize.
% ARG Y : the ratings in a numFilms x numUsers sparse matrix.
% ARG options : options for the optimization (see collabOptimiseOptions).
% RETURN model : the optimized model.
%
% SEEALSO : collabCreate, collabOptimiseOptions
%
% COPYRIGHT : Neil D. Lawrence, 2008

% COLLAB
  
  % Present data as a cell file to save space for v. large datasets.
  if iscell(model.y)
    numUsers = size(model.y, 1);
  else
    numUsers = size(model.y, 2);
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % In case we are recovering from an earlier run.
  if isfield(options, 'randState') && ~isempty(options.randState)
    rand('state', options.randState);
  end
  if isfield(options, 'randnState') && ~isempty(options.randnState)
    randn('state', options.randnState);
  end
  randState = rand('state');
  randnState = randn('state');
  order = randperm(numUsers);
  if isfield(options, 'order') && ~isempty(options.order)
    order = options.order;
  end
  param = kernExtractParam(model.kern);
  totalIters = options.numIters*numUsers;

  runIter = 0;
  if isfield(options, 'runIter') && ~isempty(options.runIter)
    runIter = options.runIter;
  end

  startUser = 1;
  startIter = 1;
  if isfield(options, 'startIter') && ~isempty(options.startIter)
    startIter = options.startIter;
  end
  if isfield(options, 'startUser') && ~isempty(options.startUser)
    startUser = options.startUser;
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  currIters = startUser-1+(startIter-1)*numUsers;
  oldIters = 0;
  tic
  for iters = startIter:options.numIters
    for user = startUser:numUsers
      
      currIters = currIters + 1;
      if iscell(model.y) && isempty(model.y{order(user), 1})
        % If there is an empty cell entry skip it.
        continue
      end
      model.currentOut = order(user);
      runIter = runIter + 1;
      if (iscell(model.y)) || nnz(model.y(:,order(user)))
        % there are ratings from this user
        model.m = collabComputeM(model);
        if model.M > 1 && iters == 1
          model = collabInitS(model);
        end
        model = collabUpdateKernels(model);
        if model.M > 1
          model = collabEstep(model, 100);
        end
        [g, g_param, g_diag] = collabLogLikeGradients(model);
        if options.optimiseParam
          model.changeParam = model.changeParam * options.paramMomentum + options.paramLearnRate*g_param;
          param = param + model.changeParam;
          
          model.kern = kernExpandParam(model.kern, param);
          if model.heteroNoise
            ind = find(g_diag);
            model.lndiagChange(ind) = model.lndiagChange(ind)*options.noiseMomentum + options.noiseLearnRate*g_diag(ind);
            model.diagvar(ind) = model.diagvar(ind).*exp(model.lndiagChange(ind));
          elseif model.M > 1 
            model.lnsigma2Change = model.lnsigma2Change*options.noiseMomentum + options.noiseLearnRate*g_diag;
            model.sigma2 = model.sigma2*exp(model.lnsigma2Change);
          end
        end
        model.change = model.change *options.momentum+options.learnRate*g;
        model.X = model.X + model.change;
      end
      if ~rem(runIter, options.saveEvery)
        try
          save([options.saveName num2str(currIters)], ...
               'model', 'iters', ...
               'user', 'randState', ...
               'randnState', 'runIter')
        catch
          warning([options.saveName num2str(currIters)  ' not saved.'])
          err = lasterror;
          fprintf(['Error message ''' err.message ''' trapped.\n'])
        end
      end
      if ~rem(runIter, options.showEvery)
        diffIters = currIters - oldIters;
        oldIters = currIters;
        iph = 3600*diffIters/toc;
        remain = (totalIters - currIters)/iph;
        fprintf('Current iter %d, total iter %d, time remain %2.4f hr.\n', currIters, totalIters, remain)
        tic
      end
    end
    if options.showLikelihood
      ll(iters) = collabLogLikelihood(model);
      fprintf('Log likelihood, %2.4f\n', ll(iters));
    end
    try
      save([options.saveName 'Iters' num2str(iters)], 'model', 'iters', ...
           'user', 'randnState', 'randState', 'runIter')
    catch 
      warning([options.saveName num2str(currIters)  ' not saved.'])
      err = lasterror;
      fprintf(['Error message ''' err.message ''' trapped.\n'])
    end
  end
end