function model = collabOptimiseEachMovie(model, Y, options)  
  
% COLLABOPTIMISEEACHMOVIE Optimise the collaborative filter for the EachMovie dataset
% FORMAT 
% DESC optimises the collaborative filter model using stochastic gradient
% descent.
% ARG model : the model to optimize.
% ARG Y : the ratings in a numFilms x numUsers sparse matrix.
% ARG options : options for the optimization.
% RETURN model : the optimized model.
%
% SEEALSO : collabCreate
%
% COPYRIGHT : Raquel Urtasun, 2008
  
  if iscell(Y)
    numUsers = size(Y, 1);
  else
    numUsers = size(Y, 2);
  end
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

  % compute the number of users in each film
  for i=1:size(Y,1)
    numUsersPerMovie(i) = length(find(Y(i,:)));
  end

				 % learnRateAdjust = size(Y,2)./numUsersPerMovie;

			      
  currIters = startUser-1+(startIter-1)*numUsers;
  oldIters = 0;
  tic
  for iters = startIter:options.numIters
    for user = startUser:numUsers
      currIters = currIters + 1;
      if iscell(Y) && isempty(Y{order(user), 1})
        continue
      end
      runIter = runIter + 1;
      if (iscell(Y)) ||   (nnz(Y(:,order(user))))
          if options.optimiseParam
            if iscell(Y)
                [g, g_param] = collabLogLikeGradients(model, Y(order(user), :));
            else
                [g, g_param] = collabLogLikeGradients(model, Y(:, order(user)));
            end

      % ?? change the optimization to do something more customized
       
            model.changeParam = model.changeParam * options.paramMomentum + options.paramLearnRate*g_param;
            param = param + model.changeParam;

            model.kern = kernExpandParam(model.kern, param);
          else
            if iscell(Y)
                    g = collabLogLikeGradients(model, Y(order(user), :));
            else 
                g = collabLogLikeGradients(model, Y(:, order(user)));
            end
          end
% ???? change the optimization to do something more customized
	      XlearnRateVector = max([options.learnRate*learnRateAdjust ones(length(order(user)),1)],2);
				 model.change = model.change *options.momentum+g.*repmat(XlearnRateVector,1,q);
          %model.change = model.change *options.momentum+options.learnRate*g;
          model.X = model.X + model.change;
	  % a bit hacky
				
      end
      if ~rem(runIter, options.saveEvery)
        try
          save([options.saveName num2str(currIters)], 'model', 'iters', ...
               'user', 'randState', 'randnState', 'runIter')
        catch
          warning([options.saveName num2str(currIters)  ' not saved.'])
          err = lasterror;
          fprintf(['Error message ''' err.message ''' trapped.\n'])
        end
      end
      if ~rem(runIter, options.showEvery)
        diffIters = currIters - oldIters;
        oldIters = currIters;
        iph = 3600*diffIters/(toc);
        remain = (totalIters - currIters)/iph;
        fprintf('Current iter %d, total iter %d, time remain %2.4f hr.\n', currIters, totalIters, remain)
        tic
      end
    end
    if options.showLikelihood
      ll(iters) = collabLogLikelihood(model, Y);
      
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
