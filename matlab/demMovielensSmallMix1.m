% DEMMOVIELENSSMALLMIX1 Try collaborative filtering on the small movielens data.

% COLLAB

randn('seed', 1e5);
rand('seed', 1e5);

experimentNo = 1;
error = 0;
for partition = 1:5
  dataSetName = ['movielensSmall' num2str(partition)];
  [Y, void, Ytest] = collabLoadData(dataSetName);
  q = 2;
  options = collabOptions;
  options.numComps = 2;
  options.kern = {'rbf', 'bias'};
  %/~
  %options.heteroNoise = true;
  %~/
  model = collabCreate(q, size(Y, 2), Y, options);
  %/~
  %model.diagvar = repmat(5.0, size(model.diagvar));
  %~/
  model.kern.comp{2}.variance = 0.11;
  model.sigma2 = 5;
  %model.kern.comp{3}.variance =  5; 
  options = collabOptimiseOptions;
  
  % set parameters
  options.momentum = 0.9;
  options.learnRate = 0.0001;
  options.paramMomentum = 0.9;
  options.paramLearnRate = 0.0001;
  options.noiseMomentum = 0.9;
  options.noiseLearnRate = 0.0001;
  options.numIters = 10;
  
  capName = dataSetName;
  capName(1) = upper(capName(1));
  options.saveName = ['dem' capName 'Mix' num2str(experimentNo) '_'];
  
  model = collabOptimise(model, Y, options);
  
  val = 0;
  tot = 0;
  for i = 1:size(Y, 2)       
    ind = find(Ytest(:, i));
    elim = find(ind>size(model.X, 1));
    tind = ind;
    tind(elim) = [];
    [mu, varsig] = collabPosteriorMeanVar(model, Y(:, i), model.X(tind, :));
    a = Ytest(tind, i) - mu; 
    a = [a; Ytest(elim, i)];
    val = val + a'*a;
    tot = tot + length(a);
  end
  error(partition) = sqrt(val/tot);
                                   % Save the results.
  capName = dataSetName;
  capName(1) = upper(capName(1));
  save(['dem' capName 'Mix_' num2str(experimentNo) '.mat'], 'model', 'error');
  
end
