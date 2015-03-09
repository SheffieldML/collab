% DEMMOVIELENS2 Try collaborative filtering on the large movielens data.

% COLLAB

randn('seed', 1e5);
rand('seed', 1e5);

experimentNo = 2;

dataSetName = 'movielens';
[Y, void, Ytest] = collabLoadData(dataSetName);

q = 4;
options = collabOptions;
model = collabCreate(q, size(Y, 2), Y, options);
model.kern.comp{2}.variance = 0.11;
model.kern.comp{3}.variance =  5; 
options = collabOptimiseOptions;
options.numIters = 10;
options.showLikelihood = false;
model = collabOptimise(model, Y, options)
  
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
error = sqrt(val/tot);

% Save the results.
capName = dataSetName;
capName(1) = upper(capName(1));
save(['dem' capName num2str(experimentNo) '.mat'], 'model', 'error');
