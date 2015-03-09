% DEMMOVIELENS4 Try collaborative filtering on the large movielens data.
% try different kernels

% COLLAB

randn('seed', 1e5);
rand('seed', 1e5);

experimentNo = 3;

dataSetName = 'movielens';
[Y, lbls, Ytest] = collabLoadData(dataSetName);

% get the extra data in the labels

q = 5;
q = q+1;
options = collabOptionsTensor;


%%%%% as in gpReversible dynamics
type = {'cmpnd', {'tensor', 'rbf', 'rbfadditional'}, 'bias', 'white'};
options.kern = kernCreate(q, type);
%keyboard;
options.kern.comp{1} = kernSetIndex(options.kern.comp{1}, 1, [1:q-1]);
options.kern.comp{1} = kernSetIndex(options.kern.comp{1}, 2, [q]);
options.kern.comp{1}.comp{2}.additional = lbls;
%options.kern.comp{1}.comp{1}.inverseWidth = 0.2;
%options.kern.comp{1}.comp{1}.variance = 0.001;
%options.kern.comp{1}.comp{2}.variance = 2/pi;
%options.kern.comp{1}.comp{2}.weightVariance = 1000;
%options.kern.comp{1}.comp{2}.biasVariance = eps;


% as previously
%options.kern = {'cmpnd', {'tensor', 'rbf', 'rbfadditional'}, 'bias', 'white'};
%options.kern.comp{1}.comp{1}.index = 1:q-1;
%options.kern.comp{1}.comp{2}.index = q; 
%options.kern.comp{1}.comp{2}.additional = lbls;
%keyboard;
model = collabCreateTensor(q, size(Y, 2), size(Y, 1), options);
% put the last component to be the index
%keyboard
model.kern.comp{2}.variance = 0.11;
model.kern.comp{3}.variance =  5;

%keyboard;
options = collabOptimiseOptions;

% set parameters
options.momentum = 0.9;
options.learnRate = 0.0001;
options.paramMomentum = 0.9;
options.paramLearnRate = 0.0001;
options.numIters = 1;
options.showLikelihood = false;

capName = dataSetName;
capName(1) = upper(capName(1));
options.saveName = ['dem' capName num2str(experimentNo) '_'];

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

