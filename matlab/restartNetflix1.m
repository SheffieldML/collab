% DEMNETFLIX1 Try collaborative filtering on the netflix data.

% COLLAB

randn('seed', 1e5);
rand('seed', 1e5);

experimentNo = 1;
dataSetName = 'netflix';

load /local/data/netFlixDataProbe.mat
load demNetflix1_1875881

options = collabOptimiseOptions;
options.numIters = 5;
options.showEvery = 400;
options.saveEvery = 20000;
options.currIters = 17*400;
options.randState = 1e5;
options.startIter = 1;
options.runIter = 1875882;
options.startUser = 1875882;
capName = dataSetName;
capName(1) = upper(capName(1));
options.saveName = ['dem' capName num2str(experimentNo) '_'];
options.showLikelihood = false;
model = collabOptimise(model, Y, options)
  
% val = 0;
% tot = 0;
% for i = 1:size(Y, 2)       
%   ind = find(Ytest(:, i));
%   elim = find(ind>size(model.X, 1));
%   tind = ind;
%   tind(elim) = [];
%   [mu, varsig] = collabPosteriorMeanVar(model, Y(:, i), model.X(tind, :));
%   a = Ytest(tind, i) - mu; 
%   a = [a; Ytest(elim, i)];
%   val = val + a'*a;
%   tot = tot + length(a);
% end
% error = sqrt(val/tot);

% Save the results.
capName = dataSetName;
capName(1) = upper(capName(1));
save(['dem' capName num2str(experimentNo) '.mat'], 'model');
