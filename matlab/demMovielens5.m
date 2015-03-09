% DEMMOVIELENS5 Try collaborative filtering on the large movielens data.
% where now the latent space is in the users, not the films

randn('seed', 1e5);
rand('seed', 1e5);

experimentNo = 3;

??? to be done

dataSetName = 'movielens';
[Y, void, Ytest] = collabLoadData(dataSetName);

% learn latent space of each user
Y = Y';
Ytest = Ytest';

q = 5;
options = collabOptions;
model = collabCreate(q, size(Y, 2), Y, options);
model.kern.comp{2}.variance = 0.11;
model.kern.comp{3}.variance =  5; 
options = collabOptimiseOptions;

% set parameters
options.momentum = 0.9;
options.learnRate = 0.0001;
options.paramMomentum = 0.9;
options.paramLearnRate = 0.0001;
options.numIters = 10; 
options.showLikelihood = false;

capName = dataSetName;
capName(1) = upper(capName(1));
options.saveName = ['dem' capName num2str(experimentNo) '_'];

model = collabOptimise(model, Y, options)

% we have to divide the test data into two sets, train and test for the
% prediction. All but one are the train

  
val_L2 = 0;
tot_L2 = 0;
val_NMAE = 0;
tot_NMAE = 0;
val_NMAE_round = 0;
tot_NMAE_round = 0;

disp('Computing test error');


for i = 1:size(Y, 2)       
  ind = find(Ytest(:, i));
  elim = find(ind>size(model.X, 1));
  tind = ind;
  tind(elim) = [];
  [mu, varsig] = collabPosteriorMeanVar(model, Y(:, i), model.X(tind, :));
  a = Ytest(tind, i) - mu; 
  a = [a; Ytest(elim, i)];
  val_L2 = val_L2 + a'*a;
  tot_L2 = tot_L2 + length(a);
  val_NMAE = val_NMAE + sum(abs(a));
  tot_NMAE = tot_NMAE + length(a);
  val_NMAE_round = val_NMAE_round + sum(abs(round(a)));
  tot_NMAE_round = tot_NMAE_round + length(a);
end
error_L2 = sqrt(val_L2/tot_L2);
error_NMAE = (val_NMAE/tot_NMAE)/1.6;
error_NMAE_round = (val_NMAE_round/tot_NMAE_round)/1.6;


% Save the results.
capName = dataSetName;
capName(1) = upper(capName(1));
save(['dem' capName num2str(experimentNo) '.mat'], 'model', 'error_L2', 'error_NMAE', 'error_NMAE_round');



