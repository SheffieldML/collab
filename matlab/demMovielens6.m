% DEMMOVIELENS5 Try collaborative filtering on the large movielens data.
% where the strong movielens experiment

randn('seed', 1e5);
rand('seed', 1e5);

experimentNo = 3;

dataSetName = 'movielens_strong_1';
[Y, void, Ytest] = collabLoadData(dataSetName);

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
options.numIters = 1; % ??? put 10 back
options.showLikelihood = false;

capName = dataSetName;
capName(1) = upper(capName(1));
options.saveName = ['dem' capName num2str(experimentNo) '_'];

model = collabOptimise(model, Y, options)

% we have to divide the test data into two sets, train and test for the
% prediction. All but one are the train

  


disp('Computing test error');

% ????? this test is to be done

keyboard


[error_L2,error_NMAE,error_NMAE_round] = computeTestErrorStrong(model,Ytest);
% 
% val_L2 = 0;
% tot_L2 = 0;
% val_NMAE = 0;
% tot_NMAE = 0;
% val_NMAE_round = 0;
% tot_NMAE_round = 0;
% 
% for i = 1:size(Ytest, 2)       
%   ind = find(Ytest(:, i));
%   elim = find(ind>size(model.X, 1));
%   tind = ind;
%   tind(elim) = [];
%   
%   if (length(tind)==0)
%       continue;
%   end
%   % in the case of STRONG experiments, the user is new, so we have to
%   % compute the prediction using the test data
%   % compute random (LOO --> leave one out)
%   indexRand = randperm(length(tind));
%   Y_train_user = Ytest(:,i);
%   Y_test_user = Y_train_user(tind(indexRand(end)));
%   Y_train_user(tind(indexRand(end)),:) = 0;
%   [mu, varsig] = collabPosteriorMeanVar(model, Y_train_user, model.X(tind(indexRand(end)), :));
%   a = Y_test_user - mu; 
%   a = [a; Ytest(elim, i)];
%   val_L2 = val_L2 + a'*a;
%   tot_L2 = tot_L2 + length(a);
%   val_NMAE = val_NMAE + sum(abs(a));
%   tot_NMAE = tot_NMAE + length(a);
%   val_NMAE_round = val_NMAE_round + sum(abs(round(a)));
%   tot_NMAE_round = tot_NMAE_round + length(a);
% end
% error_L2 = sqrt(val_L2/tot_L2);
% error_NMAE = (val_NMAE/tot_NMAE)/1.6;
% error_NMAE_round = (val_NMAE_round/tot_NMAE_round)/1.6;


% Save the results.
capName = dataSetName;
capName(1) = upper(capName(1));
save(['dem' capName num2str(experimentNo) '.mat'], 'model', 'error_L2', 'error_NMAE', 'error_NMAE_round');



