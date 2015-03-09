% DEMMOVIELENS5 Try collaborative filtering on the large movielens data.
% where the strong movielens experiment

randn('seed', 1e5);
rand('seed', 1e5);

experimentNo = 3;
substract_mean = 0;

dataSetName = 'eachmovie_weak_1';
[Y, void, Ytest] = collabLoadData(dataSetName);

numFilms = size(Y,1);
numUsers = size(Y,2);
meanFilms = zeros(numFilms,1);
stdFilms = ones(numFilms,1);
if (substract_mean)
    % do for each film independently
    for i=1:numFilms
        % compute the mean and standard deviation of each film
        ind = find(Y(i,:));
        mean_v = sum(Y(i,ind));
        mean_v = mean_v + sum(nonzeros(Ytest(i,:)));
        length_v = length(ind) + nnz(Ytest(i,:));
        mean_v = mean_v/length_v;
        std_v = (length(ind)*std(Y(i,ind)) + nnz(Ytest(i,:))*std(Ytest(i,:)))/length_v;
        Y(i,ind) = Y(i,ind) - mean_v;
        if (std_v>0) 
            Y(i,ind) = Y(i,ind)/std_v;
        end
        meanFilms(i) = mean_v;
        stdFilms(i) = std_v;
    end
end

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

%%% ?? add the model.mu and model.sd
model.mu = meanFilms;
model.sd = stdFilms;

model = collabOptimise(model, Y, options)

% we have to divide the test data into two sets, train and test for the
% prediction. All but one are the train

  


disp('Computing test error');

% ????? this test is to be done

keyboard

% ??? check if the mean is substracted...

[error_L2,error_NMAE,error_NMAE_round] = computeTestErrorWeak(model,Y,Ytest);

% val_L2 = 0;
% tot_L2 = 0;
% val_NMAE = 0;
% tot_NMAE = 0;
% val_round_NMAE = 0;
% tot_round_NMAE = 0;
% 
% for i = 1:size(Y, 2)       
%     ind = find(Ytest(:, i));
%     elim = find(ind>size(model.X, 1));
%     tind = ind;
%     tind(elim) = [];
%     [mu, varsig] = collabPosteriorMeanVar(model, Y(:, i), model.X(tind, :));
%     % normalize the values
% 
% 
%     mu = mu*model.sd(1);
%     mu = mu+model.mu(1);
%     a = Ytest(tind, i) - mu; 
%     a = [a; Ytest(elim, i)];
%     val_L2 = val_L2 + a'*a;
%     tot_L2 = tot_L2 + length(a);
%     val_NMAE = val_NMAE + sum(abs(a));
%     tot_NMAE = tot_NMAE + length(a);
%     val_round_NMAE = val_round_NMAE + sum(abs(round(a)));
%     tot_round_NMAE = tot_round_NMAE + length(a);
% end
% L2_error = sqrt(val_L2/tot_L2);
% NMAE_error = (val_NMAE/tot_NMAE)/1.6;
% NMAE_round_error = (val_round_NMAE/tot_round_NMAE)/1.6;


% Save the results.
capName = dataSetName;
capName(1) = upper(capName(1));
save(['dem' capName num2str(experimentNo) '.mat'], 'model', 'error_L2', 'error_NMAE', 'error_NMAE_round');



