function[] = RecomputeResults(perc_train, substract_mean, partNo_v, latentDim_v)
% RECOMPUTERESULTS Try collaborative filtering on the large movielens data.
%
  % RecomputeResults(perc_train, substract_mean, partNo_v, latentDim_v)
%
% perc_train -> percentage of training
% substract_mean --> bool if substract the mean
% partNo_v --> vector with the partitions to compute results
% latentDim_v --> vector with the latent dimensionalities to compute results

randn('seed', 1e5);
rand('seed', 1e5);

experimentNo = 3;


%partNo_v = [1:5];
%latentDim_v = [5, 2:4, 6];


for i_latent=1:length(latentDim_v)
    q = latentDim_v(i_latent);
    for i_part=1:length(partNo_v)
        partNo = partNo_v(i_part);

        dataSetName = ['movielens_',num2str(perc_train),'_',num2str(partNo)];
        
        disp(['Reading ... ',dataSetName]);
        
        [Y, void, Ytest] = collabLoadData(dataSetName);
        
        if (substract_mean)
            % create the total vector
            s = nonzeros(Ytest);
            ratings = [nonzeros(Y); nonzeros(Ytest)];
            meanY = mean(ratings);
            stdY = std(ratings);
            %keyboard;
            index = find(Y);
            Y(index) = Y(index) - meanY;
            Y(index) = Y(index) / stdY;
            %keyboard;
        end;

% load the model
% Save the results.
        capName = dataSetName;
        capName(1) = upper(capName(1));
        
        loadResults = [capName,'_norm_',num2str(substract_mean),'_',num2str(q),'_',num2str(partNo),'.mat'];
        disp(['Loading ... ',loadResults]);
load(loadResults);
%, 'model', 'L2_error','options','NMAE_error','NMAE_round_error');
L2_error_before = L2_error;
NMAE_error_before = NMAE_error;
NMAE_round_error_before = NMAE_round_error;


        val_L2 = 0;
        tot_L2 = 0;
        val_NMAE = 0;
        tot_NMAE = 0;
        val_round_NMAE = 0;
        tot_round_NMAE = 0;
        val_round_NMAE_2 = 0;
        tot_round_NMAE_2 = 0;

disp('Computing results');
ErrorValues = [];
ErrorValues_round = [];
        for i = 1:size(Y, 2)       
          ind = find(Ytest(:, i));
          elim = find(ind>size(model.X, 1));
          tind = ind;
          tind(elim) = [];
          [mu, varsig] = collabPosteriorMeanVar(model, Y(:, i), model.X(tind, :));
          % normalize the values
	           

mu = mu*model.sd(1);
mu = mu+model.mu(1);
          a = Ytest(tind, i) - mu; 
          a = [a; Ytest(elim, i)];
a_round = Ytest(tind, i) - round(mu); 
          a_round = [a_round; Ytest(elim, i)];
          val_L2 = val_L2 + a'*a;
          tot_L2 = tot_L2 + length(a);
          val_NMAE = val_NMAE + sum(abs(a));
          tot_NMAE = tot_NMAE + length(a);
          val_round_NMAE = val_round_NMAE + sum(abs(round(a)));
          tot_round_NMAE = tot_round_NMAE + length(a);
          val_round_NMAE_2 = val_round_NMAE_2 + sum(abs(a_round));
          tot_round_NMAE_2 = tot_round_NMAE_2 + length(a_round);
          
% ??? this doesn't work yet
	    %keyboard;
ErrorValues = [ErrorValues; full(abs(a))];
ErrorValues_round = [ErrorValues_round; full(abs(a))];

        end
        L2_error = sqrt(val_L2/tot_L2);
        NMAE_error = (val_NMAE/tot_NMAE)/1.6;
        NMAE_round_error = (val_round_NMAE/tot_round_NMAE)/1.6;
        NMAE_round_error_2 = (val_round_NMAE_2/tot_round_NMAE_2)/1.6;


[L2_error L2_error_before]
[NMAE_error NMAE_error_before]
[NMAE_round_error NMAE_round_error_before]
NMAE_round_error_2
mean(ErrorValues)
std(ErrorValues)
%keyboard;

        % Save the results.
%        capName = dataSetName;
%        capName(1) = upper(capName(1));
        
%        saveResults = [capName,'_norm_',num2str(substract_mean),'_',num2str(q),'_',num2str(partNo),'.mat'];
%        disp(['Saving ... ',saveResults]);
        %save(saveResults, 'model', 'L2_error','options','NMAE_error','NMAE_round_error');
    end
end

