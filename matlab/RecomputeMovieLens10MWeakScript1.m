function [] = RecomputeMovieLens10MWeakScript1(substract_mean, partNo_v, latentDim_v,iters, inverted)
% RECOMPUTEMOVIELENS10MWEAKSCRIPT1 Recompute the test error for the 10M Movielens database
% where the weak movielens experiment
%
  % RecomputeMovieLens10MWeakScript1(substract_mean, partNo_v,
  % latentDim_v,iters, inverted)
%
% substract_mean --> bool if substract the mean
% partNo_v --> vector with the partitions to compute results
% latentDim_v --> vector with the latent dimensionalities to compute results
% iters --> number of iterations
% if inverted = true, then learn users as examples and not movies

randn('seed', 1e5);
rand('seed', 1e5);

experimentNo = 3;


for i_latent=1:length(latentDim_v)
    q = latentDim_v(i_latent);
    for i_part=1:length(partNo_v)
        partNo = partNo_v(i_part);
        
        dataSetName = ['movielens_10M_',num2str(partNo)];
        
        disp(['Reading ... ',dataSetName]);
        
        [Y, void, Ytest] = collabLoadData(dataSetName);

        if (inverted)
            Y = Y';
            Ytest = Y';
        end
        
	numFilms = size(Y,1);
        numUsers = size(Y,2);
        meanFilms = zeros(numFilms,1);
        stdFilms = ones(numFilms,1);
        

% Save the results.
        capName = dataSetName;
        capName(1) = upper(capName(1));
        
        loadResults = [capName,'inverted_',num2str(inverted),'_norm_',num2str(substract_mean),'_',num2str(q),'_',num2str(partNo),'_iters_',num2str(iters),'.mat'];
        disp(['Loading ... ',saveResults]);
load(saveResults);, 'model', 'L2_error','options','NMAE_error','NMAE_round_error');
   
        
	% compute the test error
	  disp('Computing test error');


[L2_error,NMAE_error,NMAE_round_error] = computeTestErrorWeakCell(model,Y,Ytest)


        % Save the results.
        capName = dataSetName;
        capName(1) = upper(capName(1));
        
        saveResults = [capName,'inverted_',num2str(inverted),'_norm_',num2str(substract_mean),'_',num2str(q),'_',num2str(partNo),'_iters_',num2str(iters),'.mat'];
        disp(['Saving ... ',saveResults]);
        save(saveResults, 'model', 'L2_error','options','NMAE_error','NMAE_round_error');
    end
end



