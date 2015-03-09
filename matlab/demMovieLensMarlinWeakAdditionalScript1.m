function [] = demMovieLensMarlinWeakAdditionalScript1(substract_mean, partNo_v, latentDim_v,iters, inverted)
% DEMMOVIELENSMARLINWEAKADDITIONALSCRIPT1 Collaborative filtering on Marlin's weak partitions.
% FORMAT
% DESC Try collaborative filtering with the additional information
% on the Movielens data with Marlin's partitions for weak generalization.
% ARG  substract_mean : bool if substract the mean.
% ARG partNo :  vector with the partitions to compute results.
% ARG latentDim_v : vector with the latent dimensionalities to compute results.
% ARG iters : number of iterations.
% ARG inverted : if true, then learn users as examples and not items.
%
% SEEALSO collabCreate, collabOptimise
%
% COPYRIGHT : Raquel Urtasun, 2009

% COLLAB

randn('seed', 1e5);
rand('seed', 1e5);

experimentNo = 3;


%partNo_v = [1:5];
%latentDim_v = [5, 2:4, 6];


for i_latent=1:length(latentDim_v)
  q = latentDim_v(i_latent);
  for i_part=1:length(partNo_v)
    partNo = partNo_v(i_part);
    
    dataSetName = ['movielens_marlin_weak_',num2str(partNo)];
    
    disp(['Reading ... ',dataSetName]);
    
    [Y, lbls, Ytest] = collabLoadData(dataSetName);
    
    if (inverted)
      Y = Y';
      Ytest = Ytest';
    end
        
    numFilms = size(Y,1);
    numUsers = size(Y,2);
    meanFilms = zeros(numFilms,1);
    stdFilms = ones(numFilms,1);
    
    if (substract_mean)
      if 0
        % this substract the global mean
        % create the total vector
        s = nonzeros(Ytest);
        ratings = [nonzeros(Y); nonzeros(Ytest)];
        meanY = mean(ratings);
        stdY = std(ratings);
        %keyboard;
        index = find(Y);
        %Y(index) = Y(index) - meanY;
        %Y(index) = Y(index) / stdY;
      else
        for i=1:numFilms
          % compute the mean and standard deviation of each film
          ind = find(Y(i,:));
          mean_v = sum(Y(i,ind));
          mean_v = mean_v + sum(nonzeros(Ytest(i,:)));
          length_v = length(ind) + nnz(Ytest(i,:));
          mean_v = mean_v/length_v;
          std_v = (length(ind)*std(Y(i,ind)) + nnz(Ytest(i,:))*std(Ytest(i,:)))/length_v;
          %Y(i,ind) = Y(i,ind) - mean_v;
          %if (std_v>0) 
          %    Y(i,ind) = Y(i,ind)/std_v;
          %end
          meanFilms(i) = mean_v;
          stdFilms(i) = std_v;
        end
      end
      %keyboard;
    end
    
    
    
    
    type = {'cmpnd', {'tensor', 'rbf', 'rbfadditional'}, 'bias', 'white'};
    options.kern = kernCreate(q, type);
    options.kern.comp{1} = kernSetIndex(options.kern.comp{1}, 1, [1:q-1]);
    options.kern.comp{1} = kernSetIndex(options.kern.comp{1}, 2, [q]);
    
    %keyboard;
    options.kern.comp{1}.comp{2}.additional = lbls;
    model = collabCreateTensor(q, size(Y, 2), Y, options);
    % put the last component to be the index
    
    %keyboard;
    
    %options = collabOptions;
    %model = collabCreate(q, size(Y, 2), Y, options);
    
    if (substract_mean)
      if 0
        % this does the global mean
        model.mu = repmat(meanY,size(model.mu,1),1);
        model.sd = repmat(stdY,size(model.sd,1),1);
      else
        model.mu = meanFilms;
        model.sd = stdFilms;
      end
      
    end
    model.kern.comp{2}.variance = 0.11;
    model.kern.comp{3}.variance =  5; 
    options = collabOptimiseOptions;
    
    
    % set parameters
    options.momentum = 0.9;
    options.learnRate = 0.0001;
    options.paramMomentum = 0.9;
    options.paramLearnRate = 0.0001;
    options.numIters = iters;
    options.showLikelihood = false;
    
    capName = dataSetName;
    capName(1) = upper(capName(1));
    options.saveName = ['dem' capName num2str(experimentNo) '_'];
    
    model = collabOptimise(model, Y, options)
    
    % compute the test error
    disp('Computing test error');
    
    
    [L2_error,NMAE_error,NMAE_round_error] = computeTestErrorWeak(model,Y,Ytest)
    
    
    % Save the results.
    capName = dataSetName;
    capName(1) = upper(capName(1));
    
    saveResults = [capName,'_additional_inverted_',num2str(inverted),'_norm_',num2str(substract_mean),'_',num2str(q),'_',num2str(partNo),'_iters_',num2str(iters),'.mat'];
    disp(['Saving ... ',saveResults]);
    save(saveResults, 'model', 'L2_error','options','NMAE_error','NMAE_round_error');
  end
end



