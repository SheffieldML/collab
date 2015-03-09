function [] = demMovieLens10MLetterWeakScript1(substract_mean, partNo_v, latentDim_v,iters, inverted)

% DEMMOVIELENS10MLETTERWEAKSCRIPT1 Try collaborative filtering on the 10M movielens data set.
% FORMAT
% DESC run a script on the 10M movielens data.
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

partLetter_v = 'ab';

%partNo_v = [1:5];
%latentDim_v = [5, 2:4, 6];


for i_latent=1:length(latentDim_v)
  q = latentDim_v(i_latent);
  for i_part=1:length(partNo_v)
    partLetter = partLetter_v(partNo_v(i_part));
    
    dataSetName = ['movielens_10M_',partLetter];
    
    disp(['Reading ... ',dataSetName]);
    
    [Y, void, Ytest] = collabLoadData(dataSetName);
    
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
    
    options = collabOptions;
    model = collabCreate(q, size(Y, 2), Y, options);
    % keyboard;
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
    
    capName = dataSetName;
    capName(1) = upper(capName(1));
    
    saveResults = [capName,'inverted_',num2str(inverted),'_norm_',num2str(substract_mean),'_',num2str(q),'_',partLetter,'_iters_',num2str(iters),'.mat'];
    disp(['Saving ... ',saveResults]);
    
    save(saveResults, 'model', 'options');
    
    
    % compute the test error
    disp('Computing test error');
    
    
    [L2_error,NMAE_error,NMAE_round_error] = computeTestErrorWeakCell(model,Y,Ytest)
    
    
    % Save the results.
    disp(['Saving ... ',saveResults]);
    save(saveResults, 'model', 'L2_error','options','NMAE_error','NMAE_round_error');
  end
end



