function[] = demMovielens3Script(perc_train, substract_mean, partNo_v, latentDim_v,iters)
% DEMMOVIELENS3Script Try collaborative filtering on the large movielens data.
%
  % demMovielens3script(perc_train, substract_mean, partNo_v, latentDim_v)
%
% perc_train -> percentage of training
% substract_mean --> bool if substract the mean
% partNo_v --> vector with the partitions to compute results
% latentDim_v --> vector with the latent dimensionalities to compute results
% iters --> number of iterations

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

        options = collabOptions;
        model = collabCreate(q, size(Y, 2), Y, options);
        % keyboard;
        if (substract_mean)
	    model.mu = repmat(meanY,size(model.mu,1),1);
            model.sd = repmat(stdY,size(model.sd,1),1);
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

        val_L2 = 0;
        tot_L2 = 0;
        val_NMAE = 0;
        tot_NMAE = 0;
        val_round_NMAE = 0;
        tot_round_NMAE = 0;

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
          val_L2 = val_L2 + a'*a;
          tot_L2 = tot_L2 + length(a);
          val_NMAE = val_NMAE + sum(abs(a));
          tot_NMAE = tot_NMAE + length(a);
          val_round_NMAE = val_round_NMAE + sum(abs(round(a)));
          tot_round_NMAE = tot_round_NMAE + length(a);
        end
        L2_error = sqrt(val_L2/tot_L2);
        NMAE_error = (val_NMAE/tot_NMAE)/1.6;
        NMAE_round_error = (val_round_NMAE/tot_round_NMAE)/1.6;

%         % compute NMAE
%         val = 0;
%         tot = 0;
%         for i = 1:size(Y, 2)       
%           ind = find(Ytest(:, i));
%           elim = find(ind>size(model.X, 1));
%           tind = ind;
%           tind(elim) = [];
%           [mu, varsig] = collabPosteriorMeanVar(model, Y(:, i), model.X(tind, :));
%           % un-normalize the values
%           mu = mu*model.sd;
%           mu = mu+model.mu;
%           a = Ytest(tind, i) - mu; 
%           a = [a; Ytest(elim, i)];
%           val = val + sum(abs(a));
%           tot = tot + length(a);
%         end
%         NMAE_error = (val/tot)/1.6;
% 
%         % round NMAE
%         val = 0;
%         tot = 0;
%         for i = 1:size(Y, 2)       
%           ind = find(Ytest(:, i));
%           elim = find(ind>size(model.X, 1));
%           tind = ind;
%           tind(elim) = [];
%           [mu, varsig] = collabPosteriorMeanVar(model, Y(:, i), model.X(tind, :));
%           % un-normalize the values
%           mu = mu*model.sd;
%           mu = mu+model.mu;
%           a = Ytest(tind, i) - mu; 
%           a = [a; Ytest(elim, i)];
%           val = val + sum(abs(round(a)));
%           tot = tot + length(a);
%         end
%         NMAE_round_error = (val/tot)/1.6;


        % Save the results.
        capName = dataSetName;
        capName(1) = upper(capName(1));
        
        saveResults = [capName,'_norm_',num2str(substract_mean),'_',num2str(q),'_',num2str(partNo),'_iters_',num2str(iters),'.mat'];
        disp(['Saving ... ',saveResults]);
        save(saveResults, 'model', 'L2_error','options','NMAE_error','NMAE_round_error');
    end
end

