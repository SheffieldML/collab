function[] = ComputeVarianceScatterPlot(substract_mean, partNo_v, latentDim_v, iters, inverted, type, kern_type)
% RECOMPUTERESULTS Try collaborative filtering on the large movielens data.
%
  % ComputeVarianceScatterPlot(substract_mean, partNo_v, latentDim_v, iters, inverted, type, kern_type)
%
% substract_mean --> bool if substract the mean
% partNo_v --> vector with the partitions to compute results
% latentDim_v --> vector with the latent dimensionalities to compute results
% iters
% inverted
% type is weak or strong
% kern_type

randn('seed', 1e5);
rand('seed', 1e5);

experimentNo = 3;


%partNo_v = [1:5];
%latentDim_v = [5, 2:4, 6];


for i_latent=1:length(latentDim_v)
    q = latentDim_v(i_latent);
    for i_part=1:length(partNo_v)
        partNo = partNo_v(i_part);


dataSetName = ['movielens_marlin_',type,'_',num2str(partNo)];
        
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
        
loadResults = [capName,'inverted_',num2str(inverted),'_norm_',num2str(substract_mean),'_',num2str(q),'_',num2str(partNo),'_iters_',num2str(iters),'.mat'];
        disp(['Loading ... ',loadResults]);
load(loadResults);
%, 'model', 'L2_error','options','NMAE_error','NMAE_round_error');
L2_error_before = L2_error
NMAE_error_before = NMAE_error
NMAE_round_error_before = NMAE_round_error

% recompute the results
if (strcmp(type,'weak'))

% compute the results
    disp('Computing the train, test results');
[L2_error_new,NMAE_error_new,NMAE_round_error_new,pred_L2, pred_r_NMAE,pred_var, users, perUser_var, perUser_L2, perUser_r_NMAE, numUsers] = computeMeanVarianceWeak(model,Y,Ytest);

% get the users for training
for i=1:size(Y,2)
  numUsers(i) = nnz(Y(:,i));
end



figure(1)
  clf;
  scatter(pred_r_NMAE,pred_var,3);

font_size = 16;
set(gca,'FontSize',font_size);
set(get(gca,'Title'),'FontSize',font_size);
set(get(gca,'Xlabel'),'FontSize',font_size);
set(get(gca,'Ylabel'),'FontSize',font_size);
				xlabel('NMAE error');
				ylabel('GP variance');

%				set(gca,'XTick',latentDim_v);


figure(2)
  clf;
scatter(sqrt(pred_L2), pred_var,3);
font_size = 16;
set(gca,'FontSize',font_size);
set(get(gca,'Title'),'FontSize',font_size);
set(get(gca,'Xlabel'),'FontSize',font_size);
set(get(gca,'Ylabel'),'FontSize',font_size);
				xlabel('RMSE error');
				ylabel('GP variance');

figure(3)
  clf;
scatter(numUsers, pred_r_NMAE,3);
font_size = 16;
set(gca,'FontSize',font_size);
set(get(gca,'Title'),'FontSize',font_size);
set(get(gca,'Xlabel'),'FontSize',font_size);
set(get(gca,'Ylabel'),'FontSize',font_size);
				xlabel('Number of ratings');
				ylabel('NMAE');



figure(4)
  clf;
scatter(numUsers, pred_var);
font_size = 16;
set(gca,'FontSize',font_size);
set(get(gca,'Title'),'FontSize',font_size);
set(get(gca,'Xlabel'),'FontSize',font_size);
set(get(gca,'Ylabel'),'FontSize',font_size);
				ylabel('NMAE error');




				[numUsers_sort, i_sort] = sort(numUsers);
pred_r_NMAE_sort  = pred_r_NMAE(i_sort);
pred_var_sort  = pred_var(i_sort);

numIntervals = 5;
[div_va] = numIntervals*ones(length(pred_var),1);
perval = length(pred_var)/numIntervals;
for i=1:numIntervals
  div_va(1+(i-1)*perval:i*perval) = i*ones(perval,1);
end

for i=1:numIntervals
  text_plot{i} = [num2str(numUsers_sort(1+(i-1)*perval)),'-',num2str(numUsers_sort(i*perval))];
end

  boxplot(pred_r_NMAE_sort,div_va,'labels',text_plot);
ylabel('NMAE error');



figure(5)
  clf;
scatter(numUsers, pred_var);
font_size = 16;
set(gca,'FontSize',font_size);
set(get(gca,'Title'),'FontSize',font_size);
set(get(gca,'Xlabel'),'FontSize',font_size);
set(get(gca,'Ylabel'),'FontSize',font_size);
				ylabel('GP variance');

boxplot(pred_var_sort,div_va,'labels',text_plot);


ylabel('GP variance');



disp('Saving the plots');

for i=1:5
  nameFile = ['scatter_',num2str(i)];
				figure(i)
				saveas(gcf,[nameFile,'.fig']);
				nameFile = [nameFile,'.eps'];
				print('-depsc',nameFile);
end 


keyboard;
end

  if (strcmp(type,'strong'))
    disp('WARNING: Strong partition not yet coded');
end


disp('Computing results');
        % Save the results.
%        capName = dataSetName;
%        capName(1) = upper(capName(1));
        
%        saveResults = [capName,'_norm_',num2str(substract_mean),'_',num2str(q),'_',num2str(partNo),'.mat'];
%        disp(['Saving ... ',saveResults]);
        %save(saveResults, 'model', 'L2_error','options','NMAE_error','NMAE_round_error');
    end
end

