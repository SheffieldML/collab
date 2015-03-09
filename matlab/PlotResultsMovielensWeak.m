function [L2_error_T,NMAE_error_T,NMAE_round_error_T] = PlotResultsMovielensWeak(substract_mean,partNo_v,latentDim_v, iters, type)
%
  % [L2_error,NMAE_error,NMAE_round_error] = PlotResultsMovielensWeak(substract_mean,partNo_v,latentDim_v, iters)
%
% substract_mean --> bool if substract the mean
% partNo_v --> vector with the partitions to compute results
% latentDim_v --> vector with the latent dimensionalities to compute results
  % iters --> number of iterations
% type --> strong or weak

  numDim = length(latentDim_v);
numPartNo = length(partNo_v);

L2_error_T = -ones(numDim,numPartNo);
NMAE_error_T = -ones(numDim,numPartNo);
NMAE_round_error_T = -ones(numDim,numPartNo);

for i_latent=1:numDim
    q = latentDim_v(i_latent);
    for i_part=1:numPartNo
        partNo = partNo_v(i_part);

dataSetName = ['movielens_',type,'_',num2str(partNo)];
        


        % Save the results.
        capName = dataSetName;
        capName(1) = upper(capName(1));
        
loadResults = [capName,'_norm_',num2str(substract_mean),'_',num2str(q),'_',num2str(partNo),'_iters_',num2str(iters),'.mat'];
        disp(['Loading ... ',loadResults]);
try
        load(loadResults);
catch
continue;
end
L2_error_T(i_latent,i_part) = L2_error;
NMAE_error_T(i_latent,i_part) = NMAE_error;
NMAE_round_error_T(i_latent,i_part) = NMAE_round_error;
    end
end

% plot the results

mean_L2 = mean(L2_error_T,2);
mean_NMAE = mean(NMAE_error_T,2);
mean_NMAE_round = mean(NMAE_round_error_T,2);
%keyboard;
  for j=1:size(mean_L2,2)
    std_L2(j) = std(permute(L2_error_T(j,:),[2 1]));
std_NMAE(j) = std(permute(NMAE_error_T(j,:),[2 1]));
std_NMAE_round(j) = std(permute(NMAE_round_error_T(j,:),[2 1]));
end

%figure(1)
%  clf;
%hold on;
%for i=1:length(latentDim_v)
  % plot(perc_train_v/100,mean_NMAE_round(:,i),[getColor(i),'x']);
%errorbar(perc_train_v/100,mean_NMAE_round(:,i),std_NMAE_round(:,i),[getColor(i),'x']);
%toLeg{i} = ['Dimension ',num2str(latentDim_v(i))];
%end
%xlabel('percentage database');
%ylabel('NMAE round error');
%legend(toLeg);

end


function [value] = getColor(index)
switch index
  case 1
value = 'r-';
case 2
value = 'b-';
case 3
value = 'g--'
  case 4
value = 'm--';
case 5
value = 'k-'
end
end
