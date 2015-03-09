function [] = changeNameMovielens(mean_substraction, perc_train_v, latentDim_v, partNo_v, iters)
%
  % changeNameMovielens(mean_substraction, perc_train_v, latentDim_v, partNo_v, iters)

numDim = length(latentDim_v);
numPerc = length(perc_train_v);
numPart = length(partNo_v);

for i_perc=1:numPerc
  perc_train = perc_train_v(i_perc);
for i_latent=1:numDim
  latentDim = latentDim_v(i_latent);
for i_part = 1:numPart
  partNo = partNo_v(i_part);

toLoad = ['Movielens_',num2str(perc_train),'_',num2str(partNo),'_norm_',num2str(mean_substraction),'_',num2str(latentDim),'_',num2str(partNo),'.mat'];

toSave = ['Movielens_',num2str(perc_train),'_',num2str(partNo),'_norm_',num2str(mean_substraction),'_',num2str(latentDim),'_',num2str(partNo),'_iters_',num2str(iters),'.mat'];


disp(['Loading ... ',toLoad]);

try
load(toLoad)
catch
continue;
end

disp(['Saving ... ',toSave]);
save(toSave,'options','model','L2_error','NMAE_error','NMAE_round_error');

end
end
end
