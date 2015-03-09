% DEMMOVIELENSSMALLMIXFROMSINGLE1 Try collaborative filtering on the small movielens data.

% COLLAB

randn('seed', 1e5);
rand('seed', 1e5);

partNo_v = [1:5];
latentDim_v = [2 5 7 10];
numComps_v = [2];
experimentNo = 1;
numIters = 0;
numItersSingle = 10;
itersFinalEstep = 100;

error = 0;
for i_latent=1:length(latentDim_v)
    q = latentDim_v(i_latent);
    for i_part=1:length(partNo_v)
        partNo = partNo_v(i_part);
for i_comp = 1:length(numComps_v)
  
  dataSetName = ['movielensSmall' num2str(partNo)];
  [Y, void, Ytest] = collabLoadData(dataSetName);

  options = collabOptions;
  options.numComps = numComps_v(i_comp);
options.kern = {'rbf', 'bias'};%, 'white'};
  %/~
  %options.heteroNoise = true;
  %~/
  model = collabCreate(q, size(Y, 2), Y, options);
  %/~
  %model.diagvar = repmat(5.0, size(model.diagvar));
  %~/
  model.kern.comp{2}.variance = 0.11;
  model.sigma2 = 5;
  %model.kern.comp{3}.variance =  5; 
  options = collabOptimiseOptions;

  capName = dataSetName;
  capName(1) = upper(capName(1));
  options.saveName = ['dem' capName 'Mix' num2str(experimentNo) '_'];


  loadResults = [capName,'_',num2str(q),'_1_',num2str(partNo),'_iters_',num2str(numItersSingle),'.mat'];
  disp(['Loading ... ',loadResults]);

  % loading the model learn without a mixture
  model_single = load(loadResults);

model.X = model_single.model.X;
params_single = kernExtractParam(model_single.model.kern);
%model.kern = kernExpandParam(model.kern,params_single);
model = collabInitS(model);
model = collabUpdateKernels(model);

  
  % set parameters
  options.momentum = 0.9;
  options.learnRate = 0.0001;
  options.paramMomentum = 0.9;
  options.paramLearnRate = 0.0001;
  options.noiseMomentum = 0.9;
  options.noiseLearnRate = 0.0001;
  options.numIters = numIters;
  
options.numIters
  
  disp('Starting optimization');
  
 
  model = collabOptimise(model, Y, options);

  disp('Ending optimization');

keyboard;

% do an E-step
     model = collabUpdateKernels(model);
    disp(['Doing E-step ',num2str(itersFinalEstep)]);
model = collabEstep(model,itersFinalEstep);
  
keyboard;  
  
  disp('Computing error');


  [L2_error,NMAE_error,NMAE_round_error] = computeTestErrorWeak(model,Y,Ytest)
  [L2_error_all,NMAE_error_all,NMAE_round_error_all] = computeTestErrorWeakAllModes(model,Y,Ytest)
  [L2_error_best,NMAE_error_best,NMAE_round_error_best] = computeTestErrorWeakBestMode(model,Y,Ytest)

   
  capName = dataSetName;
  capName(1) = upper(capName(1));
  %save(['dem' capName 'Mix_' num2str(experimentNo) '.mat'], 'model', 'error');
saveResults = [capName,'_',num2str(q),'_',num2str(numComps_v(i_comp)),'_',num2str(partNo),'_iters_',num2str(numItersSingle),'_mix_',num2str(numComps_v(i_comp)),'_Estepiters_',num2str(itersFinalEstep),'_iters_',num2str(numItersSingle),'.mat'];
  disp(['Saving ... ',saveResults]);
  save(saveResults, 'model', 'L2_error','options','NMAE_error','NMAE_round_error', 'L2_error_best','NMAE_error_best','NMAE_round_error_best','L2_error_all','NMAE_error_all','NMAE_round_error_all');
  end
  end
end
