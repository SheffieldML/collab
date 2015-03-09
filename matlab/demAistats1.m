% DEMAISTATS1 Try collaborative filtering on the Aistats Reviews

% COLLAB

randn('seed', 1e5);
rand('seed', 1e5);

experimentNo = 1;

dataSetName = 'aistats';
[Y, void, Ytest] = collabLoadData(dataSetName);

numPapers = size(Y,1);
numReviewers = size(Y,2);
meanPapers = zeros(numPapers,1);
stdPapers = ones(numPapers,1);

q = 2;
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
options.numIters = 20; % ??? put 10 back
options.showLikelihood = true;

capName = dataSetName;
capName(1) = upper(capName(1));
options.saveName = ['dem' capName num2str(experimentNo)];

model.mu = meanPapers;
model.sd = stdPapers;

model = collabOptimise(model, Y, options)



