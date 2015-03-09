% DEMMIXTOYDATA1 Demonstrate model on toy data.

% COLLAB

dataSetName = 'mixtoydata';
[Y, lbls, Ytest, X] = collabLoadData(dataSetName);

q = 2;
options = collabOptions;
options.kern = {'rbf', 'bias'}
options.numComps = 2;
model = collabCreate(q, size(Y, 2), Y, options);
options = collabOptimiseOptions();
options.momentum = 0.9;
options.learnRate = 0.0001;
options.paramMomentum = 0.9;
options.paramLearnRate = 0.0001;
options.numIters = 5;
model = collabOptimise(model, Y, options);
%model.X = X;
%model.kern.comp{1}.variance = 1;
%model.kern.comp{2}.variance = 0.4;
%model.sigma2 = 0.4;  
%model = collabEstep(model, 100);
