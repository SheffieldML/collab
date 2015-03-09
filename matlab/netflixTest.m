% Load in the probe set and test netflix performance.

load /local/data/netFlixDataProbe.mat
counter = 0;
totalCount = 0;
totalse = 0;
totalse2 = 0;
totalMeanSe = 0;
rmseOne = spalloc(length(Y), 1, 480000);
rmseTwo = spalloc(length(Y), 1, 480000);
Ypred = cell(size(Yprobe, 1), 3);
for i = 1:length(Y)
  if ~isempty(Y{i, 1})
    counter = counter + 1;
    if counter > 1000
      break
    else
      if length(Yprobe{i, 1})>0
        ind = Y{i, 1};
        if length(ind)<3000
          yprime = (double(Y{i, 2}) - model.mu(ind))./model.sd(ind);
          K = kernCompute(model.kern, model.X(ind, :));
          invK = pdinv(K);
          
          testInd = Yprobe{i, 1};
          diagK = kernDiagCompute(model.kern, model.X(testInd, :));
          Kx = kernCompute(model.kern,model.X(ind, :), model.X(testInd, :));
          KinvK = invK*Kx;
          sd = model.sd(testInd);
          Ypred{i, 1} = (KinvK'*yprime).*sd + model.mu(testInd);
          
          Ypred{i, 2} = (diagK - sum(Kx.*KinvK, 1)').*sd.*sd;
          thisMu = Ypred{i, 1};
          thisSd = sqrt(Ypred{i, 2});
          a = (1-thisMu)./thisSd;
          b = (5-thisMu)./thisSd;
          Ypred{i, 3} = thisMu ...
              + (gaussOverDiffCumGaussian(b, a, 2) ...
              - gaussOverDiffCumGaussian(b, a, 1)).*thisSd;

          vals = double(Yprobe{i, 2}) - Ypred{i, 1};
          vals2 = double(Yprobe{i, 2}) - Ypred{i, 3};
          dum = double(Yprobe{i, 2}) - model.mu(testInd);

          dumValsSq = dum'*dum;
          vals2Sq = vals2'*vals2;
          valsSq = vals'*vals;

          rmse1(i) = sqrt(valsSq/length(vals));
          rmse2(i) = sqrt(vals2Sq/length(vals));
          rmseDum(i) = sqrt(dumValsSq/length(vals));
          totalMeanSe = totalMeanSe + dumValsSq; 
          totalCount = totalCount + length(vals);
          totalse = totalse+valsSq;
          totalse2 = totalse2+vals2Sq;
        end
      end
    end
  end
end
rmseCorrected= sqrt(totalse2/totalCount);
rmse = sqrt(totalse/totalCount);
mrmse = sqrt(totalMeanSe/totalCount);