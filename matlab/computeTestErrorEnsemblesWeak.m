function [L2_error,NMAE_error,NMAE_round_error] = computeTestErrorEnsemblesWeak(allModels,Y,Ytest)
%
% [error_L2,error_NMAE,error_NMAE_round] = computeTestErrorEnsemblesWeak(allModels,Y,Ytest)

 
val_L2 = 0;
tot_L2 = 0;
val_NMAE = 0;
tot_NMAE = 0;
val_round_NMAE = 0;
tot_round_NMAE = 0;
accum = [];

for i = 1:size(Y, 2)       
    ind = find(Ytest(:, i));
elim = find(ind>size(allModels{1}.X, 1));
    tind = ind;
    tind(elim) = [];
mu_T = 0;
for j=1:length(allModels)
  [mu, varsig] = collabPosteriorMeanVar(allModels{j}, Y(:, i), allModels{j}.X(tind, :));
    % normalize the values

    if (length(mu)>0)
      mu = mu.*allModels{j}.sd(tind);
mu = mu+allModels{j}.mu(tind);
    end
	mu_T = mu_T + mu;
end
    mu_T = mu_T/length(allModels);
    a = Ytest(tind, i) - mu_T; 
    a = [a; Ytest(elim, i)];
    val_L2 = val_L2 + a'*a;
    tot_L2 = tot_L2 + length(a);
    val_NMAE = val_NMAE + sum(abs(a));
    tot_NMAE = tot_NMAE + length(a);
    val_round_NMAE = val_round_NMAE + sum(abs(round(a)));
    tot_round_NMAE = tot_round_NMAE + length(a);
    accum = [accum; abs(a)];
end
L2_error = sqrt(val_L2/tot_L2);
NMAE_error = (val_NMAE/tot_NMAE)/1.6;
NMAE_round_error = (val_round_NMAE/tot_round_NMAE)/1.6;
